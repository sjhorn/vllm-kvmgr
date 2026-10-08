"""kvmgr.control -- the HTTP control plane for the hot knobs.

*** WHY THIS EXISTS: A HOT KNOB MUST NOT DEPEND ON DISK OR CONTAINER
INJECTION. *** `/var/tmp/vllm-kv-flags/flags.json` works and is why every
policy knob is hot today, but as the only way in it is a bind mount (so the
control plane depends on the container spec), it is PER NODE (writing it on the
head does not change the worker), it returns no acknowledgement, and it is not
atomic -- three processes each re-stat it on their own 1s TTL, so one change
lands at three different instants. The file is demoted to the BOOT-TIME
DEFAULT, which is a real job: a restart has to start somewhere.

SHAPE COPIED FROM vLLM'S OWN DEV ROUTES (`entrypoints/serve/*/api_router.py`),
read rather than guessed:

  - one `APIRouter()` per concern, with `attach_router(app)`;
  - THE GATE IS AT ATTACH TIME, so off means the route is ABSENT rather than
    403 -- which is this repo's own `METER_TAP_PORT` convention ("off means
    nothing is bound") arrived at independently by upstream;
  - POST for mutation, query params for scalars, a JSON body for structure;
  - dependencies off `request.app.state`, never module globals;
  - only JSON-simple args cross the RPC boundary, as upstream puts it "for
    security reason".

TWO CAVEATS IN THEIR CODE WE DELIBERATELY DO NOT COPY. Upstream's
`/reset_prefix_cache` says *"we currently do not check if the prefix cache is
successfully reset"* and `/sleep` says *"FIXME: the sleep command is sent but
does not finish yet when we return a response"*. Both are fire and forget.
**Ours returns the verdict**, per rank, because a knob that silently did
nothing is the failure this repo keeps paying for -- and because the whole
reason for a control plane over a file is that a file cannot answer.

*** AND WE SHARE THE ENGINE'S PORT, WHICH IS A DELIBERATE, TEMPORARY RISK. ***
`Caddyfile:116` is a bare `handle {}` catch-all that reverse-proxies every
unmatched path to `[meter 9098, tunnel 8888]`, so these routes are reachable
through the mTLS front door by any holder of a client certificate. That is
accepted for now, on the operator's explicit decision, so the plane can be
tested and exercised; the intended fix is a path gate in Caddy rather than a
second listener. **Until that lands, treat every route here as reachable by any
client certificate** -- which is why `/kv/reload` re-imports OUR modules only
and there is deliberately no general `exec`-shaped route, unlike upstream's
`/collective_rpc`.

THREE PROCESSES READ THESE KNOBS and that is the whole difficulty: the API
server (where the route runs), the engine core (manager and policy), and both
workers (the disk handlers). So a set fans out:

    route (API server)            -> set_override locally
      engine_client.collective_rpc("kvmgr_set_knob", ...)
        EngineCore.collective_rpc  -> WRAPPED: applies locally, then forwards
          executor                 -> both workers' kvmgr_set_knob

One wrap and one worker method covers all three. No new listener, no IPC of our
own, and the answer comes back per rank -- which the file can never give.
"""

from __future__ import annotations

import importlib
from typing import Any

from vllm.logger import init_logger

# `vllm.` prefix: a logger named for this module alone has no handler and an
# effective level of 30, so every info() would be silently dropped.
logger = init_logger("vllm." + __name__)

_RPC_PREFIX = "kvmgr_"

# *** THESE IMPORTS MUST BE MODULE-LEVEL, AND IT IS `from __future__ import
# annotations` THAT MAKES IT SO. *** Measured on the box 2026-10-08: with
# `Request` imported inside `_router()`, all five routes registered and every
# one answered
#
#   400  1 validation error: {'loc': ('query', 'raw_request'),
#                             'msg': 'Field required'}
#
# because the future import turns every annotation into a STRING, and FastAPI
# resolves it with `get_type_hints()` against the function's MODULE GLOBALS --
# not its closure. So `raw_request: Request` read as an unresolvable name and
# FastAPI fell back to treating it as a query parameter. The routes existed and
# could not be called: a mechanism present, correct-looking and producing
# nothing, which is this repo's signature failure in a new costume.
#
# Guarded because the offline harness has no fastapi, and the whole value of
# the pure functions above is that they are testable without it. `_router()`
# is never reached when this fails, so a missing fastapi costs the routes and
# nothing else.
try:
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse

    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover - the laptop harness
    APIRouter = Request = JSONResponse = None  # type: ignore[assignment]
    _HAVE_FASTAPI = False


# --- the per-process primitive, called in the workers over RPC --------------


def apply_knob(name: str, value: Any) -> dict:
    """Set one knob in THIS process. The body of the worker-side RPC method."""
    from . import flags

    try:
        act = flags.set_override(name, value)
    except (KeyError, ValueError) as e:
        return {"ok": False, "name": name, "error": str(e)}
    act["ok"] = True
    act.update(_derived(name))
    return act


def clear_knob(name: str | None) -> dict:
    """Drop one override, or all, in THIS process."""
    from . import flags

    try:
        dropped = flags.clear_override(name)
    except KeyError as e:
        return {"ok": False, "error": str(e)}
    out = {"ok": True, "cleared": dropped}
    if name:
        out["now"] = flags.raw(name)
        out["source"] = flags.source_of(name)
    return out


def _derived(name: str) -> dict:
    """The DERIVED figure for a knob, not just the value echoed back.

    A response that repeats what you sent tells you the request parsed, which
    is not the question. For a cap-shaped knob the useful answer is the cap in
    blocks AND tokens -- and this repo already has the case where a hot knob
    read on an IDLE engine shows the OLD derived figure, because the reconcile
    runs in `new_step_starts()`. So the figure is reported with a note rather
    than silently, and a stale one reads as stale instead of as a knob that
    failed to take.
    """
    if name not in ("protected_fraction", "protect_alloc"):
        return {}
    try:
        from .manager import DSparkMLAAttentionManager as M

        mgrs = tuple(getattr(M, "all_managers", ()) or ())
        caps = []
        for i, m in enumerate(mgrs):
            cap = getattr(m, "_cap_blocks_cached", None)
            if cap is None and hasattr(m, "_max_protected_prompt_blocks"):
                try:
                    cap = m._max_protected_prompt_blocks()
                except Exception:
                    cap = None
            bs = getattr(m, "block_size", None)
            caps.append({"group": i, "cap_blocks": cap,
                         "cap_tokens": (cap * bs) if (cap and bs) else None})
        return {"derived": caps,
                "derived_note": (
                    "The reconcile runs in new_step_starts(), so on an IDLE "
                    "engine these are the PREVIOUS figures until one request "
                    "goes through. A stale figure here is not a knob that "
                    "failed to take.")}
    except Exception:
        return {}


# --- rebinding live instances, which is what makes a reload possible --------


def live_instances() -> dict[str, list]:
    """Every object a reload must re-point, by role.

    Gathered from the two registries that already exist for other reasons --
    `all_managers` (stashed by the coordinator wrap for the pool-wide cache
    drop) and diskio's handler list -- rather than a registry invented for
    this, because a second registry is a second thing to forget to populate.
    """
    out: dict[str, list] = {"managers": [], "disk_manager": [], "handlers": []}
    try:
        from .manager import DSparkMLAAttentionManager as M

        out["managers"] = list(getattr(M, "all_managers", ()) or ())
    except Exception:
        pass
    try:
        from . import diskspec

        if diskspec._LIVE_MANAGER is not None:
            out["disk_manager"] = [diskspec._LIVE_MANAGER]
    except Exception:
        pass
    try:
        from . import diskio

        out["handlers"] = list(getattr(diskio, "_LIVE_HANDLERS", ()) or ())
    except Exception:
        pass
    return out


def rebind(obj) -> str:
    """Point `obj` at the freshly imported version of its own class.

    *** `obj.__class__ = NewClass` IS THE WHOLE TRICK, and it is better than
    rebinding methods one at a time: *** every method moves at once, including
    ones added since, and the instance keeps all of its state. Per-method
    rebinding silently misses any method the reload ADDED, which is exactly the
    edit a reload is usually for.

    What it CANNOT do is give the instance attributes the new code expects and
    the old `__init__` never set. So a reload is for POLICY LOGIC, and the
    response names every object it re-pointed so a caller can see the blast
    radius rather than infer it.
    """
    cls = type(obj)
    # *** ONE UNRELOADABLE CLASS MUST NOT FAIL THE WHOLE RELOAD. *** A class
    # defined somewhere `importlib` cannot re-import -- `__main__`, an exec'd
    # namespace, a C extension -- raises here, and raising made `ok: False` for
    # the entire operation even though every other instance re-pointed fine.
    # That is a refusal with the wrong scope: the useful answer is "these
    # moved, this one could not, and here is why".
    try:
        mod = importlib.import_module(cls.__module__)
        importlib.reload(mod)
    except Exception as e:
        return f"{cls.__name__}: NOT reloadable ({cls.__module__}: {e})"
    new = getattr(mod, cls.__name__, None)
    if new is None or new is cls:
        return f"{cls.__name__}: unchanged"
    obj.__class__ = new
    return f"{cls.__name__}: rebound"


def reload_modules() -> dict:
    """Re-import our modules and re-point every live instance. 6d."""
    report: dict[str, Any] = {"reloaded": [], "rebound": [], "errors": []}

    # *** SNAPSHOT THE LIVE INSTANCES BEFORE RELOADING ANYTHING, because the
    # reload ERASES THE REGISTRY IT IS ABOUT TO READ. *** Measured on the box
    # 2026-10-08: the first live `/kv/reload` reported `instances: []` on both
    # ranks while a store handler was demonstrably running. `importlib.reload`
    # re-executes the module, so `_LIVE_HANDLERS: list = []` runs again and the
    # list is empty by the time `live_instances()` looks -- the handler objects
    # are still alive and serving, and nothing can find them to re-point.
    #
    # Same class as the dashboard's priming code being a destructive overwrite:
    # a step that rebuilds the thing it then consumes. Gathering first costs
    # one dict and makes the ordering impossible to get wrong by accident.
    targets = live_instances()

    # *** AND SNAPSHOT THE OVERRIDES, FOR THE SAME REASON ONE LINE UP. ***
    # `importlib.reload` re-executes the module, so `flags._OVERRIDE: dict = {}`
    # runs again and EVERY knob set through the control plane is silently
    # discarded. Measured on the box 2026-10-08: the boot sync pushed 12 knobs
    # to both workers, `POST /kv/reload` ran, and the ranks were split again --
    # rank 1 back on its stale file, because the reload had thrown away the
    # override that fixed it.
    #
    # The same destructive-re-execution bug as `_LIVE_HANDLERS`, in the same
    # function, which I fixed for the handlers and did not carry across. The
    # general rule for this module: a reload must restore every module-level
    # piece of STATE it re-initialises, and the only safe way to know is to
    # enumerate them here.
    saved_overrides: dict = {}
    try:
        from . import flags as _f

        saved_overrides = dict(_f._OVERRIDE)
    except Exception as e:
        report["errors"].append(f"snapshotting overrides: {e}")

    # Order matters: leaves before the modules that import them, so a reloaded
    # parent picks up the reloaded child rather than the stale one it holds.
    for name in ("kvmgr.flags", "kvmgr.policy", "kvmgr.diskio",
                 "kvmgr.diskspec", "kvmgr.manager", "kvmgr.stats",
                 "kvmgr.control"):
        try:
            import sys

            if name in sys.modules:
                importlib.reload(sys.modules[name])
                report["reloaded"].append(name)
        except Exception as e:
            report["errors"].append(f"{name}: {e}")
    for role, objs in targets.items():
        for obj in objs:
            try:
                report["rebound"].append(f"{role}/{rebind(obj)}")
            except Exception as e:
                report["errors"].append(f"{role}: {e}")
    # Put the handlers back, because the reload emptied the list they register
    # themselves into and nothing else ever will -- they register in their own
    # constructors, and these objects are already constructed.
    try:
        from . import diskio

        diskio._LIVE_HANDLERS[:] = targets.get("handlers", [])
    except Exception as e:
        report["errors"].append(f"re-registering handlers: {e}")
    # Put the overrides back, or a reload quietly reverts every knob to the
    # file -- which on this cluster means reverting the WORKER to a stale file
    # and re-splitting the ranks.
    try:
        from . import flags as _f2

        _f2._OVERRIDE.update(saved_overrides)
        report["overrides_restored"] = sorted(saved_overrides)
    except Exception as e:
        report["errors"].append(f"restoring overrides: {e}")
    report["found"] = {k: len(v) for k, v in targets.items()}
    report["ok"] = not report["errors"]
    logger.info("kvmgr: reload -- %d module(s), %d instance(s), %d error(s)",
                len(report["reloaded"]), len(report["rebound"]),
                len(report["errors"]))
    return report


# --- the worker-side methods, patched onto Worker --------------------------


def install_worker_methods() -> None:
    """Add `kvmgr_*` methods to the Worker class, for `collective_rpc`.

    `MultiprocExecutor.collective_rpc` resolves a STRING method name on the
    worker object, so a method on the class is all an RPC target needs. Patched
    rather than subclassed because the worker class is chosen by vLLM's own
    executor and substituting it is a far larger change than adding two
    methods.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:
        logger.exception("kvmgr: could not import Worker; the control plane "
                         "will not reach the workers")
        return
    if getattr(Worker, "_kvmgr_control", False):
        return

    def kvmgr_set_knob(self, name: str, value: Any) -> dict:
        out = apply_knob(name, value)
        out["rank"] = getattr(self, "rank", None)
        return out

    def kvmgr_clear_knob(self, name: str | None = None) -> dict:
        out = clear_knob(name)
        out["rank"] = getattr(self, "rank", None)
        return out

    def kvmgr_reload(self) -> dict:
        out = reload_modules()
        out["rank"] = getattr(self, "rank", None)
        return out

    def kvmgr_snapshot(self) -> dict:
        from . import flags

        out = {"rank": getattr(self, "rank", None),
               "knobs": {n: flags.raw(n) for n in flags.DEFAULTS},
               "sources": {n: flags.source_of(n) for n in flags.DEFAULTS}}
        # *** THE HANDLER STATS LIVE HERE AND NOWHERE ELSE. *** The store and
        # load handlers are constructed in the WORKER; `/kv` runs in the API
        # server, so its disk section can only ever report the API server's
        # (empty) view. The fence counters -- wait_calls, wait_blocked_ms,
        # wait_max_pending -- are the numbers that say whether moving the fence
        # from disk to staging is worth the design change, and this RPC is the
        # only way they reach a caller.
        try:
            from . import diskio

            out["handlers"] = [
                dict(h.snapshot(), cls=type(h).__name__)
                for h in getattr(diskio, "_LIVE_HANDLERS", ())
            ]
        except Exception as e:
            out["handlers_error"] = str(e)
        return out

    Worker.kvmgr_set_knob = kvmgr_set_knob
    Worker.kvmgr_clear_knob = kvmgr_clear_knob
    Worker.kvmgr_reload = kvmgr_reload
    Worker.kvmgr_snapshot = kvmgr_snapshot
    Worker._kvmgr_control = True
    logger.info("kvmgr: control methods installed on Worker "
                "(set_knob, clear_knob, reload, snapshot)")


# --- the engine-core hook, so the core applies a knob too ------------------


def wrap_engine_core_rpc() -> None:
    """Make `EngineCore.collective_rpc` apply `kvmgr_*` LOCALLY, then forward.

    *** THE ENGINE CORE IS NOT A WORKER, and that is the gap this closes. ***
    `MultiprocExecutor.collective_rpc` broadcasts to the WORKERS; the engine
    core process itself never runs the method. But the engine core is where our
    managers live and where `protect_alloc`, `evict_order` and
    `protected_fraction` are read -- so without this hook a POST would change
    the workers' view of a knob and not the core's, which is the worst possible
    outcome: a knob half-applied, with nothing saying which half.

    One wrap covers it, because `EngineCore.collective_rpc` is the single place
    the core delegates to the executor. Only `kvmgr_`-prefixed methods are
    touched; everything else passes straight through untouched.
    """
    try:
        from vllm.v1.engine.core import EngineCore
    except Exception:
        logger.exception("kvmgr: could not import EngineCore; a knob set over "
                         "HTTP will reach the workers but NOT the scheduler")
        return
    original = EngineCore.collective_rpc
    if getattr(original, "_kvmgr_wrapped", False):
        return

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if isinstance(method, str) and method.startswith(_RPC_PREFIX):
            try:
                _apply_locally(method, args, kwargs or {})
            except Exception:
                logger.exception(
                    "kvmgr: applying %s in the engine core failed; the workers "
                    "will still be asked, so expect a SPLIT view", method)
        return original(self, method, timeout=timeout, args=args,
                        kwargs=kwargs)

    collective_rpc._kvmgr_wrapped = True  # type: ignore[attr-defined]
    EngineCore.collective_rpc = collective_rpc  # type: ignore[method-assign]
    logger.info("kvmgr: EngineCore.collective_rpc wrapped, so a knob set over "
                "HTTP reaches the scheduler as well as both workers")


def sync_knobs_to_workers(engine_core) -> dict:
    """Push the ENGINE CORE's resolved knobs to every worker, once, at boot.

    *** THE FLAGS FILE IS PER NODE AND THE WORKER'S GOES STALE, which was
    measured rather than feared. *** 2026-10-08, after a `/kv/knob/clear`,
    rank 0 reported `source=file` and rank 1 reported `source=DEFAULT`:

        head    Oct 8 07:59   12 keys   disk_store_paused=1
        worker  Oct 7 20:29    9 keys   disk_store_paused ABSENT

    So every knob set through the file reached rank 0 only. The consequence
    that matters: a deploy made "safe" by pausing the disk store path had it
    paused on rank 0 and RUNNING on rank 1 -- the half-applied knob this whole
    phase exists to prevent.

    The control plane fixes a knob set over HTTP, because that fans out. It
    does NOT fix the BOOT defaults, which each node reads from its own file --
    so without this, every restart begins with the two ranks disagreeing and
    the only symptom is behaviour that differs by node.

    So the head's file becomes the single source of truth for the cluster: at
    boot the core pushes every knob the file or the env actually SET, as an
    override, to both workers. Knobs at their default are deliberately NOT
    pushed -- sending them would make `source_of()` report `override` for a
    knob nobody set, which trades a split view for a misleading one.

    Best-effort and late: the executor exists by the end of `EngineCore
    .__init__`, and a failure here costs the sync and nothing else -- the
    engine serves, and the knobs are still settable over HTTP.
    """
    from . import flags

    pushed, errors = [], []
    for name in flags.DEFAULTS:
        try:
            src = flags.source_of(name)
            if src not in ("file", "env"):
                continue
            value = flags.raw(name)
            engine_core.collective_rpc(
                "kvmgr_set_knob", args=(), kwargs={"name": name,
                                                   "value": value})
            pushed.append(f"{name}={value!r}({src})")
        except Exception as e:
            errors.append(f"{name}: {e}")
    if pushed or errors:
        logger.info(
            "kvmgr: boot knob sync -- pushed %d knob(s) to the workers so both "
            "ranks start from the HEAD's file rather than their own: %s%s",
            len(pushed), pushed,
            f"; {len(errors)} error(s): {errors}" if errors else "")
    return {"pushed": pushed, "errors": errors}


def wrap_engine_core_init() -> None:
    """Fan the boot knobs out ONCE, on whichever step entry point actually runs.

    *** TWO HOOKS WRONG IN A ROW, AND THE EVIDENCE WAS ALREADY IN HAND BOTH
    TIMES. ***

    Attempt 1 wrapped `EngineCore.__init__`, whose FIRST STATEMENT is
    `load_general_plugins()` -- us (`v1/engine/core.py:106`). A wrap installed
    from inside the plugin lands on the class attribute while the only call
    that matters is already executing the original frame. **You cannot wrap the
    function that loads you.**

    Attempt 2 wrapped `EngineCore.step`, which this engine never calls. The
    busy loop goes through `self.step_fn()`, a reference CAPTURED at
    construction and set to `step_with_batch_queue` under async scheduling --
    which this engine runs. The crash traceback from earlier the same day
    showed `_process_engine_step()` calling `self.step_fn()` in as many words,
    so the right name was already on screen and I wrapped the plausible one
    instead.

    So: wrap EVERY candidate, sharing ONE one-shot flag, and let whichever runs
    first do the work. Three names rather than one is not belt-and-braces for
    its own sake -- a scheduling-mode change must not be able to silently
    disable this, and the symptom when it does is behaviour differing by node
    with nothing in any log.
    """
    installed = []
    try:
        from vllm.v1.engine.core import EngineCore, EngineCoreProc
    except Exception:
        logger.exception("kvmgr: no EngineCore to hook; the boot knob sync is "
                         "off and the ranks may start from different files")
        return

    def _oneshot(original):
        def wrapper(self, *a, **kw):
            if not getattr(self, "_kvmgr_synced", False):
                # Set FIRST: a raising sync must not become a retry on every
                # step forever. A missed sync that says so beats a hot path
                # that keeps trying.
                self._kvmgr_synced = True
                try:
                    sync_knobs_to_workers(self)
                except Exception:
                    logger.exception(
                        "kvmgr: the boot knob sync failed, so the ranks may be "
                        "starting from DIFFERENT files. Knobs set over HTTP "
                        "still fan out; only the boot defaults may disagree.")
            return original(self, *a, **kw)

        wrapper._kvmgr_sync = True  # type: ignore[attr-defined]
        return wrapper

    for cls, name in ((EngineCore, "step"),
                      (EngineCore, "step_with_batch_queue"),
                      (EngineCoreProc, "_process_engine_step")):
        try:
            original = getattr(cls, name, None)
            if original is None or getattr(original, "_kvmgr_sync", False):
                continue
            setattr(cls, name, _oneshot(original))
            installed.append(f"{cls.__name__}.{name}")
        except Exception:
            logger.exception("kvmgr: could not wrap %s.%s", cls.__name__, name)
    logger.info(
        "kvmgr: boot knob sync armed on %s -- one shot, whichever runs first. "
        "NOT __init__, which is what loads this plugin, and not `step` alone, "
        "which async scheduling never calls.", installed)


def _apply_locally(method: str, args: tuple, kwargs: dict) -> None:
    if method == "kvmgr_set_knob":
        name = kwargs.get("name", args[0] if args else None)
        value = kwargs.get("value", args[1] if len(args) > 1 else None)
        apply_knob(name, value)
    elif method == "kvmgr_clear_knob":
        clear_knob(kwargs.get("name", args[0] if args else None))
    elif method == "kvmgr_reload":
        reload_modules()


# --- the router, attached to the engine's own app -------------------------


def _router():
    """The router. Uses the MODULE-LEVEL fastapi names, deliberately -- see the
    note at those imports for the 400 that teaches why a local import here
    cannot work under `from __future__ import annotations`."""
    if not _HAVE_FASTAPI:
        raise RuntimeError(
            "kvmgr: fastapi is not importable, so the control plane has no "
            "router. Nothing else is affected.")
    router = APIRouter()

    def engine_client(request: Request):
        # Off `app.state`, as upstream's routers do, never a module global.
        return request.app.state.engine_client

    async def _fanout(request: Request, method: str, kwargs: dict) -> dict:
        """Apply here, then to the core and both workers, and REPORT EACH.

        The local apply comes first and separately because the API server
        process reads these knobs too (nothing policy-shaped yet, but the
        precedence must not differ by process). The engine core applies itself
        inside the wrap; the workers answer over RPC.
        """
        local = _apply_locally_reported(method, kwargs)
        ranks: Any
        try:
            ranks = await engine_client(request).collective_rpc(
                method=method, args=(), kwargs=kwargs)
        except Exception as e:
            # A failed fan-out is the dangerous case -- the knob may be set
            # HERE and not there, which is a split view. Say so loudly rather
            # than returning a 200 with a local success in it.
            return {"ok": False, "api_server": local,
                    "error": f"fan-out to the workers failed: {e}",
                    "warning": ("the knob may now differ between the API "
                                "server and the workers; re-issue or clear it")}
        return {"ok": True, "api_server": local, "ranks": ranks}

    @router.get("/kv")
    async def kv_state(raw_request: Request):
        from . import flags, stats

        snap = stats.snapshot()
        snap["sources"] = {n: flags.source_of(n) for n in flags.DEFAULTS}
        return JSONResponse(content=snap)

    @router.post("/kv/knob")
    async def kv_knob(raw_request: Request):
        body = await raw_request.json()
        name, value = body.get("name"), body.get("value")
        if name is None:
            return JSONResponse(status_code=400, content={
                "ok": False,
                "error": "missing 'name'",
                "known": sorted(__import__(
                    "kvmgr.flags", fromlist=["DEFAULTS"]).DEFAULTS)})
        out = await _fanout(raw_request, "kvmgr_set_knob",
                            {"name": name, "value": value})
        return JSONResponse(status_code=200 if out.get("ok") else 409,
                            content=out)

    @router.post("/kv/knob/clear")
    async def kv_knob_clear(raw_request: Request):
        name = raw_request.query_params.get("name")
        out = await _fanout(raw_request, "kvmgr_clear_knob", {"name": name})
        return JSONResponse(status_code=200 if out.get("ok") else 409,
                            content=out)

    @router.get("/kv/workers")
    async def kv_workers(raw_request: Request):
        """Per-rank knobs, sources and HANDLER stats, straight off the workers.

        `GET /kv` can only report the API server's view, and the handlers --
        with the fence counters on them -- live in the workers. This is the
        only route that can see them.
        """
        try:
            ranks = await engine_client(raw_request).collective_rpc(
                method="kvmgr_snapshot", args=(), kwargs={})
        except Exception as e:
            return JSONResponse(status_code=409,
                                content={"ok": False, "error": str(e)})
        return JSONResponse(content={"ok": True, "ranks": ranks})

    @router.post("/kv/reload")
    async def kv_reload(raw_request: Request):
        out = await _fanout(raw_request, "kvmgr_reload", {})
        return JSONResponse(status_code=200 if out.get("ok") else 409,
                            content=out)

    @router.post("/kv/drop_gpu_cache")
    async def kv_drop(raw_request: Request):
        """Bump `gpu_cache_epoch`, which is the existing live-drop mechanism.

        Deliberately a knob bump rather than a direct call: the drop acts on
        the flag-change EDGE in `new_step_starts()`, which is also why it needs
        a scheduler step to land -- and why the response says so rather than
        implying the cache is already gone. Upstream's own
        `/reset_prefix_cache` has the same property and does NOT say so
        ("we currently do not check if the prefix cache is successfully
        reset"); ours does, and the verdict appears in the engine log with the
        free-block movement and `reset OK` / `reset REFUSED`.
        """
        from . import flags

        nxt = int(flags.raw("gpu_cache_epoch") or 0) + 1
        out = await _fanout(raw_request, "kvmgr_set_knob",
                            {"name": "gpu_cache_epoch", "value": nxt})
        out["epoch"] = nxt
        out["note"] = (
            "The drop lands on the next SCHEDULER STEP, so an idle engine has "
            "not dropped anything yet -- send one request. The verdict ("
            "released N/M managers, free blocks before -> after, reset "
            "OK|REFUSED) is logged by the engine core. REFUSED means blocks "
            "are still referenced: let it go idle and bump again.")
        return JSONResponse(status_code=200, content=out)

    return router


def _apply_locally_reported(method: str, kwargs: dict) -> dict:
    if method == "kvmgr_set_knob":
        return apply_knob(kwargs.get("name"), kwargs.get("value"))
    if method == "kvmgr_clear_knob":
        return clear_knob(kwargs.get("name"))
    if method == "kvmgr_reload":
        return reload_modules()
    return {"ok": False, "error": f"unknown method {method!r}"}


def attach_router(app) -> None:
    """Attach ours to the engine's app, gated AT ATTACH TIME.

    Upstream's own shape (`entrypoints/serve/rpc/api_router.py`):

        def attach_router(app: FastAPI):
            if not envs.VLLM_SERVER_DEV_MODE:
                return
            app.include_router(router)

    So when it is off the routes are ABSENT, not 403 -- the same statement
    `METER_TAP_PORT` makes by binding nothing, and a stronger one than a
    boolean refusing. Ours gates on `kvmgr_control_plane`, read from the flags
    file at ATTACH time, which is the one knob that genuinely cannot be hot:
    a route either exists in the app or it does not, and the app is built once.
    """
    from . import flags

    if not flags.raw("kvmgr_control_plane"):
        logger.info("kvmgr: control plane NOT attached (kvmgr_control_plane is "
                    "off, so the routes are absent rather than refusing). This "
                    "is the one knob that cannot be hot -- the app is built "
                    "once, so enabling it needs a restart.")
        return
    try:
        app.include_router(_router())
    except Exception:
        logger.exception("kvmgr: could not attach the control plane router")
        return
    logger.warning(
        "kvmgr: SECURITY -- the control plane is ATTACHED to the engine's own "
        "app, and Caddyfile's catch-all proxies every unmatched path to this "
        "port. GET /kv, POST /kv/knob, /kv/knob/clear, /kv/reload and "
        "/kv/drop_gpu_cache are therefore reachable by ANY holder of a client "
        "certificate until a path gate lands in Caddy. Accepted deliberately "
        "while this is being exercised.")


def install_router_hook() -> None:
    """Attach ours whenever vLLM registers its own serve routers.

    `api_server.py:182` does the import INSIDE the function that calls it, so
    the lookup happens at call time against the module -- which is why patching
    the module attribute works even though our plugin loads earlier.
    """
    try:
        import vllm.entrypoints.serve as serve
    except Exception:
        logger.exception("kvmgr: could not import vllm.entrypoints.serve; the "
                         "control plane has nothing to attach to")
        return
    original = serve.register_vllm_serve_api_routers
    if getattr(original, "_kvmgr_wrapped", False):
        return

    def register(app):
        out = original(app)
        try:
            attach_router(app)
        except Exception:
            logger.exception("kvmgr: attaching the control plane failed; the "
                             "engine is unaffected and serves normally")
        return out

    register._kvmgr_wrapped = True  # type: ignore[attr-defined]
    serve.register_vllm_serve_api_routers = register  # type: ignore[assignment]
