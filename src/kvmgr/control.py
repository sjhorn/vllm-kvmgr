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
client certificate** -- which is why `/kv/reload` (a diagnostic, absent unless
`diag_hot_reload` is on) re-imports OUR modules only
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

from . import hooks

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

    def kvmgr_verify_presence(self, keys: "list[str] | None" = None) -> dict:
        """Does THIS rank hold each key's file, at exactly the right size?

        *** THIS IS THE HALF THE BOOT WALK CANNOT DO. *** The index is a claim
        about two filesystems; `_load_manifest`'s prune runs in the engine core
        and `get_file_name` is rank-dependent, so the 371s spent at every boot
        checks ONE node. A file lost on the other one is not a miss -- upstream
        asserts on a failed transfer, so it is a dead engine core.

        *** SIZE, NOT EXISTENCE, AND THE SYSCALL ALREADY RETURNS IT. *** The
        boot walk calls `os.path.exists`, which is a `stat` whose size is
        thrown away -- so a truncated or foreign-sized file passes the gate and
        dies at load as a `ValueError` on the worker's critical path
        (`read_chunks` refuses before reading a byte, which is right, but by
        then there is no safe answer left). The expected size is a property of
        the GROUP GEOMETRY, which this process holds: `group_file_sizes` is
        exactly what `read_chunks` itself computes.

        A wrong-sized file is UNLINKED here. It is provably unloadable, and
        `_write_batch`'s `exists()` skip would otherwise preserve it for ever --
        so leaving it would make the self-heal unable to heal.

        Returns hex keys, because an OffloadKey is raw bytes and this reply
        crosses msgpack. `free_bytes` rides along so the engine core's
        free-space floor can see THIS node's filesystem; it is the whole reason
        the keep-alive round exists.
        """
        import os as _os
        import time as _time

        out: dict = {"rank": getattr(self, "rank", None), "missing": [],
                     "wrong_size": [], "checked": 0, "ms": 0,
                     "free_bytes": None, "unlinked": 0}
        t0 = _time.monotonic()
        try:
            from . import diskio
            from vllm.v1.kv_offload.base import get_offload_group_idx

            h = None
            for cand in list(getattr(diskio, "_LIVE_HANDLERS", ())):
                if getattr(cand, "_mapper", None) is not None and \
                        getattr(cand, "_refs", None) is not None:
                    h = cand
                    break
            if h is None:
                # *** NOT AN EMPTY ANSWER. *** "No handler" must not read as
                # "nothing is missing" -- that is the manufactured zero this
                # repo keeps paying for, and here it would vouch for a whole
                # node on the strength of having failed to look.
                out["error"] = ("no live disk handler on this rank, so nothing "
                                "could be checked")
                return out
            sizes = diskio.group_file_sizes(h._refs)
            try:
                vfs = _os.statvfs(getattr(h._spec, "root_dir", None)
                                  or _os.path.dirname(h._mapper.get_file_name(
                                      bytes.fromhex((keys or ["00" * 32])[0]))))
                out["free_bytes"] = int(vfs.f_bavail) * int(vfs.f_frsize)
            except Exception:
                out["free_bytes"] = None
            for hexkey in (keys or ()):
                try:
                    k = bytes.fromhex(hexkey)
                except Exception:
                    out["missing"].append(hexkey)
                    continue
                out["checked"] += 1
                path = h._mapper.get_file_name(k)
                try:
                    stt = _os.stat(path)
                except OSError:
                    out["missing"].append(hexkey)
                    continue
                try:
                    g = int(get_offload_group_idx(k))
                    want = int(sizes[g])
                except Exception:
                    # A key naming a group this engine does not have is not
                    # something to guess about.
                    out["wrong_size"].append(hexkey)
                    continue
                if int(stt.st_size) != want:
                    out["wrong_size"].append(hexkey)
                    try:
                        _os.unlink(path)
                        out["unlinked"] += 1
                    except Exception:
                        pass
        except Exception as e:
            out["error"] = repr(e)
        out["ms"] = int((_time.monotonic() - t0) * 1000)
        return out

    Worker.kvmgr_verify_presence = kvmgr_verify_presence
    Worker.kvmgr_set_knob = kvmgr_set_knob
    Worker.kvmgr_clear_knob = kvmgr_clear_knob
    Worker.kvmgr_snapshot = kvmgr_snapshot
    Worker._kvmgr_control = True
    logger.info("kvmgr: control methods installed on Worker "
                "(set_knob, clear_knob, snapshot, verify_presence; reload only "
                "when diag_hot_reload installed it)")


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

    def _verify_pump(engine_core) -> None:
        """Drive the cross-rank presence verification, once per step.

        *** THIS RIDES THE EXISTING WRAP RATHER THAN ADDING AN INSTALLER, and
        *** that is deliberate on two counts. *** It avoids the installer
        audit (`register.py`'s `_install_*` table, which found two unarmed
        re-arms the moment it was written), and more importantly it inherits
        the hard-won answer to "which step entry point actually runs" -- the
        one this project got wrong twice, once by wrapping the function that
        loads the plugin and once by wrapping a `step` async scheduling never
        calls.
        
        A repeating `collective_rpc` from the scheduler step is exactly what
        `_enforce_budget`'s comment forbids ("the blocking-the-critical-path
        mistake that has already cost this project an NCCL kill") -- so note
        what makes this different: it is DEMAND-DRIVEN and bounded. With
        nothing deferring and the background rate at its default, it is one
        bounded RPC a minute; with a request deferring, it is one chunk per
        step until that request is answered. The alternative it replaces is a
        919,669-key walk, not nothing.

        It never raises EXCEPT on an RPC timeout, which is unrecoverable --
        see DiskPresenceManager.pump.
        """
        try:
            from . import diskspec
            m = diskspec._LIVE_MANAGER
            if m is None:
                return
        except Exception:
            return
        m.pump(engine_core)

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
            _verify_pump(self)
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
        # Hot reload is a diagnostic (kvmgr.diag.reload); a no-op unless on.
        hooks.call("reload")
    elif method == "kvmgr_verify_presence":
        # *** AN EXPLICIT NO-OP, BECAUSE THIS FUNCTION IGNORES UNKNOWN NAMES
        # *** SILENTLY AND A READER WOULD ASSUME IT RAN. *** The presence
        # verification is a question about a WORKER's filesystem: each rank
        # stats its own slice, and the engine core has no files of its own to
        # check -- the core's node is rank 0's node, already covered by rank
        # 0's own reply. Applying it locally would double-count rank 0 and
        # make `verify_verdict`'s rank arithmetic wrong.
        pass


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

    # ABSENT, not 403, unless the reload diagnostic installed itself -- the
    # same gate-at-attach shape as the plane itself. Defined HERE rather than
    # in kvmgr.diag.reload because FastAPI resolves `Request` against this
    # module's globals (the `from __future__ import annotations` trap).
    if hooks.active("reload"):
        @router.post("/kv/reload")
        async def kv_reload(raw_request: Request):
            out = await _fanout(raw_request, "kvmgr_reload", {})
            return JSONResponse(status_code=200 if out.get("ok") else 409,
                                content=out)

    # *** POST /kv/verify USED TO BE HERE, AND IT WAS STRUCTURALLY DEAD. ***
    #
    # It needs the disk index to re-queue keys, and the index lives in the
    # ENGINE CORE while this router is built and attached in the API SERVER
    # process -- so every call answered 409 "no disk manager in THIS process",
    # measured 2026-10-12. The same split makes `GET /kv` on :8888 report
    # `disk: null`, which this repo's digest already records as a field named
    # for a state it does not report.
    #
    # Routing it through `collective_rpc` like `/kv/knob` -- the obvious fix --
    # was rejected: the wrap applies `kvmgr_*` in the core and then forwards to
    # the workers, but it RETURNS the workers' replies, so the core's own
    # result is dropped on the floor. A lever that cannot report what it did is
    # how a refusal nobody can see becomes a capability nobody checked, and the
    # workers would need a no-op method purely to keep the fan-out from
    # raising.
    #
    # So it moved to the listener the engine core already owns, where the index
    # is a local variable: `POST 127.0.0.1:9199/kv/verify` (kvmgr/serve.py).
    # That is also this repo's stated Stage 6 rule -- hot levers go on
    # :9199, outside the traffic path, because Caddyfile's catch-all proxies
    # every unmatched path to :8888 and would expose a mutating route to any
    # holder of a client certificate.

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

        # *** THE SCHEDULER-SIDE GUARD IS NOT ENOUGH, AND THE DROP CRASHED THE
        # *** HEAD AGAIN ON 2026-10-09 AT 17:51 WITH THAT GUARD LIVE.
        #
        # Reproduced deliberately (local-ops/kvdroploop.sh): eight drop/restore
        # cycles at a 20K prefix were clean, and the FIRST cycle at 100K
        # faulted -- Xid 32, Worker_TP0, `unspecified launch failure` at
        # synchronize_input_prep, two seconds after a drop that logged
        # `reset OK`. So SCALE is the trigger, and the log named the cause in
        # two lines twenty seconds apart:
        #
        #   17:50:58 (Worker_TP0) store path busy -- 2 job(s) outstanding
        #   17:51:18 (EngineCore)  drop -- free 22280 -> 22755, reset OK
        #
        # manager.py refuses on `offload_sched_snapshot()`, which reads
        # `cs._jobs` -- the SCHEDULER's dict. But our handler queues a job of
        # any size and pumps it over ROUNDS from `get_finished()` (the chunking
        # fix Phase 1c needed), keeping its own `_jobs`, `_writing` and
        # `_reading` in the WORKER PROCESS. Those are the copies actually
        # touching GPU memory, and nothing scheduler-side can see them. This is
        # the file's own "idle by request count is not idle by transfer" trap
        # one layer deeper: idle by SCHEDULER JOB SET is not idle by WORKER
        # TRANSFER either, and each time the remedy has been to ask the thing
        # that actually knows.
        #
        # `kvmgr_snapshot` already returns both handlers' counts per rank and
        # `/kv/workers` already serves them, so nothing is being built here --
        # the quantity existed and the dangerous lever never asked for it.
        #
        # Refuse rather than wait, as manager.py does: a drop is an operator
        # action with a human behind it and the remedy is seconds of patience.
        # And refuse on an UNREADABLE answer too -- ABSENT IS NOT ZERO, and for
        # this lever being wrong costs a dead worker and a lost prefill, so
        # "I cannot tell" must never read as "nothing in flight".
        try:
            ranks = await engine_client(raw_request).collective_rpc(
                method="kvmgr_snapshot", args=(), kwargs={})
        except Exception as e:
            return JSONResponse(status_code=409, content={
                "ok": False, "refused": "worker_transfers_unknown",
                "error": str(e),
                "note": "Could not read the workers' handler state, so whether "
                        "a DMA is reading GPU blocks right now is UNKNOWN, not "
                        "zero. Refusing: this is the lever that killed a worker "
                        "twice. Retry, or look at /kv/workers."})
        # A RAISING GUARD MUST NOT BECOME A 500 -- and it must not become a
        # permitted drop either. Found by mutation M6 in
        # test_phase7_dropguard.py: breaking the non-dict-rank branch makes the
        # helper raise instead of returning, and an unhandled raise here would
        # surface as a server error on the one lever whose failure costs a
        # worker. Treat it as the strongest form of "I cannot tell".
        try:
            busy = _transfers_busy(ranks)
        except Exception as e:
            return JSONResponse(status_code=409, content={
                "ok": False, "refused": "worker_transfers_unreadable",
                "error": repr(e),
                "note": "Reading the per-rank transfer counts RAISED, so "
                        "whether a DMA is in flight is unknown. Refusing, for "
                        "the same reason as the unknown case above."})
        if busy:
            return JSONResponse(status_code=409, content={
                "ok": False, "refused": "worker_transfers_outstanding",
                "busy": busy,
                "note": "One or more RANKS still hold KV transfer jobs. A store "
                        "copies GPU -> pinned staging; free its source block "
                        "mid-copy and the DMA reads what is no longer there -- "
                        "Xid 32, with the traceback landing at "
                        "synchronize_input_prep, nowhere near the cause. Wait "
                        "for these to reach zero and bump again. The "
                        "scheduler-side count can be 0 while these are not, "
                        "which is exactly how 2026-10-09 17:51 happened."})

        nxt = int(flags.raw("gpu_cache_epoch") or 0) + 1
        out = await _fanout(raw_request, "kvmgr_set_knob",
                            {"name": "gpu_cache_epoch", "value": nxt})
        out["worker_transfers"] = "drained -- checked per rank before the bump"
        out["epoch"] = nxt
        out["note"] = (
            "The drop lands on the next SCHEDULER STEP, so an idle engine has "
            "not dropped anything yet -- send one request. The verdict ("
            "released N/M managers, free blocks before -> after, reset "
            "OK|REFUSED) is logged by the engine core. REFUSED means blocks "
            "are still referenced: let it go idle and bump again.")
        return JSONResponse(status_code=200, content=out)

    return router


# The counted fields, named once rather than copied. A handler holding ANY of
# these is touching GPU memory or about to: `jobs_open` is queued-or-running
# rounds, `writes_open` are store rounds whose file write has not finished (so
# their staging slots are still held), `reads_open` the same on the load side.
_BUSY_FIELDS = ("jobs_open", "writes_open", "reads_open")


def _transfers_busy(ranks) -> list:
    """Per-rank outstanding transfer counts, or an UNREADABLE marker.

    *** THE SHAPE IS A LIST UNDER "handlers", KEYED BY "cls", AND MY FIRST
    *** VERSION LOOKED FOR "disk_store"/"disk_load" DICT KEYS. ***
    It would have found nothing on every rank, returned empty, and let every
    drop through -- a refusal that cannot fire, which is this repo's
    invisible-icon failure in a new costume and precisely the class of bug this
    guard exists to prevent. Found by reading `kvmgr_snapshot` rather than
    assuming its shape, and pinned by test_phase7_dropguard.py, whose first
    mutation is exactly that wrong key.

    Pure, and separate from the route, because the one thing that must never
    happen is an empty return for any reason other than "every rank is idle".
    An unreadable rank -- a non-dict payload, a missing handler list, a non-int
    count -- is therefore reported as BUSY with `unreadable: True` rather than
    skipped. A rank we cannot parse is a rank we cannot clear.
    """
    busy: list = []
    if not isinstance(ranks, (list, tuple)):
        return [{"rank": None, "unreadable": True,
                 "why": "collective_rpc returned %s, not a per-rank list"
                        % type(ranks).__name__}]
    if not ranks:
        return [{"rank": None, "unreadable": True,
                 "why": "collective_rpc returned NO ranks, which cannot mean "
                        "that every rank is idle"}]
    for i, r in enumerate(ranks):
        if not isinstance(r, dict):
            busy.append({"rank": i, "unreadable": True,
                         "why": "rank payload is %s, not a dict"
                                % type(r).__name__})
            continue
        rank = r.get("rank", i)
        handlers = r.get("handlers")
        if handlers is None:
            # kvmgr_snapshot records a failure in place of the list, so this is
            # "we could not look", never "there are no handlers".
            busy.append({"rank": rank, "unreadable": True,
                         "why": "no 'handlers' in the rank snapshot (%s)"
                                % (r.get("handlers_error") or "no reason given")})
            continue
        if not isinstance(handlers, (list, tuple)):
            busy.append({"rank": rank, "unreadable": True,
                         "why": "'handlers' is %s, not a list"
                                % type(handlers).__name__})
            continue
        counts: dict = {}
        for h in handlers:
            if not isinstance(h, dict):
                busy.append({"rank": rank, "unreadable": True,
                             "why": "a handler entry is %s, not a dict"
                                    % type(h).__name__})
                continue
            cls = h.get("cls") or "handler"
            for f in _BUSY_FIELDS:
                if f not in h:
                    continue
                v = h.get(f)
                # bool is an int subclass, so True here would be a SHAPE change
                # reported as a quantity. Unreadable, not 1.
                if isinstance(v, bool) or not isinstance(v, int):
                    busy.append({"rank": rank, "unreadable": True,
                                 "why": "%s.%s is %r, not an int" % (cls, f, v)})
                    continue
                if v:
                    counts["%s.%s" % (cls, f)] = v
        if counts:
            busy.append({"rank": rank, "counts": counts})
    return busy


def _apply_locally_reported(method: str, kwargs: dict) -> dict:
    if method == "kvmgr_set_knob":
        return apply_knob(kwargs.get("name"), kwargs.get("value"))
    if method == "kvmgr_clear_knob":
        return clear_knob(kwargs.get("name"))
    if method == "kvmgr_reload":
        out = hooks.first("reload")
        if out is None:
            return {"ok": False, "error": "hot reload is off (diag_hot_reload)"}
        return out
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
        "port. GET /kv, POST /kv/knob, /kv/knob/clear, /kv/drop_gpu_cache "
        "(and /kv/reload when diag_hot_reload is on) are therefore reachable "
        "by ANY holder of a client "
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
