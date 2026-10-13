"""kvmgr.diag.reload -- re-import kvmgr inside a running engine.

A restart of this engine costs minutes of boot plus a cold prefill, so being
able to swap kvmgr's code in place was worth building -- and every defect it
has had was a piece of state the reload destroyed (module globals, class
attributes, closures held by other modules). ``_PRESERVE`` and
``_PRESERVE_CLASS`` are the tables of that state; tests audit the package
against them.

Moved from kvmgr.control on 2026-10-13. OFF by default (``diag_hot_reload``):
with it off, ``Worker.kvmgr_reload`` is not installed and ``POST /kv/reload``
does not exist. Core reaches it only through the ``reload`` hook.
"""

from __future__ import annotations

import importlib
from typing import Any

from vllm.logger import init_logger

from .. import hooks

logger = init_logger("vllm." + __name__)


# --- rebinding live instances, which is what makes a reload possible --------


# *** EVERY MODULE-LEVEL PIECE OF STATE A RELOAD WOULD DESTROY, AS A TABLE. ***
# `importlib.reload` re-executes the module, so anything initialised at module
# level is reset and whatever it held is gone. This bug has now appeared FOUR
# times -- `diskio._LIVE_HANDLERS` and `flags._OVERRIDE` found live within
# minutes of each other, `diskspec._LIVE_MANAGER` found by reading, and
# `stats._groups` / `offload._real_odirect` / `serve._server` found by the
# audit that replaced the hand-written list.
#
# A table rather than three more try/excepts, because the thing that kept
# failing was *remembering*, and `test_phase6` audits the kvmgr modules against
# this table: a new stateful global is an assertion failure, not something
# somebody notices afterwards.
#
# `stats._groups` is the one worth naming: it holds every per-group hit/miss/
# evict counter, so a reload without it silently zeroes all observability --
# the instrument resetting itself, which is the failure this project keeps
# paying for in other costumes.
_PRESERVE: tuple[tuple[str, str, str], ...] = (
    ("kvmgr.diskio", "_LIVE_HANDLERS", "the live store/load handler objects"),
    # The watchdog's cross-thread fail request, and its DEDUPE. A reload
    # resetting it to None would re-raise the loud "FAILING THE STALLED JOB"
    # line for a stall already acted on -- self-healing within 5s rather than
    # dangerous, but a duplicated fatal-looking log line is exactly the sort of
    # noise that makes an incident harder to read. The audit in test_phase6
    # caught this the moment the global was added, which is what the table is
    # for; it is in here rather than exempted because preserving a float costs
    # nothing and needs no argument.
    ("kvmgr.diskio", "_STALL_FAIL_REQUEST",
     "the watchdog's stall-fail request + its dedupe key"),
    ("kvmgr.flags", "_OVERRIDE", "every knob the control plane has set"),
    # Every diagnostic's registration. A reload re-executes hooks.py, and an
    # empty table would switch every diagnostic off while /kv still listed it.
    ("kvmgr.hooks", "_HOOKS", "the diagnostics registered into core"),
    # Derived from VLLM_KV_FLAGS_FILE at import, so a reload normally re-derives
    # the same path -- preserved anyway, because a path set at RUNTIME (the
    # test harness does) would otherwise snap back to /etc silently.
    ("kvmgr.flags", "FLAGS_PATH", "the flags file this process reads"),
    ("kvmgr.diskspec", "_LIVE_MANAGER", "the disk index, for GET /kv"),
    ("kvmgr.stats", "_groups", "all per-group counters -- the observability"),
    ("kvmgr.diag.tiering", "_real_odirect", "the saved platform O_DIRECT value"),
    ("kvmgr.serve", "_server", "the stats HTTP server handle"),
    # *** AND THE LOCK THAT GUARDS IT, because a reload rebinds it. *** A
    # fresh Lock means a thread holding the OLD one no longer excludes anybody,
    # so start and stop could both believe they hold it. Found by auditing the
    # reload list rather than by a failure: `offload` and `serve` were never
    # reloaded at all, so their _PRESERVE entries protected state across a
    # reload that did not happen -- which is itself the tell.
    ("kvmgr.serve", "_lock", "the lock guarding the server handle"),
    # A reload re-executes the module, so this would go back to False and
    # _start_watchdog() would start a SECOND thread on the next handler --
    # every reload leaking one more, each dumping the same stacks. Fifth
    # instance of the reload-destroys-module-state family, and the first one
    # caught by remembering the table exists rather than by a live failure.
    ("kvmgr.diskio", "_watchdog_started", "the stall watchdog's one-shot flag"),
    # Published by the __init__ wrap in register.py, which runs ONCE when the
    # connector is built -- so a reload is the only way to lose it and nothing
    # re-derives it. Same shape as all_managers, one layer out.
    ("kvmgr.diskspec", "_LIVE_OFFLOAD_SCHED",
     "upstream's offload scheduler -- the open job set behind the fence"),
    ("kvmgr.diskspec", "_SCHED_FOUND_BY", "how that handle was obtained"),
    ("kvmgr.diskspec", "_ARRIVAL",
     "completion-arrival counters -- losing them restarts the measurement"),
    ("kvmgr.diskio", "_LIVE_CONN_WORKER",
     "upstream's connector worker -- the unshipped completion queue"),
    # *** A FAULT MUST NOT BE CLEARABLE BY A RELOAD. *** It means a rank could
    # not be asked whether it holds a key's file, so presence refuses for the
    # rest of the boot; a reload re-executing the module would set it back to
    # None and silently re-enable claims nobody can vouch for. It clears only
    # on an explicit reset, which is the whole design.
    ("kvmgr.diskspec", "_VERIFY_FAULT",
     "the cross-rank verification fault -- a rank that could not answer"),
    # *** THE HANDOFF, AND LOSING IT LOSES A REQUEST SILENTLY. *** The listener
    # thread rebinds `_VERIFY_REQUEST` and the scheduler thread takes it; a
    # reload between those two moments re-executes the module and sets it back
    # to None, so an operator's `POST /kv/verify` would 202 and then never
    # happen -- the worst shape available, because the caller was told yes.
    ("kvmgr.diskspec", "_VERIFY_REQUEST",
     "the listener's pending re-verify request, not yet applied"),
    # The outcome, which is the only thing that makes the 202 checkable.
    ("kvmgr.diskspec", "_VERIFY_LAST",
     "what the pump did with the last request -- the 202's receipt"),
    # Monotonic. A reload restarting it at 0 would hand out a seq already in
    # `_VERIFY_LAST`, so a caller polling for its own request could match
    # somebody else's receipt.
    ("kvmgr.diskspec", "_VERIFY_SEQ", "the request sequence, so a seq is unique"),
)


# *** AND CLASS ATTRIBUTES ARE A SECOND KIND OF RELOAD CASUALTY. ***
# `_PRESERVE` above covers module globals. These live on a CLASS, and a reload
# creates a NEW class object, so they vanish just as silently -- measured live
# 2026-10-08, when a reload left `all_managers` unset and the next GPU cache
# drop logged "no manager list stashed, so only THIS group can be released" and
# released 1 of 5. The drop still said `reset OK` because the pool happened to
# be empty, so a broken lever reported success.
#
# They are stamped by the coordinator wrap in register.py, which runs ONCE at
# boot -- so nothing re-derives them and a reload is the only way to lose them.
_PRESERVE_CLASS: tuple[tuple[str, str, str, str], ...] = (
    ("kvmgr.manager", "DSparkMLAAttentionManager", "all_managers",
     "every group's manager -- without it the GPU cache drop releases 1 of 5"),
    ("kvmgr.manager", "DSparkMLAAttentionManager", "group_count",
     "the real group count from the KVCacheConfig"),
    ("kvmgr.manager", "DSparkMLAAttentionManager", "group_block_sizes",
     "per-group block sizes, which by_need sizes caps from"),
    ("kvmgr.manager", "DSparkMLAAttentionManager", "group_sliding_windows",
     "which groups are windowed -- the by_need discriminator"),
)


def _snapshot_class_state() -> tuple[dict, list[str]]:
    """Grab the class attributes in `_PRESERVE_CLASS` before the reload."""
    import importlib as _il
    saved, errs = {}, []
    for mod, cls, attr, _why in _PRESERVE_CLASS:
        try:
            k = getattr(_il.import_module(mod), cls, None)
            if k is not None and hasattr(k, attr):
                saved[(mod, cls, attr)] = getattr(k, attr)
        except Exception as exc:
            errs.append("snapshot %s.%s.%s: %r" % (mod, cls, attr, exc))
    return saved, errs


def _restore_class_state(saved: dict) -> tuple[list[str], list[str]]:
    import importlib as _il
    done, errs = [], []
    for (mod, cls, attr), val in saved.items():
        try:
            k = getattr(_il.import_module(mod), cls, None)
            if k is None:
                errs.append("restore %s.%s: class is gone" % (mod, cls))
                continue
            setattr(k, attr, val)
            done.append("%s.%s" % (cls, attr))
        except Exception as exc:
            errs.append("restore %s.%s.%s: %r" % (mod, cls, attr, exc))
    return done, errs


def _snapshot_state() -> tuple[dict, list[str]]:
    """Grab everything in `_PRESERVE` before the reload wipes it."""
    import importlib as _il

    saved, errs = {}, []
    for mod, attr, _why in _PRESERVE:
        try:
            m = _il.import_module(mod)
        except Exception:
            # Not importable in this process (or in the offline harness) is
            # not an error: there is simply nothing to save. Reporting it
            # would make a healthy reload read as failed.
            continue
        try:
            saved[(mod, attr)] = getattr(m, attr)
        except Exception as e:
            errs.append(f"snapshot {mod}.{attr}: {e}")
    return saved, errs


def _restore_state(saved: dict) -> tuple[list[str], list[str]]:
    """Put it all back, in place where the object identity matters."""
    import importlib as _il

    done, errs = [], []
    for mod, attr, _why in _PRESERVE:
        if (mod, attr) not in saved:
            continue
        try:
            m = _il.import_module(mod)
            old = saved[(mod, attr)]
            cur = getattr(m, attr, None)
            # A list or dict is mutated IN PLACE, because other modules may
            # already hold a reference to the new (empty) container --
            # rebinding the name would leave them looking at the empty one.
            if isinstance(old, list) and isinstance(cur, list):
                cur[:] = old
            elif isinstance(old, dict) and isinstance(cur, dict):
                cur.update(old)
            else:
                setattr(m, attr, old)
            done.append(f"{mod.split('.')[-1]}.{attr}")
        except Exception as e:
            errs.append(f"restore {mod}.{attr}: {e}")
    return done, errs


def live_instances() -> dict[str, list]:
    """Every object a reload must re-point, by role.

    Gathered from the two registries that already exist for other reasons --
    `all_managers` (stashed by the coordinator wrap for the pool-wide cache
    drop) and diskio's handler list -- rather than a registry invented for
    this, because a second registry is a second thing to forget to populate.
    """
    out: dict[str, list] = {"managers": [], "disk_manager": [], "handlers": []}
    try:
        from ..manager import DSparkMLAAttentionManager as M

        out["managers"] = list(getattr(M, "all_managers", ()) or ())
    except Exception:
        pass
    try:
        from .. import diskspec

        if diskspec._LIVE_MANAGER is not None:
            out["disk_manager"] = [diskspec._LIVE_MANAGER]
    except Exception:
        pass
    try:
        from .. import diskio

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
    saved_state, _serrs = _snapshot_state()
    report["errors"].extend(_serrs)
    saved_cls, _cerrs = _snapshot_class_state()
    report["errors"].extend(_cerrs)

    # Order matters: leaves before the modules that import them, so a reloaded
    # parent picks up the reloaded child rather than the stale one it holds.
    # *** `kvmgr.register` WAS MISSING FROM THIS LIST, so no change to it was
    # *** EVER hot. *** `import_module("kvmgr.register")` below returns
    # whatever is in sys.modules, which for an un-reloaded module is the OLD
    # one -- so the re-arm calls below ran the OLD installers and logged
    # nothing. The hang dump and watchdog re-arms hid it perfectly, because
    # they are idempotent no-ops whose old and new versions behave identically;
    # it only surfaced when a CHANGE to register.py had to take effect. Three
    # reloads reported success while running the previous code.
    # The reloader reloads ITSELF LAST: the running frame keeps the old code,
    # so a change to this function takes effect on the NEXT reload.
    for name in ("kvmgr.hooks", "kvmgr.flags", "kvmgr.policy", "kvmgr.diskio",
                 "kvmgr.diskspec", "kvmgr.manager", "kvmgr.stats",
                 "kvmgr.diag.hangdump", "kvmgr.diag.ledger",
                 "kvmgr.diag.serializer", "kvmgr.diag.geometry",
                 "kvmgr.diag.tiering", "kvmgr.serve", "kvmgr.compat.dspark",
                 "kvmgr.register", "kvmgr.control", "kvmgr.diag.reload"):
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
        from .. import diskio

        diskio._LIVE_HANDLERS[:] = targets.get("handlers", [])
    except Exception as e:
        report["errors"].append(f"re-registering handlers: {e}")
    # Put it all back. See _PRESERVE for why this is a table.
    _restored, _rerrs = _restore_state(saved_state)
    report["errors"].extend(_rerrs)
    report["state_restored"] = _restored
    _crestored, _crerrs = _restore_class_state(saved_cls)
    report["errors"].extend(_crerrs)
    report["class_state_restored"] = _crestored
    report["overrides_restored"] = sorted(saved_state.get(
        ("kvmgr.flags", "_OVERRIDE"), {}))
    report["found"] = {k: len(v) for k, v in targets.items()}

    # *** RE-ARM THE CRASH WITNESS, because otherwise a hot deploy cannot
    # *** install one. `_install_hang_dump` runs from `register()` at plugin
    # load, so faulthandler.enable() -- the thing whose absence lost us the
    # 2026-10-08 traceback -- would otherwise only arrive on the next engine
    # start, i.e. after the very crash it exists to witness. A reload fans out
    # to both ranks and the API server, which is exactly the reach needed.
    #
    # Safe to repeat: faulthandler.enable() and .register() are both idempotent,
    # and _install_hang_dump is best-effort by construction. Reported so a
    # reload that could NOT arm it says so rather than implying it did.
    try:
        # *** `from kvmgr import register` GIVES THE FUNCTION, NOT THE MODULE. ***
        # __init__.py does `from .register import register`, so the package
        # attribute `register` is the entry-point FUNCTION and it shadows the
        # submodule of the same name. The obvious spelling therefore raised
        # AttributeError("'function' object has no attribute
        # '_install_hang_dump'") on all three processes -- found live, and only
        # because this is a WARNING: the reload still reported ok=True and named
        # the cause, instead of a green reload that had silently armed nothing.
        # import_module always resolves the module, whatever the package
        # re-exports.
        import importlib as _il2
        _reg = _il2.import_module("kvmgr.register")
        report["diagnostics_rearmed"] = _reg._install_diagnostics()
        report["hang_dump_rearmed"] = bool(
            report["diagnostics_rearmed"].get("hangdump"))
        # *** AND START THE WATCHDOG, because a RELOAD NEVER RUNS A
        # *** CONSTRUCTOR. `_start_watchdog()` is called from the store/load
        # handler __init__, and rebind() re-points a live object with
        # `obj.__class__ = new` -- no __init__ anywhere. So a hot deploy of the
        # watchdog would have installed the code and armed nothing, which is
        # this repo's signature failure: a mechanism present, correct-looking
        # and producing nothing. Idempotent via _watchdog_started, which is in
        # _PRESERVE, so a reload on an already-watched process is a no-op.
        # *** AND RE-INSTALL THE SERIALIZER SHIM, which was missing here and
        # *** made the shim un-updatable without a restart. *** The shim is
        # installed from `register()`, which a reload never calls, so a hot
        # deploy of a change to it installed the code and changed nothing --
        # the signature failure this file's own comment names three lines up,
        # committed in the same function. It cost two four-minute restarts in
        # one session. Safe to call repeatedly: the installer unwraps its own
        # previous wrapper before re-wrapping, and the test that proves it
        # replaced rather than stacked is in test_phase1d_disk.py.
        _reg._install_serializer_shim()
        report["serializer_shim_reinstalled"] = True
        # *** AND RE-WIRE THE FORK BRIDGE. *** patch6's pending-blocks hook
        # holds a reference to `diskspec.pending_block_ids`, and its `_flag`
        # bridge closes over `flags` -- both REBOUND by the reload, so patch6
        # would keep calling the previous modules' functions. They would still
        # work, which is the trap: a stale closure that behaves correctly is
        # invisible until the behaviour changes. The installer audit in
        # test_phase1d_disk.py is what requires this line.
        _reg._install_fork_bridges()
        report["fork_bridges_rewired"] = True
        _dio = _il2.import_module("kvmgr.diskio")
        _dio._start_watchdog()
        report["watchdog_armed"] = bool(_dio._watchdog_started)
    except Exception as exc:
        # A WARNING, NOT AN ERROR, and the distinction is the point: the
        # reload itself succeeded, and failing to arm a diagnostic must never
        # be reported as a failed code deploy. Putting it in `errors` made
        # `ok` False for a reload that had worked perfectly -- an instrument
        # breaking the thing it measures, which is the rule this file already
        # carries twice. `ok` means "the reload worked"; the warning says the
        # witness is not armed, which is a separate fact and still visible.
        report["hang_dump_rearmed"] = False
        report.setdefault("warnings", []).append(
            "could not re-arm the crash witness (faulthandler): %r -- the "
            "reload itself was fine; a fatal signal will not leave a stack "
            "until the next engine start" % (exc,))

    report["ok"] = not report["errors"]
    logger.info("kvmgr: reload -- %d module(s), %d instance(s), %d error(s)",
                len(report["reloaded"]), len(report["rebound"]),
                len(report["errors"]))
    return report


def install() -> None:
    """Register the reload with core, and give the Worker its RPC target."""
    hooks.register("reload", reload_modules)
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:
        return  # not a worker-capable process; the hook is enough here

    def kvmgr_reload(self) -> dict:
        out = reload_modules()
        out["rank"] = getattr(self, "rank", None)
        return out

    Worker.kvmgr_reload = kvmgr_reload
