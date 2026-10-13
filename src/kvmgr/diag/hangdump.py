"""kvmgr.diag.hangdump -- every thread's Python stack, with no ptrace.

Moved verbatim from kvmgr.register on 2026-10-13. ON by default
(diag_hang_dump): it costs nothing until a signal arrives, and diagnosing
a wedged worker without it cost two wedges.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)


def install() -> None:
    """Make a HANG dumpable with no ptrace, because twice now it was not.

    *** THE PROBLEM THIS SOLVES. *** The store path has wedged the worker twice
    (Phase 1c, and again 2026-10-07 at the staging-saturation point). Both times
    the only evidence available was "the worker stopped consuming shm broadcasts"
    plus thread wchans, because the container has no `SYS_PTRACE` -- `gdb -p`
    answers `ptrace: Operation not permitted` and there is no py-spy. So a
    deadlock in OUR code was diagnosable only by reading it and guessing, which
    is how the Phase 1c attempt came to be built on a wrong premise.

    `faulthandler` needs no ptrace and no attach: the signal handler runs inside
    the process and walks every thread's Python stack itself. Output goes to
    stderr, which is what `docker logs` shows, so the dump lands beside the log
    lines that led up to it.

        docker exec <c> kill -USR1 <worker pid>   # every thread's Python stack

    SIGUSR1 rather than SIGABRT or SIGQUIT: those two kill the process, and a
    wedged engine holding a 700K-token prefix is worth more alive than dead --
    the dump is for diagnosis, not for a postmortem. Registered per process from
    the plugin, so the worker, the engine core and the API server all answer it.

    *** `chain=True` KILLS THE PROCESS, AND THE TEST IS WHAT FOUND IT. ***
    faulthandler chains to whatever handler was installed when it registered,
    and for SIGUSR1 that is normally `SIG_DFL` -- whose default action is
    TERMINATE. So the obvious spelling dumps the stack and then kills the
    engine, which on a box holding a 700K-token prefix would destroy the thing
    being diagnosed in the act of diagnosing it. Measured offline: the child
    exited with `-30`, i.e. died of its own diagnostic signal.

    So chain only to a REAL handler -- one somebody actually installed -- and
    never to the default. Checked in the running image 2026-10-07: vLLM
    installs no SIGUSR1 or SIGUSR2 handler anywhere (0 hits across its tree),
    so in practice nothing is chained and nothing is stolen; the check is there
    so that a future vLLM taking SIGUSR1 keeps working rather than losing it.

    The whole thing is best-effort: an instrument that can break the boot is
    worse than no instrument.
    """
    try:
        import faulthandler
        import signal
        import sys as _sys

        if not hasattr(signal, "SIGUSR1"):
            return
        # *** THE PASSIVE NET, AND ITS ABSENCE COST US THE 2026-10-08
        # *** TRACEBACK. `register(SIGUSR1)` only answers a signal somebody
        # SENDS, so it is useless for a process that dies on its own -- and on
        # 2026-10-08 `Worker proc VllmWorker-0 died unexpectedly` with NO Python
        # traceback anywhere, which is the signature of a fatal signal. There
        # was no kernel log either (dmesg needs root here), so the cause is
        # still unknown and cannot now be recovered.
        #
        # `faulthandler.enable()` installs handlers for SIGSEGV, SIGBUS,
        # SIGFPE, SIGABRT and SIGILL, dumps every thread's Python stack to
        # stderr -- i.e. into `docker logs` -- and then RE-RAISES, so the
        # process still dies exactly as it would have. It is a witness, not a
        # mitigation: it changes nothing about behaviour and cannot keep a
        # broken process alive.
        #
        # Unlike register(), chaining is not a hazard here: enable() always
        # re-raises on the default handler for a genuinely fatal signal, which
        # is the correct outcome. The register() trap below is specifically
        # about SIGUSR1, whose default action is terminate.
        faulthandler.enable(file=_sys.stderr, all_threads=True)

        existing = signal.getsignal(signal.SIGUSR1)
        chain = callable(existing) and existing not in (
            signal.SIG_DFL, signal.SIG_IGN)
        faulthandler.register(signal.SIGUSR1, file=_sys.stderr,
                              all_threads=True, chain=chain)
        logger.info(
            "kvmgr: hang dump armed -- faulthandler.enable() is the PASSIVE "
            "net (a fatal signal now dumps every thread's Python stack to the "
            "container log and then re-raises, so a death is no longer "
            "silent), and `kill -USR1 <pid>` is the ACTIVE one for a process "
            "that is wedged but alive. No ptrace needed, which is the point: "
            "gdb cannot attach in this container and the store path has wedged "
            "the worker twice with no stack available. chain=%s (a real "
            "handler was already installed: %s).",
            chain, chain,
        )
    except Exception:
        logger.exception("kvmgr: could not arm the hang dump (continuing)")
