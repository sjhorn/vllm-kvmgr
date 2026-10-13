"""kvmgr.hooks -- where diagnostics plug in, so core never imports them.

Core calls a hook by name; a diagnostic in ``kvmgr.diag`` registers a function
for it when its knob is on. With nothing registered a hook is a no-op, which is
the whole contract: turning a diagnostic off can never change what core does.

    hooks.register("ledger_note", fn)   # diag, at install
    hooks.call("ledger_note", h, ...)   # core: every registered fn, results list
    hooks.first("ledger_snapshot", h)   # core: the first non-None result, or None

A hook that raises is logged and skipped: an instrument must never take down
the path it observes.
"""

from __future__ import annotations

from typing import Any, Callable

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_HOOKS: dict[str, list[Callable]] = {}


def register(name: str, fn: Callable) -> None:
    """Add ``fn`` under ``name``, replacing one with the same qualified name
    (so a re-install after a reload swaps rather than stacks)."""
    key = getattr(fn, "__module__", ""), getattr(fn, "__qualname__", repr(fn))
    fns = [f for f in _HOOKS.get(name, [])
           if (getattr(f, "__module__", ""), getattr(f, "__qualname__", repr(f))) != key]
    fns.append(fn)
    _HOOKS[name] = fns


def unregister(name: str) -> None:
    _HOOKS.pop(name, None)


def active(name: str) -> bool:
    return bool(_HOOKS.get(name))


def call(name: str, *args: Any, **kwargs: Any) -> list:
    out = []
    for fn in list(_HOOKS.get(name, ())):
        try:
            out.append(fn(*args, **kwargs))
        except Exception:
            logger.exception("kvmgr: diagnostic hook %r raised (skipped)", name)
    return out


def first(name: str, *args: Any, **kwargs: Any) -> Any:
    for value in call(name, *args, **kwargs):
        if value is not None:
            return value
    return None
