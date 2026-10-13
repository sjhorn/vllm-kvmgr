"""kvmgr.compat.dspark -- the bridge to the DSpark fork's patch6.

patch6 (`vllm/v1/core/single_type_kv_cache_manager.py` in the DSpark recipe)
carries three mechanisms that are NOT kvmgr -- MNT-6b SWA recycle, MNT-8
free-queue prepend, MNT-9 static-head pin -- and the flags-file reader kvmgr
used to import. kvmgr runs without it now. When it IS present, this module does
four things, and nothing at all otherwise:

1. **One cache over one file.** patch6's ``_flags`` is pointed at
   ``kvmgr.flags.file_flags``, so the two readers cannot see the file at
   different moments on different TTLs.
2. **The override reaches the fork's knobs.** patch6's ``_flag`` consults
   kvmgr's in-memory override first. Before this bridge, ``POST /kv/knob
   pin_static_head=1`` answered ok, read back as ``source=override`` -- and
   patch6, reading env and file only, never saw it: a knob reported as set
   that governed nothing. ``_protected_fraction`` is pointed at
   ``flags.raw`` for the same reason, so MNT-6a's own path (``kvmgr_enabled``
   off) honours an override too.
3. **The fork's knobs are declared**, so ``/kv`` lists them and the
   control plane accepts them. On a stock vLLM they are absent rather than
   listed as knobs that govern nothing.
4. **patch6's pending-blocks hook gets kvmgr's provider**, so releasing a
   protected block skips blocks with a KV transfer in flight.

``install()`` is idempotent and reload-safe: patch6's original readers are
saved once, on patch6's own module, and a second install never wraps a wrapper.
"""

from __future__ import annotations

import sys
from typing import Any

from vllm.logger import init_logger

from .. import flags

logger = init_logger("vllm." + __name__)

STM = "vllm.v1.core.single_type_kv_cache_manager"

# The fork's knobs, with patch6's own defaults. Read only by patch6/patch8.
FORK_DEFAULTS: dict[str, Any] = {
    "swa_recycle": True,
    "swa_recycle_cached": False,
    "free_queue_prepend_uncached": False,
    "pin_static_head": False,
    "head_pin_fraction": 0.05,
    # Read by patch6's `_maybe_pin_static_head` with an inline default and
    # declared nowhere until 2026-10-09 -- found by auditing reads against
    # declarations, which is why the declaration lives with the bridge now.
    "head_pin_partial": False,
}

_INSTALLED_ON: object | None = None


def detect():
    """patch6's module if it is the one loaded, else None.

    Detected by its flag reader, which stock vLLM does not have. Imports the
    module, which is safe: vLLM has always imported it before any plugin can
    build a manager.
    """
    mod = sys.modules.get(STM)
    if mod is None:
        try:
            import importlib

            mod = importlib.import_module(STM)
        except Exception:
            return None
    if callable(getattr(mod, "_flag", None)) and callable(getattr(mod, "_flags", None)):
        return mod
    return None


def _bridged_flag(original):
    def _flag(name: str, default):
        if name in flags._OVERRIDE:
            value = flags._OVERRIDE[name]
            if isinstance(default, bool):
                return bool(value)
            try:
                return type(default)(value)
            except (TypeError, ValueError):
                return default
        return flags._env_or_file(name, default)

    _flag.__doc__ = ("kvmgr.compat.dspark bridge over patch6's _flag: the "
                     "in-memory override first, then env, then the file.")
    return _flag


def install() -> bool:
    """Bridge patch6 if present. Returns True when the fork was found."""
    global _INSTALLED_ON
    stm = detect()
    if stm is None:
        return False

    flags.declare(FORK_DEFAULTS, "the DSpark fork bridge (patch6)")

    # The originals are saved ONCE, on patch6's module, so a second install
    # (register() twice, or a reload of this module) never wraps a wrapper.
    if not hasattr(stm, "_kvmgr_originals"):
        stm._kvmgr_originals = {
            "_flag": stm._flag, "_flags": stm._flags,
            "_protected_fraction": stm._protected_fraction}
    stm._flag = _bridged_flag(stm._kvmgr_originals["_flag"])
    stm._flags = flags.file_flags
    stm._protected_fraction = lambda: float(flags.raw("protected_fraction"))

    setter = getattr(stm, "set_pending_block_ids_provider", None)
    if setter is not None:
        try:
            from .. import diskspec

            setter(diskspec.pending_block_ids)
        except Exception:
            logger.exception("kvmgr: could not install patch6's pending-blocks "
                             "provider (release will not consult transfers)")

    if _INSTALLED_ON is not stm:
        logger.info(
            "kvmgr: DSpark fork detected (patch6) -- its flag reader now shares "
            "kvmgr's cache and honours overrides; fork knobs declared: %s; "
            "pending-blocks provider %s",
            sorted(FORK_DEFAULTS),
            "installed" if setter is not None else "not offered by this patch6")
    _INSTALLED_ON = stm
    return True


def uninstall() -> None:
    """Restore patch6's own readers. For tests that prove the bridge matters."""
    global _INSTALLED_ON
    stm = detect()
    if stm is None or not hasattr(stm, "_kvmgr_originals"):
        return
    for name, fn in stm._kvmgr_originals.items():
        setattr(stm, name, fn)
    _INSTALLED_ON = None
