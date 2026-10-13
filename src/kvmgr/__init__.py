"""kvmgr -- a KV cache manager of our own for the DSpark engine.

Phase 1: group 0's manager, behaviour-identical to patch6, with every policy
knob hot.  See docs/history/KVMGR-PHASE0-20261007.md for what the reading settled
and patches/kvmgr_plugin/README.md for the shape and the deploy.
"""

from .register import register

__all__ = ["register"]
