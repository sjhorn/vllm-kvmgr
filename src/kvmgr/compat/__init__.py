"""kvmgr.compat -- optional bridges to forks of vLLM.

Nothing in kvmgr's core imports from here. Each bridge detects its fork and
does nothing when the fork is absent, so the package runs on a vLLM without it.
"""
