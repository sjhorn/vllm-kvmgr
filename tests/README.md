# vllm-kvmgr tests — no GPU, no cluster, a few seconds

```sh
bash run-all.sh                                   # every suite, BOTH modes; exits non-zero on any failure
MODES=pristine bash run-all.sh                    # one mode
KVMGR_BUILD_PY=/path/to/python bash run-all.sh    # also build the wheel and test it installed
python3.12 test_phase1.py --with-dspark-patches   # one suite, one mode
```

**Use `python3.12`.** 3.9 cannot parse the evaluated `X | None` annotations,
and the failure looks like a broken harness.

**Read the exit status, not the last line.** `run-all.sh` ends in `exit $T`
because an earlier sweep ended in an `echo`, and `sweep && git commit`
committed a red suite.

## The two modes

| mode | loads | proves |
|---|---|---|
| `pristine` (default) | `upstream/` only: the vLLM sources of the build kvmgr is proven on (`upstream/MANIFEST.txt`) | kvmgr works on a vLLM with **no patch of ours**. The harness asserts that no patch6 symbol is reachable |
| `--with-dspark-patches` | the DSpark recipe's `patch6/7/8` over the same sources, from `KVMGR_DSPARK_REPO` (default: the repo this package sits in) | what the box runs, including the fork bridge (`kvmgr.compat.dspark`) |

Sections that assert facts *about patch6* run only in the second mode. In the
first mode they print `SKIP [pristine mode] …`, and `run-all.sh` counts those
lines next to the suite's result. A skip is never folded into "ok".

## What is real and what is stubbed

The vLLM sources under test are real: the block pool, the manager, the
coordinator, `kv_offload/{base,worker,factory,file_mapper,cpu}` and the
offloading connector. They are vendored in `upstream/` with a sha256 per file.
The leaves around them (`vllm.logger`, `envs`, `config`, the hashing utils, the
request type) and **numpy and torch** are stubbed in `vllm_stubs.py`. The stub
header says what is and is not modelled, so no test asserts on the stub
instead of on vLLM.

## The suites

| suite | what it pins |
|---|---|
| `test_phase0` | facts about patch6 the manager design rests on (DSpark mode only) |
| `test_phase0_disk` | the `kv_offload` contract our disk spec implements, against the real upstream files |
| `test_phase1` | the manager's cap arithmetic, parity with patch6, and that **every knob is hot** (a probe pair per declared knob) |
| `test_phase1_disk` | the scheduler-side disk spec: index, lookup, budget, fingerprint, and the **on-disk layout pin** |
| `test_phase1b_disk` | the pure core of the write and read paths: file plans, pairing, bounds, the pump, the transfer ledger |
| `test_phase1d_disk` | the fingerprint shared across processes, the serializer shim, and the reload audits (every module and installer covered) |
| `test_phase2` | counters and attribution; **every knob read is declared, and every declared knob is published** |
| `test_phase3` | `by_need` and `lru_tail_first`, the GPU-cache drop guard, patch6's pending-blocks hook (DSpark mode) |
| `test_phase4` | the tiering shims (`kvmgr.diag.tiering`) against a stub tiering spec |
| `test_phase6` | the control plane's in-memory override and fan-out, hot reload, and the `_PRESERVE` state audit |
| `test_phase7_dropguard` | the drop refuses while any rank holds a transfer |
| `test_phase8_journal` | the presence journal: append, replay, compaction, torn tails, the crash windows |
| `test_phase9_verify` | cross-rank verification, boot check, self-heal, the free-space floor |
| `test_phase10_standalone` | **no patch required**, and the fork bridge carries overrides into patch6 (proven by uninstalling it) |
| `test_phase11_diag` | diagnostics isolated: a lint, the defaults, and **no diag module imported** with every diag knob off (fresh process) |
| `test_phase12_packaging` | the generated dist-info is in sync; the wheel's RECORD is complete; the installed wheel registers through its entry point |

## The discipline these suites run on

- **Verify every guard by breaking it.** Most of the guards here passed
  vacuously the first time, and only a mutation found that out. A guard needs a
  partner check that fails when nothing happened at all, such as "the lint
  found the module", "the tuple was located" or "the audit read more than ten
  files".
- **Restore a mutation from a copy, never with `git checkout --`**, which
  reverts to HEAD and silently discards an uncommitted fix. Clear
  `__pycache__` afterwards, or a same-second `.pyc` keeps the mutation alive.
- **Enumerate from the declaration.** Knobs, modules, installers and preserved
  state are all audited against the code rather than against a list somebody
  maintains, and three of those audits found a real omission the moment they
  were written.
