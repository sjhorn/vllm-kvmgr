"""Phase 7: the GPU-cache drop must refuse while any RANK holds a transfer.

*** WHY THIS FILE EXISTS: THE DROP KILLED THE HEAD AGAIN ON 2026-10-09 17:51
*** WITH THE SCHEDULER-SIDE GUARD ALREADY LIVE. ***

Reproduced on purpose with local-ops/kvdroploop.sh: eight drop/restore cycles
at a 20K prefix were clean, and the FIRST cycle at 100K faulted -- Xid 32,
Worker_TP0, `unspecified launch failure` at synchronize_input_prep, two seconds
after a drop that logged `reset OK`. The engine log named the cause in two
lines twenty seconds apart:

    17:50:58 (Worker_TP0) store path busy -- 2 job(s) outstanding
    17:51:18 (EngineCore)  drop -- free 22280 -> 22755, reset OK

manifest.py's guard reads `offload_sched_snapshot()`, which reads `cs._jobs` --
the SCHEDULER's dict. Our handler queues a job of any size and pumps it over
ROUNDS from `get_finished()`, keeping its own `_jobs`/`_writing`/`_reading` in
the WORKER process, and those are the copies actually touching GPU memory. So
"idle by scheduler job set" is not idle by worker transfer, which is this
project's "idle by request count is not idle by transfer" trap one layer down.

`_transfers_busy` is pure and is tested here rather than on the box, because
the box costs a four-minute boot and a cold prefill per attempt -- and because
the failure mode is a guard that cannot fire, which no live run can distinguish
from a guard that had nothing to refuse.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # the package root
CONTROL = ROOT / "src" / "kvmgr" / "control.py"

checks = 0


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        print("FAIL: %s" % what)
        sys.exit(1)


def load():
    """Exec only `_BUSY_FIELDS` and `_transfers_busy` in a clean namespace.

    control.py imports FastAPI and vllm; the guard depends on neither, so
    lifting the two definitions keeps this runnable on a laptop and also
    asserts the function really is self-contained -- if somebody gives it a
    dependency, this stops working and says so.
    """
    tree = ast.parse(CONTROL.read_text())
    want = {"_BUSY_FIELDS", "_transfers_busy"}
    picked, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            picked.append(node); found.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in want:
                    picked.append(node); found.add(t.id)
    check(found == want,
          "both names located in control.py (missing %s)" % (want - found))
    ns: dict = {}
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(CONTROL), "exec"), ns)
    return ns


ns = load()
busy = ns["_transfers_busy"]
FIELDS = ns["_BUSY_FIELDS"]

def first(out):
    """out[0] or {} -- so a mutation that EMPTIES the list FAILS BY NAME.

    Three of six mutations originally died with an IndexError/KeyError here
    instead of printing which guard broke. A test that aborts detects the bug
    and tells you nothing about it, which this project has paid for more than
    once; the discriminator has to survive the thing it discriminates.
    """
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], dict):
        return out[0]
    return {}


# --- THE REAL SHAPE, taken from kvmgr_snapshot: a LIST under "handlers",
# --- each entry carrying "cls". My first version looked for "disk_store" and
# --- "disk_load" DICT KEYS and would have returned empty on every rank.
def rank(i, store=None, load=None, **extra):
    hs = []
    if store is not None:
        hs.append(dict(store, cls="DiskStoreHandler"))
    if load is not None:
        hs.append(dict(load, cls="DiskLoadHandler"))
    return dict({"rank": i, "handlers": hs}, **extra)


IDLE_S = {"jobs_open": 0, "writes_open": 0}
IDLE_L = {"jobs_open": 0, "reads_open": 0}

# 1. every rank idle -> empty, which is the ONLY way empty may happen.
check(busy([rank(0, IDLE_S, IDLE_L), rank(1, IDLE_S, IDLE_L)]) == [],
      "two idle ranks produce no refusal")

# 2. the measured failure: rank 0's store has 2 jobs outstanding.
out = busy([rank(0, {"jobs_open": 2, "writes_open": 0}, IDLE_L),
            rank(1, IDLE_S, IDLE_L)])
check(len(out) == 1 and first(out).get("rank") == 0, "the busy rank is named")
check(first(out).get("counts") == {"DiskStoreHandler.jobs_open": 2},
      "the count and its handler are named: %r" % (out[0],))

# 3. a held staging slot counts even with no job queued -- writes_open means
#    the slot is still ours and a file write is running against it.
check(busy([rank(0, {"jobs_open": 0, "writes_open": 1}, IDLE_L)]),
      "writes_open alone refuses")
check(busy([rank(0, IDLE_S, {"jobs_open": 0, "reads_open": 3})]),
      "reads_open alone refuses (the LOAD side)")

# 4. ABSENT IS NOT ZERO, five ways. Each must refuse, not pass.
check(first(busy("nope")).get("unreadable"), "a non-list reply is unreadable")
check(first(busy([])).get("unreadable"), "NO ranks cannot mean every rank is idle")
check(first(busy([None])).get("unreadable"), "a non-dict rank is unreadable")
check(first(busy([{"rank": 0}])).get("unreadable"),
      "a rank with no 'handlers' is unreadable, not idle")
check(first(busy([{"rank": 0, "handlers": "x"}])).get("unreadable"),
      "'handlers' that is not a list is unreadable")
check(first(busy([{"rank": 0, "handlers": [1]}])).get("unreadable"),
      "a non-dict handler entry is unreadable")

# 5. a bool is an int subclass, so True must not silently count as 1 -- that
#    would be a shape change reported as a quantity.
check(first(busy([rank(0, {"jobs_open": True}, None)])).get("unreadable"),
      "a bool count is unreadable rather than 1")

# 6. the error reason is carried through, so a refusal says where to look.
o = busy([{"rank": 1, "handlers_error": "no _LIVE_HANDLERS"}])
check("no _LIVE_HANDLERS" in (first(o).get("why") or ""),
      "the snapshot's own error reason reaches the refusal: %r" % (o,))

# 7. a rank absent from the reply cannot be inferred idle: one rank reported
#    while world_size is 2 is the half-empty-file failure in a new place.
#    The route refuses on ANY non-empty list, so an unreadable rank suffices;
#    what this pins is that a SHORT list is not silently accepted as idle.
check(busy([rank(0, IDLE_S, IDLE_L)]) == [],
      "a short list of IDLE ranks is empty here BY DESIGN -- the per-rank "
      "count is upstream's; see the note below")

# 8. the field list is the declaration, not a hand copy.
check(set(FIELDS) == {"jobs_open", "writes_open", "reads_open"},
      "the counted fields are exactly the three handler counters: %r" % (FIELDS,))

print("phase7 drop guard: %d assertions passed" % checks)
