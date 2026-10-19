#!/usr/bin/env python3
"""Exact-echo canary for the local 0.31 spike (scripts/local031.sh).

  canary031.py run [--tag NAME]   send the canary, print the reply and the
                                  restore counters that moved; saves the reply
                                  to local/canary-<tag>.json
  canary031.py compare A B        the two saved replies must be identical

The prompt is FIXED text (never generated at run time): numbered filler lines
around a sentinel, then the instruction to repeat it. Greedy, so the same
prefix must give the same reply whether it was computed or restored from disk.
A pass needs the reply to contain the sentinel AND (for a restore) the
external-hit counter to have moved -- a correct answer with no restore proves
nothing about restores.
"""
import json
import os
import sys
import time
import urllib.request

API = os.environ.get("API", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
SENTINEL = "7f3a9c21e4b85d06"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "local")


def prompt() -> str:
    lines = ["Line %04d: the quick brown fox jumps over the lazy dog, then "
             "counts the stones on the path and writes the total down." % i
             for i in range(1, 161)]
    lines.insert(40, "IMPORTANT: the secret code is %s. Remember it." % SENTINEL)
    return ("Read the following log carefully.\n\n" + "\n".join(lines)
            + "\n\nQuestion: what is the secret code? Answer with the code "
              "only.\nAnswer: ")


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode()


def metric(name):
    """Sum of a Prometheus counter across labels, or None if absent."""
    tot, seen = 0.0, False
    for ln in get(API + "/metrics").splitlines():
        if ln.startswith(name) and not ln.startswith("#"):
            try:
                tot += float(ln.rsplit(" ", 1)[1])
                seen = True
            except ValueError:
                pass
    return tot if seen else None


def load_counters():
    try:
        workers = json.loads(get(API + "/kv/workers")).get("ranks") or []
    except Exception as e:
        return {"why": repr(e)[:120]}
    out = {}
    for w in workers:
        for h in (w.get("handlers") or []):
            if h.get("cls") == "DiskLoadHandler":
                out = {k: h.get(k) for k in ("blocks_to_gpu", "bytes", "files",
                                             "checksum_ok", "checksum_absent")}
    return out


def run(tag):
    names = ("vllm:external_prefix_cache_hits", "vllm:prefix_cache_hits")
    before = {n: metric(n) for n in names}
    lb = load_counters()
    body = json.dumps({"model": MODEL, "prompt": prompt(), "max_tokens": 24,
                       "temperature": 0}).encode()
    t0 = time.time()
    req = urllib.request.Request(API + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        rep = json.loads(r.read())
    dt = time.time() - t0
    after = {n: metric(n) for n in names}
    la = load_counters()
    text = rep["choices"][0]["text"]
    res = {"tag": tag, "seconds": round(dt, 2), "text": text,
           "prompt_tokens": rep.get("usage", {}).get("prompt_tokens"),
           "echo": SENTINEL in text,
           "counters_moved": {n: (None if before[n] is None else after[n] - before[n])
                              for n in names},
           "load_before": lb, "load_after": la}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "canary-%s.json" % tag), "w"), indent=1)
    print(json.dumps(res, indent=1))


def compare(a, b):
    ra = json.load(open(os.path.join(OUT, "canary-%s.json" % a)))
    rb = json.load(open(os.path.join(OUT, "canary-%s.json" % b)))
    same = ra["text"] == rb["text"]
    print("identical replies:", same, "| echo:", ra["echo"], rb["echo"])
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        run(sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--tag" else "x")
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
