#!/usr/bin/env bash
# The kvmgr 0.31 cycle on one CUDA node: prime, restart, restore, Gate B, heal.
set -u
T=~/kvmgr031-test; B=$T/bin
export API=http://127.0.0.1:18000 STATS=18899 MODEL=${MODEL:-Qwen/Qwen3-0.6B}
export IMAGE=eugr/spark-vllm:latest SERVE="vllm serve" POLICY=recompute EXTRA="--safetensors-load-strategy eager"
G=${G:-0}; BLK=${BLK:-18}; P=${P:-q}
boot() { $B/cuda031.sh down; $B/cuda031.sh up >/dev/null; t0=$(date +%s)
  until curl -sf $API/health >/dev/null; do docker ps --format '{{.Names}}' | grep -qx kvmgr031 || { echo "ENGINE DIED AT BOOT"; docker logs kvmgr031 2>&1 | grep -E "Error|Traceback" | tail -5; exit 1; }; sleep 3; done
  echo "boot $(( $(date +%s)-t0 ))s; $(docker logs kvmgr031 2>&1 | grep 'BOOT WALK' | tail -1 | cut -c60-240)"; }
can() { python3 $B/canary031.py run --tag "$P-$1" | python3 -c "import json,sys;d=json.load(sys.stdin);print('$1',d['seconds'],'s',repr(d['text'][:24]),'tok',d['prompt_tokens'],'ext_hits',d['counters_moved']['vllm:external_prefix_cache_hits'],'load',d['load_after'])" || echo "$1 FAILED"; }
keys() { docker exec -i -e STATS=$STATS kvmgr031 python3 - < $B/keys.py | grep group | cut -c1-90; }
files() { docker exec kvmgr031 sh -c 'find /kvoffload/dspark-kv-v1 -name "*.bin" | wc -l; du -sh /kvoffload/dspark-kv-v1'; }
echo "== 1. prime"; can prime; keys; files
echo "== 2. reboot, restore"; boot; can restore; keys
echo "== 3. corrupt group $G block $BLK, reboot"; docker exec -i -e STATS=$STATS kvmgr031 python3 - $G $BLK < $B/corrupt.py; boot
echo "== 4. canary after corruption"; can corrupt
docker ps --format '{{.Names}}' | grep -qx kvmgr031 && echo "engine: UP" || echo "engine: DEAD"
docker logs kvmgr031 2>&1 | grep -E "FAILED its checksum|restore job|request-level|failed to load|RuntimeError|AssertionError|Xid|illegal" | cut -c1-200
keys
echo "== 5. canary again"; can heal; keys; files
echo "== 6. reboot, restore whole"; boot; can final; keys
for t in restore corrupt heal final; do python3 $B/canary031.py compare $P-prime $P-$t; done
docker logs kvmgr031 2>&1 | grep -iE "compute wait|could not arm|HostCopy" | sort | uniq -c | head -5 | cut -c1-200
