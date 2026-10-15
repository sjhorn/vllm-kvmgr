#!/usr/bin/env bash
# kvmgr on vLLM 0.31 in a Linux arm64 container (Apple `container`), CPU only.
# Spike branch spike/vllm-0.31-local. Never touches the cluster.
#
#   scripts/local031.sh up      start the engine (detached)
#   scripts/local031.sh logs    follow its log
#   scripts/local031.sh down    stop and remove it (the tier volume survives)
#   scripts/local031.sh wipe    also delete the tier volume
#
# Layout inside the container, the same shape as DSpark's compose:
#   /opt/kvmgr/kvmgr                       <- this worktree's src/kvmgr (ro)
#   /opt/kvmgr/vllm_kvmgr-<ver>.dist-info  <- its generated entry point (ro)
#   PYTHONPATH=/opt/kvmgr, so vLLM's plugin loader finds the entry point
#   /kvoffload                             <- a container VOLUME, not a macOS
#                                             bind mount, so xattrs (checksums) work
#   /etc/vllm-kv-flags/flags.json          <- local/flags.json (ro)
set -euo pipefail
W="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-vllm/vllm-openai-cpu:v0.31.0-arm64}"
NAME="${NAME:-kvmgr031}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
VOL="${VOL:-kvlocal}"
DIST="$(ls -d "$W"/deploy/vllm_kvmgr-*.dist-info | head -1)"
# TP=2: EXTRA="--tensor-parallel-size 2" CPUBIND="0-3|4-7" (one NUMA node
# here, so vLLM's CPU backend needs each worker's threads bound by hand).
# POLICY=recompute adds vLLM's kv_load_failure_policy (default "fail").
KVPOL=""
[ -n "${POLICY:-}" ] && KVPOL="\"kv_load_failure_policy\":\"$POLICY\","
KVCFG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",${KVPOL}\"kv_connector_extra_config\":{\"spec_name\":\"DSparkDiskOffloadingSpec\",\"staging_blocks\":64,\"staging_load_blocks\":64,\"write_threads\":4,\"root_dir\":\"/kvoffload\"}}"

case "${1:-up}" in
  up)
    container volume inspect "$VOL" >/dev/null 2>&1 || container volume create "$VOL" >/dev/null
    mkdir -p "$HOME/.cache/huggingface"
    container delete "$NAME" >/dev/null 2>&1 || true   # a stopped leftover
    # Not --rm: a failed start keeps its logs. --shm-size: vLLM's engine
    # message queue lives in /dev/shm, and the default is too small.
    container run -d --name "$NAME" \
      --cpus "${CPUS:-8}" --memory "${MEM:-16G}" --shm-size "${SHM:-4G}" \
      -p 8000:8000 -p 8899:8899 \
      -v "$W/src/kvmgr:/opt/kvmgr/kvmgr:ro" \
      -v "$DIST:/opt/kvmgr/$(basename "$DIST"):ro" \
      -v "$W/local:/etc/vllm-kv-flags:ro" \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -v "$VOL:/kvoffload" \
      -e PYTHONPATH=/opt/kvmgr \
      -e PYTHONHASHSEED=20261007 \
      -e VLLM_KV_FLAGS_FILE=/etc/vllm-kv-flags/flags.json \
      -e VLLM_CPU_KVCACHE_SPACE="${KVSPACE:-4}" \
      ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
      ${CPUBIND:+-e VLLM_CPU_OMP_THREADS_BIND="$CPUBIND"} \
      "$IMAGE" \
      "$MODEL" --host 0.0.0.0 --port 8000 \
      --max-model-len "${MAXLEN:-8192}" --enable-prefix-caching \
      --kv-transfer-config "$KVCFG" ${EXTRA:-}
    echo "started $NAME: logs with '$0 logs'; API :8000, kvmgr stats :8899"
    ;;
  logs) container logs -f "$NAME" ;;
  down) container stop "$NAME" 2>/dev/null || true; container delete "$NAME" ;;
  wipe) "$0" down 2>/dev/null || true; container volume delete "$VOL" ;;
  *) echo "usage: $0 up|logs|down|wipe" >&2; exit 2 ;;
esac
