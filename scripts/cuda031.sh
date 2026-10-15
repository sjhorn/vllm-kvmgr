#!/usr/bin/env bash
# kvmgr on vLLM 0.31 with CUDA, single GPU, in plain docker. Spike branch only.
# Written for a one-off test on a cluster node WITHOUT touching what is there:
#
#   * everything lives under $T (default ~/kvmgr031-test): the code copy, the
#     flags, the tier root and the HF cache. Not ~/vllm-kvmgr, not ~/kvoffload.
#   * the container is `kvmgr031`, never the DSpark engine's name, so
#     engine-supervisor (which matches its exact name) ignores it.
#   * host network, but the API and the stats port bind 127.0.0.1 on
#     18000/18899, not the engine's ports. A small GPU share (GPU_UTIL).
#
# On a GB10 (DGX Spark, sm_121) the stock image has no 12.1 kernels and stalls
# in JIT at the first CUDA copy; use an image built for 12.1a, e.g.
#   IMAGE=eugr/spark-vllm:latest SERVE="vllm serve"   (its entrypoint is NVIDIA's)
#
#   cuda031.sh up | logs | down        (down leaves $T; `rm -rf $T` is the cleanup)
set -euo pipefail
T="${T:-$HOME/kvmgr031-test}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.31.0}"
NAME="${NAME:-kvmgr031}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DIST="$(ls -d "$T"/deploy/vllm_kvmgr-*.dist-info | head -1)"
KVPOL=""
[ -n "${POLICY:-}" ] && KVPOL="\"kv_load_failure_policy\":\"$POLICY\","
KVCFG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",${KVPOL}\"kv_connector_extra_config\":{\"spec_name\":\"DSparkDiskOffloadingSpec\",\"staging_blocks\":64,\"staging_load_blocks\":64,\"write_threads\":4,\"root_dir\":\"/kvoffload\"}}"

case "${1:-up}" in
  up)
    mkdir -p "$T/kvoffload" "$T/hf"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all --network host --shm-size 4g --ulimit memlock=-1 \
      -v "$T/src/kvmgr:/opt/kvmgr/kvmgr:ro" \
      -v "$DIST:/opt/kvmgr/$(basename "$DIST"):ro" \
      -v "$T/local:/etc/vllm-kv-flags:ro" \
      -v "$T/hf:/root/.cache/huggingface" \
      -v "$T/kvoffload:/kvoffload" \
      -e PYTHONPATH=/opt/kvmgr \
      -e PYTHONHASHSEED=20261007 \
      -e VLLM_KV_FLAGS_FILE=/etc/vllm-kv-flags/flags.json \
      "$IMAGE" ${SERVE:-} \
      "$MODEL" --host 127.0.0.1 --port 18000 \
      --gpu-memory-utilization "${GPU_UTIL:-0.25}" \
      --max-model-len "${MAXLEN:-8192}" --enable-prefix-caching \
      --kv-transfer-config "$KVCFG" ${EXTRA:-} >/dev/null
    echo "started $NAME: API 127.0.0.1:18000, kvmgr stats 127.0.0.1:18899"
    ;;
  logs) docker logs -f "$NAME" ;;
  down) docker rm -f "$NAME" >/dev/null 2>&1 || true ;;
  *) echo "usage: $0 up|logs|down" >&2; exit 2 ;;
esac
