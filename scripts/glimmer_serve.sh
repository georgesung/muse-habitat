#!/usr/bin/env bash
# Serve Muse Glimmer 30B (bf16) with vLLM (../envs/vllm) on one GPU, DFlash speculative decoding with the 5-layer
# assistant, for the PARTNR backend (conf/llm/glimmer_vllm.yaml). Weights from the HF cache, offline.
#   scripts/glimmer_serve.sh <gpu> [port]      log: runs/glimmer_serve.log (glimmer_serve_<port>.log off 8011)
# GLIMMER_MEM: vLLM's GPU memory share (default 0.92; less leaves room for Habitat on the same GPU).
set -euo pipefail
gpu=$1 port=${2:-8011}
log=glimmer_serve.log; [ "$port" = 8011 ] || log=glimmer_serve_$port.log
root="$(cd "$(dirname "$0")/.." && pwd)"
hub=~/.cache/huggingface/hub
model=$hub/models--meta-models--Muse-Glimmer-30B/snapshots/cec94fcfd6512f2cfe708138c32533b346ea4f69
draft=$hub/models--meta-models--Muse-Glimmer-30B-assistant/snapshots/c676cce91658f9568aaf264a416211e2782735b7
mkdir -p "$root/runs"
export CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec "$root/../envs/vllm/bin/vllm" serve "$model" --served-model-name glimmer --host 127.0.0.1 --port "$port" \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization ${GLIMMER_MEM:-0.92} \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"$draft\",\"num_speculative_tokens\":15}" \
    > "$root/runs/$log" 2>&1
