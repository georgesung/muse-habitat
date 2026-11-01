#!/usr/bin/env bash
# Serve a Llama 3.x 8B Instruct (bf16, from the HF cache, offline) with vLLM (../envs/vllm) for the PARTNR backend
# (conf/llm/llama31_vllm.yaml, llama3_vllm.yaml). Context is 32k for both: Llama 3 (8k trained) is allowed past
# 8k as PARTNR's HF path lets it run past it.
#   scripts/llama_serve.sh <gpu> <port> <gpu memory fraction> <HF repo> <served name>    log: runs/<name>_serve.log
#   scripts/llama_serve.sh 0 8014 0.40 NousResearch/Meta-Llama-3.1-8B-Instruct llama31
set -euo pipefail
gpu=$1 port=$2 mem=$3 repo=$4 name=$5
root="$(cd "$(dirname "$0")/.." && pwd)"
model=$(ls -d ~/.cache/huggingface/hub/models--${repo//\//--}/snapshots/*/ | head -1)
mkdir -p "$root/runs"
export CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
exec "$root/../envs/vllm/bin/vllm" serve "$model" --served-model-name "$name" --host 127.0.0.1 --port "$port" \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization "$mem" \
    > "$root/runs/${name}_serve.log" 2>&1
