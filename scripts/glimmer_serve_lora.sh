#!/usr/bin/env bash
# Serve Muse Glimmer 30B (bf16) with vLLM for RL rollouts: LoRA adapters loaded at runtime
# (POST /v1/load_lora_adapter {lora_name, lora_path}). DFLASH=1 adds DFlash speculative decoding (SPEC_TOKENS, default 15, from the
# 5-layer assistant; the drafter has no adapter, the target model verifies with it). vLLM env: scripts/env.sh.
# Text only (--language-model-only: the vision encoder is not loaded). Base model served as "glimmer". Weights from the
# HF cache, offline.
#   scripts/glimmer_serve_lora.sh <gpu> [port]      log: runs/glimmer_serve_lora_<port>.log
# GLIMMER_MEM: vLLM's GPU memory share (default 0.85); MAX_LORA_RANK (default 64); MAX_LORAS per batch (default 4).
set -euo pipefail
gpu=$1 port=${2:-8021}
root="$(cd "$(dirname "$0")/.." && pwd)"
. "$root/scripts/env.sh"
draft=~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-assistant/snapshots/c676cce91658f9568aaf264a416211e2782735b7
spec=()
# with LoRA, vLLM needs max_num_batched_tokens >= max_num_seqs * (SPEC_TOKENS + 1): cap the batch at MAX_NUM_SEQS (default 64)
[ "${DFLASH:-0}" = 1 ] && spec=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$draft\",\"num_speculative_tokens\":${SPEC_TOKENS:-15}}"
                               --max-num-seqs "${MAX_NUM_SEQS:-64}")
model=~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/cec94fcfd6512f2cfe708138c32533b346ea4f69
mkdir -p "$root/runs"
export CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
exec "$VLLM_ENV/bin/vllm" serve "$model" --served-model-name glimmer --host 127.0.0.1 --port "$port" \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization ${GLIMMER_MEM:-0.85} \
    --language-model-only --enable-lora --max-lora-rank ${MAX_LORA_RANK:-64} --max-loras ${MAX_LORAS:-4} "${spec[@]}" \
    > "$root/runs/glimmer_serve_lora_$port.log" 2>&1
