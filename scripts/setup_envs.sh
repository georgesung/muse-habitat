#!/usr/bin/env bash
# Build the project's envs inside it from requirements/:
#   envs/partnr  PARTNR: micromamba env (Python 3.9.2, habitat-sim 0.3.3 headless + Bullet) + pip set, PARTNR and its
#                habitat-lab / transformers-CFG submodules editable from third_party/partnr
#   envs/vllm    vLLM 0.32.0 (torch 2.13)
#   envs/trl     training: torch 2.13 cu130, transformers, PEFT, TRL, flash-attn built from source, muse_habitat
#                editable
#   scripts/setup_envs.sh [partnr] [vllm] [trl]        (default: all three)
# uv and micromamba: on PATH or in tools/. FLASH_ATTN_CUDA_ARCHS (default 90) picks the flash-attn build targets.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
UV=$(command -v uv || echo "$root/tools/uv")
MM=$(command -v micromamba || echo "$root/tools/micromamba")
P=third_party/partnr
mkdir -p envs runs

partnr() {
    MAMBA_ROOT_PREFIX="$root/envs/mamba" "$MM" create -y -p envs/partnr --file requirements/partnr-conda-explicit.txt
    envs/partnr/bin/python -m pip install --no-deps -r requirements/partnr-pip.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128
    envs/partnr/bin/python -m pip install --no-deps -e $P/third_party/habitat-lab/habitat-lab \
        -e $P/third_party/habitat-lab/habitat-baselines -e $P/third_party/transformers-CFG -e $P
    envs/partnr/bin/python -m pip check || true
}

vllm() {
    "$UV" venv -q --python 3.12 envs/vllm
    VIRTUAL_ENV=envs/vllm "$UV" pip install -r requirements/vllm-pip.txt
}

trl() {
    "$UV" venv -q --python 3.12 envs/trl
    VIRTUAL_ENV=envs/trl "$UV" pip install -r requirements/train-pip.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
    VIRTUAL_ENV=envs/trl "$UV" pip install -e .
    local cuda=${CUDA_HOME:-$(ls -d /usr/local/cuda-13* | sort -V | tail -1)}
    VIRTUAL_ENV=envs/trl CUDA_HOME=$cuda PATH=$cuda/bin:$root/envs/trl/bin:$PATH \
        FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS:-90} MAX_JOBS=${MAX_JOBS:-32} NVCC_THREADS=2 \
        FLASH_ATTENTION_FORCE_BUILD=TRUE "$UV" pip install --no-build-isolation flash-attn==2.8.3.post2
}

for env in "${@:-partnr vllm trl}"; do
    for e in $env; do
        echo "== $e $(date -u '+%F %H:%M:%S')"
        $e
    done
done
echo "SETUP_DONE $(date -u '+%F %H:%M:%S')"
