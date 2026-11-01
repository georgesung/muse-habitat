# Sourced by the scripts (with $root set to the project): where PARTNR and the envs are.
# Two layouts: inside the project (third_party/partnr, envs/partnr, envs/vllm), or next to it
# (../partnr, ../envs/partnr, ../envs/vllm). The first that exists wins; MUSE_PARTNR, MUSE_PARTNR_ENV and
# MUSE_VLLM_ENV override.
_first() { for d; do [ -e "$d" ] && { echo "$d"; return; }; done; echo "$1"; }
PARTNR_DIR=${MUSE_PARTNR:-$(_first "$root/third_party/partnr" "$root/../partnr")}
PARTNR_ENV=${MUSE_PARTNR_ENV:-$(_first "$root/envs/partnr" "$root/../envs/partnr")}
VLLM_ENV=${MUSE_VLLM_ENV:-$(_first "$root/envs/vllm" "$root/../envs/vllm")}
# HF hub cache, as huggingface_hub resolves it
HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
