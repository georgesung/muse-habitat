#!/usr/bin/env bash
# One PARTNR planner_demo run in ../partnr with its own env (../envs/partnr), logs under runs/<name>.
#   scripts/partnr_run.sh <name> <gpu> <baseline config> [hydra overrides...]
#   scripts/partnr_run.sh heuristic_smoke 1 baselines/heuristic_full_obs.yaml '+episode_indices=[0,1]'
# Episode set: PARTNR_SPLIT (default val_mini; val = all 1000).
set -euo pipefail
name=$1 gpu=$2 cfg=$3; shift 3
split=${PARTNR_SPLIT:-val_mini}
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/runs/$name"
mkdir -p "$out"
cd "$root/../partnr"
export CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
# Our backends (src/muse_habitat) and llm / instruct configs (conf/)
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
../envs/partnr/bin/python -m habitat_llm.examples.planner_demo --config-name "$cfg" \
    habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/$split.json.gz \
    evaluation.save_video=False hydra.run.dir="$out" "hydra.searchpath=[file://$root/conf]" "$@" > "$out/run.log" 2>&1
echo "PARTNR_DONE $name"
