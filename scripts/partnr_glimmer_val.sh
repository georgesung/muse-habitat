#!/usr/bin/env bash
# Glimmer 30B, setup C1 (native thinking + one tool call per generation, low strength), on all 1000 PARTNR val
# episodes: one Glimmer server (port 8011) and <shards> planner_demo processes on one GPU.
# Logs: runs/<name>_s<k>, server runs/glimmer_serve.log.
#   scripts/partnr_glimmer_val.sh [gpu] [shards] [name]
set -euo pipefail
gpu=${1:-0} shards=${2:-8} name=${3:-glimmer_tools1_low_val}
root="$(cd "$(dirname "$0")/.." && pwd)"
export PARTNR_SPLIT=val GLIMMER_MEM=${GLIMMER_MEM:-0.85}

"$root/scripts/glimmer_serve.sh" "$gpu" 8011 &
until curl -sf http://127.0.0.1:8011/v1/models > /dev/null; do sleep 20; done
echo "server up $(date -u +%H:%M)"

"$root/scripts/partnr_shards.sh" "$name" "$gpu" "$shards" "$(seq -s, 0 999)" \
    baselines/decentralized_zero_shot_react_summary.yaml \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=glimmer_vllm_tools1 \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=glimmer_vllm_tools1 \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=zero_shot_native_tools \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=zero_shot_native_tools \
    evaluation.agents.agent_0.planner._target_=muse_habitat.native_tools.NativeToolsPlanner \
    evaluation.agents.agent_1.planner._target_=muse_habitat.native_tools.NativeToolsPlanner
echo "PARTNR_VAL_DONE $name $(date -u +%H:%M)"
