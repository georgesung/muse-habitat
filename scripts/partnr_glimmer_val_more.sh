#!/usr/bin/env bash
# More shards for partnr_glimmer_val.sh (setup C1 on PARTNR val) on another GPU / Glimmer server: <episodes> over
# <shards> planner_demo processes on <gpu>, against the server on <port> (started here if it does not answer).
# Used to finish a val run on two GPUs: the episodes not yet in any shard's results, split between port 8011 (GPU 0)
# and 8012 (GPU 1). Logs: runs/<name>_s<k>.
#   scripts/partnr_glimmer_val_more.sh <gpu> <port> <shards> <name> <comma-separated episodes>
set -euo pipefail
gpu=$1 port=$2 shards=$3 name=$4 eps=$5
root="$(cd "$(dirname "$0")/.." && pwd)"
export PARTNR_SPLIT=val GLIMMER_MEM=${GLIMMER_MEM:-0.85}

if ! curl -sf http://127.0.0.1:$port/v1/models > /dev/null; then
    "$root/scripts/glimmer_serve.sh" "$gpu" "$port" &
    until curl -sf http://127.0.0.1:$port/v1/models > /dev/null; do sleep 20; done
fi
echo "server up $(date -u +%H:%M)"

over=()
for a in 0 1; do
    p=evaluation.agents.agent_$a.planner
    over+=(llm@$p.plan_config.llm=glimmer_vllm_tools1 instruct@$p.plan_config.instruct=zero_shot_native_tools
           $p._target_=muse_habitat.native_tools.NativeToolsPlanner $p.plan_config.llm.port=$port)
done
"$root/scripts/partnr_shards.sh" "$name" "$gpu" "$shards" "$eps" \
    baselines/decentralized_zero_shot_react_summary.yaml "${over[@]}"
echo "PARTNR_VAL_DONE $name $(date -u +%H:%M)"
