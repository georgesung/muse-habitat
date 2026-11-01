#!/usr/bin/env bash
# Sanity check: Glimmer C1 low on the 10-episode val_mini study set (docs/reports/baseline.md: 10/10). One Glimmer
# server (started here if <port> does not answer) and <shards> planner_demo processes on <gpu>; logs runs/<name>_s<k>. Stop the server by PID afterwards.
#   scripts/val_mini_check.sh <gpu> [port] [shards] [name]
set -euo pipefail
gpu=$1 port=${2:-8011} shards=${3:-5} name=${4:-val_mini_c1_low}
root="$(cd "$(dirname "$0")/.." && pwd)"
export PARTNR_SPLIT=val_mini GLIMMER_MEM=${GLIMMER_MEM:-0.85}

if ! curl -sf http://127.0.0.1:$port/v1/models > /dev/null; then
    "$root/scripts/glimmer_serve.sh" "$gpu" "$port" &
    echo "server pid $!"
    until curl -sf http://127.0.0.1:$port/v1/models > /dev/null; do sleep 20; done
fi
echo "server up $(date -u +%H:%M)"

over=()
for a in 0 1; do
    p=evaluation.agents.agent_$a.planner
    over+=(llm@$p.plan_config.llm=glimmer_vllm_tools1 instruct@$p.plan_config.instruct=zero_shot_native_tools
           $p._target_=muse_habitat.native_tools.NativeToolsPlanner $p.plan_config.llm.port=$port)
done
"$root/scripts/partnr_shards.sh" "$name" "$gpu" "$shards" 20,132,155,183,197,207,215,244,248,261 \
    baselines/decentralized_zero_shot_react_summary.yaml "${over[@]}"
PYTHONPATH="$root/src" python3 -m muse_habitat.results "$root"/runs/${name}_s* --expect 10
echo "VAL_MINI_CHECK_DONE $name $(date -u +%H:%M)"
