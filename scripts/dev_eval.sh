#!/usr/bin/env bash
# Greedy C1 low eval of one served model (base "glimmer" or a loaded LoRA adapter) on the dev sample
# (data/splits/dev.json: 300 episodes of the held-out train scenes) or its first <n>; logs runs/<name>_s<k>, summary at
# the end. The server on <port> must serve <model>.
#   scripts/dev_eval.sh <name> <gpu> <port> <model> [shards] [n]
set -euo pipefail
name=$1 gpu=$2 port=$3 model=$4 shards=${5:-12} n=${6:-300}
root="$(cd "$(dirname "$0")/.." && pwd)"
eps=$(python3 -c "import json; print(','.join(str(e[0]) for e in json.load(open('$root/data/splits/dev.json'))[:$n]))")
mkdir -p "$root/runs/$name"
echo "$model" > "$root/runs/$name/policy"
export PARTNR_SPLIT=train MUSE_POLICY_FILE="$root/runs/$name/policy"
over=()
for a in 0 1; do
    p=evaluation.agents.agent_$a.planner
    over+=(llm@$p.plan_config.llm=glimmer_vllm_tools1 instruct@$p.plan_config.instruct=zero_shot_native_tools
           $p._target_=muse_habitat.native_tools.NativeToolsPlanner $p.plan_config.llm.port=$port)
done
"$root/scripts/partnr_shards.sh" "$name" "$gpu" "$shards" "$eps" baselines/decentralized_zero_shot_react_summary.yaml \
    "${over[@]}"
PYTHONPATH="$root/src" python3 -m muse_habitat.results "$root"/runs/${name}_s*/results/episode_result_log.csv --expect "$n"
