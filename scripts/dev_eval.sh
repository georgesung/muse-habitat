#!/usr/bin/env bash
# Greedy C1 low eval of one served model (base "glimmer" or a loaded LoRA adapter) on the dev sample
# (data/splits/dev.json: 300 episodes of the held-out train scenes) or its first <n>; with SPLIT=val, on val episodes
# 0..<n>-1 (n=1000: full val). Episodes are dealt round-robin over the comma-separated <gpus> and <ports> (one
# server per GPU's shards); logs runs/<name>_h<j>_s<k>, summary at the end. Each server must serve <model>.
# EPISODES (comma-separated indices into the split's episodes) replaces the default set; n is then their count.
#   [SPLIT=val] [EPISODES=i,j,...] scripts/dev_eval.sh <name> <gpus> <ports> <model> [shards per gpu] [n]
set -euo pipefail
name=$1 shards=${5:-12}
IFS=, read -ra gpus <<< "$2"
IFS=, read -ra ports <<< "$3"
model=$4 split=${SPLIT:-dev}
[ "$split" = val ] && n=${6:-1000} || n=${6:-300}
root="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${EPISODES:-}" ]; then
    all=$EPISODES n=$(tr , '\n' <<< "$all" | wc -l)
elif [ "$split" = val ]; then
    all=$(seq -s, 0 $((n - 1)))
else
    all=$(python3 -c "import json; print(','.join(str(e[0]) for e in json.load(open('$root/data/splits/dev.json'))[:$n]))")
fi
IFS=, read -ra eps <<< "$all"
m=${#gpus[@]}
for ((j = 0; j < m; j++)); do
    part=()
    for ((i = j; i < ${#eps[@]}; i += m)); do part+=("${eps[i]}"); done
    run=${name}_h$j
    mkdir -p "$root/runs/$run"
    echo "$model" > "$root/runs/$run/policy"
    over=()
    for a in 0 1; do
        p=evaluation.agents.agent_$a.planner
        over+=(llm@$p.plan_config.llm=glimmer_vllm_tools1 instruct@$p.plan_config.instruct=zero_shot_native_tools
               $p._target_=muse_habitat.native_tools.NativeToolsPlanner $p.plan_config.llm.port=${ports[j]})
    done
    [ "$split" = val ] && s=val || s=train
    PARTNR_SPLIT=$s MUSE_POLICY_FILE="$root/runs/$run/policy" "$root/scripts/partnr_shards.sh" "$run" "${gpus[j]}" \
        "$shards" "$(IFS=,; echo "${part[*]}")" baselines/decentralized_zero_shot_react_summary.yaml "${over[@]}" &
done
wait
PYTHONPATH="$root/src" python3 -m muse_habitat.results "$root"/runs/${name}_h*_s*/results/episode_result_log.csv \
    --expect "$n"
