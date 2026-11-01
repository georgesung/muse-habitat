#!/usr/bin/env bash
# GRPO-style groups: <samples> rollouts of each episode in a list, C1 low with trajectory logging, against one
# Glimmer server on <port> (scripts/glimmer_serve_lora.sh). Each sample k runs <procs> planner_demo processes over the
# whole list (no process sees an episode twice); process j of sample k uses GPU ${GPUS[(k*procs+j) % n]} for Habitat.
# Logs runs/<name>/k<k>_s<j> (results), trajectories runs/<name>/traj_k<k>/.
#   PARTNR_SPLIT=train GPUS="0 1" TEMPERATURE=1.0 scripts/partnr_groups.sh <name> <port> <samples> <procs> <episodes>
set -euo pipefail
name=$1 port=$2 samples=$3 procs=$4 eps=$5
root="$(cd "$(dirname "$0")/.." && pwd)"
read -ra gpus <<< "${GPUS:-0}"
temp=${TEMPERATURE:-1.0}
IFS=, read -ra all <<< "$eps"
mkdir -p "$root/runs/$name"
echo "start $(date -u '+%F %H:%M:%S') samples $samples procs $procs episodes ${#all[@]} temperature $temp gpus ${gpus[*]}" \
    > "$root/runs/$name/info"
over=()
for a in 0 1; do
    p=evaluation.agents.agent_$a.planner
    over+=(llm@$p.plan_config.llm=glimmer_vllm_tools1 instruct@$p.plan_config.instruct=zero_shot_native_tools
           $p._target_=muse_habitat.native_tools.NativeToolsPlanner $p.plan_config.llm.port=$port
           $p.plan_config.llm.generation_params.temperature=$temp)
done
n=0
for ((k = 0; k < samples; k++)); do
    mkdir -p "$root/runs/$name/traj_k$k"
    for ((j = 0; j < procs; j++)); do
        part=()
        for ((i = j; i < ${#all[@]}; i += procs)); do part+=("${all[i]}"); done
        list=$(IFS=,; echo "${part[*]}")
        gpu=${gpus[$(( n % ${#gpus[@]} ))]}
        MUSE_TRAJ_DIR="$root/runs/$name/traj_k$k" \
            "$root/scripts/partnr_run.sh" "$name/k${k}_s$j" "$gpu" baselines/decentralized_zero_shot_react_summary.yaml \
            "+episode_indices=[$list]" "${over[@]}" &
        n=$((n + 1))
    done
done
wait
echo "end $(date -u '+%F %H:%M:%S')" >> "$root/runs/$name/info"
echo "PARTNR_GROUPS_DONE $name"
