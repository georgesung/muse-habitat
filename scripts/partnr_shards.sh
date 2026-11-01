#!/usr/bin/env bash
# Run one PARTNR setup over an episode list split into parallel shards (one planner_demo process each, all sharing
# the same LLM server); logs under runs/<name>_s<k>.
#   scripts/partnr_shards.sh <name> <gpu> <shards> <comma-separated episodes> <baseline config> [hydra overrides...]
set -euo pipefail
name=$1 gpu=$2 shards=$3 eps=$4 cfg=$5; shift 5
root="$(cd "$(dirname "$0")/.." && pwd)"
IFS=, read -ra all <<< "$eps"
for ((k = 0; k < shards; k++)); do
    part=()
    for ((i = k; i < ${#all[@]}; i += shards)); do part+=("${all[i]}"); done
    list=$(IFS=,; echo "${part[*]}")
    "$root/scripts/partnr_run.sh" "${name}_s$k" "$gpu" "$cfg" "+episode_indices=[$list]" "$@" &
done
wait
echo "PARTNR_SHARDS_DONE $name"
