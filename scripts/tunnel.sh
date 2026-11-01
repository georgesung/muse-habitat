#!/usr/bin/env bash
# Forward local ports to the same ports on another machine (vLLM servers there, rollouts and trainer here); reconnects
# if the connection drops. Stop it by PID.
#   scripts/tunnel.sh <ssh target> <port>[,<port>...] [ssh options...]
#   scripts/tunnel.sh user@host 8021,8022 -i ~/.ssh/key -o IdentitiesOnly=yes
set -uo pipefail
target=$1 ports=$2; shift 2
fwd=()
for p in ${ports//,/ }; do fwd+=(-L "$p:127.0.0.1:$p"); done
while true; do
    ssh -N "${fwd[@]}" -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$@" "$target"
    echo "tunnel down $(date -u +%H:%M:%S), reconnecting" >&2
    sleep 5
done
