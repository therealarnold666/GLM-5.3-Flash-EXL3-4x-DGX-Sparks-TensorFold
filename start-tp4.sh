#!/usr/bin/env bash
# Four DGX Sparks, one TensorFold GLM-5.3-Flash TP4 engine on a direct CX7 ring.
# Rank order must follow the physical cycle: head -> WORKER -> WORKER2 -> WORKER3 -> head.
# Set the site-specific workers, HCA pins, GIDs and patched NCCL paths in scripts/local.sh.
# Usage: ./start-tp4.sh [restart] [tensorfold serve arguments]
#        ./start-tp4.sh stop
#        DRY_RUN=1 ./start-tp4.sh
set -euo pipefail
export TP=4 TOPOLOGY=switchless-ring COMM=nccl
root=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
if [[ "${1:-}" == stop ]]; then
  shift
  [[ $# == 0 ]] || { echo 'stop takes no extra arguments' >&2; exit 2; }
  exec "$root/stop.sh"
fi
exec "$root/start.sh" "$@"
