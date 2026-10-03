#!/usr/bin/env bash
# Health check for the optional systemd timer in deploy/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source scripts/config.sh

# While the four ranks load, start-tp4.sh owns recovery. A completed service is
# restarted only after two failed probes so one transient busy response does not
# interrupt a healthy inference cluster.
[[ "$(systemctl --user show glm53-tf-tp4.service -p ActiveState --value)" == active ]] || exit 0
healthy() {
  curl -fsS --max-time 8 "http://127.0.0.1:$PORT/health" |
    python3 -c 'import json,sys; assert json.load(sys.stdin)["ok"] is True' >/dev/null
}
healthy && exit 0
sleep 5
healthy && exit 0
echo "TensorFold TP4 health failed twice; restarting four ranks" >&2
systemctl --user restart glm53-tf-tp4.service
