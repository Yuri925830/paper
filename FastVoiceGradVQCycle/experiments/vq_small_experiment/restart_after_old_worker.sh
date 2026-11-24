#!/usr/bin/env bash
set -euo pipefail
OLD_PID=${1:?old pid required}
kill -TERM "$OLD_PID" 2>/dev/null || true
while kill -0 "$OLD_PID" 2>/dev/null; do sleep 5; done
sleep 3
exec /root/FastVoiceGrad/research_suite/vq_small_experiment_5090_20261116/run_queue.sh
