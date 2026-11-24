#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/FastVoiceGrad
SUITE=$ROOT/research_suite/vq_small_experiment_5090_20261116
BASE=$ROOT/runs/vq_small_experiment_5090_20261116
OUT=$ROOT/runs/vq_small_experiment_5090_20261116_v2
PY=/root/miniconda3/envs/fastvoicegrad5090/bin/python
PARENT=$ROOT/project/exp/diagnostics/student_development_20260906/acdd_mc1_longtrain_from3327_v1/latest_full.pt
K=$(tr -d '[:space:]' < "$BASE/codebook_probe/SELECTED_K.txt")
CB=$BASE/codebook_probe/codebook_k${K}.npy
mkdir -p "$OUT"
printf '{"status":"training_vq_only","selected_k":%s,"updated":"%s"}\n' "$K" "$(date -Is)" > "$OUT/queue_status.json"
"$PY" -u "$SUITE/vq_train_runner.py" --vq-codebook "$CB" --project-root "$ROOT/project" --run-root "$OUT/vq_only" --arm base_post --seed 1234 --lambda-cycle 0 --parent "$PARENT" --screen-passes 2 --metrics-endpoint --run-approved > "$OUT/vq_only.log" 2>&1
printf '{"status":"training_vq_cycle","selected_k":%s,"updated":"%s"}\n' "$K" "$(date -Is)" > "$OUT/queue_status.json"
"$PY" -u "$SUITE/vq_train_runner.py" --vq-codebook "$CB" --project-root "$ROOT/project" --run-root "$OUT/vq_cycle" --arm cycle_post --seed 1234 --lambda-cycle 2.5 --parent "$PARENT" --screen-passes 2 --metrics-endpoint --run-approved > "$OUT/vq_cycle.log" 2>&1
printf '{"status":"complete","selected_k":%s,"finished":"%s"}\n' "$K" "$(date -Is)" > "$OUT/queue_status.json"
