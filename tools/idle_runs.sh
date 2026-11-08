#!/bin/bash
# Time-budgeted training runs that need an otherwise idle workstation (equal seconds per family only
# compare fairly when nothing else competes for the CPU). In order:
#   1. the within-project run of record (13 projects x 3 families, 60 s each) and its evaluation
#   2. the cross-project arm (config/apachejit_d6.yaml) and its evaluation
#   3. the three sensitivity variants with the family the run of record selected, and their evaluations
# Usage: tools/idle_runs.sh        (from the repository root; logs to logs/idle-<time>.log)
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v colima >/dev/null && colima status >/dev/null 2>&1; then
  echo "colima is running: stop it first (colima stop) so the run is not timed against a busy CPU"; exit 1
fi
LOG="logs/idle-$(date -u +%Y%m%dT%H%M%SZ).log"; mkdir -p logs
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
echo "start $(date -u +%FT%TZ) code $(git rev-parse --short HEAD)"

$PY -m riskrank.run_within --seconds 60 --seed 42
REC=$(ls -td runs/within-2*Z | head -1)
$PY -m riskrank.evaluate "$REC"
FAM=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['chosen_family'])" "$REC/selection.json")
echo "run of record $REC chose $FAM"

$PY -m riskrank.run_cross --seconds 60 --seed 42
$PY -m riskrank.evaluate "$(ls -td runs/cross-* | head -1)"

for V in exclude_relabelled window_1y from_2010; do
  $PY -m riskrank.run_within --seconds 60 --seed 42 --families "$FAM" --variant "$V"
  $PY -m riskrank.evaluate "$(ls -td runs/within-*-$V | head -1)"
done
echo "done $(date -u +%FT%TZ)"
