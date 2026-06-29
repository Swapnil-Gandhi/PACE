#!/usr/bin/env bash
# PACE goodput/SLO eval sweep (repo-resident). Runs the eval worker once per mode
# (one process each for clean TP state), then analyzes. Override via env vars.
#
#   HF_HOME=/path/to/hf COEFFS=./pace_coeffs.json MODES="eager pace off cap2" \
#   CONCURRENCY=256 DECOMP_FRAC=0.8 SLO=30 bash run_eval.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"                       # .../sglang
WORKER="$REPO/test/srt/pace/_pace_eval_worker.py"
PY="${PY:-python}"
: "${HF_HOME:?set HF_HOME to your model cache}"; export HF_HOME
MODEL="${MODEL:-Multiverse4FM/Multiverse-32B}"
COEFFS="${COEFFS:-$HERE/pace_coeffs.json}"
OUT="${OUT:-/tmp/pace_sweep}"; mkdir -p "$OUT"
MODES="${MODES:-eager pace off cap2}"
CONCURRENCY="${CONCURRENCY:-256}"; DECOMP_FRAC="${DECOMP_FRAC:-0.8}"
MRR="${MRR:-1024}"; DURATION="${DURATION:-50}"; WARMUP="${WARMUP:-10}"
MNT="${MNT:-500}"; SLO="${SLO:-30}"; MEMFRAC="${MEMFRAC:-0.80}"; UTILITY="${UTILITY:-linear}"

for m in $MODES; do
  mode="$m"; cap=0
  case "$m" in cap2) mode=cap; cap=2;; cap5) mode=cap; cap=5;; esac
  echo "=== running $m ==="
  PACE_CAP=$cap "$PY" "$WORKER" --mode "$mode" --out "$OUT/$m.json" \
    --model "$MODEL" --load closed --concurrency "$CONCURRENCY" \
    --max-running-requests "$MRR" --duration "$DURATION" --warmup "$WARMUP" \
    --decomp-frac "$DECOMP_FRAC" --max-new-tokens "$MNT" --slo-ms "$SLO" \
    --mem-fraction-static "$MEMFRAC" --pace-utility "$UTILITY" --coeffs "$COEFFS" \
    || echo "RUN FAILED: $m (rc=$?)"
done

echo "=== analysis ==="
"$PY" "$HERE/analyze_sweep.py" "$OUT"
