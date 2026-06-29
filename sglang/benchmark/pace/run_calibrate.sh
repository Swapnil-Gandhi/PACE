#!/usr/bin/env bash
# Calibrate the PACE latency predictor T(S)=a+b*n_tokens+c*L_context (repo-resident).
# Sweeps a (batch_size x input_len) grid via bench_one_batch, then OLS-fits.
#   HF_HOME=/path/to/hf OUT=./pace_coeffs.json bash run_calibrate.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"   # .../sglang
PY="${PY:-python}"
: "${HF_HOME:?set HF_HOME to your model cache}"; export HF_HOME
MODEL="${MODEL:-Multiverse4FM/Multiverse-32B}"
OUT="${OUT:-$HERE/pace_coeffs.json}"
CALIB="${CALIB:-/tmp/pace_calib.jsonl}"; rm -f "$CALIB"

"$PY" -m sglang.bench_one_batch --model-path "$MODEL" --tp-size "${TP:-8}" \
  --dtype bfloat16 --mem-fraction-static "${MEMFRAC:-0.85}" --output-len "${OUTLEN:-8}" \
  --batch-size ${BATCH_SIZES:-1 2 4 8 16 32 64} \
  --input-len ${INPUT_LENS:-128 512 1024 2048} \
  --result-filename "$CALIB"

"$PY" "$REPO/scripts/pace_calibrate.py" --results "$CALIB" --out "$OUT"
echo "wrote coeffs -> $OUT"
