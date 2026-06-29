#!/usr/bin/env python
"""Fit the PACE OLS latency predictor (DESIGN.md §10).

Model:  T(S) = a + b * n_tokens + c * L_context
where for a decode step n_tokens = batch size and L_context = aggregate context
length (sum over rows). We obtain calibration points from sglang.bench_one_batch,
which times a static decode batch at a controlled (batch_size, input_len) grid —
this decorrelates the two predictor features.

Workflow:
  python -m sglang.bench_one_batch --model-path <model> --tp-size 8 \
      --dtype bfloat16 --output-len 8 \
      --batch-size 1 2 4 8 16 32 64 --input-len 128 512 1024 2048 \
      --result-filename calib.jsonl
  python scripts/pace_calibrate.py --results calib.jsonl --out pace_coeffs.json

The resulting json {a, b, c} is loaded by the predictor via --pace-coeffs-path.
"""

import argparse
import json

import numpy as np


def fit(results_path: str):
    X, y, rows = [], [], []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "median_decode_latency" not in r:
                continue
            bs = r["batch_size"]
            il = r["input_len"]
            t = r["median_decode_latency"]
            # n_tokens = #sequences advancing; L_context = aggregate context.
            X.append([1.0, float(bs), float(bs * il)])
            y.append(float(t))
            rows.append((bs, il, t))
    if len(y) < 3:
        raise SystemExit(f"need >=3 calibration points, got {len(y)}")
    X = np.asarray(X)
    y = np.asarray(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = (float(v) for v in coef)
    pred = X @ coef
    resid = np.abs((pred - y) / y)
    mape = float(np.mean(resid) * 100.0)
    return a, b, c, mape, rows, pred, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="bench_one_batch jsonlines file")
    ap.add_argument("--out", required=True, help="output coeffs json {a,b,c}")
    args = ap.parse_args()

    a, b, c, mape, rows, pred, resid = fit(args.results)
    with open(args.out, "w") as f:
        json.dump({"a": a, "b": b, "c": c, "mape": mape, "n": len(rows)}, f, indent=2)
    print(
        f"PACE coeffs: a={a:.6e}  b={b:.6e}  c={c:.6e}  "
        f"MAPE={mape:.2f}%  n={len(rows)}  -> {args.out}"
    )
    order = np.argsort(-resid)[:5]
    print("worst-fit points:")
    for i in order:
        bs, il, t = rows[i]
        print(
            f"  bs={bs:<4} il={il:<5} actual={t*1000:7.2f}ms "
            f"pred={pred[i]*1000:7.2f}ms err={resid[i]*100:5.1f}%"
        )


if __name__ == "__main__":
    main()
