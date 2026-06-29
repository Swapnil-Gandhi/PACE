# PACE evaluation

Repo-resident scripts to calibrate the PACE latency predictor and run the
goodput/SLO comparison vs the baselines (IRP-Off / IRP-C2 / IRP-C5 / IRP-Eager).
See `../../docs/pace/DESIGN.md` and `../../docs/pace/STATUS.md` for context.

## 1. Calibrate the predictor (once per hardware)

```bash
HF_HOME=/path/to/hf OUT=./pace_coeffs.json bash run_calibrate.sh
```

Sweeps a `(batch_size × input_len)` grid via `sglang.bench_one_batch` and OLS-fits
`T(S) = a + b·n_tokens + c·L_context` (writes `pace_coeffs.json`). On 8×H200 / TP8 / bf16 this
fits with ~3.5% MAPE.

## 2. Run the goodput/SLO sweep

```bash
HF_HOME=/path/to/hf COEFFS=./pace_coeffs.json OUT=/tmp/pace_sweep \
  MODES="eager pace off cap2" CONCURRENCY=256 DECOMP_FRAC=0.8 SLO=30 bash run_eval.sh
```

Runs `_pace_eval_worker.py` once per mode (one process each for clean TP state), then prints the
comparison via `analyze_sweep.py`. Per-request TPOTs are saved, so the SLO can be swept post-hoc.

Env knobs: `MODES` (eager/pace/off/cap2/cap5), `CONCURRENCY`, `DECOMP_FRAC`, `MRR`
(max-running-requests), `DURATION`, `WARMUP`, `MNT` (max-new-tokens), `SLO`, `MEMFRAC`, `UTILITY`
(linear/concave/priority).

## 3. Analyze / plot

```bash
python analyze_sweep.py /tmp/pace_sweep     # goodput / SLO-attainment / victim-TPOT vs SLO
python plot_results.py  /tmp/pace_sweep     # goodput-vs-SLO + victim-TPOT figures (needs matplotlib)
```

## Notes

- Keep `CONCURRENCY` under the req-slot pool (auto-sized ≤4096; branches each consume a req slot).
  PACE regulates *latency*, not memory — deferred branches keep their KV resident.
- PACE's win scales with branch externality, which is large only in the **branch-dominated regime**
  (high parallel-duty / large-fanout workloads); with low-duty prompts branches stay a batch
  minority and the gain is modest (see STATUS.md §7.1).
