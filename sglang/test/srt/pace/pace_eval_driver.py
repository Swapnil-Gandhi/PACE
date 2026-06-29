"""PACE goodput/SLO sweep driver (DESIGN.md §11.3/§17).

Runs _pace_eval_worker.py once per (operating-point x mode), one process per run
so each gets clean TP/GPU state (and an OOM/gloo death in one run can't poison the
rest). Aggregates the per-run JSON summaries into one comparison table + JSON so
PACE's goodput/SLO benefit is visible per operating point.

Operating points come from the calibration model (DESIGN.md §17 "drive into the
decode-step-bound regime"):

    T(S) = a + b*n_tokens + c*L_context
         = 6.99e-3 + 3.94e-5 * batch + 1.43e-8 * aggregate_context_tokens   [s]

Per-row marginal step cost at context L tok/row is (b + c*L). To push the loaded
decode step from the base ~7 ms up toward an SLO we need a running batch of:

    batch ~= (SLO - a) / (b + c * L_avg)

With L_avg ~= 1000 tok/row:  b + c*L = 3.94e-5 + 1.43e-5 = 5.37e-5 s/row, so
    SLO 25 ms -> ~335 rows,  SLO 50 ms -> ~800 rows,  SLO 100 ms -> ~1730 rows.

That is why bring-up (peak batch ~53) saw externality ~0. The closed-loop points
below target concurrency in the hundreds so the resident batch reaches that range
and eager's loaded step latency genuinely exceeds the SLO (the regime where PACE
defers and protects victims). Tune concurrency upward until sched.peak_resident
in the output approaches the target batch for the chosen SLO.

Usage:
  python pace_eval_driver.py --out-dir /tmp/pace-env/sweep \
      --modes off cap2 cap5 eager pace --load closed \
      --points "c=128,slo=50" "c=256,slo=50" "c=384,slo=100"

A point string is comma-separated k=v overrides applied on top of the load
defaults; recognized keys: c|concurrency, slo|slo-ms, dur|duration, warmup,
decomp-frac, rate, num-decomp, num-simple, mnt|max-new-tokens, rho,
mem|mem-fraction-static, cp|chunked-prefill-size, mrr|max-running-requests.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "_pace_eval_worker.py")

# Calibration coefficients (H200, TP8, bf16) for the operating-point estimate.
CAL_A, CAL_B, CAL_C = 6.99e-3, 3.94e-5, 1.43e-8

# Default operating points: closed-loop concurrency sweep that walks the resident
# batch from "still cheap" through "loaded step ~= SLO" so the benefit window is
# bracketed. Each is a point string parsed by _parse_point.
DEFAULT_POINTS = [
    "c=128,slo=50",   # warm-up regime; PACE ~= eager (sanity: no regression)
    "c=256,slo=50",   # loaded step approaching SLO; PACE should start winning
    "c=384,slo=50",   # eager step over SLO; PACE protects victims (headline)
    "c=512,slo=100",  # higher SLO, larger batch; gain persists
]

# key aliases -> worker flag
KEY_ALIAS = {
    "c": "concurrency", "concurrency": "concurrency",
    "slo": "slo-ms", "slo-ms": "slo-ms",
    "dur": "duration", "duration": "duration",
    "warmup": "warmup",
    "decomp-frac": "decomp-frac", "df": "decomp-frac",
    "rate": "rate",
    "num-decomp": "num-decomp", "nd": "num-decomp",
    "num-simple": "num-simple", "ns": "num-simple",
    "mnt": "max-new-tokens", "max-new-tokens": "max-new-tokens",
    "rho": "rho",
    "mem": "mem-fraction-static", "mem-fraction-static": "mem-fraction-static",
    "cp": "chunked-prefill-size", "chunked-prefill-size": "chunked-prefill-size",
    "mrr": "max-running-requests", "max-running-requests": "max-running-requests",
}

# mode label -> (worker --mode, PACE_CAP env or None)
MODE_MAP = {
    "baseline": ("baseline", None),
    "off": ("off", None),
    "eager": ("eager", None),
    "pace": ("pace", None),
    "cap2": ("cap", "2"),
    "cap5": ("cap", "5"),
}


def est_batch_for_slo(slo_ms, l_avg=1000.0):
    slo = slo_ms / 1000.0
    return max(0.0, (slo - CAL_A) / (CAL_B + CAL_C * l_avg))


def _parse_point(s):
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition("=")
        k = k.strip()
        if k not in KEY_ALIAS:
            raise SystemExit(f"unknown point key {k!r} in {s!r}; known: {sorted(KEY_ALIAS)}")
        out[KEY_ALIAS[k]] = v.strip()
    return out


def run_one(mode_label, point, load, base, out_dir, dry):
    worker_mode, cap = MODE_MAP[mode_label]
    tag = f"{mode_label}__{'_'.join(f'{k}{v}' for k, v in sorted(point.items()))}"
    out_path = os.path.join(out_dir, f"{tag}.json")
    flags = dict(base)  # load defaults
    flags.update(point)  # per-point overrides
    timeout = flags.pop("_timeout", 3600)
    cmd = [sys.executable, WORKER, "--mode", worker_mode, "--out", out_path,
           "--load", load]
    for k, v in flags.items():
        if k.startswith("_"):
            continue  # internal-only key, not a worker flag
        cmd += [f"--{k}", str(v)]
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/tmp/pace-env/hf")
    if cap is not None:
        env["PACE_CAP"] = cap
    print("RUN", tag, " ".join(cmd))
    if dry:
        return {"tag": tag, "mode": mode_label, "point": point, "out": out_path,
                "ok": None}
    try:
        subprocess.run(cmd, env=env, check=True, timeout=timeout)
        with open(out_path) as f:
            summary = json.load(f)["summary"]
        return {"tag": tag, "mode": mode_label, "point": point, "out": out_path,
                "ok": True, "summary": summary}
    except Exception as e:  # one run dying must not abort the sweep
        print("FAIL", tag, repr(e))
        return {"tag": tag, "mode": mode_label, "point": point, "out": out_path,
                "ok": False, "error": repr(e)}


def _g(summary, *path):
    cur = summary
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def print_table(results):
    cols = [
        ("mode", lambda s: None),
        ("point", lambda s: None),
        ("gp_tok_s", lambda s: _g(s, "goodput_tok_s")),
        ("tp_tok_s", lambda s: _g(s, "throughput_tok_s")),
        ("slo%", lambda s: _g(s, "slo_attainment")),
        ("victim_p50", lambda s: _g(s, "victim", "victim_tpot_ms_p50")),
        ("victim_slo%", lambda s: _g(s, "victim", "victim_slo_attainment")),
        ("ser_gp", lambda s: _g(s, "serial", "goodput_tok_s")),
        ("dec_gp", lambda s: _g(s, "decomp", "goodput_tok_s")),
        ("peak_batch", lambda s: _g(s, "sched", "peak_resident")),
        ("ext_p99ms", lambda s: _g(s, "sched", "branch_externality_ms_p99")),
        ("defer%", lambda s: _g(s, "sched", "frac_steps_deferred")),
    ]
    print("\n=== PACE sweep ===")
    hdr = " | ".join(f"{c[0]:>12}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        s = r.get("summary") or {}
        cells = []
        for name, fn in cols:
            if name == "mode":
                cells.append(f"{r['mode']:>12}")
            elif name == "point":
                cells.append(f"{','.join(f'{k}={v}' for k,v in r['point'].items()):>12}")
            else:
                v = fn(s)
                cells.append(f"{v:>12.3f}" if isinstance(v, (int, float)) else f"{'na':>12}")
        print(" | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/pace-env/sweep")
    ap.add_argument("--modes", nargs="+",
                    default=["off", "cap2", "cap5", "eager", "pace"],
                    help=f"subset of {sorted(MODE_MAP)}")
    ap.add_argument("--load", choices=["open", "closed"], default="closed")
    ap.add_argument("--points", nargs="+", default=DEFAULT_POINTS,
                    help="point strings, e.g. 'c=256,slo=50'")
    # load defaults (overridable per-point); closed-loop steady-state run
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--decomp-frac", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--per-run-timeout", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    base = {
        "duration": a.duration,
        "warmup": a.warmup,
        "decomp-frac": a.decomp_frac,
        "max-new-tokens": a.max_new_tokens,
        "_timeout": a.per_run_timeout,
    }
    points = [_parse_point(s) for s in a.points]

    # Report the operating-point estimate so the operator can sanity-check batch.
    print("operating-point estimate (L_avg=1000 tok/row):")
    for slo in (25, 50, 100):
        print(f"  SLO {slo:>3} ms -> resident batch ~= {est_batch_for_slo(slo):.0f} rows")

    results = []
    for pt in points:
        for mode in a.modes:
            if mode not in MODE_MAP:
                raise SystemExit(f"unknown mode {mode!r}; known: {sorted(MODE_MAP)}")
            results.append(run_one(mode, pt, a.load, base, a.out_dir, a.dry_run))

    agg_path = os.path.join(a.out_dir, "sweep_summary.json")
    with open(agg_path, "w") as f:
        json.dump(results, f, indent=2)
    if not a.dry_run:
        print_table(results)
    print("SWEEP_DONE", agg_path)


if __name__ == "__main__":
    main()
