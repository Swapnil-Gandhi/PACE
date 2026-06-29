"""Post-hoc analysis of a PACE eval sweep: goodput / SLO-attainment vs SLO per
mode, regime metrics, and victim-serial TPOT. Decouples SLO choice from the GPU
runs (per-request TPOTs are saved in each run's JSON).

Usage:  python analyze_sweep.py <dir-of-per-mode-jsons>
"""

import glob
import json
import os
import sys

SLOS = [15, 20, 25, 30, 40, 50, 75, 100]
MODE_ORDER = ["off", "cap2", "cap5", "pace", "eager"]


def _label(summary):
    m = summary.get("mode")
    cap = summary.get("cap")
    return f"cap{cap}" if (m == "cap" and cap) else m


def load_runs(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f).startswith(("_agg", "sweep_summary")):
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if "summary" in data and "requests" in data:
            runs[_label(data["summary"])] = data
    return runs


def goodput_attain(d, slo_ms, kind=None):
    slo = slo_ms / 1000.0
    reqs = [r for r in d["requests"] if r.get("tpot") is not None]
    if kind:
        reqs = [r for r in reqs if r.get("kind") == kind]
    wall = d["summary"].get("wall_s") or 1.0
    met = [r for r in reqs if r["tpot"] <= slo]
    return sum(r["n_tok"] for r in met) / wall, (len(met) / len(reqs) if reqs else 0.0)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = load_runs(d)
    if not runs:
        print("no runs found in", d)
        return
    ms = [m for m in MODE_ORDER if m in runs] + [m for m in runs if m not in MODE_ORDER]
    print("modes:", ms, "\n")

    print("=== regime / scheduler metrics ===")
    print(f"{'mode':6} {'thru':>8} {'peak_res':>9} {'mean_res':>9} {'frac_def':>9} {'ext_med':>8} {'ext_peak':>9}")
    for m in ms:
        s = runs[m]["summary"]
        def f(k, dec=1):
            v = s.get(k)
            if v is None:  # scheduler metrics are nested under summary["sched"]
                v = s.get("sched", {}).get(k)
            return f"{v:.{dec}f}" if isinstance(v, (int, float)) else "na"
        print(f"{m:6} {f('throughput_tok_s',0):>8} {f('peak_resident',0):>9} "
              f"{f('mean_resident_lifetime',0):>9} {f('frac_steps_deferred',2):>9} "
              f"{f('branch_externality_ms_median',2):>8} {f('branch_externality_ms_peak',1):>9}")

    print("\n=== victim-serial TPOT (serial reqs overlapping a decomposable) ===")
    for m in ms:
        print(f"  {m:6}", runs[m]["summary"].get("victim"))

    for kind, label in [(None, "OVERALL"), ("serial", "SERIAL"), ("decomp", "DECOMP")]:
        print(f"\n=== goodput tok/s vs SLO [{label}] ===")
        print("SLO  " + " ".join(f"{m:>9}" for m in ms))
        for slo in SLOS:
            print(f"{slo:>3}  " + " ".join(f"{goodput_attain(runs[m], slo, kind)[0]:>9.0f}" for m in ms))
        print(f"--- SLO attainment %% vs SLO [{label}] ---")
        for slo in SLOS:
            print(f"{slo:>3}  " + " ".join(f"{goodput_attain(runs[m], slo, kind)[1]*100:>9.1f}" for m in ms))


if __name__ == "__main__":
    main()
