"""Plot a PACE eval sweep: goodput-vs-SLO and victim-TPOT-vs-mode.

Usage:  python plot_results.py <sweep-dir> [--out fig.png]
Reads the per-mode JSONs written by _pace_eval_worker.py (each has summary +
per-request records) and renders comparison curves. Requires matplotlib.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLOS = list(range(15, 101, 5))
ORDER = ["off", "cap2", "cap5", "pace", "eager"]


def _label(s):
    return f"cap{s['cap']}" if (s.get("mode") == "cap" and s.get("cap")) else s.get("mode")


def load(d):
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


def goodput(d, slo_ms):
    slo = slo_ms / 1000.0
    wall = d["summary"].get("wall_s") or 1.0
    return sum(r["n_tok"] for r in d["requests"]
               if r.get("tpot") is not None and r["tpot"] <= slo) / wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = load(a.sweep_dir)
    if not runs:
        print("no runs in", a.sweep_dir)
        return
    ms = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]
    out = a.out or os.path.join(a.sweep_dir, "pace_results.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for m in ms:
        ax1.plot(SLOS, [goodput(runs[m], s) for s in SLOS], marker="o", ms=3, label=m)
    ax1.set_xlabel("TPOT SLO (ms)"); ax1.set_ylabel("goodput (tok/s)")
    ax1.set_title("Goodput vs SLO"); ax1.legend(); ax1.grid(alpha=0.3)

    labels, p50, p90, p99 = [], [], [], []
    for m in ms:
        v = runs[m]["summary"].get("victim") or {}
        labels.append(m)
        p50.append(v.get("victim_tpot_ms_p50") or 0)
        p90.append(v.get("victim_tpot_ms_p90") or 0)
        p99.append(v.get("victim_tpot_ms_p99") or 0)
    x = range(len(labels)); w = 0.25
    ax2.bar([i - w for i in x], p50, w, label="p50")
    ax2.bar(list(x), p90, w, label="p90")
    ax2.bar([i + w for i in x], p99, w, label="p99")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels)
    ax2.set_ylabel("victim TPOT (ms)"); ax2.set_title("Victim (co-batched serial) TPOT")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout(); fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
