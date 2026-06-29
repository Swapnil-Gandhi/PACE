"""PACE goodput/SLO eval: one serving config under concurrent load (DESIGN.md §11.3/§17).

Submits a mix of decomposable (Multiverse, branching) and non-decomposable
("serial") prompts under load, streams each, measures per-request effective-TPOT,
then aggregates throughput / SLO-attainment / goodput per kind and overall. Run
one config per process; aggregate across modes with the driver (pace_eval_driver.py).

Two load generators (DESIGN.md §17 "drive into the decode-step-bound regime"):

  * ``--load open``   : Poisson arrivals at ``--rate`` req/s over a fixed pool of
    requests. Default rate is > 0 (staggered) to avoid the all-at-once gloo crash
    seen in bring-up. ``--rate 0`` reproduces the pathological burst (debug only).
  * ``--load closed`` : a closed-loop generator that keeps ``--concurrency`` N
    requests in flight for ``--duration`` seconds. As soon as one finishes another
    is launched, so the running decode batch stays high and step latency climbs
    toward the SLO (the regime where branch externality is non-zero and PACE has
    something to protect). This is the headline generator for reproducing gains.

Metrics that ISOLATE branch externality (DESIGN.md §17):
  * per-kind effective-TPOT (serial vs decomposable) and goodput;
  * victim-serial-TPOT: TPOT of serial (non-branching) requests that overlapped
    in time with >=1 in-flight decomposable request (the cohort PACE protects);
  * mean/peak resident decode-batch size + branch externality + fraction of steps
    with deferral, pulled from the scheduler via the async get_internal_state()
    (also re-exported on the HTTP /get_server_info endpoint as ["pace"]).
"""

import argparse
import asyncio
import json
import os
import random
import time

import sglang as sgl

PROMPT_DIR = "/home/swgandhi/pace/multiverse-engine/example/prompt"
BRANCH_STOP_TOKEN_IDS = [151670, 151674]
SIMPLE_PROMPTS = [
    "What is the capital of France?",
    "List three primary colors.",
    "Who wrote the play Hamlet?",
    "What is the boiling point of water at sea level in Celsius?",
    "Name the largest planet in the solar system.",
    "What gas do plants absorb from the atmosphere?",
    "Summarize the water cycle in two sentences.",
    "What is the chemical symbol for gold?",
]


def construct_prompt(user_query: str) -> str:
    sysp = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    return (
        f"<|im_start|>system\n{sysp}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}\n<|im_end|>\n<|im_start|>assistant\n"
    )


def _load_decomp_texts():
    texts = []
    for f in sorted(os.listdir(PROMPT_DIR)):
        with open(os.path.join(PROMPT_DIR, f)) as fh:
            texts.append(fh.read())
    return texts


def build_open_workload(num_decomp, num_simple, rate, seed):
    """Fixed request pool with Poisson (staggered) arrivals; rate>0 by default."""
    rng = random.Random(seed)
    decomp_texts = _load_decomp_texts()
    items = []
    for i in range(num_decomp):
        items.append(("decomp", construct_prompt(decomp_texts[i % len(decomp_texts)])))
    for i in range(num_simple):
        items.append(("simple", construct_prompt(SIMPLE_PROMPTS[i % len(SIMPLE_PROMPTS)])))
    rng.shuffle(items)
    t = 0.0
    workload = []
    for kind, prompt in items:
        workload.append((t, kind, prompt))
        if rate and rate > 0:
            t += rng.expovariate(rate)  # inter-arrival ~ Exp(rate)
        # rate<=0 => all at t=0 (open-loop burst; debug only, can crash a TP worker)
    return workload


def _sample_request(rng, decomp_texts, decomp_frac):
    """Draw one (kind, prompt) for the closed-loop generator."""
    if rng.random() < decomp_frac:
        return "decomp", construct_prompt(rng.choice(decomp_texts))
    return "simple", construct_prompt(rng.choice(SIMPLE_PROMPTS))


# --------------------------------------------------------------------------- #
# Per-request execution + measurement
# --------------------------------------------------------------------------- #
async def one_request(engine, kind, prompt, sp, rec):
    """Stream one request; append a measurement record to ``rec``.

    Records absolute start/end (monotonic, shared origin) so overlap with
    decomposable requests can be computed post-hoc for the victim-TPOT metric.
    """
    t_start = time.monotonic()
    t_first = None
    n_tok = 0
    gen = engine.async_generate(prompt, sampling_params=sp, stream=True)
    if asyncio.iscoroutine(gen):
        gen = await gen
    async for chunk in gen:
        now = time.monotonic()
        if t_first is None:
            t_first = now
        mi = chunk.get("meta_info", {}) if isinstance(chunk, dict) else {}
        ct = mi.get("completion_tokens")
        if ct is not None:
            n_tok = ct
    t_end = time.monotonic()
    ttft = (t_first - t_start) if t_first else None
    # effective-TPOT: decode wall-time / (decode tokens). Uses (n_tok-1) because
    # the first token's latency is TTFT, not a decode step (DESIGN.md D3).
    tpot = ((t_end - t_first) / (n_tok - 1)) if (t_first and n_tok > 1) else None
    rec.append(
        {
            "kind": kind,
            "t_start": t_start,
            "t_first": t_first,
            "t_end": t_end,
            "ttft": ttft,
            "tpot": tpot,
            "n_tok": n_tok,
            "dur": t_end - t_start,
        }
    )


# --------------------------------------------------------------------------- #
# Open-loop driver (fixed pool, Poisson arrivals)
# --------------------------------------------------------------------------- #
async def run_open(engine, workload, sp):
    out = []
    t_origin = time.monotonic()

    async def _arrival(arrival, kind, prompt):
        delay = arrival - (time.monotonic() - t_origin)
        if delay > 0:
            await asyncio.sleep(delay)
        await one_request(engine, kind, prompt, sp, out)

    tasks = [
        asyncio.create_task(_arrival(a, k, p)) for (a, k, p) in workload
    ]
    await asyncio.gather(*tasks)
    wall = time.monotonic() - t_origin
    return out, wall, t_origin


# --------------------------------------------------------------------------- #
# Closed-loop driver (maintain N in-flight for a fixed duration)
# --------------------------------------------------------------------------- #
async def run_closed(engine, sp, concurrency, duration, decomp_frac, seed,
                     warmup=0.0):
    """Keep ``concurrency`` requests in flight until ``duration`` s elapse.

    A pool of ``concurrency`` worker coroutines each loops: launch one request,
    await it, launch the next, until the deadline. This holds the running decode
    batch near steady state so step latency reflects the loaded regime.

    ``warmup`` seconds at the start are excluded from the measured records so
    fill/cold-start steps do not pollute goodput.
    """
    rng = random.Random(seed)
    decomp_texts = _load_decomp_texts()
    out = []
    t_origin = time.monotonic()
    deadline = t_origin + duration
    measure_after = t_origin + warmup

    async def worker(wid):
        wrng = random.Random((seed << 16) ^ wid)
        while time.monotonic() < deadline:
            kind, prompt = _sample_request(wrng, decomp_texts, decomp_frac)
            local = []
            await one_request(engine, kind, prompt, sp, local)
            # Only count requests that STARTED after warmup (steady state).
            if local and local[0]["t_start"] >= measure_after:
                out.extend(local)

    workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    await asyncio.gather(*workers)
    # Measured window = wall since warmup ended (>=0; guards duration<=warmup).
    measured_wall = max(0.0, time.monotonic() - measure_after)
    return out, measured_wall, t_origin


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _pcts(vals_ms):
    s = sorted(vals_ms)

    def pct(p):
        return s[min(len(s) - 1, int(p * len(s)))] if s else None

    return {"p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99)}


def _kind_summary(recs, wall, slo):
    """Throughput / goodput / SLO-attainment / TPOT pcts for a record subset.

    goodput = (tokens of requests whose effective-TPOT <= SLO) / measured wall.
    SLO-attainment = fraction of requests meeting the TPOT SLO (TPOT<=SLO).
    Requests with no TPOT (n_tok<=1) are excluded from attainment denominators
    but their tokens still count toward raw throughput.
    """
    n = len(recs)
    total_tok = sum(r["n_tok"] for r in recs)
    have_tpot = [r for r in recs if r["tpot"] is not None]
    met = [r for r in have_tpot if r["tpot"] <= slo]
    good_tok = sum(r["n_tok"] for r in met)
    tpot_ms = [r["tpot"] * 1000 for r in have_tpot]
    ttft_ms = [r["ttft"] * 1000 for r in recs if r["ttft"] is not None]
    pc = _pcts(tpot_ms)
    return {
        "n_requests": n,
        "total_tok": total_tok,
        "throughput_tok_s": total_tok / wall if wall else 0,
        "goodput_tok_s": good_tok / wall if wall else 0,
        "slo_attainment": (len(met) / len(have_tpot)) if have_tpot else None,
        "tpot_ms_p50": pc["p50"],
        "tpot_ms_p90": pc["p90"],
        "tpot_ms_p99": pc["p99"],
        "ttft_ms_p50": _pcts(ttft_ms)["p50"],
    }


def _victim_serial(recs, slo):
    """victim-serial-TPOT: TPOT of serial requests whose decode window overlapped
    >=1 in-flight decomposable request — the cohort PACE is meant to protect.

    Overlap test: [t_first, t_end] of the serial request intersects [t_first,
    t_end] of any decomposable request. Returns the same summary shape (no wall
    => throughput fields omitted)."""
    decomp = [
        (r["t_first"], r["t_end"])
        for r in recs
        if r["kind"] == "decomp" and r["t_first"] is not None
    ]
    decomp.sort()
    victims = []
    for r in recs:
        if r["kind"] != "simple" or r["t_first"] is None or r["tpot"] is None:
            continue
        s, e = r["t_first"], r["t_end"]
        if any(ds <= e and s <= de for (ds, de) in decomp):
            victims.append(r)
    tpot_ms = [r["tpot"] * 1000 for r in victims]
    met = sum(1 for r in victims if r["tpot"] <= slo)
    pc = _pcts(tpot_ms)
    return {
        "n_victims": len(victims),
        "victim_tpot_ms_p50": pc["p50"],
        "victim_tpot_ms_p90": pc["p90"],
        "victim_tpot_ms_p99": pc["p99"],
        "victim_slo_attainment": (met / len(victims)) if victims else None,
    }


def summarize(out, wall, slo_ms, pace_snap):
    slo = slo_ms / 1000.0
    overall = _kind_summary(out, wall, slo)
    serial = _kind_summary([r for r in out if r["kind"] == "simple"], wall, slo)
    decomp = _kind_summary([r for r in out if r["kind"] == "decomp"], wall, slo)
    summary = {
        "slo_ms": slo_ms,
        "wall_s": wall,
        **overall,  # overall throughput/goodput/tpot pcts (back-compat keys)
        "serial": serial,
        "decomp": decomp,
        "victim": _victim_serial(out, slo),
    }
    # Scheduler-side externality-isolation metrics (DESIGN.md §17).
    if pace_snap:
        summary["sched"] = {
            "mean_resident_lifetime": pace_snap.get("mean_resident_lifetime"),
            "peak_resident": pace_snap.get("peak_resident"),
            "mean_width_lifetime": pace_snap.get("mean_width_lifetime"),
            "peak_width": pace_snap.get("peak_width"),
            "frac_steps_deferred": pace_snap.get("frac_steps_deferred"),
            "branch_externality_ms_median": pace_snap.get("branch_externality_ms_median"),
            "branch_externality_ms_p99": pace_snap.get("branch_externality_ms_p99"),
            "branch_externality_ms_peak": pace_snap.get("branch_externality_ms_peak"),
            "steps_planned": pace_snap.get("steps_planned"),
            "predictor_mape": pace_snap.get("predictor_mape"),
        }
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, help="baseline|off|cap|eager|pace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--load", choices=["open", "closed"], default="closed",
                    help="open=Poisson pool; closed=maintain N in-flight (default)")
    # open-loop pool
    ap.add_argument("--num-decomp", type=int, default=16)
    ap.add_argument("--num-simple", type=int, default=16)
    ap.add_argument("--rate", type=float, default=8.0,
                    help="open-loop req/s; >0 staggers arrivals. 0 = burst (debug)")
    # closed-loop
    ap.add_argument("--concurrency", type=int, default=64,
                    help="closed-loop: in-flight requests held constant")
    ap.add_argument("--duration", type=float, default=120.0,
                    help="closed-loop: measured-window seconds")
    ap.add_argument("--warmup", type=float, default=15.0,
                    help="closed-loop: seconds excluded from metrics at start")
    ap.add_argument("--decomp-frac", type=float, default=0.5,
                    help="closed-loop: fraction of requests that are decomposable")
    # generation / SLO
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--slo-ms", type=float, default=50.0)
    ap.add_argument("--model", default="Multiverse4FM/Multiverse-32B")
    ap.add_argument("--coeffs", default="/tmp/pace-env/pace_coeffs.json")
    ap.add_argument("--seed", type=int, default=0)
    # stability knobs (DESIGN.md §17: harden burst stability on H200)
    ap.add_argument("--mem-fraction-static", type=float, default=0.83)
    ap.add_argument("--chunked-prefill-size", type=int, default=8192)
    ap.add_argument("--max-running-requests", type=int, default=256)
    ap.add_argument("--max-num-reqs", type=int, default=None,
                    help="req-slot pool size; raise for high concurrency + branches")
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--pace-utility", default="linear",
                    help="linear | concave | priority")
    ap.add_argument("--overlap", action="store_true",
                    help="enable overlap scheduler (default off; v1 targets normal mode)")
    a = ap.parse_args()

    kw = dict(
        model_path=a.model,
        tp_size=8,
        dtype="bfloat16",
        disable_overlap_schedule=not a.overlap,  # v1 default = normal mode (DESIGN.md D4)
        log_level="info",
        random_seed=0,
        # --- stability at high concurrency on H200 (DESIGN.md §17) ---
        mem_fraction_static=a.mem_fraction_static,
        chunked_prefill_size=a.chunked_prefill_size,
        max_running_requests=a.max_running_requests,
    )
    # NOTE: req-slot pool size (max_num_reqs) is runtime-derived (≤4096) and is
    # NOT a ServerArgs field, so it cannot be set here. Keep resident reqs
    # (concurrency + live branches) under ~4096 to avoid alloc_req_slots OOM.
    if a.mode != "baseline":
        kw["pace_enable"] = True
        kw["pace_mode"] = a.mode
        kw["pace_tpot_slo_ms"] = a.slo_ms
        kw["pace_coeffs_path"] = a.coeffs
        kw["pace_rho"] = a.rho
        kw["pace_utility"] = a.pace_utility
        if a.mode == "cap":
            kw["pace_cap"] = int(os.environ.get("PACE_CAP", "2"))

    engine = sgl.Engine(**kw)
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()

    sp = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": a.max_new_tokens,
        "skip_special_tokens": False,
        "stop_token_ids": BRANCH_STOP_TOKEN_IDS,
    }

    # Sample scheduler-side PACE metrics (decode-batch size, deferral fraction,
    # externality) LIVE during the run and keep the last good snapshot. A grab
    # AFTER a heavy closed-loop run is flaky (loop/communicator state), so we
    # sample mid-run via a background coroutine on the same loop.
    pace_holder = {"snap": None}

    async def _sampler(stop_evt):
        while not stop_evt.is_set():
            try:
                st = await engine.tokenizer_manager.get_internal_state()
                if isinstance(st, dict) and st.get("pace"):
                    pace_holder["snap"] = st["pace"]
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    async def _orchestrate():
        stop_evt = asyncio.Event()
        sampler = asyncio.create_task(_sampler(stop_evt))
        try:
            if a.load == "closed":
                res = await run_closed(engine, sp, a.concurrency, a.duration,
                                       a.decomp_frac, a.seed, warmup=a.warmup)
            else:
                wl = build_open_workload(a.num_decomp, a.num_simple, a.rate, a.seed)
                res = await run_open(engine, wl, sp)
        finally:
            stop_evt.set()
            try:
                await sampler
            except Exception:
                pass
        return res

    out, wall, _ = loop.run_until_complete(_orchestrate())
    pace_snap = pace_holder["snap"]

    summary = summarize(out, wall, a.slo_ms, pace_snap)
    summary["mode"] = a.mode
    summary["load"] = a.load
    summary["cap"] = kw.get("pace_cap")
    summary["concurrency"] = a.concurrency if a.load == "closed" else None
    summary["rate"] = a.rate if a.load == "open" else None
    summary["rho"] = a.rho
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "requests": out}, f)
    print("EVAL_DONE", a.mode, json.dumps(summary))
    try:
        engine.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
