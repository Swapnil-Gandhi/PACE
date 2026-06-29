<div align="center">
<h1><img src="assets/pace-logo.gif" height="40px" align="top"/> PACE: Regulating Branch Parallelism in LLM Serving</h1>
</div>

PACE is a per-step **branch-admission controller** for intra-request (branch) parallelism in LLM
serving. Built on the [Multiverse Engine](https://github.com/Multiverse4FM/Multiverse-Engine) (an
[SGLang](https://github.com/sgl-project/sglang) fork that provides the Map → Process → Reduce branch
substrate), it decides at every decode step which forked branches advance — admitting an extra branch
only when its predicted latency cost (its **branch externality**) fits the batch's current slack
budget. The payoff: higher goodput than eager admission, while protecting the latency SLO of every
co-batched request. For more information, please refer to our research paper: [<a href="https://arxiv.org/abs/2605.06914">📄 Regulating Branch Parallelism in LLM Serving </a>].

## 🧠 How it works

Each decode step, PACE evaluates a calibrated step-latency predictor
`T(S) = a + b·n_tokens + c·L_context`, then greedily admits the highest utility-per-cost branch while
predicted latency stays within the slack budget `T_0 + ρ·B_t`. The rest are deferred and re-admitted
on a later step.

- **Branch externality** — extra branches inflate the *shared* decode step that every co-batched
  request pays; PACE admits one only when it fits the slack budget.
- **The safe width is dynamic** — it shifts with batch composition, context lengths, and accumulated
  slack, so PACE re-decides per step instead of using a fixed cap.
- **Regulation is cheap** — branches share the request's prefix KV, so changing width needs no memory
  reclamation; deferral is pure batch-membership control (`ScheduleBatch.pace_split`).
- **Policies are pluggable** — subclass `AdmissionPolicy`, register it, and select with `--pace-mode <name>`
(built-ins: `off` / `cap` / `eager` / `pace`).

## 🚀 Quick start

```bash
conda create -n pace python=3.11 && conda activate pace
git clone https://github.com/Multiverse4FM/Multiverse-Engine && cd Multiverse-Engine
bash install.sh
# torch is pinned to 2.6.0 (cu124); if install.sh pulls newer transitive deps, pin these:
pip install "transformers==4.51.1" "torchao==0.9.0" "compressed-tensors==0.9.4"
```

PACE is **opt-in** and a strict no-op unless enabled (`--pace-enable false` reproduces the stock
engine):

```bash
python -m sglang.launch_server \
  --model-path Multiverse4FM/Multiverse-32B --tp 8 --dtype bfloat16 \
  --disable-overlap-schedule \
  --pace-enable --pace-mode pace --pace-tpot-slo-ms 25 \
  --pace-coeffs-path pace_coeffs.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--pace-enable` | `false` | master switch |
| `--pace-mode` | `pace` | `off` / `cap` / `eager` / `pace` |
| `--pace-rho` | `0.8` | fraction of slack budget to spend |
| `--pace-tpot-slo-ms` | `50` | TPOT SLO target (drives deadlines) |
| `--pace-coeffs-path` | `null` | offline-calibrated `(a, b, c)` json |

Calibrate the predictor on your hardware first — `scripts/pace_calibrate.py` fits `(a, b, c)` from a
`sglang.bench_one_batch` sweep — and pass the result via `--pace-coeffs-path`. On 8×H200 (TP=8, bf16)
this fits `T(S)` to ~3.5% MAPE, with < 1 ms/step planner overhead.

## 🧪 Testing

```bash
python -m unittest discover -s test/srt/pace -p "test_*.py"
```

GPU-free unit tests cover the predictor, greedy planner, slack budget, branch grouping, and deadline
model; heavy GPU correctness gates live in `test_pace_e2e.py` (run with `PACE_RUN_E2E=1`).

## 📚 Citation

```bibtex
@misc{gandhi2026,
      title={Regulating Branch Parallelism in LLM Serving},
      author={Gandhi, Swapnil and Hari, Siva and Dally, William J. and Kozyrakis, Christos},
      year={2026},
      eprint={2605.06914},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2605.06914},
}
```

## 🙏 Acknowledgments

PACE is built on the [Multiverse Engine](https://github.com/Multiverse4FM/Multiverse-Engine), which
provides the Map → Process → Reduce branch execution substrate, and on
[SGLang](https://github.com/sgl-project/sglang). We thank their authors.
