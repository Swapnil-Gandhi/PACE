"""Run one serving config end-to-end and dump generated text/hashes.

Used by the schedule-invariance integration test (DESIGN.md §11.2): the same
prompt under different --pace-mode values must yield identical output tokens
(paper Lemma 3.1). Run one config per process so each gets a clean GPU/TP state.

Example:
  HF_HOME=/tmp/pace-env/hf python _pace_e2e_worker.py --mode off \
      --out /tmp/pace-env/inv_off.json --max-new-tokens 3000 --num-prompts 1
"""

import argparse
import asyncio
import hashlib
import json
import os

import sglang as sgl

PROMPT_DIR = "/home/swgandhi/pace/multiverse-engine/example/prompt"
# </Goal>, </Path> token ids for the Multiverse-32B tokenizer (from example.py).
BRANCH_STOP_TOKEN_IDS = [151670, 151674]


def construct_prompt(user_query: str) -> str:
    sysp = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    return (
        f"<|im_start|>system\n{sysp}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}\n<|im_end|>\n<|im_start|>assistant\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, help="baseline|off|eager|cap|pace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    ap.add_argument("--num-prompts", type=int, default=1)
    ap.add_argument("--model", default="Multiverse4FM/Multiverse-32B")
    ap.add_argument("--tp", type=int, default=8)
    a = ap.parse_args()

    kw = dict(
        model_path=a.model,
        tp_size=a.tp,
        dtype="bfloat16",
        disable_overlap_schedule=True,  # v1 targets normal mode
        log_level="info",
        random_seed=0,
        mem_fraction_static=0.85,
    )
    if a.mode != "baseline":
        kw["pace_enable"] = True
        kw["pace_mode"] = a.mode
        if a.mode == "cap":
            kw["pace_cap"] = 2

    engine = sgl.Engine(**kw)

    # The Engine constructor installs the uvloop event-loop policy; engine.generate()
    # then calls asyncio.get_event_loop(), which raises on a bare main thread.
    # Install a current loop AFTER construction.
    asyncio.set_event_loop(asyncio.new_event_loop())

    files = sorted(os.listdir(PROMPT_DIR))[: a.num_prompts]
    prompts = []
    for f in files:
        with open(os.path.join(PROMPT_DIR, f)) as fh:
            prompts.append(construct_prompt(fh.read()))

    sampling_params = {
        "temperature": 0.0,  # greedy => deterministic, for invariance comparison
        "top_p": 1.0,
        "max_new_tokens": a.max_new_tokens,
        "skip_special_tokens": False,
        "stop_token_ids": BRANCH_STOP_TOKEN_IDS,
    }

    texts = []
    for p in prompts:
        out = engine.generate(p, sampling_params)
        texts.append(out["text"])

    result = {
        "mode": a.mode,
        "files": files,
        "hashes": [hashlib.sha256(t.encode()).hexdigest() for t in texts],
        "lens": [len(t) for t in texts],
        "texts": texts,
    }
    with open(a.out, "w") as fh:
        json.dump(result, fh)
    print("WORKER_DONE", a.mode, "hashes=", result["hashes"], "lens=", result["lens"])
    engine.shutdown()


if __name__ == "__main__":
    main()
