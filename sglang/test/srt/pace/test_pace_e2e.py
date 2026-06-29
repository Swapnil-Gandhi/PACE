"""PACE end-to-end correctness tests (DESIGN.md §11.2, §12).

These are HEAVY (load Multiverse-32B on 8×H200) and are skipped unless
PACE_RUN_E2E=1. They drive `_pace_e2e_worker.py` as subprocesses (one clean
TP/GPU state per config) and assert the correctness gates established during
bring-up:

  * plumbing gate:  baseline  ≡  pace-eager   (bit-identical; eager keeps full
    width ⇒ identical batch composition ⇒ identical FP). This is the strong
    correctness signal on a non-batch-invariant engine.
  * determinism:    pace-off  ≡  pace-off      (no random corruption).

Exact match across DIFFERENT widths (off vs eager) is intentionally NOT asserted:
the engine has no batch-invariant kernels, so different compositions diverge by
floating point (Lemma 3.1 is exact-arithmetic). Deferred-branch state integrity
is checked separately by running a worker with PACE_DEBUG=1 (0 violations).

Run:  PACE_RUN_E2E=1 python -m unittest test.srt.pace.test_pace_e2e -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "_pace_e2e_worker.py")
RUN_E2E = os.environ.get("PACE_RUN_E2E") == "1"
MODEL = os.environ.get("PACE_E2E_MODEL", "Multiverse4FM/Multiverse-32B")
MNT = os.environ.get("PACE_E2E_MAX_NEW_TOKENS", "1500")


def _run_worker(mode: str, out_path: str, env_extra=None):
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/tmp/pace-env/hf")
    if env_extra:
        env.update(env_extra)
    cmd = [
        sys.executable, WORKER,
        "--mode", mode, "--out", out_path,
        "--max-new-tokens", MNT, "--num-prompts", "1", "--model", MODEL,
    ]
    subprocess.run(cmd, env=env, check=True)
    with open(out_path) as f:
        return json.load(f)


@unittest.skipUnless(RUN_E2E, "set PACE_RUN_E2E=1 to run heavy GPU e2e tests")
class TestPaceEndToEnd(unittest.TestCase):
    def test_baseline_equals_eager_bit_identical(self):
        """Plumbing gate: full-width PACE is a faithful no-op vs no-PACE."""
        with tempfile.TemporaryDirectory() as d:
            base = _run_worker("baseline", os.path.join(d, "base.json"))
            eager = _run_worker("eager", os.path.join(d, "eager.json"))
        self.assertEqual(base["hashes"], eager["hashes"])

    def test_pace_off_deterministic(self):
        """No random corruption: pace-off is reproducible run-to-run."""
        with tempfile.TemporaryDirectory() as d:
            a = _run_worker("off", os.path.join(d, "off1.json"))
            b = _run_worker("off", os.path.join(d, "off2.json"))
        self.assertEqual(a["hashes"], b["hashes"])

    def test_pace_off_no_state_violations(self):
        """Deferred-branch state integrity: PACE_DEBUG reports 0 violations."""
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "dbg.json")
            log = os.path.join(d, "dbg.log")
            env = dict(os.environ)
            env.setdefault("HF_HOME", "/tmp/pace-env/hf")
            env["PACE_DEBUG"] = "1"
            with open(log, "w") as fh:
                subprocess.run(
                    [sys.executable, WORKER, "--mode", "off", "--out", out,
                     "--max-new-tokens", MNT, "--num-prompts", "1", "--model", MODEL],
                    env=env, check=True, stdout=fh, stderr=subprocess.STDOUT,
                )
            with open(log) as fh:
                violations = sum("state violation" in line for line in fh)
        self.assertEqual(violations, 0)


if __name__ == "__main__":
    unittest.main()
