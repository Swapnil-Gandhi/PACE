"""GPU-free unit tests for PaceController feature extraction:
L_context raw vs dedup (D2) and p2c_map-based branch grouping."""

import json
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
from _pace_import import PaceConfig, PaceController  # noqa: E402


class FakeReq:
    def __init__(self, rid, rmp=0, spi=-1):
        self.rid = rid
        self.right_most_pos = rmp
        self.start_path_idx = spi
        self.output_ids = []


class FakeBatch:
    def __init__(self, reqs, p2c=None):
        self.reqs = reqs
        self.p2c_map = p2c or {}


def ctrl(context_mode="raw"):
    return PaceController(PaceConfig(enable=True, context_mode=context_mode))


class TestContextLen(unittest.TestCase):
    def test_raw_uses_right_most_pos(self):
        self.assertEqual(ctrl("raw")._context_len(FakeReq("a", rmp=500, spi=100)), 500.0)

    def test_dedup_subtracts_shared_prefix(self):
        # dedup counts only the branch's private suffix (rmp - start_path_idx)
        self.assertEqual(ctrl("dedup")._context_len(FakeReq("a", rmp=500, spi=100)), 400.0)

    def test_dedup_serial_no_start_path_falls_back_to_raw(self):
        self.assertEqual(ctrl("dedup")._context_len(FakeReq("a", rmp=500, spi=-1)), 500.0)

    def test_dedup_floors_at_one(self):
        self.assertEqual(ctrl("dedup")._context_len(FakeReq("a", rmp=100, spi=100)), 1.0)


class TestGrouping(unittest.TestCase):
    def test_reverse_p2c_and_build_rows(self):
        c = ctrl()
        b = FakeBatch(
            [FakeReq("c1", rmp=10), FakeReq("c2", rmp=10), FakeReq("s1", rmp=10)],
            p2c={"parent": ["c1", "c2"]},
        )
        rev = c._reverse_p2c(b)
        self.assertEqual(rev.get("c1"), "parent")
        self.assertEqual(rev.get("c2"), "parent")
        self.assertIsNone(rev.get("s1"))
        rows = c._build_rows(b, now=100.0)
        gid = {r.batch_index: r.group_id for r in rows}
        self.assertEqual(gid[0], "parent")  # branch -> parent group
        self.assertEqual(gid[1], "parent")
        self.assertEqual(gid[2], "s1")  # serial -> own rid


class TestGoodput(unittest.TestCase):
    """record_finished wires the (previously dead) scheduler-side goodput /
    SLO-attainment counters, mirroring the eval harness's effective-TPOT
    goodput from the per-row timing PACE already stamps."""

    @staticmethod
    def _req(n, created, last):
        # output_ids length = generated-token count; pace_created/pace_last_token
        # are stamped by the controller during planning.
        return SimpleNamespace(
            output_ids=list(range(n)), pace_created=created, pace_last_token=last
        )

    def _ctrl(self, **kw):
        return PaceController(PaceConfig(enable=True, tpot_slo_ms=50.0, **kw))

    def test_slo_met_counts_tokens_and_request(self):
        c = self._ctrl()
        # 10 tokens over 0.09 s -> 10 ms/token <= 50 ms SLO -> met
        c.record_finished(self._req(10, 0.0, 0.09))
        m = c.metrics
        self.assertEqual(m.requests_completed, 1)
        self.assertEqual(m.requests_slo_met, 1)
        self.assertEqual(m.slo_met_tokens, 10)
        self.assertEqual(m.slo_missed_tokens, 0)
        self.assertEqual(m.slo_attainment, 1.0)

    def test_slo_missed(self):
        c = self._ctrl()
        # 10 tokens over 1.0 s -> ~111 ms/token > 50 ms SLO -> missed
        c.record_finished(self._req(10, 0.0, 1.0))
        m = c.metrics
        self.assertEqual(m.requests_completed, 1)
        self.assertEqual(m.requests_slo_met, 0)
        self.assertEqual(m.slo_met_tokens, 0)
        self.assertEqual(m.slo_missed_tokens, 10)
        self.assertEqual(m.slo_attainment, 0.0)

    def test_too_short_to_measure_tpot_counts_completed_no_penalty(self):
        c = self._ctrl()
        c.record_finished(self._req(1, 0.0, 0.0))  # single token: no TPOT defined
        m = c.metrics
        self.assertEqual(m.requests_completed, 1)
        self.assertEqual(m.requests_slo_met, 1)
        self.assertEqual(m.slo_met_tokens, 1)
        self.assertEqual(m.slo_missed_tokens, 0)

    def test_missing_timing_is_safe(self):
        c = self._ctrl()
        c.record_finished(SimpleNamespace(output_ids=[1, 2, 3]))  # no pace_* stamps
        m = c.metrics
        self.assertEqual(m.requests_completed, 1)
        self.assertEqual(m.slo_met_tokens, 3)

    def test_disabled_is_noop(self):
        c = PaceController(PaceConfig(enable=False, tpot_slo_ms=50.0))
        c.record_finished(self._req(10, 0.0, 1.0))
        m = c.metrics
        self.assertEqual(m.requests_completed, 0)
        self.assertEqual(m.slo_met_tokens, 0)
        self.assertEqual(m.slo_missed_tokens, 0)


class TestSnapshotJsonSafe(unittest.TestCase):
    """The pace snapshot is re-exported verbatim by the HTTP /get_server_info
    endpoint (which spreads **internal_states), so it must be JSON-serializable.
    Guards against a future non-primitive (e.g. a numpy value) sneaking into
    snapshot() and 500-ing the whole endpoint."""

    def _ctrl_with_data(self):
        c = PaceController(PaceConfig(enable=True, tpot_slo_ms=50.0))
        c.record_observation(8, 1000.0, 0.009)  # predictor obs + step latency
        c.record_finished(  # populate the goodput counters
            SimpleNamespace(
                output_ids=list(range(10)), pace_created=0.0, pace_last_token=0.09
            )
        )
        return c

    def test_snapshot_is_json_serializable(self):
        snap = self._ctrl_with_data().snapshot()
        s = json.dumps(snap)  # must not raise
        self.assertEqual(json.loads(s)["slo_met_tokens"], 10)
        self.assertIn("predictor_coeffs", snap)
        self.assertIn("slo_attainment", snap)

    def test_get_server_info_merge_exposes_pace(self):
        # Mirror entrypoints get_server_info: {**server_args, **scheduler_info,
        # **internal_states, "version": ...}, internal_states carrying "pace".
        snap = self._ctrl_with_data().snapshot()
        internal_states = {"last_gen_throughput": 0.0, "pace": snap}
        merged = {"served_model_name": "m", **internal_states, "version": "x"}
        out = json.loads(json.dumps(merged))  # the HTTP response serialization
        self.assertIn("pace", out)
        self.assertEqual(out["pace"]["slo_met_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
