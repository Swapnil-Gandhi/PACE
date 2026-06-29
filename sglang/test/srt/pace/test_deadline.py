"""Unit tests for the deadline-model refinement (DESIGN.md §6.3, area=deadline-fix).

Root cause #3 from the H200 eval (§17): a row that re-enters the decode batch
after a serial/fork/merge stall carries a stale ``pace_last_token``. Its deadline
(``pace_last_token + tpot_slo``) is then far in the past, so it dominates
``min_slack = min(deadline) - now``, driving it negative. That collapses the
slack budget ``B_t = max(0, min_slack - T0)`` to 0 (budget = T0), rejecting all
opportunistic branches -- even when steps are cheap and the true branch
externality is ~0. These tests pin the controller-side ``_deadline`` logic that
fixes this WITHOUT touching the planner's B_t semantics or breaking liveness.

Pure: drives PaceController._deadline / _build_rows / plan_and_apply against a
tiny fake Req/ScheduleBatch (no torch, no GPU).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from _pace_import import (  # noqa: E402
    PaceConfig,
    PaceController,
    PaceLatencyPredictor,
)


# --------------------------------------------------------------------------- #
# Minimal engine-shaped fakes (only the attributes the controller touches)
# --------------------------------------------------------------------------- #
class FakeReq:
    def __init__(self, rid, right_most_pos=600, output_ids=()):
        self.rid = rid
        self.right_most_pos = right_most_pos
        self.start_path_idx = -1
        self.output_ids = list(output_ids)
        # pace_* attributes are stamped lazily by the controller, exactly as on a
        # real Req. We deliberately do NOT pre-create them.


class FakeBatch:
    def __init__(self, reqs, p2c_map=None):
        self.reqs = list(reqs)
        self.p2c_map = p2c_map or {}
        self.split_calls = []

    def is_empty(self):
        return not self.reqs

    def pace_split(self, admitted_indices):
        # Record the admitted set; return a stub "deferred batch" (truthy).
        self.split_calls.append(list(admitted_indices))
        return object()


# H200 calibration (TP8, bf16): T(S) = a + b*n + c*L.
CALIB = dict(a=0.006991055847584451, b=3.9378319196954575e-05, c=1.427312349692437e-08)


def calibrated_predictor():
    return PaceLatencyPredictor(**CALIB)


def controller(mode="pace", slo_ms=25.0, rho=0.8):
    cfg = PaceConfig(enable=True, mode=mode, rho=rho, tpot_slo_ms=slo_ms)
    c = PaceController(cfg)
    c.predictor = calibrated_predictor()  # force calibrated, deterministic
    return c


class TestDeadlineReentry(unittest.TestCase):
    def test_first_seen_row_gets_now_based_deadline(self):
        # A brand-new row (no pace_last_token, not in _seen_rids) must be timed
        # from `now`, not from 0 / created-in-the-past.
        c = controller()
        req = FakeReq("r0")
        now = 100.0
        d = c._deadline(req, now)
        self.assertAlmostEqual(d, now + c.cfg.tpot_slo_s)
        self.assertAlmostEqual(req.pace_last_token, now)

    def test_reentrant_row_is_restamped_not_poisoned(self):
        # Row last advanced at t=100; it then stalled in a fork/merge round trip
        # for 200ms and re-enters decode at now=100.2. Because its rid is NOT in
        # _seen_rids (absent last step), the clock is reset to `now` => deadline
        # is now + slo (far future), NOT 100 + slo (175ms in the past).
        c = controller(slo_ms=25.0)
        req = FakeReq("r0")
        req.pace_last_token = 100.0  # stale, pre-stall
        now = 100.2  # 200ms later
        self.assertNotIn(req.rid, c._seen_rids)
        d = c._deadline(req, now)
        self.assertAlmostEqual(d, now + c.cfg.tpot_slo_s)
        # And the naive (poisoned) deadline would have been in the past:
        naive = 100.0 + c.cfg.tpot_slo_s
        self.assertLess(naive - now, 0.0)
        self.assertGreater(d - now, 0.0)

    def test_continuously_decoding_row_keeps_its_clock(self):
        # A row present last step (in _seen_rids) is NOT restamped on read: its
        # deadline stays anchored to its real last-token time so the budget still
        # reflects genuine urgency.
        c = controller(slo_ms=25.0)
        req = FakeReq("r0")
        req.pace_last_token = 100.0
        c._seen_rids = {"r0"}
        now = 100.010  # 10ms since last token, still within SLO
        d = c._deadline(req, now)
        self.assertAlmostEqual(d, 100.0 + c.cfg.tpot_slo_s)  # unchanged anchor
        self.assertAlmostEqual(req.pace_last_token, 100.0)  # not restamped


class TestStaleDeadlineClamp(unittest.TestCase):
    def test_clamp_floors_min_slack_contribution(self):
        # A continuously-decoding row hit by one transient long step can fall
        # arbitrarily far past deadline. The clamp bounds how far below `now` its
        # deadline may sit at ~2*slo, so min_slack can't run away to -inf.
        c = controller(slo_ms=25.0)
        req = FakeReq("r0")
        req.pace_last_token = 50.0  # ages ago
        c._seen_rids = {"r0"}
        now = 100.0  # ~50s overdue
        d = c._deadline(req, now)
        slack = d - now
        self.assertGreaterEqual(slack, -2.0 * c.cfg.tpot_slo_s - 1e-9)
        self.assertLess(slack, 0.0)  # still negative (urgent), just bounded

    def test_clamp_preserves_strict_urgency_order(self):
        # Two equally-resident rows, one more overdue than the other, must keep
        # strict ordering after clamping so liveness (§6.6) still advances the
        # least-recently-advanced branch as the group's baseline row.
        c = controller(slo_ms=25.0)
        c._seen_rids = {"old", "newer"}
        older = FakeReq("old")
        older.pace_last_token = 10.0
        newer = FakeReq("newer")
        newer.pace_last_token = 40.0
        now = 100.0  # both far overdue, both clamped
        d_old = c._deadline(older, now)
        d_new = c._deadline(newer, now)
        self.assertLess(d_old, d_new)  # more overdue => earlier deadline

    def test_clamp_is_continuous_passthrough_above_floor(self):
        # Deadlines at/above (now - slo) are returned unchanged (no compression).
        c = controller(slo_ms=25.0)
        req = FakeReq("r0")
        req.pace_last_token = 100.0
        c._seen_rids = {"r0"}
        now = 100.020  # deadline = 125ms, well above the floor (now - slo)
        d = c._deadline(req, now)
        self.assertAlmostEqual(d, 100.0 + c.cfg.tpot_slo_s)


class TestPlanAndApplyEndToEnd(unittest.TestCase):
    """The payoff: cheap steps + one re-entrant row => extras still admitted."""

    def _decomp_batch(self, n_groups=8, branches_per_group=4, ctx=600):
        """n_groups logical requests, each with 1 baseline + (b-1) extra branch
        rows sharing a parent (group) id via p2c_map."""
        reqs = []
        p2c = {}
        idx = 0
        for g in range(n_groups):
            parent = f"p{g}"
            children = []
            for b in range(branches_per_group):
                rid = f"{parent}_b{b}"
                reqs.append(FakeReq(rid, right_most_pos=ctx))
                children.append(rid)
                idx += 1
            p2c[parent] = children
        return FakeBatch(reqs, p2c_map=p2c)

    def test_burst_all_fresh_admits_all_when_steps_cheap(self):
        # First planned step: nobody is in _seen_rids, so every row is first-seen
        # and gets a now-based deadline => huge slack => with externality ~0 the
        # planner admits the full width (no spurious deferral).
        c = controller(slo_ms=25.0, rho=0.8)
        batch = self._decomp_batch(n_groups=8, branches_per_group=4)
        now = 100.0
        p = c.plan_and_apply(batch, now=now)
        self.assertIsNotNone(p)
        self.assertEqual(p.num_deferred, 0, "cheap step + fresh rows must not defer")
        self.assertEqual(p.num_admitted, 32)
        # externality is genuinely negligible at this width:
        self.assertLess(p.branch_externality, 0.002)  # < 2 ms
        self.assertEqual(batch.split_calls, [])  # pace_split never called

    def test_one_reentrant_row_does_not_poison_admission(self):
        # Step 1 establishes _seen_rids. Step 2: one branch was absent (stalled in
        # fork/merge) and re-enters with a stale pace_last_token. Under the OLD
        # logic that single row's past deadline would collapse the budget to T0
        # and defer all extras. With the refinement it is restamped, slack stays
        # large, and extras remain admitted.
        c = controller(slo_ms=25.0, rho=0.8)
        batch1 = self._decomp_batch(n_groups=8, branches_per_group=4)
        c.plan_and_apply(batch1, now=100.0)  # populate _seen_rids + stamp

        # Build step-2 batch: same rids, but one row (p0_b3) "stalled" -- we
        # simulate the poison by making its pace_last_token stale AND removing it
        # from the previous membership so it counts as a re-entrant.
        batch2 = self._decomp_batch(n_groups=8, branches_per_group=4)
        for r in batch2.reqs:
            r.pace_last_token = 100.0  # all advanced at t=100
        stale = next(r for r in batch2.reqs if r.rid == "p0_b3")
        stale.pace_last_token = 99.0  # 1s stale -- would be ~975ms past deadline
        c._seen_rids.discard("p0_b3")  # it was absent last step (the stall)

        now2 = 100.010  # 10ms later, a normal cheap decode cadence
        p = c.plan_and_apply(batch2, now=now2)
        self.assertEqual(
            p.num_deferred, 0,
            "a single re-entrant stale row must not collapse the budget",
        )
        self.assertEqual(p.num_admitted, 32)

    def test_naive_deadline_would_have_poisoned(self):
        # Guard/oracle: demonstrate that the *naive* deadline (no restamp, no
        # clamp) on the same scenario DOES poison min_slack -> budget == T0. This
        # documents what the refinement fixes (and fails loudly if someone
        # reverts to last + slo with no re-entry handling).
        from _pace_import import build_groups, plan as planner_plan, BranchRow

        slo = 0.025
        now = 100.010
        # 8 baselines + 24 extras, all ctx 600; one row stale by ~1s.
        rows = []
        bi = 0
        for g in range(8):
            for b in range(4):
                last = 99.0 if (g == 0 and b == 3) else 100.0
                naive_deadline = last + slo  # the OLD _deadline, no restamp/clamp
                rows.append(BranchRow(bi, f"p{g}", 600.0, naive_deadline))
                bi += 1
        groups = build_groups(rows)
        cfg = PaceConfig(enable=True, mode="pace", rho=0.8, tpot_slo_ms=25.0)
        p = planner_plan(groups, calibrated_predictor(), cfg, now)
        # Poisoned: min_slack < 0 => budget collapses to T0 => baseline only.
        self.assertLess(p.min_slack, 0.0)
        self.assertAlmostEqual(p.budget, p.t0)
        self.assertEqual(p.num_admitted, 8)  # all 24 extras wrongly deferred


class TestEagerBitIdentical(unittest.TestCase):
    """Eager never consults the budget, so the deadline refinement cannot change
    its admitted set (the baseline == pace-eager bit-identical gate, §12.1)."""

    def _batch(self):
        reqs = [FakeReq(f"p0_b{b}", right_most_pos=600) for b in range(4)]
        reqs += [FakeReq(f"p1_b{b}", right_most_pos=600) for b in range(4)]
        p2c = {"p0": ["p0_b0", "p0_b1", "p0_b2", "p0_b3"],
               "p1": ["p1_b0", "p1_b1", "p1_b2", "p1_b3"]}
        return FakeBatch(reqs, p2c_map=p2c)

    def test_eager_admits_all_regardless_of_stale_deadlines(self):
        c = controller(mode="eager", slo_ms=25.0)
        batch = self._batch()
        # Inject maximally-poisoned, stale, re-entrant timestamps.
        for r in batch.reqs:
            r.pace_last_token = 1.0  # ~99s past deadline
        c._seen_rids = {r.rid for r in batch.reqs}  # treated as continuous
        p = c.plan_and_apply(batch, now=100.0)
        self.assertEqual(p.num_admitted, 8)
        self.assertEqual(p.num_deferred, 0)
        self.assertEqual(batch.split_calls, [])  # never splits => identical FP

    def test_eager_identical_with_and_without_refined_deadlines(self):
        # The admitted index list must be identical whether deadlines are fresh
        # or poisoned -- eager's output is deadline-independent by construction.
        c = controller(mode="eager", slo_ms=25.0)
        b_fresh = self._batch()
        p_fresh = c.plan_and_apply(b_fresh, now=100.0)

        c2 = controller(mode="eager", slo_ms=25.0)
        b_poison = self._batch()
        for r in b_poison.reqs:
            r.pace_last_token = 1.0
        p_poison = c2.plan_and_apply(b_poison, now=100.0)
        self.assertEqual(p_fresh.admitted_indices, p_poison.admitted_indices)


class TestLivenessRotation(unittest.TestCase):
    """Under sustained zero-slack the group must still rotate width-1 across all
    branches (no branch starves) -- the clamp must not break this (§6.6)."""

    def test_deferred_rows_rotate_under_pressure(self):
        # Tiny SLO so the budget is always T0 (baseline only). Across steps the
        # baseline (= least-recently-advanced row) must rotate through the group
        # so every branch eventually advances.
        c = controller(mode="pace", slo_ms=0.1, rho=0.8)  # 0.1ms => always tight
        reqs = [FakeReq(f"p0_b{b}", right_most_pos=600) for b in range(4)]
        batch = FakeBatch(reqs, p2c_map={"p0": [r.rid for r in reqs]})
        now = 100.0
        advanced = set()
        for step in range(12):
            now += 0.010  # 10ms per step
            # Re-find the rows (same objects; engine keeps them resident).
            p = c.plan_and_apply(batch, now=now)
            # The one admitted row this step is the group's baseline; record it.
            self.assertGreaterEqual(p.num_admitted, 1)
            for i in p.admitted_indices:
                advanced.add(batch.reqs[i].rid)
        # Every branch advanced at least once over the window => no starvation.
        self.assertEqual(advanced, {r.rid for r in reqs})


if __name__ == "__main__":
    unittest.main()
