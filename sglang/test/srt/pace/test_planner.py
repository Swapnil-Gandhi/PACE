"""Unit tests for the PACE greedy planner + pluggable policies (Algorithm 1).

Uses a deterministic linear predictor so expected admissions are hand-computable.
With a=0, b=1, c=0 the predicted step latency equals the number of admitted rows,
so the slack budget maps directly to "how many extra branches fit".
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from _pace_import import (  # noqa: E402
    BranchRow,
    PaceConfig,
    PaceLatencyPredictor,
    build_groups,
    plan,
)


def pred(a=0.0, b=1.0, c=0.0):
    return PaceLatencyPredictor(a=a, b=b, c=c)


def cfg(mode="pace", rho=1.0, cap=None, utility="linear"):
    return PaceConfig(enable=True, mode=mode, rho=rho, cap=cap, utility=utility)


def two_groups_three_rows(deadline=1000.0):
    """2 groups, each 1 baseline + 2 extras (6 rows). Indices 0..5."""
    rows = [
        BranchRow(0, "g0", 1, deadline),
        BranchRow(1, "g0", 1, deadline),
        BranchRow(2, "g0", 1, deadline),
        BranchRow(3, "g1", 1, deadline),
        BranchRow(4, "g1", 1, deadline),
        BranchRow(5, "g1", 1, deadline),
    ]
    return build_groups(rows)


class TestPolicies(unittest.TestCase):
    def test_off_admits_baseline_only(self):
        p = plan(two_groups_three_rows(), pred(), cfg(mode="off"), now=0.0)
        self.assertEqual(p.num_admitted, 2)  # one row per group
        self.assertEqual(p.num_deferred, 4)

    def test_eager_admits_all_regardless_of_slack(self):
        # Tight deadlines: eager ignores the budget entirely.
        p = plan(two_groups_three_rows(deadline=1.0), pred(), cfg(mode="eager"), now=0.0)
        self.assertEqual(p.num_admitted, 6)
        self.assertEqual(p.num_deferred, 0)

    def test_pace_huge_slack_admits_all(self):
        p = plan(two_groups_three_rows(deadline=1000.0), pred(), cfg(mode="pace"), now=0.0)
        self.assertEqual(p.num_admitted, 6)

    def test_pace_tight_slack_baseline_only(self):
        # deadline == t0 (=2): budget = t0, no extra fits (3 > 2).
        p = plan(two_groups_three_rows(deadline=2.0), pred(), cfg(mode="pace"), now=0.0)
        self.assertEqual(p.num_admitted, 2)

    def test_pace_negative_slack_baseline_only(self):
        p = plan(two_groups_three_rows(deadline=0.5), pred(), cfg(mode="pace"), now=0.0)
        self.assertEqual(p.num_admitted, 2)

    def test_pace_partial_admission_counts(self):
        # t0 = 2 groups; deadline 4 => budget 4 => admit exactly 2 extras (n<=4).
        p = plan(two_groups_three_rows(deadline=4.0), pred(), cfg(mode="pace"), now=0.0)
        self.assertEqual(p.num_admitted, 4)
        self.assertEqual(p.t0, 2.0)
        self.assertEqual(p.budget, 4.0)
        self.assertEqual(p.min_slack, 4.0)
        self.assertEqual(p.predicted_latency, 4.0)
        self.assertEqual(p.branch_externality, 2.0)

    def test_cap_limits_width_per_group(self):
        # cap=2 => baseline (1) + at most 1 extra per group, even with huge slack.
        p = plan(two_groups_three_rows(deadline=1000.0), pred(), cfg(mode="cap", cap=2), now=0.0)
        self.assertEqual(p.num_admitted, 4)  # 2 baseline + 1 extra each
        # the third row of each group (idx 2 and 5) is deferred
        self.assertEqual(sorted(p.deferred_indices), [2, 5])

    def test_complement_is_partition(self):
        p = plan(two_groups_three_rows(deadline=3.0), pred(), cfg(mode="pace"), now=0.0)
        admitted = set(p.admitted_indices)
        deferred = set(p.deferred_indices)
        self.assertEqual(admitted | deferred, {0, 1, 2, 3, 4, 5})
        self.assertEqual(admitted & deferred, set())

    def test_admission_respects_budget(self):
        for d in (2.0, 3.0, 4.0, 5.0, 7.0):
            p = plan(two_groups_three_rows(deadline=d), pred(), cfg(mode="pace"), now=0.0)
            self.assertLessEqual(p.predicted_latency, p.budget + 1e-9)

    def test_determinism(self):
        groups1 = two_groups_three_rows(deadline=4.0)
        groups2 = two_groups_three_rows(deadline=4.0)
        a = plan(groups1, pred(), cfg(mode="pace"), now=0.0)
        b = plan(groups2, pred(), cfg(mode="pace"), now=0.0)
        self.assertEqual(a.admitted_indices, b.admitted_indices)

    def test_serial_singleton_groups(self):
        rows = [BranchRow(0, "a", 1, 5.0), BranchRow(1, "b", 1, 5.0)]
        p = plan(build_groups(rows), pred(), cfg(mode="pace"), now=0.0)
        self.assertEqual(p.num_admitted, 2)
        self.assertEqual(p.num_deferred, 0)

    def test_empty(self):
        p = plan([], pred(), cfg(mode="pace"), now=0.0)
        self.assertEqual(p.num_admitted, 0)
        self.assertEqual(p.num_deferred, 0)

    def test_priority_utility_prefers_high_priority_branch(self):
        # 2 groups, each baseline+1 extra; budget allows exactly ONE extra.
        rows = [
            BranchRow(0, "hi", 1, 3.0),
            BranchRow(1, "hi", 1, 3.0),
            BranchRow(2, "lo", 1, 3.0),
            BranchRow(3, "lo", 1, 3.0),
        ]
        groups = build_groups(rows, priorities={"hi": 5.0, "lo": 1.0})
        # t0=2 groups, deadline 3 => budget 3 => room for exactly one extra.
        p = plan(groups, pred(), cfg(mode="pace", utility="priority"), now=0.0)
        self.assertEqual(p.num_admitted, 3)
        self.assertIn(1, p.admitted_indices)  # hi's extra admitted
        self.assertIn(3, p.deferred_indices)  # lo's extra deferred

    def test_greedy_prefers_cheaper_branch(self):
        # cost-aware: with c=1 (T=L), prefer the lower-context extra (max du/dt).
        rows = [
            BranchRow(0, "a", 10, 30.0),  # baseline a
            BranchRow(1, "a", 5, 30.0),  # cheap extra (context 5)
            BranchRow(2, "b", 10, 30.0),  # baseline b
            BranchRow(3, "b", 100, 30.0),  # expensive extra (context 100)
        ]
        groups = build_groups(rows)
        predictor = pred(a=0.0, b=0.0, c=1.0)  # T = L_context
        # baseline L0 = 20 => t0=20. budget chosen to fit only the cheap extra.
        # deadline 30 => min_slack 30 => budget = 20 + 1*(30-20) = 30.
        # add cheap: L=25 (ok); add expensive: L=120 (>30).
        p = plan(groups, predictor, cfg(mode="pace", rho=1.0), now=0.0)
        self.assertEqual(p.budget, 30.0)
        self.assertIn(1, p.admitted_indices)  # cheap extra admitted
        self.assertIn(3, p.deferred_indices)  # expensive extra deferred

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            plan(two_groups_three_rows(), pred(), cfg(mode="does_not_exist"), now=0.0)


class TestCustomPolicyPlugin(unittest.TestCase):
    def test_register_and_use_custom_policy(self):
        from _pace_import import planner_mod

        @planner_mod.register_policy("admit_nothing_extra")
        class AdmitNothingExtra(planner_mod.AdmissionPolicy):
            def select(self, groups, predictor, cfg_, now):
                groups = [g for g in groups if g.rows]
                baseline, _, _, t0 = planner_mod._baseline(groups, predictor)
                all_rows = planner_mod._all_rows(groups)
                min_slack = min(r.deadline for r in all_rows) - now
                return planner_mod._finalize(
                    groups, baseline, predictor, t0, t0, min_slack, self.name
                )

        self.assertIn("admit_nothing_extra", planner_mod.available_policies())
        p = plan(
            two_groups_three_rows(deadline=1000.0),
            pred(),
            cfg(mode="admit_nothing_extra"),
            now=0.0,
        )
        self.assertEqual(p.num_admitted, 2)
        self.assertEqual(p.policy, "admit_nothing_extra")


if __name__ == "__main__":
    unittest.main()
