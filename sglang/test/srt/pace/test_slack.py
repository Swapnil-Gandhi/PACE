"""Unit tests for the slack-budget math: B_t = max(0, min_slack - T_0),
budget = T_0 + rho * B_t (DESIGN.md §6.3 / §2)."""

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


def pred():
    return PaceLatencyPredictor(a=0.0, b=1.0, c=0.0)  # T = n_tokens


def groups(deadline):
    # 3 singleton groups => t0 = 3 (T = n_tokens). No extras (pure budget math).
    rows = [BranchRow(i, f"g{i}", 1, deadline) for i in range(3)]
    return build_groups(rows)


class TestSlack(unittest.TestCase):
    def test_min_slack_reported(self):
        # deadlines differ; min_slack = min(deadline) - now.
        rows = [
            BranchRow(0, "a", 1, 10.0),
            BranchRow(1, "b", 1, 4.0),
            BranchRow(2, "c", 1, 7.0),
        ]
        p = plan(build_groups(rows), pred(), PaceConfig(mode="pace", rho=0.8), now=1.0)
        self.assertAlmostEqual(p.min_slack, 4.0 - 1.0)

    def test_budget_formula_positive_slack(self):
        # t0 = 3, deadline 10, now 0 => min_slack 10.
        # budget = 3 + 0.5 * (10 - 3) = 6.5
        p = plan(groups(10.0), pred(), PaceConfig(mode="pace", rho=0.5), now=0.0)
        self.assertEqual(p.t0, 3.0)
        self.assertAlmostEqual(p.budget, 3.0 + 0.5 * (10.0 - 3.0))

    def test_budget_clamped_when_slack_below_t0(self):
        # min_slack (2) < t0 (3) => B_t = 0 => budget = t0.
        p = plan(groups(2.0), pred(), PaceConfig(mode="pace", rho=0.8), now=0.0)
        self.assertEqual(p.budget, p.t0)

    def test_budget_clamped_when_slack_negative(self):
        p = plan(groups(1.0), pred(), PaceConfig(mode="pace", rho=0.8), now=5.0)
        self.assertLess(p.min_slack, 0.0)
        self.assertEqual(p.budget, p.t0)

    def test_rho_scales_budget(self):
        b_low = plan(groups(13.0), pred(), PaceConfig(mode="pace", rho=0.2), now=0.0).budget
        b_high = plan(groups(13.0), pred(), PaceConfig(mode="pace", rho=0.9), now=0.0).budget
        self.assertLess(b_low, b_high)
        # rho=0.2: 3 + 0.2*10 = 5 ; rho=0.9: 3 + 0.9*10 = 12
        self.assertAlmostEqual(b_low, 5.0)
        self.assertAlmostEqual(b_high, 12.0)


if __name__ == "__main__":
    unittest.main()
