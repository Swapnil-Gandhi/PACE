"""Unit tests for branch grouping, liveness ordering, and ready-branch counts."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from _pace_import import BranchGroup, BranchRow, build_groups  # noqa: E402


class TestGrouping(unittest.TestCase):
    def test_build_groups_preserves_first_seen_order(self):
        rows = [
            BranchRow(0, "a", 10, 1.0),
            BranchRow(1, "b", 10, 1.0),
            BranchRow(2, "a", 10, 2.0),
        ]
        groups = build_groups(rows)
        self.assertEqual([g.group_id for g in groups], ["a", "b"])
        ga = next(g for g in groups if g.group_id == "a")
        self.assertEqual(len(ga.rows), 2)

    def test_rows_sorted_by_deadline_for_liveness(self):
        g = BranchGroup(
            "a",
            [
                BranchRow(0, "a", 1, 5.0),
                BranchRow(1, "a", 1, 2.0),
                BranchRow(2, "a", 1, 9.0),
            ],
        )
        # baseline = earliest deadline = most starved row.
        self.assertEqual(g.baseline_row.batch_index, 1)
        self.assertEqual([r.batch_index for r in g.extra_rows], [0, 2])

    def test_tiebreak_equal_deadline_equal_progress_by_index(self):
        g = BranchGroup(
            "a",
            [BranchRow(5, "a", 1, 3.0), BranchRow(3, "a", 1, 3.0), BranchRow(7, "a", 1, 3.0)],
        )
        # equal deadlines + equal progress -> deterministic tie-break by
        # batch_index (DESIGN.md §6.6 order: deadline, right_most_pos, index).
        self.assertEqual([r.batch_index for r in g.rows], [3, 5, 7])

    def test_tiebreak_equal_deadline_prefers_least_progress(self):
        # §6.6: with equal deadlines (e.g. siblings restamped on simultaneous
        # re-entry), the least-advanced branch becomes the baseline so all
        # branches rotate instead of pinning one forever.
        g = BranchGroup(
            "a",
            [
                BranchRow(0, "a", 1, 3.0, right_most_pos=50),
                BranchRow(1, "a", 1, 3.0, right_most_pos=10),
                BranchRow(2, "a", 1, 3.0, right_most_pos=30),
            ],
        )
        self.assertEqual(g.baseline_row.batch_index, 1)  # least right_most_pos
        self.assertEqual([r.batch_index for r in g.rows], [1, 2, 0])

    def test_ready_branches(self):
        self.assertEqual(
            BranchGroup("a", [BranchRow(0, "a", 1, 1.0)]).ready_branches, 0
        )
        self.assertEqual(
            BranchGroup(
                "a", [BranchRow(0, "a", 1, 1.0), BranchRow(1, "a", 1, 2.0)]
            ).ready_branches,
            1,
        )

    def test_priority_passthrough(self):
        groups = build_groups(
            [BranchRow(0, "hi", 1, 1.0), BranchRow(1, "lo", 1, 1.0)],
            priorities={"hi": 5.0},
        )
        gp = {g.group_id: g.priority for g in groups}
        self.assertEqual(gp["hi"], 5.0)
        self.assertEqual(gp["lo"], 1.0)


if __name__ == "__main__":
    unittest.main()
