"""Unit tests for PaceLatencyPredictor (T(S) = a + b*n_tokens + c*L_context)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from _pace_import import PaceLatencyPredictor  # noqa: E402


class TestPredictor(unittest.TestCase):
    def test_predict_linear(self):
        p = PaceLatencyPredictor(a=1.0, b=2.0, c=3.0)
        self.assertAlmostEqual(p.predict(4, 5), 1.0 + 8.0 + 15.0)

    def test_predict_clamped_nonnegative(self):
        p = PaceLatencyPredictor(a=-100.0, b=0.0, c=0.0)
        self.assertEqual(p.predict(1, 1), 0.0)

    def test_is_calibrated(self):
        self.assertFalse(PaceLatencyPredictor().is_calibrated)
        self.assertTrue(PaceLatencyPredictor(a=0.1).is_calibrated)

    def test_fit_recovers_known_coeffs(self):
        a, b, c = 0.5, 0.01, 1e-4
        p = PaceLatencyPredictor(window=2000, min_samples=10)
        for n in range(1, 40):
            for l in range(100, 2000, 100):
                p.record(n, l, a + b * n + c * l)
        self.assertTrue(p.refit())
        aa, bb, cc = p.coeffs
        self.assertAlmostEqual(aa, a, places=4)
        self.assertAlmostEqual(bb, b, places=5)
        self.assertAlmostEqual(cc, c, places=6)
        self.assertTrue(p.is_calibrated)
        self.assertLess(p.mape(), 1e-6)

    def test_window_eviction(self):
        p = PaceLatencyPredictor(window=10)
        for i in range(50):
            p.record(i + 1, i + 1, i + 1)
        self.assertEqual(p.num_observations, 10)

    def test_refit_below_min_samples_fails(self):
        p = PaceLatencyPredictor(window=100, min_samples=20)
        for i in range(5):
            p.record(i + 1, i + 1, i + 1)
        self.assertFalse(p.refit())

    def test_record_ignores_nonpositive_latency(self):
        p = PaceLatencyPredictor()
        p.record(1, 1, 0.0)
        p.record(1, 1, -5.0)
        self.assertEqual(p.num_observations, 0)

    def test_maybe_refit_respects_interval(self):
        a, b, c = 0.1, 0.02, 3e-4
        p = PaceLatencyPredictor(window=2000, refit_interval=10.0, min_samples=10)
        for n in range(1, 30):
            for l in range(100, 1000, 100):
                p.record(n, l, a + b * n + c * l)
        self.assertTrue(p.maybe_refit(now=100.0))  # first fit
        self.assertFalse(p.maybe_refit(now=105.0))  # within interval
        self.assertTrue(p.maybe_refit(now=120.0))  # interval elapsed

    def test_load_dump_round_trip(self):
        p = PaceLatencyPredictor(a=1.0, b=2.0, c=3.0)
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            p.dump_coeffs(path)
            q = PaceLatencyPredictor()
            self.assertFalse(q.is_calibrated)
            q.load_coeffs(path)
            self.assertEqual(q.coeffs, (1.0, 2.0, 3.0))
            self.assertTrue(q.is_calibrated)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
