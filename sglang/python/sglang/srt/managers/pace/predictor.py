"""PACE step-latency predictor.

Implements the paper's linear model (DESIGN.md §2 / §6.2):

    T(S) = a + b * n_tokens + c * L_context

where ``n_tokens`` is the number of sequences advancing one token this step and
``L_context`` is their aggregate context length. Coefficients are fit offline by
OLS on a profiling grid (scripts/pace_calibrate.py) and refreshed online from a
rolling window of recently observed step latencies to track thermal/workload
drift.

This module depends only on numpy + stdlib so it is unit-testable without a GPU.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class PaceLatencyPredictor:
    """Linear OLS predictor of decode-step latency.

    Args:
        a, b, c: initial coefficients (seconds, seconds/seq, seconds/token).
            All-zero means "uncalibrated" until coefficients are loaded or fit.
        window: rolling-window capacity for online observations.
        refit_interval: seconds between online OLS refits.
        min_samples: minimum observations before a refit / before
            ``is_calibrated`` can become True via online data.
    """

    def __init__(
        self,
        a: float = 0.0,
        b: float = 0.0,
        c: float = 0.0,
        window: int = 200,
        refit_interval: float = 600.0,
        min_samples: int = 32,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.refit_interval = float(refit_interval)
        self.min_samples = int(min_samples)
        # Each entry: (n_tokens, L_context, latency_seconds)
        self._obs: Deque[Tuple[float, float, float]] = deque(maxlen=int(window))
        self._last_refit_time: Optional[float] = None
        # Calibrated if explicit non-zero coefficients were provided.
        self._calibrated_from_coeffs = any(v != 0.0 for v in (a, b, c))
        self._calibrated_from_fit = False

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, n_tokens: float, l_context: float) -> float:
        """Predicted step latency in seconds. Clamped to be non-negative."""
        t = self.a + self.b * n_tokens + self.c * l_context
        return t if t > 0.0 else 0.0

    @property
    def coeffs(self) -> Tuple[float, float, float]:
        return (self.a, self.b, self.c)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated_from_coeffs or self._calibrated_from_fit

    @property
    def num_observations(self) -> int:
        return len(self._obs)

    # ------------------------------------------------------------------ #
    # Online observation + refit
    # ------------------------------------------------------------------ #
    def record(self, n_tokens: float, l_context: float, latency: float) -> None:
        """Record one observed decode step. Ignores non-positive latency."""
        if latency is None or latency <= 0.0:
            return
        self._obs.append((float(n_tokens), float(l_context), float(latency)))

    def refit(self) -> bool:
        """Refit (a, b, c) by OLS over the current window. Returns success."""
        if len(self._obs) < self.min_samples:
            return False
        obs = np.asarray(self._obs, dtype=np.float64)  # [N, 3]
        n = obs[:, 0]
        l = obs[:, 1]
        y = obs[:, 2]
        # Design matrix [1, n_tokens, L_context].
        x = np.column_stack([np.ones_like(n), n, l])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        self.a, self.b, self.c = float(coef[0]), float(coef[1]), float(coef[2])
        self._calibrated_from_fit = True
        return True

    def maybe_refit(self, now: float) -> bool:
        """Refit if ``refit_interval`` has elapsed. Returns True if refit ran."""
        if self._last_refit_time is None or (
            now - self._last_refit_time >= self.refit_interval
        ):
            if self.refit():
                self._last_refit_time = now
                return True
            # Even if we couldn't fit (too few samples), advance the clock so we
            # don't try every step.
            if self._last_refit_time is None:
                self._last_refit_time = now
        return False

    def mape(self) -> Optional[float]:
        """Mean absolute percentage error of current coeffs over the window."""
        if not self._obs:
            return None
        obs = np.asarray(self._obs, dtype=np.float64)
        pred = self.a + self.b * obs[:, 0] + self.c * obs[:, 1]
        actual = obs[:, 2]
        nonzero = actual != 0.0
        if not nonzero.any():
            return None
        return float(
            np.mean(np.abs((pred[nonzero] - actual[nonzero]) / actual[nonzero]))
        )

    # ------------------------------------------------------------------ #
    # Offline coefficients I/O
    # ------------------------------------------------------------------ #
    def load_coeffs(self, path: str) -> None:
        """Load offline-calibrated coefficients from a json {a, b, c}."""
        with open(path, "r") as f:
            data = json.load(f)
        self.a = float(data["a"])
        self.b = float(data["b"])
        self.c = float(data["c"])
        self._calibrated_from_coeffs = True

    def dump_coeffs(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"a": self.a, "b": self.b, "c": self.c}, f)

    @classmethod
    def from_config(cls, cfg) -> "PaceLatencyPredictor":
        """Construct from a PaceConfig, loading offline coeffs if present."""
        pred = cls(
            window=cfg.window,
            refit_interval=cfg.refit_interval,
        )
        if getattr(cfg, "coeffs_path", None):
            pred.load_coeffs(cfg.coeffs_path)
        return pred
