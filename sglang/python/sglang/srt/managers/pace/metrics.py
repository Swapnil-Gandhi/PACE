"""PACE metrics (DESIGN.md §9).

Lightweight counters/aggregates the scheduler can read and export: goodput
(SLO-meeting tokens/s), branch externality, planner overhead, predictor health,
and width/deferral stats. Pure stdlib so it imports without torch.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional


class PaceMetrics:
    def __init__(self, window: int = 1000) -> None:
        # planner overhead (seconds) per step
        self._overhead: Deque[float] = deque(maxlen=window)
        # branch externality (seconds) per step
        self._externality: Deque[float] = deque(maxlen=window)
        # admitted-row count per step
        self._width: Deque[int] = deque(maxlen=window)
        self._deferred: Deque[int] = deque(maxlen=window)
        # observed step latency (seconds)
        self._latency: Deque[float] = deque(maxlen=window)

        self.steps_planned: int = 0
        self.steps_uncalibrated: int = 0
        # Lifetime counters (survive the rolling window; needed for whole-run
        # fractions and peaks that the eval harness reports — DESIGN.md §17).
        self.steps_with_deferral: int = 0  # steps that deferred >=1 row
        self.peak_width: int = 0  # max admitted rows in any single step
        self.peak_resident: int = 0  # max (admitted + deferred) rows in any step
        self.sum_width: int = 0  # for lifetime mean width
        self.sum_resident: int = 0  # for lifetime mean resident (decode-batch) size
        self.peak_externality_s: float = 0.0  # max branch externality observed (s)
        # goodput accounting
        self.slo_met_tokens: int = 0
        self.slo_missed_tokens: int = 0
        self.requests_completed: int = 0
        self.requests_slo_met: int = 0

    # ---- per-step recording ----
    def record_plan(self, plan, overhead_s: float) -> None:
        self.steps_planned += 1
        self._overhead.append(overhead_s)
        ext = max(0.0, plan.branch_externality)
        self._externality.append(ext)
        self._width.append(plan.num_admitted)
        self._deferred.append(plan.num_deferred)
        # Lifetime aggregates (decode-batch sizing + deferral fraction). The
        # resident decode batch is admitted+deferred (deferred rows keep KV).
        resident = plan.num_admitted + plan.num_deferred
        if plan.num_deferred > 0:
            self.steps_with_deferral += 1
        self.peak_width = max(self.peak_width, plan.num_admitted)
        self.peak_resident = max(self.peak_resident, resident)
        self.sum_width += plan.num_admitted
        self.sum_resident += resident
        self.peak_externality_s = max(self.peak_externality_s, ext)

    def record_plan_overhead(self, overhead_s: float) -> None:
        self._overhead.append(overhead_s)

    def note_uncalibrated(self) -> None:
        self.steps_uncalibrated += 1

    def record_latency(self, latency_s: float) -> None:
        if latency_s and latency_s > 0:
            self._latency.append(latency_s)

    # ---- goodput accounting ----
    def record_token(self, slo_met: bool, count: int = 1) -> None:
        if slo_met:
            self.slo_met_tokens += count
        else:
            self.slo_missed_tokens += count

    def record_request_finished(self, slo_met: bool) -> None:
        self.requests_completed += 1
        if slo_met:
            self.requests_slo_met += 1

    # ---- summaries ----
    @staticmethod
    def _median(d: Deque) -> Optional[float]:
        if not d:
            return None
        s = sorted(d)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    @staticmethod
    def _p(d: Deque, q: float) -> Optional[float]:
        if not d:
            return None
        s = sorted(d)
        idx = min(len(s) - 1, int(q * len(s)))
        return s[idx]

    @property
    def slo_attainment(self) -> Optional[float]:
        if self.requests_completed == 0:
            return None
        return self.requests_slo_met / self.requests_completed

    def snapshot(self) -> dict:
        sp = self.steps_planned or 1  # avoid div-by-zero in lifetime fractions
        return {
            "steps_planned": self.steps_planned,
            "steps_uncalibrated": self.steps_uncalibrated,
            "planner_overhead_ms_median": _ms(self._median(self._overhead)),
            "planner_overhead_ms_p99": _ms(self._p(self._overhead, 0.99)),
            "branch_externality_ms_median": _ms(self._median(self._externality)),
            "branch_externality_ms_p99": _ms(self._p(self._externality, 0.99)),
            "branch_externality_ms_peak": _ms(self.peak_externality_s),
            "mean_width": (sum(self._width) / len(self._width)) if self._width else None,
            "mean_deferred": (
                sum(self._deferred) / len(self._deferred) if self._deferred else None
            ),
            # Lifetime decode-batch sizing + deferral fraction (isolate whether
            # PACE is operating in the decode-step-bound regime — DESIGN.md §17).
            "mean_width_lifetime": self.sum_width / sp,
            "mean_resident_lifetime": self.sum_resident / sp,
            "peak_width": self.peak_width,
            "peak_resident": self.peak_resident,
            "frac_steps_deferred": self.steps_with_deferral / sp,
            "slo_met_tokens": self.slo_met_tokens,
            "slo_missed_tokens": self.slo_missed_tokens,
            "slo_attainment": self.slo_attainment,
        }


def _ms(x: Optional[float]) -> Optional[float]:
    return None if x is None else x * 1000.0
