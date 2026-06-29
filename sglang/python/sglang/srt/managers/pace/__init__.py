"""PACE: per-step branch-admission control for the Multiverse Engine.

It implements the per-step branch-admission controller from "Regulating
Branch Parallelism in LLM Serving" (arXiv:2605.06914).

This package contains the GPU-free core (latency predictor, greedy per-step
planner, config) plus the scheduler-side controller that applies per-step
branch width. See docs/pace/DESIGN.md for the full design.
"""

from sglang.srt.managers.pace.config import PaceConfig
from sglang.srt.managers.pace.controller import PaceController
from sglang.srt.managers.pace.metrics import PaceMetrics
from sglang.srt.managers.pace.planner import (
    AdmissionPolicy,
    BranchGroup,
    BranchRow,
    PacePlan,
    available_policies,
    build_groups,
    get_policy,
    plan,
    register_policy,
    register_utility,
)
from sglang.srt.managers.pace.predictor import PaceLatencyPredictor

__all__ = [
    "PaceConfig",
    "PaceController",
    "PaceMetrics",
    "PaceLatencyPredictor",
    "BranchRow",
    "BranchGroup",
    "PacePlan",
    "build_groups",
    "plan",
    "AdmissionPolicy",
    "register_policy",
    "get_policy",
    "available_policies",
    "register_utility",
]
