"""PACE greedy per-step admission planner (paper Algorithm 1) + pluggable policies.

Pure, GPU-free, engine-free. The scheduler-side controller (controller.py, M2)
adapts live ``Req`` rows into :class:`BranchRow` / :class:`BranchGroup` objects
and calls :func:`plan`; everything here is unit-testable in isolation.

Concept mapping (DESIGN.md §4):
  * a *group* is one logical request; its *rows* are the resident, unfinished
    branch ``Req``s that share its prefix KV (``parent_rid``). A serial request
    is a singleton group.
  * the *baseline* advances exactly one row per group (every request makes one
    token of progress); extra rows are opportunistic.
  * width = number of admitted rows; deferred rows keep their KV resident.

Liveness (DESIGN.md §6.6): the baseline row of a group is its most-urgent
(earliest-deadline = least-recently-advanced) row, so under sustained pressure a
group degrades to rotating width-1 and no branch starves.

Extensibility (DESIGN.md §6.9): admission policies are pluggable. Subclass
:class:`AdmissionPolicy`, decorate with ``@register_policy("name")``, and select
it via ``--pace-mode name``. The five paper policies are registered below.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class BranchRow:
    """One resident decode row (a branch ``Req``) considered for this step."""

    batch_index: int  # index into ScheduleBatch.reqs
    group_id: str  # logical request id (parent_rid, or own rid if serial)
    context_len: float  # aggregate-context contribution for T(S) (e.g. right_most_pos)
    deadline: float  # absolute wall-clock deadline for this row's next token (s)
    right_most_pos: float = 0.0  # decode progress; §6.6 tie-break (least-advanced first)


@dataclass
class BranchGroup:
    """All resident rows of one logical request."""

    group_id: str
    rows: List[BranchRow]
    priority: float = 1.0

    def __post_init__(self) -> None:
        # Liveness order: earliest deadline first (most starved advances first);
        # ties broken by least decode progress, then batch index (DESIGN.md §6.6),
        # so siblings restamped to identical deadlines still rotate rather than
        # pinning the baseline on one row forever.
        self.rows = sorted(
            self.rows, key=lambda r: (r.deadline, r.right_most_pos, r.batch_index)
        )

    @property
    def baseline_row(self) -> BranchRow:
        return self.rows[0]

    @property
    def extra_rows(self) -> List[BranchRow]:
        return self.rows[1:]

    @property
    def ready_branches(self) -> int:
        """Opportunistic branches available beyond the baseline (paper n_r-1)."""
        return max(0, len(self.rows) - 1)


def build_groups(
    rows: List[BranchRow], priorities: Optional[Dict[str, float]] = None
) -> List[BranchGroup]:
    """Group rows by ``group_id``, preserving first-seen group order."""
    priorities = priorities or {}
    order: List[str] = []
    buckets: Dict[str, List[BranchRow]] = {}
    for r in rows:
        if r.group_id not in buckets:
            buckets[r.group_id] = []
            order.append(r.group_id)
        buckets[r.group_id].append(r)
    return [
        BranchGroup(gid, buckets[gid], priorities.get(gid, 1.0)) for gid in order
    ]


# --------------------------------------------------------------------------- #
# Utility registry (DESIGN.md §6.7)
# --------------------------------------------------------------------------- #
def _utility_linear(group: BranchGroup, k: int) -> float:
    return float(k)


def _utility_concave(group: BranchGroup, k: int) -> float:
    return math.log1p(k)


def _utility_priority(group: BranchGroup, k: int) -> float:
    return float(k) * group.priority


_UTILITY: Dict[str, Callable[[BranchGroup, int], float]] = {
    "linear": _utility_linear,
    "concave": _utility_concave,
    "priority": _utility_priority,
}


def register_utility(name: str, fn: Callable[[BranchGroup, int], float]) -> None:
    _UTILITY[name] = fn


# --------------------------------------------------------------------------- #
# Plan result
# --------------------------------------------------------------------------- #
@dataclass
class PacePlan:
    """Output of one planning step."""

    admitted_indices: List[int]
    deferred_indices: List[int]
    n_tokens: int  # admitted row count (== width sum)
    l_context: float  # aggregate context length of admitted rows
    t0: float  # baseline predicted latency T(S_0)
    budget: float  # T_0 + rho * B_t
    min_slack: float  # min_r(deadline_r) - now
    predicted_latency: float  # T(admitted)
    branch_externality: float  # predicted_latency - t0
    num_groups: int
    policy: str = ""

    @property
    def num_admitted(self) -> int:
        return len(self.admitted_indices)

    @property
    def num_deferred(self) -> int:
        return len(self.deferred_indices)


# --------------------------------------------------------------------------- #
# Shared planning primitives (used by policies)
# --------------------------------------------------------------------------- #
def _all_rows(groups: List[BranchGroup]) -> List[BranchRow]:
    return [r for g in groups for r in g.rows]


def _baseline(groups: List[BranchGroup], predictor):
    rows = [g.baseline_row for g in groups]
    n0 = len(rows)
    l0 = float(sum(r.context_len for r in rows))
    t0 = predictor.predict(n0, l0)
    return rows, n0, l0, t0


def _finalize(
    groups, admitted_rows, predictor, t0, budget, min_slack, policy_name
) -> PacePlan:
    all_indices = [r.batch_index for g in groups for r in g.rows]
    admitted_idx = [r.batch_index for r in admitted_rows]
    admitted_set = set(admitted_idx)
    deferred_idx = [i for i in all_indices if i not in admitted_set]
    n = len(admitted_rows)
    l = float(sum(r.context_len for r in admitted_rows))
    predicted = predictor.predict(n, l)
    return PacePlan(
        admitted_indices=admitted_idx,
        deferred_indices=deferred_idx,
        n_tokens=n,
        l_context=l,
        t0=t0,
        budget=budget,
        min_slack=min_slack,
        predicted_latency=predicted,
        branch_externality=predicted - t0,
        num_groups=len(groups),
        policy=policy_name,
    )


def greedy_admit(groups, predictor, cfg, now, extra_limit, policy_name) -> PacePlan:
    """Paper Algorithm 1: greedily admit the max utility-per-cost branch while
    the predicted step latency stays within the slack budget.

    ``extra_limit`` caps extra branches per group (None = unbounded; used by the
    capped baselines).
    """
    baseline_rows, n0, l0, t0 = _baseline(groups, predictor)
    all_rows = _all_rows(groups)
    min_slack = min(r.deadline for r in all_rows) - now
    budget = t0 + cfg.rho * max(0.0, min_slack - t0)
    util = _UTILITY[cfg.utility]

    group_by_id: Dict[str, BranchGroup] = {g.group_id: g for g in groups}
    granted: Dict[str, int] = {g.group_id: 0 for g in groups}
    pending: Dict[str, List[BranchRow]] = {}
    for g in groups:
        extras = g.extra_rows
        if extra_limit is not None:
            extras = extras[: max(0, extra_limit)]
        if extras:
            pending[g.group_id] = list(extras)

    admitted_rows = list(baseline_rows)
    n, l, t_cur = n0, l0, t0

    while pending:
        best = None  # (score, gid, row, n2, l2, t2)
        infeasible: List[str] = []
        for gid, queue in pending.items():
            row = queue[0]
            n2 = n + 1
            l2 = l + row.context_len
            t2 = predictor.predict(n2, l2)
            if t2 > budget:
                # Adding any branch only grows (n, L) and hence T, so this
                # group's next branch cannot become feasible later.
                infeasible.append(gid)
                continue
            g = group_by_id[gid]
            du = util(g, granted[gid] + 1) - util(g, granted[gid])
            dt = t2 - t_cur
            score = du / (EPS + (dt if dt > 0.0 else 0.0))
            if best is None or score > best[0]:
                best = (score, gid, row, n2, l2, t2)
        for gid in infeasible:
            pending.pop(gid, None)
        if best is None or best[0] <= 0.0:
            break
        score, gid, row, n2, l2, t2 = best
        admitted_rows.append(row)
        n, l, t_cur = n2, l2, t2
        granted[gid] += 1
        queue = pending[gid]
        queue.pop(0)
        if not queue:
            pending.pop(gid)

    return _finalize(groups, admitted_rows, predictor, t0, budget, min_slack, policy_name)


# --------------------------------------------------------------------------- #
# Pluggable admission policies (DESIGN.md §6.9)
# --------------------------------------------------------------------------- #
class AdmissionPolicy(ABC):
    """Decide which branch rows advance this decode step.

    Subclass, decorate with ``@register_policy("name")``, and select via
    ``--pace-mode name``. Implementations should be stateless (one cached
    instance per name is reused across steps).
    """

    name: str = ""

    @abstractmethod
    def select(self, groups: List[BranchGroup], predictor, cfg, now: float) -> PacePlan:
        ...


_POLICY_CLASSES: Dict[str, type] = {}
_POLICY_INSTANCES: Dict[str, AdmissionPolicy] = {}


def register_policy(name: str):
    def deco(cls):
        cls.name = name
        _POLICY_CLASSES[name] = cls
        return cls

    return deco


def get_policy(name: str) -> AdmissionPolicy:
    if name not in _POLICY_INSTANCES:
        if name not in _POLICY_CLASSES:
            raise ValueError(
                f"unknown pace policy {name!r}; available: {sorted(_POLICY_CLASSES)}"
            )
        _POLICY_INSTANCES[name] = _POLICY_CLASSES[name]()
    return _POLICY_INSTANCES[name]


def available_policies() -> List[str]:
    return sorted(_POLICY_CLASSES)


@register_policy("off")
class OffPolicy(AdmissionPolicy):
    """IRP-Off: width 1 per request (baseline only, no opportunistic branches)."""

    def select(self, groups, predictor, cfg, now):
        groups = [g for g in groups if g.rows]
        if not groups:
            return PacePlan([], [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, self.name)
        baseline_rows, _, _, t0 = _baseline(groups, predictor)
        all_rows = _all_rows(groups)
        min_slack = min(r.deadline for r in all_rows) - now
        return _finalize(groups, baseline_rows, predictor, t0, t0, min_slack, self.name)


@register_policy("eager")
class EagerPolicy(AdmissionPolicy):
    """IRP-Eager: advance every resident branch (the engine's default today)."""

    def select(self, groups, predictor, cfg, now):
        groups = [g for g in groups if g.rows]
        if not groups:
            return PacePlan([], [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, self.name)
        _, _, _, t0 = _baseline(groups, predictor)
        all_rows = _all_rows(groups)
        min_slack = min(r.deadline for r in all_rows) - now
        return _finalize(
            groups, all_rows, predictor, t0, math.inf, min_slack, self.name
        )


@register_policy("pace")
class PacePolicy(AdmissionPolicy):
    """PACE: slack-budget-driven greedy admission (paper Algorithm 1)."""

    def select(self, groups, predictor, cfg, now):
        groups = [g for g in groups if g.rows]
        if not groups:
            return PacePlan([], [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, self.name)
        return greedy_admit(groups, predictor, cfg, now, extra_limit=None, policy_name=self.name)


@register_policy("cap")
class CapPolicy(AdmissionPolicy):
    """IRP-C{N}: greedy admission but width capped at ``cfg.cap`` per request."""

    def select(self, groups, predictor, cfg, now):
        groups = [g for g in groups if g.rows]
        if not groups:
            return PacePlan([], [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, self.name)
        # baseline already gives width 1, so allow cap-1 extra rows per group.
        extra_limit = (cfg.cap - 1) if cfg.cap else 0
        return greedy_admit(
            groups, predictor, cfg, now, extra_limit=extra_limit, policy_name=self.name
        )


# --------------------------------------------------------------------------- #
# Entry point used by the controller
# --------------------------------------------------------------------------- #
def plan(groups: List[BranchGroup], predictor, cfg, now: float) -> PacePlan:
    """Resolve the policy named by ``cfg.mode`` and run it for this step."""
    return get_policy(cfg.mode).select(groups, predictor, cfg, now)
