"""PaceController: the scheduler-side glue for PACE (DESIGN.md §5, §7).

Lives between the engine's decode-batch formation and the forward pass. Each
decode step the scheduler:

    1. controller.merge_deferred(batch)     # re-admit last step's deferred rows
    2. batch.filter_batch()                  # existing: drop finished / fork / merge
    3. (return early if is_zombie/is_merge)  # existing
    4. controller.plan_and_apply(batch)      # PACE: trim to admitted, stash deferred
    5. batch.prepare_for_decode()            # existing: advance admitted rows only

and after the forward:

    6. controller.record_observation(n_tokens, l_context, latency)  # refresh T(S)

The controller never frees KV: deferral is just batch-membership control, so a
deferred branch keeps its prefix + branch-local KV resident and is re-admitted
for free next step (validated by the no-reclamation test).

To keep engine surgery minimal it touches NO Req constructor: branch grouping is
derived from the existing ``batch.p2c_map`` (parent_rid -> [child_rids]); per-row
timing is stored as lazily-stamped instance attributes (``pace_created`` /
``pace_last_token``).
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from sglang.srt.managers.pace.config import PaceConfig
from sglang.srt.managers.pace.metrics import PaceMetrics
from sglang.srt.managers.pace.planner import BranchRow, PacePlan, build_groups, plan
from sglang.srt.managers.pace.predictor import PaceLatencyPredictor

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)


def _now() -> float:
    # Monotonic clock for deadlines/slack (immune to wall-clock jumps).
    return time.monotonic()


class PaceController:
    def __init__(self, cfg: PaceConfig):
        self.cfg = cfg
        self.predictor = PaceLatencyPredictor.from_config(cfg)
        self.metrics = PaceMetrics()
        # Deferred branch rows held across one step (a ScheduleBatch or None).
        self.deferred_batch: Optional["ScheduleBatch"] = None
        # Optional per-group priority overrides (group_id -> weight).
        self.priorities: Dict[str, float] = {}
        # rids that were present in the decode batch on the *previous* planned
        # step. A rid absent here but present now has just (re)entered decode
        # after a serial/fork/merge stall: its pace_last_token clock is stale, so
        # we restamp it to "now" before building its deadline (DESIGN.md §6.3
        # re-entry rule). This is what keeps a one-off fork/merge stall from
        # poisoning min_slack and forcing over-conservative deferral.
        self._seen_rids: set = set()
        # PACE_DEBUG: verify defer->re-admit preserves per-branch state.
        self._dbg = bool(os.environ.get("PACE_DEBUG"))
        self._dbg_state: Dict[str, tuple] = {}  # rid -> (rmp, n_out, admitted_last)
        self._dbg_violations = 0
        logger.info(
            "PACE enabled: mode=%s rho=%s tpot_slo_ms=%s utility=%s calibrated=%s",
            cfg.mode,
            cfg.rho,
            cfg.tpot_slo_ms,
            cfg.utility,
            self.predictor.is_calibrated,
        )

    # ------------------------------------------------------------------ #
    # Per-row feature extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reverse_p2c(batch: "ScheduleBatch") -> Dict[str, str]:
        """child_rid -> parent_rid from the engine's parent->children map."""
        rev: Dict[str, str] = {}
        p2c = getattr(batch, "p2c_map", None) or {}
        for parent_rid, child_rids in p2c.items():
            for crid in child_rids:
                rev[crid] = parent_rid
        return rev

    def _context_len(self, req: "Req") -> float:
        rp = getattr(req, "right_most_pos", 0) or 0
        if self.cfg.context_mode == "dedup":
            spi = getattr(req, "start_path_idx", -1)
            if spi and spi > 0:
                # Count only this branch's private suffix; the shared prefix is
                # counted once via the group's baseline row.
                return float(max(1, rp - spi))
        return float(max(1, rp))

    def _deadline(self, req: "Req", now: float) -> float:
        """Absolute deadline for this row's next token: last_token + tpot_slo.

        Two refinements over the naive ``pace_last_token + tpot_slo`` keep a
        one-off fork/merge stall from poisoning ``min_slack`` (DESIGN.md §6.3):

        1. Re-entry restamp. A row that just (re)entered the decode batch this
           step (first-seen, or absent from the previous step while it sat in a
           serial/fork/merge phase) has a stale ``pace_last_token`` from before
           the stall. We reset its clock to ``now`` so its next-token deadline is
           measured from when it actually re-started decoding -- not from a
           timestamp that predates the stall. A row that re-enters has, by
           definition, just made progress, so it is correctly NOT the most
           starved (it gets a fresh, far-future deadline ``now + tpot_slo`` and
           will not be chosen as a group's baseline row -- liveness still picks
           the genuinely least-recently-advanced row, §6.6).
        2. Stale-deadline clamp. Even for a row that has been continuously
           decoding, a single transient long step can leave its deadline up to
           one extra SLO interval in the past. We bound how far below ``now`` a
           deadline may sit at one SLO interval, so any single row can pull
           ``min_slack`` down by at most one SLO interval -- never to a runaway
           negative value -- while a runaway-stale (pre-stall) clock is already
           reset by rule 1. The clamp is *order-preserving*: rows past the floor
           are mapped through a strictly monotonic (logarithmic) compression of
           the overdue region, so a more-overdue row still sorts strictly before
           a less-overdue one. This keeps the liveness rotation intact -- the
           least-recently-advanced (most overdue) branch is still each group's
           baseline row (DESIGN.md §6.6) -- it only bounds the magnitude that
           feeds the budget, not the ordering.
        """
        seen = req.rid in self._seen_rids
        last = getattr(req, "pace_last_token", None)
        if last is None or not seen:
            # First-seen OR re-entered after a stall: (re)stamp the clock to now.
            req.pace_last_token = now
            if getattr(req, "pace_created", None) is None:
                req.pace_created = now  # keep first-seen time for diagnostics
            last = now
        slo = self.cfg.tpot_slo_s
        deadline = last + slo
        # Order-preserving clamp: deadlines at/above (now - slo) pass through
        # unchanged; deadlines below it are compressed into the band
        # (now - 2*slo, now - slo] via log1p, which is strictly increasing in the
        # overdue amount. min_slack is thus floored at -2*slo (a one-off stall
        # can no longer pin the budget) while strict urgency order is preserved.
        floor = now - slo
        if deadline >= floor:
            return deadline
        overdue = floor - deadline  # > 0, grows with staleness
        return floor - slo * math.log1p(overdue / slo) / (1.0 + math.log1p(overdue / slo))

    def _build_rows(self, batch: "ScheduleBatch", now: float) -> List[BranchRow]:
        rev = self._reverse_p2c(batch)
        rows: List[BranchRow] = []
        for i, req in enumerate(batch.reqs):
            rows.append(
                BranchRow(
                    batch_index=i,
                    group_id=rev.get(req.rid, req.rid),
                    context_len=self._context_len(req),
                    deadline=self._deadline(req, now),
                    right_most_pos=float(getattr(req, "right_most_pos", 0) or 0),
                )
            )
        return rows

    # ------------------------------------------------------------------ #
    # Step hooks (called from Scheduler.update_running_batch)
    # ------------------------------------------------------------------ #
    def merge_deferred(self, batch: "ScheduleBatch") -> None:
        """Re-admit rows deferred on the previous step into this step's batch."""
        if self.deferred_batch is not None and not self.deferred_batch.is_empty():
            batch.merge_batch(self.deferred_batch)
        self.deferred_batch = None

    def plan_and_apply(
        self, batch: "ScheduleBatch", now: Optional[float] = None
    ) -> Optional[PacePlan]:
        """Run the planner and trim ``batch`` to the admitted subset in place.

        Returns the PacePlan (None if PACE is disabled/empty or in cold-start,
        in which case the batch is left at full width).
        """
        if not self.cfg.enable or batch is None:
            return None
        if batch.is_empty():
            self._seen_rids = set()
            return None
        if now is None:
            now = _now()
        t_start = time.perf_counter()

        # Cold start: until the predictor is calibrated, run full width (eager)
        # but keep gathering observations so it can fit. Off/eager modes don't
        # need the predictor and proceed normally.
        if not self.predictor.is_calibrated and self.cfg.mode not in ("off", "eager"):
            self.metrics.note_uncalibrated()
            self.metrics.record_plan_overhead(time.perf_counter() - t_start)
            # Keep decode-batch membership fresh even on this early return, so
            # resident rows are not spuriously treated as re-entrants (and mass-
            # restamped to identical deadlines) next step — verify REQUIRED 2c,
            # which closes the tie-break starvation hazard (DESIGN.md §6.6).
            self._seen_rids = {req.rid for req in batch.reqs}
            return None

        # _build_rows applies the re-entry restamp (rows absent last step get a
        # fresh now-based clock), reading the previous step's membership from
        # self._seen_rids. Refresh _seen_rids to *this* step's full candidate set
        # (all resident rows, admitted or deferred) so a row deferred here is
        # still "seen" and is NOT spuriously treated as a re-entrant next step --
        # deferred rows must keep their (aging) clock to stay on the liveness
        # rotation (DESIGN.md §6.6).
        rows = self._build_rows(batch, now)
        self._seen_rids = {req.rid for req in batch.reqs}
        groups = build_groups(rows, self.priorities)
        p = plan(groups, self.predictor, self.cfg, now)

        # Stamp admitted rows' last-token time -> their next deadline. Deferred
        # rows keep an older time, so they grow more urgent and rotate in next
        # step (the liveness mechanism, DESIGN.md §6.6).
        admitted_set = set(p.admitted_indices)
        for i, req in enumerate(batch.reqs):
            if i in admitted_set:
                req.pace_last_token = now

        if self._dbg:
            self._dbg_check(batch, admitted_set)

        if p.num_deferred > 0:
            # Trim batch to admitted rows; stash the deferred rows (KV stays
            # resident — no free/evict). Re-admitted next step via merge_deferred.
            self.deferred_batch = batch.pace_split(p.admitted_indices)

        self.metrics.record_plan(p, time.perf_counter() - t_start)
        return p

    def _dbg_check(self, batch, admitted_set) -> None:
        """Assert defer->re-admit preserves per-branch state (PACE_DEBUG)."""
        for i, req in enumerate(batch.reqs):
            rmp = getattr(req, "right_most_pos", 0)
            n_out = len(req.output_ids)
            prev = self._dbg_state.get(req.rid)
            if prev is not None:
                p_rmp, p_out, p_adm = prev
                if n_out < p_out:
                    # rid reused for a new incarnation (Multiverse fork/merge
                    # rebuilds a parent Req with the same rid + reset output_ids);
                    # not a deferral event — reset baseline, don't flag.
                    ok = True
                elif p_adm:  # advanced last step -> +1
                    ok = (rmp == p_rmp + 1) and (n_out == p_out + 1)
                else:  # deferred last step -> unchanged
                    ok = (rmp == p_rmp) and (n_out == p_out)
                if not ok:
                    self._dbg_violations += 1
                    logger.error(
                        "PACE_DEBUG state violation rid=%s adm_last=%s "
                        "rmp %s->%s n_out %s->%s",
                        req.rid, p_adm, p_rmp, rmp, p_out, n_out,
                    )
            self._dbg_state[req.rid] = (rmp, n_out, i in admitted_set)

    # ------------------------------------------------------------------ #
    # Post-forward hooks
    # ------------------------------------------------------------------ #
    def record_observation(
        self,
        n_tokens: int,
        l_context: float,
        latency: float,
        now: Optional[float] = None,
    ) -> None:
        """Feed an observed decode-step latency to the online predictor."""
        if not self.cfg.enable:
            return
        self.predictor.record(n_tokens, l_context, latency)
        self.predictor.maybe_refit(now if now is not None else _now())
        self.metrics.record_latency(latency)

    def record_finished(self, req: "Req") -> None:
        """Account a completed request toward live goodput / SLO-attainment.

        Mirrors the eval harness's effective-TPOT goodput (DESIGN.md §2; D3):
        a request meets SLO iff its mean decode inter-token latency is within
        the TPOT SLO, and its generated tokens are then counted as SLO-meeting.
        Reuses the per-row timing PACE already stamps — ``pace_created`` (first
        decode step) and ``pace_last_token`` (last advance) — plus the
        generated-token count, so it needs no extra per-request state. This is
        what makes the scheduler-side goodput counters live (exposed via
        ``snapshot`` → ``get_internal_state()["pace"]`` and the decode log).
        """
        if not self.cfg.enable:
            return
        n = len(getattr(req, "output_ids", None) or ())
        created = getattr(req, "pace_created", None)
        last = getattr(req, "pace_last_token", None)
        if created is None or last is None or n < 2 or last <= created:
            # Too short to define a TPOT (finished in prefill, or never planned
            # by PACE): count it completed without penalizing goodput.
            self.metrics.record_request_finished(True)
            if n > 0:
                self.metrics.record_token(True, count=n)
            return
        eff_tpot = (last - created) / (n - 1)
        slo_met = eff_tpot <= self.cfg.tpot_slo_s
        self.metrics.record_request_finished(slo_met)
        self.metrics.record_token(slo_met, count=n)

    def snapshot(self) -> dict:
        snap = self.metrics.snapshot()
        snap["predictor_coeffs"] = self.predictor.coeffs
        snap["predictor_calibrated"] = self.predictor.is_calibrated
        snap["predictor_mape"] = self.predictor.mape()
        return snap
