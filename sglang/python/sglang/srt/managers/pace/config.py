"""PACE configuration.

A standalone dataclass so the GPU-free core can be tested without the engine.
`PaceConfig.from_server_args` reads the (forthcoming) `--pace-*` server args via
``getattr`` with defaults, so this module does not depend on the server_args
edits landing first (M1 before M2).

See docs/pace/DESIGN.md §8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Admission policy modes. A single code path realizes all five paper policies
# (DESIGN.md §6.8): off=IRP-Off, cap+--pace-cap N=IRP-C{N}, eager=IRP-Eager,
# pace=the full slack-driven planner.
PACE_MODES = ("off", "cap", "eager", "pace")

# How aggregate context length L_context is measured (DESIGN.md §6.2 / D2).
PACE_CONTEXT_MODES = ("raw", "dedup")

# Supported utility functions (DESIGN.md §6.7).
PACE_UTILITIES = ("linear", "concave", "priority")


@dataclass
class PaceConfig:
    """Tunables for the PACE per-step branch-admission controller."""

    enable: bool = False
    mode: str = "pace"
    cap: Optional[int] = None
    rho: float = 0.8
    tpot_slo_ms: float = 50.0
    utility: str = "linear"
    window: int = 200
    refit_interval: float = 600.0
    coeffs_path: Optional[str] = None
    context_mode: str = "raw"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        # Built-in modes are PACE_MODES, but custom policies may be registered
        # (planner.register_policy) and selected by name, so we only require a
        # non-empty string here; planner.get_policy raises a clear error if the
        # name is not registered.
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError(f"pace mode must be a non-empty string, got {self.mode!r}")
        if self.context_mode not in PACE_CONTEXT_MODES:
            raise ValueError(
                f"pace context_mode must be one of {PACE_CONTEXT_MODES}, "
                f"got {self.context_mode!r}"
            )
        if self.utility not in PACE_UTILITIES:
            raise ValueError(
                f"pace utility must be one of {PACE_UTILITIES}, got {self.utility!r}"
            )
        if not (0.0 < self.rho <= 1.0):
            raise ValueError(f"pace rho must be in (0, 1], got {self.rho}")
        if self.tpot_slo_ms <= 0:
            raise ValueError(f"pace tpot_slo_ms must be > 0, got {self.tpot_slo_ms}")
        if self.mode == "cap":
            if self.cap is None or self.cap < 1:
                raise ValueError("pace mode 'cap' requires --pace-cap >= 1")
        if self.window < 1:
            raise ValueError(f"pace window must be >= 1, got {self.window}")

    @property
    def tpot_slo_s(self) -> float:
        return self.tpot_slo_ms / 1000.0

    @classmethod
    def from_server_args(cls, server_args) -> "PaceConfig":
        """Build from a ServerArgs-like object, tolerating missing attributes."""
        g = lambda name, default: getattr(server_args, name, default)
        return cls(
            enable=g("pace_enable", False),
            mode=g("pace_mode", "pace"),
            cap=g("pace_cap", None),
            rho=g("pace_rho", 0.8),
            tpot_slo_ms=g("pace_tpot_slo_ms", 50.0),
            utility=g("pace_utility", "linear"),
            window=g("pace_window", 200),
            refit_interval=g("pace_refit_interval", 600.0),
            coeffs_path=g("pace_coeffs_path", None),
            context_mode=g("pace_context_mode", "raw"),
        )
