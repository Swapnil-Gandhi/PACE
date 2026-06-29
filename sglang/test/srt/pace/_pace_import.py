"""Import shim for PACE unit tests.

Tries the normal package import first (works once sglang is installed). Falls
back to loading the GPU-free core modules directly by file path so the unit
tests run with only numpy installed (no torch / no full sglang package), which
is the M1 promise: the core is testable in isolation.
"""

import importlib.util
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
# .../sglang/test/srt/pace -> .../sglang/python/sglang/srt/managers/pace
_CORE = os.path.normpath(
    os.path.join(
        _THIS, "..", "..", "..", "python", "sglang", "srt", "managers", "pace"
    )
)


def _load(modname, filename):
    full = f"_pace_core_{modname}"
    spec = importlib.util.spec_from_file_location(full, os.path.join(_CORE, filename))
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass (under `from __future__ import
    # annotations`) can resolve the module's namespace via sys.modules.
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_controller_standalone():
    """Load controller.py without importing torch/full sglang.

    controller.py uses absolute imports (``from sglang.srt.managers.pace.X``),
    so we register the already-loaded GPU-free core modules under those names in
    ``sys.modules`` before exec'ing it. ``Req``/``ScheduleBatch`` are only used
    under ``TYPE_CHECKING`` in controller.py, so no torch is pulled in.
    """
    pkg = "sglang.srt.managers.pace"
    sys.modules.setdefault(pkg + ".config", config_mod)
    sys.modules.setdefault(pkg + ".predictor", predictor_mod)
    sys.modules.setdefault(pkg + ".planner", planner_mod)
    sys.modules.setdefault(pkg + ".metrics", _load("metrics", "metrics.py"))
    return _load("controller", "controller.py")


try:  # normal path once the package is importable
    from sglang.srt.managers.pace import config as config_mod  # type: ignore
    from sglang.srt.managers.pace import controller as controller_mod  # type: ignore
    from sglang.srt.managers.pace import planner as planner_mod  # type: ignore
    from sglang.srt.managers.pace import predictor as predictor_mod  # type: ignore
except Exception:  # pragma: no cover - fallback for lightweight unit runs
    config_mod = _load("config", "config.py")
    predictor_mod = _load("predictor", "predictor.py")
    planner_mod = _load("planner", "planner.py")
    controller_mod = _load_controller_standalone()

PaceConfig = config_mod.PaceConfig
PaceController = controller_mod.PaceController
PaceLatencyPredictor = predictor_mod.PaceLatencyPredictor
BranchRow = planner_mod.BranchRow
BranchGroup = planner_mod.BranchGroup
PacePlan = planner_mod.PacePlan
build_groups = planner_mod.build_groups
plan = planner_mod.plan
