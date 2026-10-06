"""Face/gaze detector for the team's shared event contract.

The integration entry point needs the application's root ``events.py``.
Geometry, standalone demo and calibration do not require that module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .analyzer import (
    FACE_TOO_CLOSE_RATIO,
    FACE_TOO_CLOSE_SECONDS,
    AnalysisResult,
    AnalyzerConfig,
    FaceMetrics,
    GazeAnalyzer,
    HeadPose,
    calibrate_default_analyzer,
    get_calibration_status,
    get_face_width_ratio,
    get_last_result,
    reset_default_analyzer,
    reset_default_timers,
)
from .analyzer import analyze as _analyze_observations

if TYPE_CHECKING:
    from events import ProctorEvent

__all__ = [
    "FACE_TOO_CLOSE_RATIO",
    "FACE_TOO_CLOSE_SECONDS",
    "AnalysisResult",
    "AnalyzerConfig",
    "FaceMetrics",
    "GazeAnalyzer",
    "HeadPose",
    "analyze",
    "calibrate_default_analyzer",
    "get_calibration_status",
    "get_face_width_ratio",
    "get_last_result",
    "reset_default_analyzer",
    "reset_default_timers",
]


def analyze(frame: np.ndarray) -> list[ProctorEvent]:
    """Return instances of the application's shared ProctorEvent class."""
    from events import normalize_events

    return normalize_events(_analyze_observations(frame), default_source="gaze")
