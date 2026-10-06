"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Risk policy. Keep these values here so the team can tune the demo without
# touching the scoring implementation.
RISK_WEIGHTS: dict[str, float] = {
    "phone_detected": 25.0,
    "phone_aimed_at_screen": 40.0,
    "multiple_faces": 30.0,
    "no_face": 20.0,
    "gaze_side": 10.0,
    "gaze_down": 10.0,
    "too_close_to_camera": 10.0,
    "window_switched": 15.0,
    "hotkey_blocked": 15.0,
    "suspicious_process": 15.0,
    "capture_protection_failed": 15.0,
    "remote_session": 30.0,
    "multiple_monitors": 20.0,
}
RISK_COMBINATION_WINDOW_SECONDS = 10.0
RISK_COMBINATION_MULTIPLIER = 1.5
RISK_DECAY_PER_SECOND = 0.2
RISK_MAXIMUM = 100.0
RISK_YELLOW_FROM = 30.0
RISK_RED_ABOVE = 60.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera_index: int
    analysis_fps: float
    preview_fps: float
    data_dir: Path
    database_path: Path
    screenshots_dir: Path
    phone_module: str
    gaze_module: str
    security_module: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        data_value = os.getenv("PROCTOR_DATA_DIR", "data")
        data_dir = Path(data_value)
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        return cls(
            camera_index=_env_int("PROCTOR_CAMERA_INDEX", 0),
            analysis_fps=max(0.5, _env_float("PROCTOR_ANALYSIS_FPS", 6.0)),
            preview_fps=max(1.0, _env_float("PROCTOR_PREVIEW_FPS", 20.0)),
            data_dir=data_dir,
            database_path=data_dir / "proctoring.db",
            screenshots_dir=data_dir / "screenshots",
            phone_module=os.getenv(
                "PROCTOR_PHONE_MODULE", "detection.phone_detector"
            ),
            gaze_module=os.getenv("PROCTOR_GAZE_MODULE", "gaze"),
            security_module=os.getenv(
                "PROCTOR_SECURITY_MODULE", "environment_protection"
            ),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

