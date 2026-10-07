"""Stateful, local Face Mesh analysis for one student's camera stream.

Input is an unmirrored OpenCV BGR uint8 frame. Events describe suspicious
episodes, not proof of cheating. Use one GazeAnalyzer per camera/session.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np

EventType = Literal["gaze_down", "gaze_side", "no_face", "multiple_faces", "too_close_to_camera"]
EVENT_TYPES: tuple[EventType, ...] = (
    "gaze_down",
    "gaze_side",
    "no_face",
    "multiple_faces",
    "too_close_to_camera",
)

FACE_TOO_CLOSE_RATIO = 0.45
FACE_TOO_CLOSE_SECONDS = 3.0
CALIBRATION_MIN_SECONDS = 2.0
CALIBRATION_MAX_SECONDS = 8.0
# Reliability settings of the single-stream convenience analyzer used by the
# application. A plain GazeAnalyzer() keeps the strict behaviour.
#
# A held look down or aside is judged frame by frame, and one frame under the
# threshold used to restart the whole episode, so a real five-second look was
# often never reported. Brief dropouts up to this long no longer restart it.
DEFAULT_GAZE_GAP_SECONDS = 0.6
# A face on a poster or a photo behind the student is much smaller than the
# student's own. Extra faces narrower than this share of the largest face are
# ignored instead of being reported as a second person.
DEFAULT_MIN_EXTRA_FACE_RATIO = 0.4
GAZE_EVENT_TYPES: frozenset[str] = frozenset({"gaze_down", "gaze_side"})
# A phone lying on the keyboard can be read in glances shorter than the
# continuous threshold. Looking down for this many seconds in total within the
# window is reported as gaze_down too, however short each glance was.
DEFAULT_DOWN_BUDGET_SECONDS = 15.0
DEFAULT_DOWN_WINDOW_SECONDS = 60.0
# Without a calibration step the analyzer has no idea where "at the screen" is
# for this camera and this student, and absolute thresholds miss a look at the
# keyboard (measured: the head tilts only about 9 degrees). The first seconds of
# a session, when the student faces the screen, become the reference.
DEFAULT_AUTO_BASELINE_SECONDS = 3.0
# Five seconds let a student read a phone on the keyboard and look back up in
# time. The application reports a continuous look down sooner.
DEFAULT_GAZE_DOWN_SECONDS = 4.0
# Face Mesh's face oval excludes iris and interior points. Its horizontal
# extremes give the visible face width, including when the head is tilted.
FACE_CONTOUR_IDS = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

# Six vertices of Google's canonical face, converted from X-right/Y-up/Z-out
# to OpenCV X-right/Y-down/Z-away. Units are cm; no inferred landmark Z is used.
# Source (Apache-2.0): google-ai-edge/mediapipe v0.10.21,
# mediapipe/modules/face_geometry/data/canonical_face_model.obj
HEAD_LANDMARK_IDS = (1, 152, 33, 263, 61, 291)
HEAD_MODEL_POINTS = np.array(
    [
        (0.0, 1.126865, -7.475604),
        (0.0, 9.403378, -4.264492),
        (-4.445859, -2.663991, -3.173422),
        (4.445859, -2.663991, -3.173422),
        (-2.456206, 4.342621, -4.283884),
        (2.456206, 4.342621, -4.283884),
    ],
    dtype=np.float64,
)
HEAD_MODEL_POINTS.setflags(write=False)


def _prepare_mediapipe_resources_for_windows(mp: Any) -> None:
    """Work around MediaPipe 0.10.21 failing on non-ASCII Windows paths."""

    if os.name != "nt":
        return
    from mediapipe.python import solution_base

    package_parent = Path(mp.__file__).resolve().parent.parent
    try:
        str(package_parent).encode("ascii")
        return
    except UnicodeEncodeError:
        pass

    version = getattr(mp, "__version__", "0.10.21").replace(".", "_")
    cache_root = Path(tempfile.gettempdir()) / f"proctoring_mediapipe_{version}"
    source_modules = package_parent / "mediapipe" / "modules"
    cached_modules = cache_root / "mediapipe" / "modules"
    required_graph = cached_modules / "face_landmark" / "face_landmark_front_cpu.binarypb"
    if not required_graph.exists():
        shutil.copytree(source_modules, cached_modules, dirs_exist_ok=True)

    fake_solution_file = cache_root / "mediapipe" / "python" / "solution_base.py"
    fake_solution_file.parent.mkdir(parents=True, exist_ok=True)
    # SolutionBase derives its resource root from this module global.
    solution_base.__file__ = str(fake_solution_file)


@dataclass(frozen=True)
class AnalyzerConfig:
    """Starting thresholds; tune on the actual camera after calibration."""

    gaze_side_seconds: float = 3.0
    gaze_down_seconds: float = 5.0
    no_face_seconds: float = 2.0
    multiple_faces_seconds: float = 1.0
    face_too_close_ratio: float = FACE_TOO_CLOSE_RATIO
    face_too_close_seconds: float = FACE_TOO_CLOSE_SECONDS
    max_sample_gap_seconds: float = 1.0
    iris_missing_grace_seconds: float = 0.3
    head_side_degrees: float = 25.0
    head_down_degrees: float = 35.0
    combined_down_degrees: float = 25.0
    calibrated_head_down_degrees: float = 20.0
    keyboard_down_margin_degrees: float = 3.0
    iris_side_threshold: float = 0.18
    iris_down_threshold: float = 0.30
    combined_iris_down_threshold: float = 0.16
    keyboard_pitch_tolerance: float = 10.0
    keyboard_yaw_tolerance: float = 10.0
    keyboard_iris_tolerance: float = 0.15
    min_eye_open_ratio: float = 0.12
    # Used only against a measured screen reference that includes eye opening.
    # Looking down lowers the upper eyelids, so a modest head tilt together with
    # visibly narrower eyes is a look down; a larger tilt is one on its own.
    eyelid_down_degrees: float = 7.0
    eyelid_down_open_ratio: float = 0.75
    relative_head_down_degrees: float = 14.0
    # Eyes lowered with the head still: the eyelids alone drop this far.
    eyes_only_down_open_ratio: float = 0.65
    max_num_faces: int = 3
    smoothing_window: int = 3
    calibration_min_samples: int = 10

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_num_faces", "smoothing_window", "calibration_min_samples"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.max_num_faces < 2:
            raise ValueError("max_num_faces must be >= 2 to detect another face")
        if self.face_too_close_ratio > 1.0:
            raise ValueError("face_too_close_ratio must be <= 1")
        if self.combined_down_degrees > self.head_down_degrees:
            raise ValueError("combined_down_degrees must not exceed head_down_degrees")
        if self.combined_iris_down_threshold > self.iris_down_threshold:
            raise ValueError("combined iris threshold must not exceed iris_down_threshold")
        if self.eyelid_down_open_ratio >= 1.0:
            raise ValueError("eyelid_down_open_ratio must be below 1")
        if self.eyes_only_down_open_ratio > self.eyelid_down_open_ratio:
            raise ValueError("eyes_only_down_open_ratio must not exceed eyelid_down_open_ratio")
        if self.eyelid_down_degrees > self.relative_head_down_degrees:
            raise ValueError("eyelid_down_degrees must not exceed relative_head_down_degrees")


@dataclass(frozen=True)
class HeadPose:
    """Degrees: +pitch down, +yaw image-left, +roll clockwise (unmirrored)."""

    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class FaceMetrics:
    head_pose: HeadPose | None
    iris_x: float | None = None
    iris_y: float | None = None
    iris_centers: tuple[tuple[float, float], ...] = ()
    # Eye height divided by eye width, averaged over both eyes.
    eye_open: float | None = None


def is_calibration_sample_valid(
    sample: FaceMetrics | None, *, target: Literal["screen", "keyboard"] = "screen"
) -> bool:
    """Keyboard tilt may hide irises; its finite head pose is still usable."""
    if target not in ("screen", "keyboard"):
        raise ValueError("target must be 'screen' or 'keyboard'")
    if sample is None:
        return False
    pose = sample.head_pose
    if pose is None:
        return False
    if not all(math.isfinite(value) for value in (pose.pitch, pose.yaw, pose.roll)):
        return False
    eyes = (sample.iris_x, sample.iris_y)
    if target == "screen":
        return all(value is not None and math.isfinite(value) for value in eyes)
    return all(value is None or math.isfinite(value) for value in eyes)


@dataclass(frozen=True)
class AnalysisResult:
    face_count: int
    metrics: FaceMetrics | None
    signals: tuple[EventType, ...]
    events: list[dict[str, Any]]
    elapsed: dict[EventType, float]
    face_width_ratio: float | None = None


class FaceMeshProtocol(Protocol):
    def process(self, image: np.ndarray) -> Any: ...

    def close(self) -> None: ...


def _points(
    landmarks: Sequence[Any], indices: Sequence[int], width: int, height: int
) -> np.ndarray:
    return np.array(
        [(landmarks[i].x * width, landmarks[i].y * height) for i in indices], dtype=np.float64
    )


def measure_face_width_ratio(landmarks: Sequence[Any]) -> float | None:
    """Visible horizontal face span divided by frame width, never centimetres.

    Face Mesh already normalizes x by frame width, so pixel dimensions cancel.
    Clip a partially visible oval to the frame; an entirely outside, collapsed
    or nonfinite contour cannot provide a valid width observation.
    """
    if len(landmarks) <= max(FACE_CONTOUR_IDS):
        return None
    try:
        xs = [float(landmarks[index].x) for index in FACE_CONTOUR_IDS]
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(x) for x in xs):
        return None
    left, right = max(0.0, min(xs)), min(1.0, max(xs))
    return right - left if right > left else None


def estimate_head_pose(landmarks: Sequence[Any], width: int, height: int) -> HeadPose | None:
    """Estimate pose using fixed 3D geometry and correctly ordered image pixels.

    The focal length is approximated by image width. This is not a calibrated
    camera measurement; neutral-pose calibration removes some systematic bias.
    """
    if width <= 0 or height <= 0 or len(landmarks) <= max(HEAD_LANDMARK_IDS):
        return None
    image_points = _points(landmarks, HEAD_LANDMARK_IDS, width, height)
    if not np.isfinite(image_points).all():
        return None
    eye_span = float(np.linalg.norm(image_points[2] - image_points[3]))
    if eye_span < 5:
        return None
    camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rotation, translation = cv2.solvePnP(
            HEAD_MODEL_POINTS,
            image_points,
            camera,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or not np.isfinite(rotation).all() or translation[2, 0] <= 0:
            return None
        projected, _ = cv2.projectPoints(
            HEAD_MODEL_POINTS, rotation, translation, camera, distortion
        )
        error = float(
            np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)))
        )
        if error > eye_span * 0.20:
            return None
        angles = cv2.RQDecomp3x3(cv2.Rodrigues(rotation)[0])[0]
    except cv2.error:
        return None
    if not all(math.isfinite(a) for a in angles):
        return None
    return HeadPose(*map(float, angles))


def _measure_eye(
    landmarks: Sequence[Any],
    indices: tuple[int, int, int, int, int],
    width: int,
    height: int,
    min_open_ratio: float,
) -> tuple[float, float] | None:
    # Local eye axes compensate for image scale, aspect ratio and head roll.
    corner_a, corner_b, top, bottom, iris = _points(landmarks, indices, width, height)
    if not np.isfinite([corner_a, corner_b, top, bottom, iris]).all():
        return None
    if corner_a[0] > corner_b[0]:
        corner_a, corner_b = corner_b, corner_a
    horizontal = corner_b - corner_a
    eye_width = float(np.linalg.norm(horizontal))
    if eye_width < 3:
        return None
    horizontal /= eye_width
    vertical = np.array([-horizontal[1], horizontal[0]])
    eye_height = float(np.dot(bottom - top, vertical))
    if eye_height / eye_width < min_open_ratio:
        return None  # Closed/occluded eye: do not classify iris position.
    x = float(np.dot(iris - corner_a, horizontal) / eye_width)
    y = float(np.dot(iris - top, vertical) / eye_height)
    if not (-0.25 <= x <= 1.25 and -0.5 <= y <= 1.5):
        return None
    return x, y


def _eye_opening(
    landmarks: Sequence[Any], indices: tuple[int, int, int, int], width: int, height: int
) -> float | None:
    corner_a, corner_b, top, bottom = _points(landmarks, indices, width, height)
    if not np.isfinite([corner_a, corner_b, top, bottom]).all():
        return None
    if corner_a[0] > corner_b[0]:
        corner_a, corner_b = corner_b, corner_a
    horizontal = corner_b - corner_a
    eye_width = float(np.linalg.norm(horizontal))
    if eye_width < 3:
        return None
    horizontal /= eye_width
    vertical = np.array([-horizontal[1], horizontal[0]])
    return max(0.0, float(np.dot(bottom - top, vertical)) / eye_width)


def measure_eye_opening(landmarks: Sequence[Any], width: int, height: int) -> float | None:
    """Mean eye height/width ratio; also defined for nearly closed eyes."""
    if len(landmarks) < 387:
        return None
    values = [
        _eye_opening(landmarks, indices, width, height)
        for indices in ((33, 133, 159, 145), (362, 263, 386, 374))
    ]
    return None if None in values else float(sum(values) / 2)


def measure_face(
    landmarks: Sequence[Any], width: int, height: int, min_eye_open_ratio: float = 0.12
) -> FaceMetrics | None:
    """Return head pose, iris location and eye opening."""
    measured = _measure_face(landmarks, width, height, min_eye_open_ratio)
    if measured is None:
        return None
    return FaceMetrics(
        measured.head_pose,
        measured.iris_x,
        measured.iris_y,
        measured.iris_centers,
        measure_eye_opening(landmarks, width, height),
    )


def _measure_face(
    landmarks: Sequence[Any], width: int, height: int, min_eye_open_ratio: float
) -> FaceMetrics | None:
    pose = estimate_head_pose(landmarks, width, height)
    if len(landmarks) < 478:
        return FaceMetrics(pose) if pose is not None else None
    right = _measure_eye(landmarks, (33, 133, 159, 145, 468), width, height, min_eye_open_ratio)
    left = _measure_eye(landmarks, (362, 263, 386, 374, 473), width, height, min_eye_open_ratio)
    centers = tuple(
        (float(landmarks[i].x), float(landmarks[i].y))
        for i in (468, 473)
        if math.isfinite(landmarks[i].x) and math.isfinite(landmarks[i].y)
    )
    if right is None or left is None:
        return FaceMetrics(pose, iris_centers=centers) if pose is not None else None
    # Strong disagreement often means occlusion or an unreliable iris estimate.
    if abs(right[0] - left[0]) > 0.25 or abs(right[1] - left[1]) > 0.40:
        return FaceMetrics(pose, iris_centers=centers) if pose is not None else None
    return FaceMetrics(pose, (right[0] + left[0]) / 2, (right[1] + left[1]) / 2, centers)


def _median_metrics(samples: Sequence[FaceMetrics]) -> FaceMetrics:
    latest = samples[-1]
    poses = [s.head_pose for s in samples if s.head_pose is not None]
    pose = (
        None
        if latest.head_pose is None
        else HeadPose(
            *(
                float(np.median([getattr(pose, axis) for pose in poses]))
                for axis in ("pitch", "yaw", "roll")
            )
        )
    )
    openings = [s.eye_open for s in samples if s.eye_open is not None]
    eye_open = None if latest.eye_open is None else float(np.median(openings))
    # Do not resurrect an iris estimate during a blink using previous frames.
    eyes = [s for s in samples if s.iris_x is not None and s.iris_y is not None]
    if latest.iris_x is None or latest.iris_y is None:
        return FaceMetrics(pose, iris_centers=latest.iris_centers, eye_open=eye_open)
    return FaceMetrics(
        pose,
        float(np.median([s.iris_x for s in eyes])),
        float(np.median([s.iris_y for s in eyes])),
        latest.iris_centers,
        eye_open,
    )


@dataclass
class _Episode:
    started_at: float | None = None
    emitted: bool = False
    last_seen: float | None = None


class GazeAnalyzer:
    """Owns the detector, calibration and timers for one sequential stream."""

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        *,
        face_mesh: FaceMeshProtocol | None = None,
        clock: Callable[[], float] = time.monotonic,
        gaze_gap_seconds: float = 0.0,
        min_extra_face_ratio: float = 0.0,
        down_budget_seconds: float = 0.0,
        down_window_seconds: float = DEFAULT_DOWN_WINDOW_SECONDS,
        auto_baseline_seconds: float = 0.0,
    ) -> None:
        self.config = config or AnalyzerConfig()
        for name, value in (
            ("gaze_gap_seconds", gaze_gap_seconds),
            ("min_extra_face_ratio", min_extra_face_ratio),
            ("down_budget_seconds", down_budget_seconds),
            ("down_window_seconds", down_window_seconds),
            ("auto_baseline_seconds", auto_baseline_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite number >= 0")
        if min_extra_face_ratio > 1.0:
            raise ValueError("min_extra_face_ratio must be <= 1")
        if gaze_gap_seconds >= self.config.max_sample_gap_seconds:
            raise ValueError("gaze_gap_seconds must be below max_sample_gap_seconds")
        if down_budget_seconds > 0 and down_window_seconds < down_budget_seconds:
            raise ValueError("down_window_seconds must be >= down_budget_seconds")
        # 0 keeps the strict behaviour: no dropout tolerance, every face counts,
        # only a continuous look down is reported.
        self._gaze_gap_seconds = float(gaze_gap_seconds)
        self._min_extra_face_ratio = float(min_extra_face_ratio)
        self._down_budget_seconds = float(down_budget_seconds)
        self._down_window_seconds = float(down_window_seconds)
        # (frame time, seconds of looking down credited to that frame)
        self._down_spans: deque[tuple[float, float]] = deque()
        self._down_seen_at: float | None = None
        # 0 keeps the fixed default reference until calibrate() is called.
        self._auto_baseline_seconds = float(auto_baseline_seconds)
        self._auto_screen: FaceMetrics | None = None
        self._auto_samples: list[tuple[float, FaceMetrics]] = []
        if face_mesh is None:
            import mediapipe as mp

            if not hasattr(mp, "solutions"):
                raise RuntimeError(
                    "Face Mesh requires mediapipe==0.10.21; install gaze/requirements.txt"
                )
            _prepare_mediapipe_resources_for_windows(mp)
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.config.max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        self._face_mesh = face_mesh
        self._clock = clock
        self._closed = False
        self._screen: FaceMetrics | None = None
        self._keyboard: FaceMetrics | None = None
        self._history: deque[FaceMetrics] = deque(maxlen=self.config.smoothing_window)
        self._episodes: dict[EventType, _Episode] = {}
        self._last_timestamp: float | None = None
        self.last_result = AnalysisResult(0, None, (), [], {})
        self.reset()

    def reset(self) -> None:
        """Begin a new episode/session timeline, retaining calibration."""
        self._episodes = {name: _Episode() for name in EVENT_TYPES}
        self._history.clear()
        self._last_timestamp = None
        self._iris_missing_since: float | None = None
        self._last_deferred_gaze: set[EventType] = set()
        self._down_spans.clear()
        self._down_seen_at = None
        # A learned reference survives, like a calibration; partial samples do not.
        self._auto_samples.clear()
        self.last_result = AnalysisResult(0, None, (), [], {})

    def get_face_width_ratio(self) -> float | None:
        """Read the latest single-face width without performing inference."""
        return self.last_result.face_width_ratio

    def get_calibration_status(self) -> dict[str, bool]:
        return {"screen": self._screen is not None, "keyboard": self._keyboard is not None}

    def export_calibration(self) -> dict[str, Any]:
        """Save reference medians only, without frames or episode timers."""

        def pack(reference: FaceMetrics | None) -> dict[str, Any] | None:
            if reference is None:
                return None
            return {
                "head_pose": asdict(reference.head_pose),
                "iris_x": reference.iris_x,
                "iris_y": reference.iris_y,
                "eye_open": reference.eye_open,
            }

        return {"version": 1, "screen": pack(self._screen), "keyboard": pack(self._keyboard)}

    def import_calibration(self, profile: dict[str, Any]) -> None:
        """Restore validated medians from the same user/camera configuration."""
        if self._closed:
            raise RuntimeError("Analyzer is closed")
        if not isinstance(profile, dict) or type(profile.get("version")) is not int:
            raise ValueError("Invalid calibration profile")
        if profile["version"] != 1 or "screen" not in profile or "keyboard" not in profile:
            raise ValueError("Unsupported or incomplete calibration profile")

        def unpack(value: Any, target: Literal["screen", "keyboard"]) -> FaceMetrics | None:
            if value is None:
                return None
            try:
                pose = value["head_pose"]
                numbers = [pose[name] for name in ("pitch", "yaw", "roll")]
                iris_x, iris_y = value["iris_x"], value["iris_y"]
                # Absent in profiles saved before eye opening was measured.
                eye_open = value.get("eye_open")
                if eye_open is not None and not (
                    isinstance(eye_open, (int, float)) and math.isfinite(eye_open)
                ):
                    raise ValueError("Profile measurements must be numbers")
                for number in (*numbers, iris_x, iris_y, eye_open):
                    if number is not None and (
                        isinstance(number, bool) or not isinstance(number, (int, float))
                    ):
                        raise ValueError("Profile measurements must be numbers")
                reference = FaceMetrics(HeadPose(*numbers), iris_x, iris_y, eye_open=eye_open)
                if not is_calibration_sample_valid(reference, target=target):
                    raise ValueError("Profile measurements are not valid")
                return reference
            except (KeyError, TypeError, AttributeError) as error:
                raise ValueError("Invalid calibration measurements") from error

        screen = unpack(profile["screen"], "screen")
        keyboard = unpack(profile["keyboard"], "keyboard")
        if keyboard is not None and screen is None:
            raise ValueError("A keyboard profile needs a screen reference")
        self._screen, self._keyboard = screen, keyboard
        self.reset()

    def get_diagnostics(self) -> dict[str, Any]:
        """JSON-friendly measurements for local threshold troubleshooting."""

        def pack(metrics: FaceMetrics | None) -> dict[str, Any] | None:
            return None if metrics is None else asdict(metrics)

        raw = self._history[-1] if self._history else None
        measured = self.last_result.metrics
        screen = self._reference()
        return {
            "baseline_source": (
                "calibrated" if self._screen else "auto" if self._auto_screen else "default"
            ),
            "eye_open": None if raw is None else raw.eye_open,
            "raw_metrics": pack(raw),
            "smoothed_metrics": pack(measured),
            "screen": pack(screen),
            "keyboard": pack(self._keyboard),
            "keyboard_zone": measured is not None and self._in_keyboard_zone(measured),
            "head_unknown": measured is None or measured.head_pose is None,
            "iris_unknown": measured is None or measured.iris_x is None or measured.iris_y is None,
            "face_count": self.last_result.face_count,
            "signals": list(self.last_result.signals),
            "elapsed": dict(self.last_result.elapsed),
            "face_width_ratio": self.last_result.face_width_ratio,
            "deferred": sorted(self._last_deferred_gaze),
            "down_accumulated_seconds": sum(span for _, span in self._down_spans),
            "down_budget_seconds": self._down_budget_seconds,
            "config": asdict(self.config),
            "episodes": {
                name: {"started_at": episode.started_at, "emitted": episode.emitted}
                for name, episode in self._episodes.items()
            },
            "down_thresholds": {
                "head_degrees": self._down_thresholds()[0],
                "combined_degrees": self._down_thresholds()[1],
                "iris": self.config.iris_down_threshold,
                "combined_iris": self.config.combined_iris_down_threshold,
            },
        }

    def calibrate(
        self, samples: Sequence[FaceMetrics], *, target: Literal["screen", "keyboard"] = "screen"
    ) -> None:
        """Learn a screen/keyboard reference from valid single-face frames."""
        if target not in ("screen", "keyboard"):
            raise ValueError("target must be 'screen' or 'keyboard'")
        if len(samples) < self.config.calibration_min_samples:
            raise ValueError(f"Need at least {self.config.calibration_min_samples} samples")
        if not all(is_calibration_sample_valid(sample, target=target) for sample in samples):
            requirement = "finite pose and both open eyes" if target == "screen" else "finite pose"
            raise ValueError(f"Calibration needs {requirement}")
        reference = _median_metrics(samples)
        if target == "keyboard":
            # Use irises only when enough frames support them; a final blink
            # must not discard an otherwise reliable keyboard-eye reference.
            eyes = [s for s in samples if s.iris_x is not None and s.iris_y is not None]
            if len(eyes) >= self.config.calibration_min_samples:
                reference = FaceMetrics(
                    reference.head_pose,
                    float(np.median([s.iris_x for s in eyes])),
                    float(np.median([s.iris_y for s in eyes])),
                )
            else:
                reference = FaceMetrics(reference.head_pose)
        if target == "screen":
            self._screen = reference
            self._keyboard = None  # A changed camera reference invalidates the old zone.
        else:
            if self._screen is None:
                raise ValueError("Calibrate the screen before the keyboard")
            self._keyboard = reference
        self.reset()

    def _significant_faces(self, faces: Sequence[Any]) -> list[Any]:
        """Largest face first; drop extra faces far smaller than it.

        Without a usable width for the largest face nothing is dropped, so an
        unmeasurable frame can never hide a second person.
        """
        faces = list(faces)
        if self._min_extra_face_ratio <= 0 or len(faces) < 2:
            return faces
        widths = [measure_face_width_ratio(face.landmark) or 0.0 for face in faces]
        largest = max(widths)
        if largest <= 0:
            return faces
        ranked = sorted(zip(widths, range(len(faces))), key=lambda item: (-item[0], item[1]))
        limit = largest * self._min_extra_face_ratio
        return [faces[index] for width, index in ranked if width >= limit]

    def _signals(self, metrics: FaceMetrics) -> tuple[EventType, ...]:
        cfg = self.config
        baseline = self._reference()
        yaw = None if metrics.head_pose is None else metrics.head_pose.yaw - baseline.head_pose.yaw
        pitch = (
            None
            if metrics.head_pose is None
            else metrics.head_pose.pitch - baseline.head_pose.pitch
        )
        iris_x = None if metrics.iris_x is None else metrics.iris_x - baseline.iris_x
        iris_y = None if metrics.iris_y is None else metrics.iris_y - baseline.iris_y
        head_down_degrees, combined_down_degrees = self._down_thresholds()
        side = (yaw is not None and abs(yaw) >= cfg.head_side_degrees) or (
            iris_x is not None and abs(iris_x) >= cfg.iris_side_threshold
        )
        down = (pitch is not None and pitch >= head_down_degrees) or (
            iris_y is not None
            and (
                iris_y >= cfg.iris_down_threshold
                or (
                    pitch is not None
                    and pitch >= combined_down_degrees
                    and iris_y >= cfg.combined_iris_down_threshold
                )
            )
        )
        if baseline.eye_open is not None:
            opening = (
                None if metrics.eye_open is None else metrics.eye_open / baseline.eye_open
            )
            down = (
                down
                or (pitch is not None and pitch >= cfg.relative_head_down_degrees)
                or (
                    opening is not None
                    and pitch is not None
                    and pitch >= cfg.eyelid_down_degrees
                    and opening <= cfg.eyelid_down_open_ratio
                )
                or (opening is not None and opening <= cfg.eyes_only_down_open_ratio)
            )
        if self._in_keyboard_zone(metrics):
            down, side = False, False
        return tuple(name for name, active in (("gaze_down", down), ("gaze_side", side)) if active)

    def _reference(self) -> FaceMetrics:
        return self._screen or self._auto_screen or FaceMetrics(HeadPose(0, 0, 0), 0.5, 0.5)

    def _learn_reference(self, measured: FaceMetrics, now: float) -> None:
        """Take the first steady seconds facing the camera as the screen reference."""
        if (
            self._auto_baseline_seconds <= 0
            or self._screen is not None
            or self._auto_screen is not None
            or measured.eye_open is None
            or not is_calibration_sample_valid(measured, target="screen")
        ):
            return
        self._auto_samples.append((now, measured))
        if (
            len(self._auto_samples) >= self.config.calibration_min_samples
            and now - self._auto_samples[0][0] >= self._auto_baseline_seconds
        ):
            self._auto_screen = _median_metrics([sample for _, sample in self._auto_samples])
            self._auto_samples.clear()

    def _down_thresholds(self) -> tuple[float, float]:
        cfg = self.config
        head = cfg.head_down_degrees
        if self._screen is not None and self._keyboard is not None:
            keyboard_delta = self._keyboard.head_pose.pitch - self._screen.head_pose.pitch
            # Start beyond the learned keyboard band. The conservative default
            # remains the cap, and applies unchanged before keyboard calibration.
            head = min(
                head,
                max(
                    cfg.calibrated_head_down_degrees,
                    keyboard_delta
                    + cfg.keyboard_pitch_tolerance
                    + cfg.keyboard_down_margin_degrees,
                ),
            )
        return head, min(cfg.combined_down_degrees, head)

    def _in_keyboard_zone(self, metrics: FaceMetrics) -> bool:
        keyboard = self._keyboard
        if keyboard is None:
            return False
        cfg = self.config
        in_zone = (
            metrics.head_pose is not None
            and abs(metrics.head_pose.pitch - keyboard.head_pose.pitch)
            <= cfg.keyboard_pitch_tolerance
            and abs(metrics.head_pose.yaw - keyboard.head_pose.yaw) <= cfg.keyboard_yaw_tolerance
        )
        if all(
            value is not None
            for value in (metrics.iris_x, metrics.iris_y, keyboard.iris_x, keyboard.iris_y)
        ):
            eyes_in_zone = (
                abs(metrics.iris_x - keyboard.iris_x) <= cfg.keyboard_iris_tolerance
                and abs(metrics.iris_y - keyboard.iris_y) <= cfg.keyboard_iris_tolerance
            )
            in_zone = eyes_in_zone if metrics.head_pose is None else in_zone and eyes_in_zone
        return in_zone

    def analyze(self, frame: np.ndarray, *, timestamp: float | None = None) -> list[dict[str, Any]]:
        """Emit each event once after continuous observation for its threshold.

        Optional timestamps are seconds on a monotonic/video timeline, not wall
        time. Gaps over max_sample_gap_seconds break all pending episodes.
        Invalid input or detector errors raise; they never become no_face.
        """
        if self._closed:
            raise RuntimeError("Analyzer is closed")
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] == 0
            or frame.shape[1] == 0
        ):
            self.reset()
            raise ValueError("frame must be a nonempty HxWx3 uint8 BGR image")
        try:
            now = float(self._clock() if timestamp is None else timestamp)
        except Exception:
            self.reset()
            raise
        if not math.isfinite(now):
            self.reset()
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None:
            if now < self._last_timestamp:
                self.reset()
                raise ValueError("timestamps must be nondecreasing; reset for a new video")
            if now - self._last_timestamp > self.config.max_sample_gap_seconds:
                self.reset()
        self._last_timestamp = now
        try:
            return self._analyze_frame(frame, now)
        except Exception:
            # No failed geometry/landmark read can carry a timer or stale ratio
            # into the next successful observation.
            self.reset()
            raise

    def _analyze_frame(self, frame: np.ndarray, now: float) -> list[dict[str, Any]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        detection = self._face_mesh.process(rgb)
        faces = self._significant_faces(detection.multi_face_landmarks or [])
        count = len(faces)
        metrics = None
        face_width_ratio = None
        deferred: set[EventType] = set()
        if count != 1:
            self._history.clear()
            self._iris_missing_since = None
            signals: tuple[EventType, ...] = ("no_face",) if count == 0 else ("multiple_faces",)
        else:
            height, width = frame.shape[:2]
            face_width_ratio = measure_face_width_ratio(faces[0].landmark)
            measured = measure_face(
                faces[0].landmark, width, height, self.config.min_eye_open_ratio
            )
            if measured is None:
                self._history.clear()
                self._iris_missing_since = None
                signals = ()
            else:
                self._learn_reference(measured, now)
                self._history.append(measured)
                metrics = _median_metrics(list(self._history))
                fresh_signals = self._signals(measured)
                signals = tuple(name for name in self._signals(metrics) if name in fresh_signals)
                if (
                    self._iris_missing_since is not None
                    and now - self._iris_missing_since > self.config.iris_missing_grace_seconds
                ):
                    for name in self._last_deferred_gaze:
                        self._episodes[name] = _Episode()
                if metrics.iris_x is None or metrics.iris_y is None:
                    if self._iris_missing_since is None:
                        self._iris_missing_since = now
                    if now - self._iris_missing_since <= self.config.iris_missing_grace_seconds:
                        # Blinking must not defeat a long iris-only episode. Keep
                        # existing episodes briefly, but require a fresh iris
                        # observation before emitting an event based on the iris.
                        deferred = (
                            set(self.last_result.signals) & {"gaze_down", "gaze_side"}
                        ) - set(signals)
                        signals = tuple(n for n in EVENT_TYPES if n in signals or n in deferred)
                else:
                    self._iris_missing_since = None
            if face_width_ratio is not None and face_width_ratio > self.config.face_too_close_ratio:
                signals = (*signals, "too_close_to_camera")
        thresholds = {
            "gaze_down": self.config.gaze_down_seconds,
            "gaze_side": self.config.gaze_side_seconds,
            "no_face": self.config.no_face_seconds,
            "multiple_faces": self.config.multiple_faces_seconds,
            "too_close_to_camera": self.config.face_too_close_seconds,
        }
        events: list[dict[str, Any]] = []
        elapsed: dict[EventType, float] = {}
        for name, episode in self._episodes.items():
            if name not in signals:
                if (
                    name in GAZE_EVENT_TYPES
                    and self._gaze_gap_seconds > 0
                    and episode.last_seen is not None
                    and now - episode.last_seen <= self._gaze_gap_seconds
                ):
                    # A brief dropout: keep the episode, but never emit on a
                    # frame that does not itself show the signal.
                    continue
                episode.started_at, episode.emitted, episode.last_seen = None, False, None
                continue
            episode.last_seen = now
            if episode.started_at is None:
                episode.started_at = now
            duration = now - episode.started_at
            elapsed[name] = duration
            if duration >= thresholds[name] and not episode.emitted and name not in deferred:
                episode.emitted = True
                if name == "gaze_down":
                    # Already reported; the same seconds must not count twice.
                    self._down_spans.clear()
                events.append(
                    self._event(name, now, episode.started_at, duration, count,
                                face_width_ratio, metrics)
                )
        accumulated = self._accumulate_down("gaze_down" in signals, now)
        down_episode = self._episodes["gaze_down"]
        if (
            accumulated is not None
            and not down_episode.emitted
            and "gaze_down" not in deferred
        ):
            started_at, total = accumulated
            self._down_spans.clear()
            # One report per look: the continuous timer stays quiet for this one.
            down_episode.emitted = True
            event = self._event(
                "gaze_down", now, started_at, total, count, face_width_ratio, metrics
            )
            event["accumulated"] = True
            event["window_seconds"] = self._down_window_seconds
            events.append(event)
        self.last_result = AnalysisResult(
            count, metrics, signals, events, elapsed, face_width_ratio
        )
        self._last_deferred_gaze = deferred
        return events

    def _accumulate_down(self, looking_down: bool, now: float) -> tuple[float, float] | None:
        """Credit this frame; return (first glance time, total) once over budget."""
        previous, self._down_seen_at = self._down_seen_at, now if looking_down else None
        if self._down_budget_seconds <= 0:
            return None
        while self._down_spans and now - self._down_spans[0][0] > self._down_window_seconds:
            self._down_spans.popleft()
        if not looking_down:
            return None
        if previous is not None and now > previous:
            self._down_spans.append((now, now - previous))
        total = sum(span for _, span in self._down_spans)
        if total < self._down_budget_seconds:
            return None
        first_time, first_span = self._down_spans[0]
        return first_time - first_span, total

    @staticmethod
    def _event(
        name: str,
        now: float,
        started_at: float | None,
        duration: float,
        count: int,
        face_width_ratio: float | None,
        metrics: FaceMetrics | None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "type": name,
            "source": "gaze",
            "timestamp": now,
            "started_at": started_at,
            "duration": duration,
            "face_count": count,
        }
        if face_width_ratio is not None:
            event["face_width_ratio"] = face_width_ratio
        if metrics is not None:
            if metrics.head_pose is not None:
                event["head_pose"] = asdict(metrics.head_pose)
            event["iris_x"], event["iris_y"] = metrics.iris_x, metrics.iris_y
        return event

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._face_mesh.close()
            finally:
                self.reset()

    def __enter__(self) -> GazeAnalyzer:
        if self._closed:
            raise RuntimeError("Analyzer is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


_default_analyzer: GazeAnalyzer | None = None


def analyze(frame: np.ndarray) -> list[dict[str, Any]]:
    """Convenience API for a single stream. Use instances for multiple students."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = GazeAnalyzer(
            AnalyzerConfig(gaze_down_seconds=DEFAULT_GAZE_DOWN_SECONDS),
            gaze_gap_seconds=DEFAULT_GAZE_GAP_SECONDS,
            min_extra_face_ratio=DEFAULT_MIN_EXTRA_FACE_RATIO,
            down_budget_seconds=DEFAULT_DOWN_BUDGET_SECONDS,
            auto_baseline_seconds=DEFAULT_AUTO_BASELINE_SECONDS,
        )
    return _default_analyzer.analyze(frame)


def reset_default_analyzer() -> None:
    """Release the convenience detector; the next analyze starts a new session."""
    global _default_analyzer
    analyzer, _default_analyzer = _default_analyzer, None
    if analyzer is not None:
        analyzer.close()


def get_last_result() -> AnalysisResult | None:
    """Read on the inference worker thread; never creates a detector."""
    return None if _default_analyzer is None else _default_analyzer.last_result


def get_diagnostics() -> dict[str, Any] | None:
    """Read current measurements and decisions without creating a detector."""
    return None if _default_analyzer is None else _default_analyzer.get_diagnostics()


def get_face_width_ratio() -> float | None:
    """Read on the inference worker thread; never opens a model or camera."""
    return None if _default_analyzer is None else _default_analyzer.get_face_width_ratio()


def get_calibration_status() -> dict[str, bool]:
    """Return calibration flags without opening any resources."""
    return (
        {"screen": False, "keyboard": False}
        if _default_analyzer is None
        else _default_analyzer.get_calibration_status()
    )


def reset_default_timers() -> None:
    """Discard calibration-time episodes on the inference worker thread."""
    if _default_analyzer is not None:
        _default_analyzer.reset()


def calibrate_default_analyzer(
    samples: Sequence[FaceMetrics], *, target: Literal["screen", "keyboard"] = "screen"
) -> None:
    """Apply samples on the worker that owns the default analyzer."""
    if _default_analyzer is None:
        raise RuntimeError("Run gaze analysis before calibration")
    _default_analyzer.calibrate(samples, target=target)
