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


def measure_face(
    landmarks: Sequence[Any], width: int, height: int, min_eye_open_ratio: float = 0.12
) -> FaceMetrics | None:
    """Return head pose and iris location; iris centers approximate pupils."""
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
    # Do not resurrect an iris estimate during a blink using previous frames.
    eyes = [s for s in samples if s.iris_x is not None and s.iris_y is not None]
    if latest.iris_x is None or latest.iris_y is None:
        return FaceMetrics(pose, iris_centers=latest.iris_centers)
    return FaceMetrics(
        pose,
        float(np.median([s.iris_x for s in eyes])),
        float(np.median([s.iris_y for s in eyes])),
        latest.iris_centers,
    )


@dataclass
class _Episode:
    started_at: float | None = None
    emitted: bool = False


class GazeAnalyzer:
    """Owns the detector, calibration and timers for one sequential stream."""

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        *,
        face_mesh: FaceMeshProtocol | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or AnalyzerConfig()
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
                for number in (*numbers, iris_x, iris_y):
                    if number is not None and (
                        isinstance(number, bool) or not isinstance(number, (int, float))
                    ):
                        raise ValueError("Profile measurements must be numbers")
                reference = FaceMetrics(HeadPose(*numbers), iris_x, iris_y)
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
        screen = self._screen or FaceMetrics(HeadPose(0, 0, 0), 0.5, 0.5)
        return {
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

    def _signals(self, metrics: FaceMetrics) -> tuple[EventType, ...]:
        cfg = self.config
        baseline = self._screen or FaceMetrics(HeadPose(0, 0, 0), 0.5, 0.5)
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
        if self._in_keyboard_zone(metrics):
            down, side = False, False
        return tuple(name for name, active in (("gaze_down", down), ("gaze_side", side)) if active)

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
        faces = detection.multi_face_landmarks or []
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
                episode.started_at, episode.emitted = None, False
                continue
            if episode.started_at is None:
                episode.started_at = now
            duration = now - episode.started_at
            elapsed[name] = duration
            if duration >= thresholds[name] and not episode.emitted and name not in deferred:
                episode.emitted = True
                event: dict[str, Any] = {
                    "type": name,
                    "source": "gaze",
                    "timestamp": now,
                    "started_at": episode.started_at,
                    "duration": duration,
                    "face_count": count,
                }
                if face_width_ratio is not None:
                    event["face_width_ratio"] = face_width_ratio
                if metrics is not None:
                    if metrics.head_pose is not None:
                        event["head_pose"] = asdict(metrics.head_pose)
                    event["iris_x"], event["iris_y"] = metrics.iris_x, metrics.iris_y
                events.append(event)
        self.last_result = AnalysisResult(
            count, metrics, signals, events, elapsed, face_width_ratio
        )
        self._last_deferred_gaze = deferred
        return events

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
        _default_analyzer = GazeAnalyzer()
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
