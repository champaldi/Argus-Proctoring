"""Local readiness check: pinned runtime, real Face Mesh and host contract."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import unittest
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .analyzer import EVENT_TYPES, GazeAnalyzer

EXPECTED_PACKAGES = {
    "mediapipe": "0.10.21",
    "numpy": "1.26.4",
    "opencv-contrib-python": "4.11.0.86",
    "jax": "0.4.38",
    "jaxlib": "0.4.38",
}


def check(host_path: Path | None = None) -> dict:
    """Never opens a camera, records an image, or sends data anywhere."""
    report = {"result": "failed", "checks": {}, "errors": []}
    checks = report["checks"]
    original_path = list(sys.path)
    if host_path is not None:
        host_path = host_path.resolve()
        if not (host_path / "events.py").is_file():
            report["errors"].append(f"Shared events.py is missing in {host_path}")
            return report
        # This gaze package is already imported. Resolve the shared contract
        # and adapters from the requested host before the current directory.
        sys.path.insert(0, str(host_path))
    try:
        installed = {package: version(package) for package in EXPECTED_PACKAGES}
        checks["packages"] = installed
        mismatches = [
            f"{package}: expected {expected}, found {installed[package]}"
            for package, expected in EXPECTED_PACKAGES.items()
            if installed[package] != expected
        ]
        if mismatches:
            raise RuntimeError("; ".join(mismatches))

        host = importlib.import_module("events")
        contract_path = Path(host.__file__).resolve()
        if host_path is not None and contract_path != (host_path / "events.py").resolve():
            # Replacing a cached module would leave existing adapters/tests
            # holding a different ProctorEvent class, so require a fresh process.
            raise RuntimeError(
                f"events.py already loaded from {contract_path}; "
                f"requested {(host_path / 'events.py').resolve()}. "
                "Run gaze.check in a fresh Python process for this host."
            )
        supported = {item.value for item in host.EventType}
        missing = set(EVENT_TYPES) - supported
        if missing:
            raise RuntimeError(f"Shared events.py has no event types: {sorted(missing)}")
        checks["host_contract"] = str(contract_path)

        # Real model and actual normalized events, with deterministic video
        # timestamps. A blank test frame is the expected no-face observation.
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emitted = []
        with GazeAnalyzer() as analyzer:
            for tick in range(5):
                emitted.extend(analyzer.analyze(frame, timestamp=tick / 2))
            if analyzer.last_result.face_count != 0:
                raise RuntimeError("Blank frame unexpectedly contains a face")
            if analyzer.get_face_width_ratio() is not None:
                raise RuntimeError("No-face observation has a stale width ratio")
        normalized = host.normalize_events(emitted, default_source="gaze")
        if len(normalized) != 1 or normalized[0].type.value != "no_face":
            raise RuntimeError("Real model/contract did not emit exactly one no_face event")
        if not isinstance(normalized[0], host.ProctorEvent):
            raise RuntimeError("Event is not the application's shared ProctorEvent")
        checks["real_face_mesh"] = "passed: 5 real frames, one shared no_face event"

        package_root = Path(__file__).resolve().parent
        suite = unittest.defaultTestLoader.discover(
            str(package_root / "tests"), top_level_dir=str(package_root.parent)
        )
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
        checks["tests"] = {
            "run": result.testsRun,
            "failed": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
        }
        if not result.wasSuccessful() or result.skipped:
            raise RuntimeError("Tests failed or were skipped; full readiness is unconfirmed")
        report["result"] = "passed"
        report["manual_camera_check"] = "required: screen/keyboard calibration and five scenarios"
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        sys.path[:] = original_path
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check gaze runtime/model/tests against the application's actual events.py."
    )
    parser.add_argument("--host-path", type=Path, help="Folder containing the shared events.py")
    parser.add_argument("--report", type=Path, help="Save a UTF-8 JSON report (no camera images)")
    args = parser.parse_args()
    report = check(args.host_path)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["result"] == "passed":
        print("Проверка модуля пройдена: зависимости, Face Mesh, события и тесты.")
        print("Для проверки на себе запустите gaze/run.cmd и настройте экран и клавиатуру.")
        return 0
    print("Проверка не пройдена:", file=sys.stderr)
    for error in report["errors"]:
        print(f"  {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
