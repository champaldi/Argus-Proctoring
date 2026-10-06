"""Compatibility test entry; gaze owns its tests in gaze/tests."""

from pathlib import Path


def load_tests(loader, tests, pattern):
    root = Path(__file__).resolve().parents[1]
    return loader.discover(str(root / "gaze/tests"), pattern or "test_*.py", str(root))
