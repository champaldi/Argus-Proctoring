"""Host selection checks without loading a model or opening a camera."""

import importlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from gaze import check as module


class HostSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.host = Path(temporary.name)
        # Deliberately incompatible: selecting the local contract would hide this.
        (self.host / "events.py").write_text(
            "from enum import Enum\nclass EventType(Enum):\n    NO_FACE = 'no_face'\n",
            encoding="utf-8",
        )
        self.path_before = list(sys.path)
        self.packages = patch.object(module, "EXPECTED_PACKAGES", {})
        self.packages.start()
        self.addCleanup(self.packages.stop)
        self.model = patch.object(
            module, "GazeAnalyzer", side_effect=RuntimeError("model loading blocked by test")
        )
        self.model.start()
        self.addCleanup(self.model.stop)

    def test_requested_host_takes_precedence_over_current_directory(self):
        with patch.dict(sys.modules):
            sys.modules.pop("events", None)
            report = module.check(self.host)
        self.assertEqual(report["result"], "failed")
        self.assertIn("Shared events.py has no event types", " ".join(report["errors"]))
        self.assertEqual(sys.path, self.path_before)

    def test_different_cached_contract_is_rejected_without_replacing_it(self):
        existing = importlib.import_module("events")
        report = module.check(self.host)
        self.assertEqual(report["result"], "failed")
        errors = " ".join(report["errors"])
        self.assertIn("already loaded", errors)
        self.assertIn(str((self.host / "events.py").resolve()), errors)
        self.assertIs(sys.modules["events"], existing)
        self.assertEqual(sys.path, self.path_before)

    def test_matching_cached_contract_can_continue_validation(self):
        existing = importlib.import_module("events")
        contract_path = Path(existing.__file__).resolve()
        report = module.check(contract_path.parent)
        self.assertEqual(report["checks"]["host_contract"], str(contract_path))
        self.assertEqual(report["errors"], ["RuntimeError: model loading blocked by test"])
        self.assertIs(sys.modules["events"], existing)
        self.assertEqual(sys.path, self.path_before)


if __name__ == "__main__":
    unittest.main()
