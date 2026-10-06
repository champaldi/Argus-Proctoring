"""Compatibility import; implementation and singleton live in gaze.analyzer."""

import sys

from gaze import analyzer as _implementation

sys.modules[__name__] = _implementation
