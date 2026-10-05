"""Optional adapter for the environment-protection teammate module."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable

from events import ProctorEvent, normalize_events


class SecurityAdapter:
    def __init__(
        self,
        module_name: str,
        event_sink: Callable[[ProctorEvent], None],
    ) -> None:
        self.module_name = module_name
        self.event_sink = event_sink
        self._module: Any = None
        self._enabled = False

    def _receive(self, raw_event: Any) -> None:
        for event in normalize_events(raw_event, default_source="security"):
            self.event_sink(event)

    def enable(self) -> tuple[bool, str]:
        try:
            self._module = importlib.import_module(self.module_name)
            enable = getattr(self._module, "enable")
            parameters = inspect.signature(enable).parameters
            if parameters:
                enable(self._receive)
            else:
                enable()
            self._enabled = True
            return True, f"{self.module_name}.enable"
        except Exception as exc:
            self._module = None
            self._enabled = False
            return False, f"{type(exc).__name__}: {exc}"

    def disable(self) -> None:
        if not self._enabled or self._module is None:
            return
        try:
            disable = getattr(self._module, "disable", None)
            if callable(disable):
                disable()
        except Exception:
            # Shutdown must continue even if a teammate module fails while
            # unregistering one of its hooks.
            pass
        finally:
            self._enabled = False

