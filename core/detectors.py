"""Adapters that isolate teammate detector modules from the main loop."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

from events import ProctorEvent, normalize_events


@dataclass(frozen=True, slots=True)
class ModuleStatus:
    name: str
    loaded: bool
    message: str


class DetectorAdapter:
    def __init__(self, name: str, module_name: str, function_name: str) -> None:
        self.name = name
        self.module_name = module_name
        self.function_name = function_name
        self._module: ModuleType | None = None
        self._function: Callable[[Any], Any] | None = None

    def load(self) -> ModuleStatus:
        try:
            self._module = importlib.import_module(self.module_name)
            function = getattr(self._module, self.function_name)
            if not callable(function):
                raise TypeError(f"{self.function_name} is not callable")
            self._function = function
        except Exception as exc:
            self._module = None
            self._function = None
            return ModuleStatus(self.name, False, f"{type(exc).__name__}: {exc}")
        return ModuleStatus(
            self.name,
            True,
            f"{self.module_name}.{self.function_name}",
        )

    @property
    def loaded(self) -> bool:
        return self._function is not None

    def analyze(self, frame: Any) -> list[ProctorEvent]:
        if self._function is None:
            return []
        raw_events = self._function(frame)
        return normalize_events(raw_events, default_source=self.name)


class DetectorCollection:
    def __init__(self, phone_module: str, gaze_module: str) -> None:
        self.adapters = (
            DetectorAdapter("phone", phone_module, "detect"),
            DetectorAdapter("gaze", gaze_module, "analyze"),
        )

    def load(self) -> list[ModuleStatus]:
        return [adapter.load() for adapter in self.adapters]

    def analyze(self, frame: Any) -> tuple[list[ProctorEvent], list[str]]:
        events: list[ProctorEvent] = []
        errors: list[str] = []
        for adapter in self.adapters:
            if not adapter.loaded:
                continue
            try:
                events.extend(adapter.analyze(frame))
            except Exception as exc:
                errors.append(f"{adapter.name}: {type(exc).__name__}: {exc}")
        return events, errors

