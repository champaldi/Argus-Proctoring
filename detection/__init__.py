"""Модуль локальной детекции телефона."""

from .phone_detector import PhoneDetector, detect, reset_default_detector

__all__ = ["PhoneDetector", "detect", "reset_default_detector"]
