"""Второй этап: сравнение найденной рамки с описаниями через CLIP."""

from __future__ import annotations

from threading import RLock
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
CROP_MARGIN = 0.15
VERIFIER_MIN_SCORE = 0.2
POSITIVE_PROMPTS = (
    "a smartphone",
    "the back of a mobile phone",
    "a phone in a case held in a hand",
    "a phone screen",
)
NEGATIVE_PROMPTS = (
    "a jar",
    "a plastic container",
    "a glasses case",
    "a hairbrush",
    "a remote control",
    "a wallet",
    "a cup",
    "a bottle",
    "an empty hand",
    "a piece of clothing",
)

_lock = RLock()
_model_data: tuple[Any, Any, torch.Tensor] | None = None
_users = 0


def _get_model_data() -> tuple[Any, Any, torch.Tensor]:
    """Один раз загружает CLIP и кодирует все описания на CPU."""
    global _model_data
    if _model_data is None:
        with _lock:
            if _model_data is None:
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    MODEL_NAME, pretrained=PRETRAINED, device="cpu"
                )
                model.eval()
                tokenizer = open_clip.get_tokenizer(MODEL_NAME)
                prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
                with torch.inference_mode():
                    text_features = model.encode_text(tokenizer(prompts))
                    text_features = torch.nn.functional.normalize(text_features, dim=-1)
                _model_data = model, preprocess, text_features
    return _model_data


def warmup() -> None:
    """Загружает CLIP и отмечает ещё один использующий его детектор."""
    global _users
    with _lock:
        _get_model_data()
        _users += 1


def release() -> None:
    """Удаляет общую ссылку на модель после закрытия последнего детектора."""
    global _model_data, _users
    with _lock:
        if _users == 0:
            return
        _users -= 1
        if _users == 0:
            _model_data = None


def verify_phone(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Возвращает суммарную вероятность телефонных описаний для рамки BGR."""
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    margin_x = (x2 - x1) * CROP_MARGIN
    margin_y = (y2 - y1) * CROP_MARGIN
    left = max(0, int(x1 - margin_x))
    top = max(0, int(y1 - margin_y))
    right = min(width, int(np.ceil(x2 + margin_x)))
    bottom = min(height, int(np.ceil(y2 + margin_y)))
    if right <= left or bottom <= top:
        raise ValueError("phone bbox has no pixels inside the frame")

    # OpenCV даёт BGR, а CLIP ожидает RGB.
    crop = Image.fromarray(cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB))
    model, preprocess, text_features = _get_model_data()
    with torch.inference_mode():
        image_features = model.encode_image(preprocess(crop).unsqueeze(0))
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return float(probabilities[0, :len(POSITIVE_PROMPTS)].sum().item())
