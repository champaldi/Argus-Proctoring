"""Проверки CLIP с заглушкой вместо загрузки весов."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from detection import verifier


class FakeClip:
    def encode_image(self, image):
        return torch.tensor([[1.0, 0.0]])


class VerifierTests(unittest.TestCase):
    def test_crop_has_margin_and_converts_bgr_to_rgb(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (10, 20, 30)
        crops = []

        def preprocess(image):
            crops.append(image)
            return torch.zeros((3, 224, 224))

        texts = torch.tensor([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 10)
        with patch.object(verifier, "_get_model_data", return_value=(FakeClip(), preprocess, texts)):
            score = verifier.verify_phone(frame, (20, 20, 40, 40))
        self.assertEqual(crops[0].size, (26, 26))
        self.assertEqual(crops[0].getpixel((0, 0)), (30, 20, 10))
        self.assertGreater(score, 0.5)

    def test_model_is_loaded_once(self):
        # Фабрику подменяем, поэтому сетевой запрос не происходит.
        import sys
        from types import SimpleNamespace

        fake = FakeClip()
        fake.eval = lambda: None
        fake.encode_text = lambda tokens: torch.ones((14, 2))
        factory = SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (fake, None, lambda x: x),
            get_tokenizer=lambda name: lambda prompts: prompts,
        )
        with patch.dict(sys.modules, {"open_clip": factory}), patch.object(verifier, "_model_data", None):
            first = verifier._get_model_data()
            second = verifier._get_model_data()
        self.assertIs(first, second)

    def test_release_keeps_shared_model_until_last_detector_closes(self):
        shared = object()
        with patch.object(verifier, "_model_data", shared), patch.object(verifier, "_users", 0):
            verifier.warmup()
            verifier.warmup()
            verifier.release()
            self.assertIs(verifier._model_data, shared)
            verifier.release()
            self.assertIsNone(verifier._model_data)


if __name__ == "__main__":
    unittest.main()
