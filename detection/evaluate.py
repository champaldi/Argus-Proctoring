"""Покадровая оценка телефонных рамок на сохранённых изображениях."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import cv2
import numpy as np
from core.image_io import read_image, write_image

from .phone_detector import (
    PHONE_CONFIDENCE,
    PHONE_MAX_ASPECT,
    USE_DISTRACTOR_CLASSES,
    analyze_frame,
    load_model,
)


SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EVALUATION_QUERY_CONFIDENCE = 0.05


@dataclass(frozen=True)
class EvaluationSummary:
    phone_found: int
    phone_total: int
    false_positives: int
    not_phone_total: int


def evaluate_samples(
    model: Any,
    samples_dir: Path = SAMPLES_DIR,
    *,
    conf: float = PHONE_CONFIDENCE,
    max_aspect: float | None = PHONE_MAX_ASPECT,
    use_distractors: bool = USE_DISTRACTOR_CLASSES,
    use_verifier: bool = False,
    verifier: Callable[[np.ndarray, tuple[int, int, int, int]], float | None] | None = None,
    save_debug: bool = False,
    output: TextIO = sys.stdout,
) -> EvaluationSummary:
    """Считает результаты каждого кадра без окна и таймеров."""
    phone_found = phone_total = false_positives = not_phone_total = 0
    verification_times: list[float] = []
    first_verification_seen = False
    debug_dir = samples_dir / "_debug"
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in ("phone", "not_phone"):
        folder = samples_dir / folder_name
        paths = sorted(path for path in folder.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        for path in paths:
            frame = read_image(path)
            if frame is None:
                raise ValueError(f"Не удалось прочитать изображение: {path}")
            result = analyze_frame(
                frame, model, phone_confidence=conf,
                max_aspect=max_aspect, use_distractors=use_distractors,
                use_verifier=use_verifier, verifier=verifier,
                query_confidence=min(conf, EVALUATION_QUERY_CONFIDENCE),
            )
            for candidate in result.phone_candidates:
                if candidate.verification_ms is not None:
                    if first_verification_seen:
                        verification_times.append(candidate.verification_ms)
                    first_verification_seen = True
            accepted = any(candidate.rejected_reason is None
                           for candidate in result.phone_candidates)
            if folder_name == "phone":
                phone_total += 1
                phone_found += int(accepted)
            else:
                not_phone_total += 1
                false_positives += int(accepted)

            descriptions = []
            for candidate in result.phone_candidates:
                score = candidate.detection.verifier_score
                score_text = f"{score:.3f}" if score is not None else "—"
                decision = ("принят" if candidate.rejected_reason is None
                            else f"отброшен ({candidate.rejected_reason})")
                descriptions.append(
                    f"{candidate.detection.confidence:.2f} "
                    f"a={candidate.detection.aspect:.2f} "
                    f"verifier_score={score_text}: {decision}"
                )
            if save_debug:
                debug_frame = frame.copy()
                for candidate in result.phone_candidates:
                    color = (0, 180, 0) if candidate.rejected_reason is None else (0, 0, 255)
                    x1, y1, x2, y2 = candidate.detection.bbox
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    score = candidate.detection.verifier_score
                    label = (f"phone {candidate.detection.confidence:.2f} "
                             f"a={candidate.detection.aspect:.2f} "
                             f"clip={score:.2f}" if score is not None else
                             f"phone {candidate.detection.confidence:.2f} "
                             f"a={candidate.detection.aspect:.2f}")
                    cv2.putText(debug_frame, label, (max(0, x1), max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                debug_path = debug_dir / f"{folder_name}_{path.name}"
                write_image(debug_path, debug_frame)
            print(f"{folder_name}/{path.name}: "
                  + ("; ".join(descriptions) if descriptions else "нет рамок телефона"),
                  file=output)

    print(f"Телефон найден: {phone_found} из {phone_total} кадров с телефоном", file=output)
    print(f"Ложные срабатывания: {false_positives} из {not_phone_total} кадров без телефона",
          file=output)
    average_ms = sum(verification_times) / len(verification_times) if verification_times else 0.0
    print(f"Среднее время проверки рамки без первого вызова: {average_ms:.1f} мс "
          f"({len(verification_times)} рамок)", file=output)
    return EvaluationSummary(phone_found, phone_total, false_positives, not_phone_total)


def main(argv: list[str] | None = None) -> int:
    """Читает пороги из командной строки и запускает оценку."""
    parser = argparse.ArgumentParser(description="Оценка YOLO по кадрам из detection/samples")
    parser.add_argument("--conf", type=float, default=PHONE_CONFIDENCE,
                        help="порог уверенности телефона")
    parser.add_argument("--max-aspect", type=float, default=PHONE_MAX_ASPECT,
                        help="максимальное отношение сторон телефона")
    parser.add_argument("--no-distractors", action="store_true",
                        help="не проверять пересечение с классами-обманками")
    parser.add_argument("--verify", action="store_true", help="проверять рамки через CLIP")
    parser.add_argument("--save-debug", action="store_true",
                        help="сохранить кадры с рамками в samples/_debug")
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1:
        parser.error("--conf должен быть от 0 до 1")
    if args.max_aspect is not None and args.max_aspect <= 0:
        parser.error("--max-aspect должен быть положительным")
    evaluate_samples(
        load_model(), conf=args.conf, max_aspect=args.max_aspect,
        use_distractors=not args.no_distractors,
        use_verifier=args.verify, save_debug=args.save_debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
