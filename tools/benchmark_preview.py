"""Compare display-only processing against main's former full-frame RGB path.

Uses synthetic pixels, without a camera or detector models. Results describe
preview preparation and UI rendering, not end-to-end camera/ML throughput.
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from ui.preview import make_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    app = QApplication.instance() or QApplication([])
    frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    target = QSize(480, 270)

    def previous():
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, 1280, 720, rgb.strides[0], QImage.Format.Format_RGB888).copy()

    def optimized():
        return make_preview(frame, target.width(), target.height())

    print("Synthetic 1280x720 BGR -> 480x270 preview (no camera or inference)")
    for name, prepare in (("Previous", previous), ("Optimized", optimized)):
        preparation, rendering = [], []
        for index in range(args.iterations + 30):
            started = time.perf_counter()
            image = prepare()
            prepared = time.perf_counter()
            pixmap = QPixmap.fromImage(image).scaled(
                target, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            rendered = time.perf_counter()
            if index >= 30:
                preparation.append((prepared - started) * 1000)
                rendering.append((rendered - prepared) * 1000)
        total = [a + b for a, b in zip(preparation, rendering)]
        print(f"{name}: prepare median {statistics.median(preparation):.3f} ms; "
              f"UI median {statistics.median(rendering):.3f} ms; "
              f"total median {statistics.median(total):.3f} ms; "
              f"image {image.sizeInBytes():,} bytes")
    # Keep the application alive until the final pixmap is released.
    del pixmap, image
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
