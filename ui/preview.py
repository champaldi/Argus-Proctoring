"""Prepare an owned display image without modifying the detector's BGR frame."""

from PySide6.QtGui import QImage


def make_preview(frame, width: int, height: int) -> QImage:
    import cv2
    import numpy as np

    source_height, source_width = frame.shape[:2]
    scale = min(1.0, max(1, width) / source_width, max(1, height) / source_height)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, int(source_width * scale)), max(1, int(source_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    frame = np.ascontiguousarray(frame)
    height, width = frame.shape[:2]
    # Qt accepts BGR directly. copy() detaches from OpenCV's temporary buffer.
    return QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888).copy()
