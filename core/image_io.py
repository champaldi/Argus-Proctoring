"""OpenCV codecs with Python filesystem I/O for Unicode Windows paths."""

from pathlib import Path
from typing import Any


def write_image(path: Path, frame: Any) -> None:
    import cv2

    ok, encoded = cv2.imencode(path.suffix, frame)
    if not ok:
        raise OSError(f"could not encode image for {path}")
    path.write_bytes(encoded.tobytes())


def read_image(path: Path) -> Any:
    import cv2
    import numpy as np

    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if not encoded.size:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
