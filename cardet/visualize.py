"""Drawing helpers shared by the evaluation reports and the web app."""

from __future__ import annotations

import cv2
import numpy as np

GREEN = (80, 200, 60)


def draw_boxes(gray: np.ndarray, boxes, color=GREEN) -> np.ndarray:
    """Return a BGR copy of `gray` with labelled detection boxes."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        cv2.rectangle(img, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
        label = f"{b.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        y = max(b.y, th + 4)
        cv2.rectangle(img, (b.x, y - th - 4), (b.x + tw + 4, y), color, -1)
        cv2.putText(
            img, label, (b.x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA
        )
    return img


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()
