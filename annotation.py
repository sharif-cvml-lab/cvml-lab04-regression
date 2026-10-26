"""Fixed infrastructure: drawing on frames (no GUI-layout code)."""
from __future__ import annotations

import cv2
import numpy as np

from config import COLOR_BADGE_OFF, COLOR_BADGE_OK, COLOR_LANDMARK

FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_placeholder(width: int, height: int, lines: list[str]) -> np.ndarray:
    """Dark panel with centered text, used when no image is available."""
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (60, 66, 76), 1)
    y = height // 2 - 12 * (len(lines) - 1)
    for line in lines:
        (text_w, _), _ = cv2.getTextSize(line, FONT, 0.7, 1)
        cv2.putText(panel, line, (max(8, (width - text_w) // 2), y), FONT, 0.7,
                    (190, 198, 208), 1, cv2.LINE_AA)
        y += 28
    return panel


def draw_landmarks(image: np.ndarray, points: np.ndarray, color=COLOR_LANDMARK, radius: int = 1) -> None:
    """Draw landmark dots in place (points are pixel coordinates of this image)."""
    height, width = image.shape[:2]
    for x, y in np.rint(points).astype(np.int32):
        if -50 <= x < width + 50 and -50 <= y < height + 50:
            cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def put_label(image, text, origin, color=(235, 235, 235), bg=(25, 25, 30), scale=0.5) -> None:
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x, y = origin
    cv2.rectangle(image, (x - 4, y - text_h - 4), (x + text_w + 4, y + baseline + 2), bg, -1)
    cv2.putText(image, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def draw_overlay(image: np.ndarray, face_detected: bool, angles, show_angles: bool, message: str = "") -> None:
    """Face badge and angle read-out, drawn AFTER any mirroring so text is readable."""
    badge = "FACE" if face_detected else "NO FACE"
    put_label(image, badge, (10, 24), (255, 255, 255),
              COLOR_BADGE_OK if face_detected else COLOR_BADGE_OFF, 0.55)
    if show_angles and angles is not None:
        pitch, yaw, roll = angles
        put_label(image, f"pitch {pitch:+6.1f}  yaw {yaw:+6.1f}  roll {roll:+6.1f} deg",
                  (10, image.shape[0] - 12), scale=0.5)
    elif message:
        put_label(image, message[:70], (10, image.shape[0] - 12), scale=0.45)
