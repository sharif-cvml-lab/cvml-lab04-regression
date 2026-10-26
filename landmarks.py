"""Fixed infrastructure: face-landmark extraction and landmark preprocessing.

Landmark convention (identical for the dataset, training, inference, examples):
  * 478 points, MediaPipe Face Landmarker order, shape (478, 2), columns (x, y)
  * pixel coordinates of the image that was given to the detector, origin at the
    top-left corner, x to the right, y downwards, truncated to integer pixels
    (this is exactly how the supplied dataset was produced)
  * the image is NEVER mirrored before detection
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import (
    CENTER_LANDMARK_IDX,
    LANDMARKER_MODEL_PATH,
    NUM_LANDMARKS,
    SCALE_LANDMARK_IDX_A,
    SCALE_LANDMARK_IDX_B,
)


class LandmarkerError(RuntimeError):
    """The face landmarker could not be created or returned unusable data."""


class FaceLandmarkExtractor:
    """MediaPipe Face Landmarker wrapper (created once and reused for every frame)."""

    def __init__(self, model_path=LANDMARKER_MODEL_PATH):
        path = Path(model_path)
        if not path.is_file():
            raise LandmarkerError(
                f"Face-landmarker model not found: {path}\n"
                "Place 'face_landmarker.task' in assets/models/."
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:
            raise LandmarkerError(
                f"MediaPipe is not installed ({exc}). Run: pip install -r requirements.txt"
            ) from exc
        try:
            # A buffer (instead of a path) avoids problems with non-ASCII Windows paths.
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=path.read_bytes()),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            raise LandmarkerError(f"Could not create the face landmarker: {exc}") from exc
        self._mp = mp

    def extract(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Return the (478, 2) pixel landmarks of the first face, or None if no face."""
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        try:
            result = self._detector.detect(mp_image)
        except Exception as exc:
            raise LandmarkerError(f"Landmark detection failed: {exc}") from exc
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        if len(face) != NUM_LANDMARKS:
            raise LandmarkerError(
                f"The landmark model returned {len(face)} points, expected {NUM_LANDMARKS}. "
                "Use the supplied face_landmarker.task."
            )
        height, width = image_bgr.shape[:2]
        normalized = np.array([[p.x, p.y] for p in face], dtype=np.float64)
        pixels = np.trunc(normalized * np.array([width, height], dtype=np.float64))
        return pixels.astype(np.float32)

    def close(self) -> None:
        detector, self._detector = getattr(self, "_detector", None), None
        if detector is not None:
            detector.close()


def normalize_landmarks(points: np.ndarray) -> np.ndarray:
    """Convert (478, 2) pixel landmarks into the 956-value feature vector.

    1. Translation: subtract landmark 99 (nose side) -> position independent.
    2. Scale: divide by the distance between landmarks 10 and 171 -> face-size
       (camera distance / image resolution) independent.
    3. Layout: [x_0 ... x_477, y_0 ... y_477].

    Raises ValueError for invalid input. Training and prediction MUST both use
    this function so that they see identical features.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (NUM_LANDMARKS, 2):
        raise ValueError(f"Expected landmarks of shape ({NUM_LANDMARKS}, 2), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Landmarks contain NaN or infinite values")
    x_values = points[:, 0] - points[CENTER_LANDMARK_IDX, 0]
    y_values = points[:, 1] - points[CENTER_LANDMARK_IDX, 1]
    scale = float(np.hypot(
        points[SCALE_LANDMARK_IDX_A, 0] - points[SCALE_LANDMARK_IDX_B, 0],
        points[SCALE_LANDMARK_IDX_A, 1] - points[SCALE_LANDMARK_IDX_B, 1],
    ))
    if scale < 1e-6:
        raise ValueError("Degenerate landmarks: normalisation distance is zero")
    return np.concatenate([x_values / scale, y_values / scale])
