from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pointer import PointerState

APP_NAME = "CVML Lab 04 - Head-Controlled Pointer"
ORGANIZATION_NAME = "Sharif CVML Lab"

# ---------------------------------------------------------------------------
# Paths (always resolved relative to the project, never the working directory)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
DATASET_DIR = ASSETS_DIR / "dataset"
MODELS_DIR = ASSETS_DIR / "models"
EXAMPLES_DIR = ASSETS_DIR / "examples"
DEFAULT_DATASET_PATH = DATASET_DIR / "AFLW2000_landmarks.csv"
LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"  # never deleted by the app
MODEL_BUNDLE_FILENAME = "head_pose_regressors.joblib"  # the ONLY file the app writes/deletes

# ---------------------------------------------------------------------------
# Data contract (shared by dataset loading, training, inference and examples)
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 478  # MediaPipe Face Landmarker (468 face + 10 iris points)
NUM_FEATURES = 2 * NUM_LANDMARKS  # 956
ANGLE_NAMES = ("pitch", "yaw", "roll")  # fixed output order everywhere
DATASET_ANGLE_UNIT = "radians"  # unit of the pitch/yaw/roll columns in the CSV
MODEL_ANGLE_UNIT = "degrees"  # unit of the models' targets and of predict_angles()

# Preprocessing (see landmarks.normalize_landmarks)
CENTER_LANDMARK_IDX = 99
SCALE_LANDMARK_IDX_A = 10
SCALE_LANDMARK_IDX_B = 171
FEATURE_LAYOUT = "x_then_y"  # [x_0..x_477, y_0..y_477]
FEATURE_SPEC = {
    "layout": FEATURE_LAYOUT,
    "center_landmark": CENTER_LANDMARK_IDX,
    "scale_landmarks": [SCALE_LANDMARK_IDX_A, SCALE_LANDMARK_IDX_B],
}

# Training
ANGLE_LIMIT_DEG = 75.0  # samples with any |angle| above this are discarded
MIN_TRAIN_SAMPLES = 50
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42

MODEL_TYPES = ("linear", "ridge", "lasso")
MODEL_LABELS = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
}
# Fixed hyper-parameters (no search; chosen with 5-fold CV on the supplied dataset).
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.01
LASSO_MAX_ITER = 5000
DEFAULT_MODEL_TYPE = "ridge"

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
TARGET_FPS = 30
DEFAULT_CAMERA_INDEX = 0
DEFAULT_THEME = "dark"
MAX_PROCESS_DIMENSION = 960
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm")

# ---------------------------------------------------------------------------
# Pointer behaviour
# ---------------------------------------------------------------------------
YAW_RANGE_MIN, YAW_RANGE_MAX, DEFAULT_YAW_RANGE = 10, 60, 30  # degrees to reach the edge
PITCH_RANGE_MIN, PITCH_RANGE_MAX, DEFAULT_PITCH_RANGE = 5, 45, 20
SMOOTHING_MIN, SMOOTHING_MAX, DEFAULT_SMOOTHING = 0, 500, 120  # time constant, ms
DEFAULT_INVERT_X = False
DEFAULT_INVERT_Y = False
DEFAULT_SHOW_TRAIL = True
DEFAULT_MIRROR = True
DEFAULT_SHOW_LANDMARKS = True
DEFAULT_SHOW_ANGLES = True

HOLD_SECONDS = 0.75  # keep the pointer still this long after the face is lost
MAX_PLAUSIBLE_ANGLE_DEG = 120.0  # predictions beyond this are rejected as invalid
MAX_ROLL_DISPLAY_DEG = 90.0
TRAIL_LENGTH = 40

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
COLOR_LANDMARK = (80, 230, 120)  # BGR
COLOR_BADGE_OK = (70, 170, 90)
COLOR_BADGE_OFF = (60, 60, 200)
FRAME_TIMEOUT_SECONDS = 1.0  # a displayed frame not acknowledged for this long is re-sent

SHORTCUTS = {
    "pause": "Space",
    "fullscreen": "F11",
    "reset": "R",
    "quit": "Q",
    "center": "C",
}


@dataclass(frozen=True)
class PointerSettings:
    yaw_range: float = DEFAULT_YAW_RANGE
    pitch_range: float = DEFAULT_PITCH_RANGE
    smoothing_ms: int = DEFAULT_SMOOTHING
    invert_x: bool = DEFAULT_INVERT_X
    invert_y: bool = DEFAULT_INVERT_Y
    show_trail: bool = DEFAULT_SHOW_TRAIL


@dataclass(frozen=True)
class DisplaySettings:
    mirror: bool = DEFAULT_MIRROR
    show_landmarks: bool = DEFAULT_SHOW_LANDMARKS
    show_angles: bool = DEFAULT_SHOW_ANGLES


@dataclass(frozen=True)
class FrameResult:
    """Everything the GUI needs to know about one processed frame."""

    kind: str = "webcam"  # "webcam" | "image" | "video"
    face_detected: bool = False
    angles: tuple[float, float, float] | None = None  # (pitch, yaw, roll) in degrees
    pointer: "PointerState | None" = None
    models_available: bool = False
    fps: float = 0.0  # 0.0 = not a running live source
    process_ms: float = 0.0
    status: str = ""


def _as_float(value, default: float) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if value is None:
        return default
    return bool(value)


def validated_pointer_settings(
    yaw_range=DEFAULT_YAW_RANGE,
    pitch_range=DEFAULT_PITCH_RANGE,
    smoothing_ms=DEFAULT_SMOOTHING,
    invert_x=DEFAULT_INVERT_X,
    invert_y=DEFAULT_INVERT_Y,
    show_trail=DEFAULT_SHOW_TRAIL,
) -> PointerSettings:
    """Clamp arbitrary inputs to safe values; fall back to defaults if invalid."""
    yaw = min(max(_as_float(yaw_range, DEFAULT_YAW_RANGE), YAW_RANGE_MIN), YAW_RANGE_MAX)
    pitch = min(max(_as_float(pitch_range, DEFAULT_PITCH_RANGE), PITCH_RANGE_MIN), PITCH_RANGE_MAX)
    smoothing = int(
        round(min(max(_as_float(smoothing_ms, DEFAULT_SMOOTHING), SMOOTHING_MIN), SMOOTHING_MAX))
    )
    return PointerSettings(
        yaw, pitch, smoothing,
        _as_bool(invert_x, DEFAULT_INVERT_X),
        _as_bool(invert_y, DEFAULT_INVERT_Y),
        _as_bool(show_trail, DEFAULT_SHOW_TRAIL),
    )


def validated_display_settings(
    mirror=DEFAULT_MIRROR,
    show_landmarks=DEFAULT_SHOW_LANDMARKS,
    show_angles=DEFAULT_SHOW_ANGLES,
) -> DisplaySettings:
    return DisplaySettings(
        _as_bool(mirror, DEFAULT_MIRROR),
        _as_bool(show_landmarks, DEFAULT_SHOW_LANDMARKS),
        _as_bool(show_angles, DEFAULT_SHOW_ANGLES),
    )
