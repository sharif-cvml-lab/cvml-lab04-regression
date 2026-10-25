"""Shared constants for the Head-Controlled Virtual Pointer lab (not student-facing)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_NAME = "CVML Lab 04 - Head-Controlled Pointer"
ORGANIZATION_NAME = "Sharif CVML Lab"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
DATASET_DIR = ASSETS_DIR / "dataset"
MODELS_DIR = ASSETS_DIR / "models"
EXAMPLES_DIR = ASSETS_DIR / "examples"

DEFAULT_DATASET_PATH = DATASET_DIR / "AFLW2000_landmarks.csv"
LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"      # never deleted by the app
TRAINED_MODELS_DIR = MODELS_DIR / "trained"                 # the only folder the app deletes from

# ---------------------------------------------------------------------------
# Landmarks, features and angle conventions
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 478
NUM_FEATURES = 2 * NUM_LANDMARKS

# Normalization used for BOTH training and inference (see face_landmarker.py):
# translate so landmark 99 is the origin, divide by the distance |10 - 171|.
CENTER_LANDMARK_IDX = 99
SCALE_LANDMARK_A = 10
SCALE_LANDMARK_B = 171
MIN_SCALE_DISTANCE = 1e-3
FEATURE_VERSION = 1

ANGLE_NAMES = ("pitch", "yaw", "roll")      # fixed output order everywhere
DATASET_ANGLE_UNIT = "radians"              # verified on the supplied CSV (std ~0.5, not ~30)
MODEL_ANGLE_UNIT = "degrees"                # what the models predict / the GUI shows
MAX_VALID_ANGLE_DEG = 75.0                  # |angle| above this is treated as an invalid label
MIN_TRAINING_SAMPLES = 50

# ---------------------------------------------------------------------------
# Regression models
# ---------------------------------------------------------------------------
MODEL_TYPES = ("Linear", "Ridge", "Lasso")
DEFAULT_MODEL_TYPE = "Ridge"
DEFAULT_ALPHA = {"Linear": 1.0, "Ridge": 1.0, "Lasso": 0.01}   # Linear ignores alpha
ALPHA_MIN = 0.0001
ALPHA_MAX = 1000.0

MODEL_FILENAMES = {
    "pitch": "pitch_model.joblib",
    "yaw": "yaw_model.joblib",
    "roll": "roll_model.joblib",
}
METADATA_FILENAME = "metadata.json"
MODEL_FORMAT_VERSION = 1

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
DEFAULT_CAMERA_INDEX = 0
MAX_CAMERA_INDEX = 9
CAMERA_MAX_READ_FAILURES = 30        # consecutive failed reads before "disconnected"
MAX_PROCESS_DIMENSION = 960          # test images are downscaled to this longest side
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
IMAGE_REFRESH_MS = 250               # static-image source is re-evaluated at ~4 Hz

# ---------------------------------------------------------------------------
# Pointer (angles in degrees)
# ---------------------------------------------------------------------------
DEFAULT_YAW_RANGE = 30.0             # +-30 deg of yaw spans the full width
DEFAULT_PITCH_RANGE = 20.0           # +-20 deg of pitch spans the full height
YAW_RANGE_LIMITS = (10, 60)
PITCH_RANGE_LIMITS = (10, 45)
DEFAULT_SMOOTHING = 60               # percent; higher = steadier but slower
SMOOTHING_LIMITS = (0, 95)
DEFAULT_INVERT_X = False
DEFAULT_INVERT_Y = False
DEFAULT_SHOW_TRAIL = True
TRAIL_LENGTH = 40
FACE_LOST_RESET_FRAMES = 10          # frames without a pose before smoothing is reset

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
DEFAULT_THEME = "dark"
DEFAULT_MIRROR = True
DEFAULT_SHOW_LANDMARKS = True

SHORTCUTS = {
    "pause": "Space",
    "fullscreen": "F11",
    "reset": "R",
    "quit": "Q",
}


@dataclass(frozen=True)
class FrameResult:
    """Everything the GUI needs to know about one processed frame."""
    face_found: bool = False
    angles: tuple[float, float, float] | None = None   # (pitch, yaw, roll) in degrees
    status: str = ""
    fps: float = 0.0                                    # 0.0 = not a running live source
    processing_ms: float = 0.0
    source: str = "camera"                              # "camera" or "image"
