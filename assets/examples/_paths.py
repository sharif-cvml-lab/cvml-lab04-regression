"""Tiny helper shared by the examples: makes the project root importable and defines paths."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_IMAGE = ROOT / "assets" / "examples" / "image00013.jpg"
LANDMARKER = ROOT / "assets" / "models" / "face_landmarker.task"
DATASET = ROOT / "assets" / "dataset" / "AFLW2000_landmarks.csv"
TRAINED_DIR = ROOT / "assets" / "models" / "trained"
OUTPUT_DIR = ROOT / "assets" / "examples" / "output"
