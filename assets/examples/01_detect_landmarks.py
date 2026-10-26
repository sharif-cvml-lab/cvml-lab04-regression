"""Example 1 - detect facial landmarks in a still image and draw them.

Run:  python assets/examples/01_detect_landmarks.py [image_path]
Output: assets/examples/output/01_landmarks.jpg
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # lets the example import the project's modules

import cv2

from annotation import draw_landmarks
from landmarks import FaceLandmarkExtractor

image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets/examples/image00013.jpg"
output_path = ROOT / "assets/examples/output/01_landmarks.jpg"

image = cv2.imread(str(image_path))  # BGR image, NOT mirrored
if image is None:
    sys.exit(f"Could not read the image: {image_path}")

extractor = FaceLandmarkExtractor()  # loads assets/models/face_landmarker.task
points = extractor.extract(image)  # (478, 2) pixel coordinates (x, y) or None
extractor.close()
if points is None:
    sys.exit("No face detected.")

print(f"Landmarks: {points.shape}, first point (x, y) = {points[0]}")
annotated = image.copy()
draw_landmarks(annotated, points, radius=1)
output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), annotated)
print(f"Saved {output_path}")
