"""Example 4 - predict pitch, yaw, roll of one image with the trained models.

Needs the models made by example 03 (assets/examples/output/). Pass --use-app-models
to use the models trained in the application (assets/models/) instead.

Run:  python assets/examples/04_predict_angles.py [image_path] [--use-app-models]
Output: prints the angles and saves assets/examples/output/04_prediction.jpg
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2

from annotation import draw_landmarks, put_label
from config import ANGLE_NAMES, MODELS_DIR
from landmarks import FaceLandmarkExtractor, normalize_landmarks
from model_store import ModelStoreError, load_model_bundle

arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
image_path = Path(arguments[0]) if arguments else ROOT / "assets/examples/image00013.jpg"
model_dir = MODELS_DIR if "--use-app-models" in sys.argv else ROOT / "assets/examples/output"

image = cv2.imread(str(image_path))
if image is None:
    sys.exit(f"Could not read the image: {image_path}")
try:
    bundle = load_model_bundle(model_dir)
except ModelStoreError as exc:
    sys.exit(f"No usable models in {model_dir}: {exc}\nRun example 03 (or train in the application) first.")

extractor = FaceLandmarkExtractor()
points = extractor.extract(image)
extractor.close()
if points is None:
    sys.exit("No face detected.")

features = normalize_landmarks(points).reshape(1, -1)  # SAME preprocessing as training
angles = [float(bundle.models[name].predict(features)[0]) for name in ANGLE_NAMES]
print(f"Model: {bundle.model_type}")
for name, value in zip(ANGLE_NAMES, angles):
    print(f"{name:5s}: {value:+6.1f} deg")

annotated = image.copy()
draw_landmarks(annotated, points)
put_label(annotated, "pitch %+.1f  yaw %+.1f  roll %+.1f deg" % tuple(angles), (10, 24))
out = ROOT / "assets/examples/output/04_prediction.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), annotated)
print(f"Saved {out}")
