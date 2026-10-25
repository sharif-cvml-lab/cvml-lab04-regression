"""Example D - estimate pitch, yaw and roll for one image with the trained models.

Run:  python assets/examples/example_d_predict_single_image.py [path/to/image.jpg]
Run example C (or press Train in the GUI) first.
"""
import sys

import _paths  # noqa: F401  (makes the project root importable; keep before project imports)
import cv2
import joblib

import model_store
from _paths import LANDMARKER, OUTPUT_DIR, SAMPLE_IMAGE, TRAINED_DIR
from face_landmarker import FaceLandmarkExtractor, annotate_face_landmarks, landmarks_to_feature_vector

image_path = sys.argv[1] if len(sys.argv) > 1 else str(SAMPLE_IMAGE)
image = cv2.imread(image_path)
if image is None:
    raise SystemExit(f"Could not read the image: {image_path}")

status, message, metadata = model_store.check_models(TRAINED_DIR)
if status != "ok":
    raise SystemExit(f"{message} Train first (example C or the GUI).")

extractor = FaceLandmarkExtractor(LANDMARKER)
landmarks = extractor.extract_landmarks(image)
extractor.close()
if landmarks is None:
    raise SystemExit("No face detected.")

features = landmarks_to_feature_vector(landmarks)          # exactly the same preprocessing as training
if features is None:
    raise SystemExit("The landmarks could not be normalized.")
x = features.reshape(1, -1)                                 # scikit-learn wants (n_samples, n_features)

paths = model_store.model_paths(TRAINED_DIR)
pitch, yaw, roll = (float(joblib.load(paths[name]).predict(x)[0]) for name in ("pitch", "yaw", "roll"))
print(f"Model: {metadata['model_type']}   pitch={pitch:.1f}  yaw={yaw:.1f}  roll={roll:.1f}  (degrees)")

annotated = annotate_face_landmarks(image, landmarks)
cv2.putText(annotated, f"pitch {pitch:.1f} yaw {yaw:.1f} roll {roll:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
OUTPUT_DIR.mkdir(exist_ok=True)
cv2.imwrite(str(OUTPUT_DIR / "prediction.jpg"), annotated)
print("Saved:", OUTPUT_DIR / "prediction.jpg")
