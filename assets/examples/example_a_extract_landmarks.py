"""Example A - detect facial landmarks in one image and draw them.

Run:  python assets/examples/example_a_extract_landmarks.py [path/to/image.jpg]
"""
import sys

import cv2

from _paths import LANDMARKER, OUTPUT_DIR, SAMPLE_IMAGE
from face_landmarker import FaceLandmarkExtractor, annotate_face_landmarks


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else str(SAMPLE_IMAGE)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read the image: {image_path}")
        return
    if not LANDMARKER.is_file():
        print(f"Missing MediaPipe model file: {LANDMARKER}")
        return

    extractor = FaceLandmarkExtractor(LANDMARKER)
    landmarks = extractor.extract_landmarks(image)        # (478, 2) pixel coordinates, or None
    extractor.close()
    if landmarks is None:
        print("No face detected.")
        return

    print(f"Found {len(landmarks)} landmarks. Landmark 1 (nose tip) is at x={landmarks[1][0]:.1f}, y={landmarks[1][1]:.1f}")

    annotated = annotate_face_landmarks(image, landmarks, radius=1)
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "annotated.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
