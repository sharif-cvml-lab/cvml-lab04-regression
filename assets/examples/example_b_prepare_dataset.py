"""Example B - understand (and build) the landmark dataset.

Inspect the supplied dataset (default):
    python assets/examples/example_b_prepare_dataset.py

Build a NEW dataset from your own images:
    python assets/examples/example_b_prepare_dataset.py build IMAGES_DIR LABELS_CSV OUTPUT_CSV
where LABELS_CSV has the columns  filename,pitch,yaw,roll  (angles in RADIANS, one row per image).
Rows are written as:  filename, pitch, yaw, roll, x_0, y_0, x_1, y_1, ..., x_477, y_477
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _paths import DATASET, LANDMARKER
from face_landmarker import FaceLandmarkExtractor


def inspect(csv_path):
    df = pd.read_csv(csv_path)
    print(f"{len(df)} samples, {df.shape[1]} columns")
    print("First columns:", list(df.columns[:8]), "...")

    # Interleaved layout: x_0, y_0, x_1, y_1, ...  -> reshape gives one (478, 2) array per sample.
    coordinate_columns = []
    for i in range(478):
        coordinate_columns += [f"x_{i}", f"y_{i}"]
    landmarks = df[coordinate_columns].to_numpy().reshape(-1, 478, 2)
    print("Landmarks array:", landmarks.shape, "(samples, landmarks, x/y)")

    angles = df[["pitch", "yaw", "roll"]].to_numpy()
    print("Angle std (radians):", np.round(angles.std(axis=0), 2), "-> the labels are in radians")
    print("Largest |angle| in degrees:", np.round(np.degrees(np.abs(angles).max(axis=0)), 1))
    wrapped = (np.abs(np.degrees(angles)) > 75).any(axis=1).sum()
    print(f"{wrapped} samples have an angle above 75 degrees (invalid/wrapped labels; training skips them)")


def build(images_dir, labels_csv, output_csv):
    labels = {}
    with open(labels_csv, newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            labels[Path(row["filename"]).stem] = (row["pitch"], row["yaw"], row["roll"])

    extractor = FaceLandmarkExtractor(LANDMARKER)
    header = ["filename", "pitch", "yaw", "roll"]
    for i in range(478):
        header += [f"x_{i}", f"y_{i}"]

    saved = 0
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for image_path in sorted(Path(images_dir).glob("*.jpg")):
            if image_path.stem not in labels:
                continue
            image = cv2.imread(str(image_path))
            landmarks = extractor.extract_landmarks(image) if image is not None else None
            if landmarks is None:
                print("Skipped (unreadable or no face):", image_path.name)
                continue
            row = [image_path.name, *labels[image_path.stem]]
            for x, y in landmarks:
                row += [f"{x:.4f}", f"{y:.4f}"]
            writer.writerow(row)
            saved += 1
    extractor.close()
    print(f"Saved {saved} samples to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        inspect(sys.argv[1] if len(sys.argv) > 1 else DATASET)
