"""Example 2 - build a dataset CSV from images with known head angles.

Inputs : a folder of images and a labels CSV with columns
         filename, pitch, yaw, roll   (angles in RADIANS, same convention as AFLW2000)
Output : a CSV with columns filename, pitch, yaw, roll, x_0, y_0, ..., x_477, y_477
         that can be selected in the application ("Browse..." next to Train).

Run:  python assets/examples/02_build_dataset.py [--images DIR] [--labels CSV] [--output CSV]
Default: the demo image + labels_demo.csv -> assets/examples/output/demo_dataset.csv
(one row only - too small to train on, it just shows the format).
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2

from config import NUM_LANDMARKS
from landmarks import FaceLandmarkExtractor

EXAMPLES = ROOT / "assets/examples"
parser = argparse.ArgumentParser()
parser.add_argument("--images", type=Path, default=EXAMPLES)
parser.add_argument("--labels", type=Path, default=EXAMPLES / "labels_demo.csv")
parser.add_argument("--output", type=Path, default=EXAMPLES / "output/demo_dataset.csv")
args = parser.parse_args()

with args.labels.open(newline="", encoding="utf-8-sig") as file:
    labels = {row["filename"]: row for row in csv.DictReader(file)}

header = ["filename", "pitch", "yaw", "roll"]
for i in range(NUM_LANDMARKS):
    header += [f"x_{i}", f"y_{i}"]  # interleaved: x_0, y_0, x_1, y_1, ...

extractor = FaceLandmarkExtractor()
args.output.parent.mkdir(parents=True, exist_ok=True)
saved = 0
with args.output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for name, label in labels.items():
        image = cv2.imread(str(args.images / name))
        if image is None:
            print(f"skip {name}: image not readable")
            continue
        points = extractor.extract(image)
        if points is None:
            print(f"skip {name}: no face detected")
            continue
        row = [name, label["pitch"], label["yaw"], label["roll"]]
        for x, y in points:
            row += [f"{x:.0f}", f"{y:.0f}"]  # integer pixel coordinates
        writer.writerow(row)
        saved += 1
extractor.close()
print(f"Wrote {saved} rows to {args.output}")
