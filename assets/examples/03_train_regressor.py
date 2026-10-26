"""Example 3 - load the dataset, train pitch/yaw/roll regressors, save them.

This is a plain walkthrough of the same pipeline the application uses. It saves
into assets/examples/output/ so it never overwrites the models of the application.

Run:  python assets/examples/03_train_regressor.py [--dataset CSV] [--model linear|ridge|lasso]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from config import (ANGLE_LIMIT_DEG, ANGLE_NAMES, DEFAULT_DATASET_PATH, LASSO_ALPHA, LASSO_MAX_ITER,
                    NUM_LANDMARKS, RANDOM_SEED, RIDGE_ALPHA)
from dataset import coordinate_columns, load_dataset_table
from landmarks import normalize_landmarks
from model_store import save_model_bundle

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
parser.add_argument("--model", choices=("linear", "ridge", "lasso"), default="ridge")
args = parser.parse_args()
output_dir = ROOT / "assets/examples/output"

table, report = load_dataset_table(args.dataset)  # validated table
print(report)

# Targets (radians in the file -> degrees) and landmarks (N, 478, 2)
angles = np.degrees(table[list(ANGLE_NAMES)].to_numpy())
points = table[coordinate_columns()].to_numpy().reshape(-1, NUM_LANDMARKS, 2)

keep = np.all(np.abs(angles) <= ANGLE_LIMIT_DEG, axis=1)  # drop impossible poses
angles, points = angles[keep], points[keep]

X = np.array([normalize_landmarks(p) for p in points])  # (N, 956)
X_train, X_val, y_train, y_val = train_test_split(X, angles, test_size=0.2, random_state=RANDOM_SEED)

models = {}
for k, name in enumerate(ANGLE_NAMES):
    if args.model == "linear":
        model = LinearRegression()
    elif args.model == "ridge":
        model = Ridge(alpha=RIDGE_ALPHA)
    else:
        model = Lasso(alpha=LASSO_ALPHA, max_iter=LASSO_MAX_ITER)
    model.fit(X_train, y_train[:, k])
    error = np.mean(np.abs(model.predict(X_val) - y_val[:, k]))
    print(f"{name:5s} validation MAE: {error:.2f} deg")
    models[name] = model

path = save_model_bundle(output_dir, args.model, models, {"num_samples": int(len(X))})
print(f"Saved {path}")
