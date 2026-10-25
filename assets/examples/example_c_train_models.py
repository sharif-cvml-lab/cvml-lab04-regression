"""Example C - train pitch / yaw / roll regressors from the landmark dataset.

Run:  python assets/examples/example_c_train_models.py [Linear|Ridge|Lasso] [alpha]
The models are saved to assets/models/trained/, where the GUI also looks for them.
NOTE FOR INSTRUCTORS: this script contains the full training solution of regression.py.
"""
import sys

import _paths  # noqa: F401  (makes the project root importable; keep before project imports)
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split

import model_store
from _paths import DATASET, TRAINED_DIR
from face_landmarker import landmarks_to_feature_vector

model_type = sys.argv[1] if len(sys.argv) > 1 else "Ridge"
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0


def make_model():
    if model_type == "Linear":
        return LinearRegression()
    if model_type == "Ridge":
        return Ridge(alpha=alpha)
    if model_type == "Lasso":
        return Lasso(alpha=alpha, max_iter=5000)
    raise SystemExit(f"Unknown model type: {model_type}")


# 1. Load, and build the column list in the file's interleaved order (x_0, y_0, x_1, ...)
df = pd.read_csv(DATASET)
columns = []
for i in range(478):
    columns += [f"x_{i}", f"y_{i}"]
landmarks = df[columns].to_numpy().reshape(-1, 478, 2)

# 2. Targets: radians in the file -> degrees for the models
angles_deg = np.degrees(df[["pitch", "yaw", "roll"]].to_numpy())

# 3. Features (same function as inference) and removal of invalid samples
features, targets = [], []
for points, angles in zip(landmarks, angles_deg):
    if not np.isfinite(angles).all() or (np.abs(angles) > 75).any():
        continue
    vector = landmarks_to_feature_vector(points)
    if vector is not None:
        features.append(vector)
        targets.append(angles)
X, Y = np.array(features), np.array(targets)
print(f"{len(X)} usable samples out of {len(df)}")

# 4. Train one model per angle; report held-out error, then fit on everything and save
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
paths = model_store.model_paths(TRAINED_DIR)
for index, name in enumerate(["pitch", "yaw", "roll"]):
    check = make_model().fit(X_train, Y_train[:, index])
    mae = np.abs(check.predict(X_val) - Y_val[:, index]).mean()
    print(f"{name:5s} held-out MAE: {mae:.2f} deg")
    joblib.dump(make_model().fit(X, Y[:, index]), paths[name])

# 5. metadata.json marks the set as complete (the GUI writes it for you when you press Train)
model_store.write_metadata(TRAINED_DIR, model_type, alpha, len(X), DATASET.resolve())
print("Saved models to", TRAINED_DIR)
