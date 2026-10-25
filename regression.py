"""Lab 04 - Regression: the ONLY file students edit.

The application around this file is complete. It already
  * extracts the 478 facial landmarks of every webcam frame (MediaPipe),
  * calls  predict_angles()  for each frame and moves the virtual pointer with the result,
  * calls  train_models()    (in a background thread) when you press the Train button.

Until you complete the two functions below, the app still runs: it shows the camera and the
landmarks, and reports "not implemented" / "models not trained yet" instead of crashing.

Conventions you must follow (they make training and inference consistent):
  * Features : face_landmarker.landmarks_to_feature_vector(points) -> shape (956,) or None.
               Use it in BOTH functions. Never write your own normalization.
  * Targets  : pitch, yaw, roll, always in this order.
  * Units    : the CSV stores angles in RADIANS.  Models are trained in DEGREES.
               Convert once, with np.degrees(), while preparing the targets in train_models().
  * Files    : save the three models with joblib.dump() to the paths returned by
               model_store.model_paths(model_dir).  Do NOT write metadata.json - the app does it.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

import model_store
from config import ANGLE_NAMES, MAX_VALID_ANGLE_DEG, MIN_TRAINING_SAMPLES, NUM_LANDMARKS
from face_landmarker import landmarks_to_feature_vector


# =============================================================================
# TASK 1 - Training
# =============================================================================
def train_models(dataset_path: str, model_type: str, model_dir: str, alpha: float = 1.0) -> dict:
    """Train one regression model each for pitch, yaw and roll, and save them.

    Args:
        dataset_path: CSV file with columns filename, pitch, yaw, roll, x_0, y_0, ..., x_477, y_477.
        model_type:   "Linear", "Ridge" or "Lasso".
        model_dir:    folder where the three .joblib files must be saved (it already exists).
        alpha:        regularization strength (used by Ridge and Lasso, ignored by Linear).

    Returns:
        {"success": True,  "message": "...", "n_samples": <number of samples used for training>}
        {"success": False, "message": "<what went wrong, in plain English>"}
    """
    # TODO 1: Load the dataset with pd.read_csv(dataset_path).
    #         If it cannot be read, return {"success": False, "message": "..."}.

    # TODO 2: Build the list of coordinate column names in the correct (interleaved) order:
    #             x_0, y_0, x_1, y_1, ..., x_477, y_477
    #         (use a for-loop over range(NUM_LANDMARKS)).  Check that these columns and
    #         "pitch", "yaw", "roll" exist; if some are missing, return a failure message.

    # TODO 3: Get the coordinates as a NumPy array and reshape it to (number_of_samples, NUM_LANDMARKS, 2).
    #         Because the columns are interleaved, df[coordinate_columns].to_numpy().reshape(-1, NUM_LANDMARKS, 2)
    #         gives landmarks[i, j] = (x_j, y_j) of sample i.

    # TODO 4: Read the three label columns, convert radians -> degrees with np.degrees().

    # TODO 5: Loop over the samples. Skip a sample when
    #           - one of its angles is not finite (np.isfinite), or
    #           - one of its angles has an absolute value above MAX_VALID_ANGLE_DEG
    #             (the dataset contains a few wrapped / invalid labels), or
    #           - landmarks_to_feature_vector(points) returns None.
    #         Collect the feature vectors and the (pitch, yaw, roll) targets of the kept samples.
    #         Convert both lists to NumPy arrays X (n, 956) and Y (n, 3).
    #         If fewer than MIN_TRAINING_SAMPLES samples remain, return a failure message.

    # TODO 6: For each of the three angles (index 0, 1, 2 in ANGLE_NAMES):
    #           a) create the estimator for model_type:
    #                "Linear" -> LinearRegression()
    #                "Ridge"  -> Ridge(alpha=alpha)
    #                "Lasso"  -> Lasso(alpha=alpha, max_iter=5000)
    #              (any other model_type -> failure message)
    #           b) train it:      model.fit(X, Y[:, index])
    #           c) save it:       joblib.dump(model, model_store.model_paths(model_dir)[name])

    # TODO 7: Return {"success": True, "message": "...", "n_samples": X.shape[0]}.

    return {
        "success": False,
        "message": "train_models() is not implemented yet. Complete the TODOs in regression.py.",
    }


# =============================================================================
# TASK 2 - Inference
# =============================================================================
def predict_angles(landmarks: np.ndarray, model_dir: str):
    """Estimate (pitch, yaw, roll) in DEGREES from the landmarks of ONE frame.

    Args:
        landmarks: array of shape (478, 2) with pixel coordinates (row i = landmark i).
        model_dir: folder that contains the trained models.

    Returns:
        (pitch, yaw, roll) as three Python floats, or None if no estimate can be made
        (invalid landmarks, models not trained / not loadable, invalid prediction).
        Never return made-up numbers.
    """
    # TODO 1: Convert the landmarks to a feature vector with landmarks_to_feature_vector().
    #         If it returns None, return None.

    # TODO 2: Get the three trained models with model_store.load_models(model_dir).
    #         It returns (pitch_model, yaw_model, roll_model) or None. If None, return None.
    #         (It caches the models, so it is cheap to call for every frame.)

    # TODO 3: scikit-learn expects a 2-D array: features.reshape(1, -1).
    #         Predict with each model:  float(model.predict(x)[0])  -> pitch, yaw, roll.

    # TODO 4: If any prediction is not finite (np.isfinite), return None.
    #         Otherwise return (pitch, yaw, roll).

    return None
