"""Lab 04 - head-pose regression.   >>> THIS IS THE ONLY FILE STUDENTS MODIFY <<<

The application detects 478 facial landmarks (fixed code), calls YOUR functions
to learn / predict the head angles, and moves a virtual pointer with the result.

Conventions you must know (all verified on the supplied dataset):
  * Landmarks: np.ndarray of shape (478, 2), columns (x, y), pixel coordinates.
  * Dataset angles (pitch, yaw, roll) are stored in RADIANS.
  * The models are trained in and predict DEGREES.
  * Output order is ALWAYS (pitch, yaw, roll) - see ANGLE_NAMES.

Provided helpers you may (and should) use:
  landmarks.normalize_landmarks(points)  -> feature vector of 956 values.
        Centers on landmark 99, divides by the distance between landmarks 10
        and 171, returns [x_0..x_477, y_0..y_477]. Raises ValueError on bad input.
  dataset.load_dataset_table(path)       -> (pandas table, report text).
        Validates the CSV; raises DatasetError (a ValueError) with a readable message.
  model_store.save_model_bundle(...)     -> saves the 3 models safely (atomic).
  model_store.load_model_bundle(...)     -> loads (and caches) them. Raises ModelStoreError.
Do not edit any other file.
"""
from __future__ import annotations

import numpy as np

# Allowed building blocks (import what you need; extend this list if necessary).
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # noqa: F401
from sklearn.model_selection import train_test_split  # noqa: F401

from config import (  # noqa: F401
    ANGLE_LIMIT_DEG,
    ANGLE_NAMES,
    LASSO_ALPHA,
    LASSO_MAX_ITER,
    MIN_TRAIN_SAMPLES,
    MODEL_TYPES,
    NUM_LANDMARKS,
    RANDOM_SEED,
    RIDGE_ALPHA,
    VALIDATION_FRACTION,
)
from dataset import DatasetError, coordinate_columns, load_dataset_table  # noqa: F401
from landmarks import normalize_landmarks  # noqa: F401
from model_store import ModelStoreError, load_model_bundle, save_model_bundle  # noqa: F401


def train_models(
    dataset_path: str,
    model_type: str,
    model_dir: str,
) -> tuple[bool, str]:
    """
    TODO: Train one regressor per angle (pitch, yaw, roll) and save them.

    IMPORTANT:
    - Keep this function signature unchanged.
    - Return (True, "message") on success and (False, "reason") on failure.
      Failure is a normal answer: never let an exception escape (use try/except
      around the work and return (False, str(error))).
    - Do not touch the GUI, the camera or any file other than through
      save_model_bundle().

    Arguments:
      dataset_path : path of the CSV file chosen in the GUI.
      model_type   : "linear", "ridge" or "lasso" (see MODEL_TYPES).
      model_dir    : directory where the models must be saved.

    Expected implementation sequence:

    1. Validate the arguments. Fail if model_type is not in MODEL_TYPES.

    2. Load the data with load_dataset_table(dataset_path). It raises
       DatasetError for missing files / columns / bad values - turn that into
       (False, message). Keep the returned `report` text for your final message.

    3. Separate FEATURES from TARGETS:
         - targets : the three columns in ANGLE_NAMES, shape (N, 3), in RADIANS;
         - landmarks : the columns from coordinate_columns() (x_0, y_0, x_1, ...),
           reshaped to (N, NUM_LANDMARKS, 2). The column order is interleaved
           x/y, so a plain reshape gives the (x, y) pair of every landmark.

    4. Convert the targets to DEGREES (np.degrees).

    5. Remove extreme samples: keep a sample only if ALL THREE absolute angles
       are <= ANGLE_LIMIT_DEG. (The raw data contains a few impossible values.)
       Fail if fewer than MIN_TRAIN_SAMPLES samples remain.

    6. Build the feature matrix X of shape (N_kept, 956): use a loop over the
       kept samples and call normalize_landmarks(...) on each (478, 2) array.
       If it raises ValueError, skip that sample and count it.
       Why this step? (position and face size would otherwise dominate)

    7. Split X and the (N_kept, 3) targets into training and validation parts
       with train_test_split(..., test_size=VALIDATION_FRACTION,
       random_state=RANDOM_SEED).

    8. For EACH angle (pitch, yaw, roll) create a NEW estimator for model_type:
         "linear" -> LinearRegression()
         "ridge"  -> Ridge(alpha=RIDGE_ALPHA)
         "lasso"  -> Lasso(alpha=LASSO_ALPHA, max_iter=LASSO_MAX_ITER)
       and fit it on the training part with that angle as the target.
       Keep the three fitted estimators in a dict {"pitch": ..., "yaw": ..., "roll": ...}.

    9. Evaluate: predict the validation part and compute the mean absolute
       error in degrees for each angle (np.mean(np.abs(prediction - truth))).

    10. Save with save_model_bundle(model_dir, model_type, models, metadata)
        where metadata is a dict such as
        {"num_samples": ..., "validation_mae_deg": {"pitch": ..., "yaw": ..., "roll": ...}}.
        It raises ModelStoreError on failure.

    11. Return (True, text) where text mentions the number of samples used,
        any rows/samples that were skipped, and the three validation errors.
    """
    # TODO: Delete this placeholder and implement the steps above.
    return False, (
        "train_models() is not implemented yet. Open regression.py and complete "
        "the TODO in train_models()."
    )


def predict_angles(
    landmarks: np.ndarray,
    model_dir: str,
) -> tuple[float, float, float] | None:
    """
    TODO: Predict (pitch, yaw, roll) in DEGREES from one face's landmarks.

    IMPORTANT:
    - Keep this function signature unchanged.
    - It is called for every video frame: do not read files yourself and do not
      print. load_model_bundle() already caches the models in memory.
    - Return None (never raise) if the input is invalid or no usable models exist.

    Arguments:
      landmarks : np.ndarray, shape (478, 2), pixel (x, y) coordinates of ONE face,
                  in the same convention as the dataset.
      model_dir : directory that contains the saved models.

    Expected implementation sequence:

    1. Validate `landmarks`: it must be a numpy array of shape (NUM_LANDMARKS, 2)
       with only finite values. Otherwise return None.

    2. Get the trained models: bundle = load_model_bundle(model_dir).
       It raises ModelStoreError when there are no (valid) models -> return None.
       The estimators are in bundle.models["pitch" / "yaw" / "roll"].

    3. Preprocess EXACTLY like in training: features = normalize_landmarks(landmarks)
       (catch ValueError -> return None). Reshape to (1, 956) because
       scikit-learn expects a 2-D array of samples.

    4. For each name in ANGLE_NAMES call .predict(features) on that angle's
       estimator and take the single value as a float.
       Return None if any value is not finite.

    5. Return (pitch, yaw, roll) as three Python floats in DEGREES, in this order.
    """
    # TODO: Delete this placeholder and implement the steps above.
    return None
