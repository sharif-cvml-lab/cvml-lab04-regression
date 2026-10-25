# Lab 04 - Regression: Head-Controlled Virtual Pointer

Welcome to the fourth assignment of the Computer Vision and Machine Learning lab course!

This project is a real-time PyQt6 application in which your head moves a **virtual pointer**. MediaPipe finds 478 facial landmarks in every webcam frame; three regression models you train (one each for **pitch, yaw and roll**) turn those landmarks into head-pose angles; the angles move the pointer. Everything around the models is provided. Your job is the regression part.

## Assignment objectives

Complete two functions in `regression.py` (the only file you edit):

```python
def train_models(dataset_path: str, model_type: str, model_dir: str, alpha: float = 1.0) -> dict
def predict_angles(landmarks: np.ndarray, model_dir: str) -> tuple[float, float, float] | None
```

You will practice: loading a labeled dataset, separating features from targets, normalizing landmarks, converting units (radians to degrees), cleaning invalid samples, training Linear / Ridge / Lasso regression with scikit-learn (`.fit()`), saving models, and predicting on new data with *exactly* the same preprocessing.

Do not modify any other file. The detailed step-by-step TODOs are inside `regression.py`.

### Pipeline

```
Dataset CSV --> landmark columns (478 x 2) --> landmarks_to_feature_vector() (fixed) --> X (n, 956)
                angle columns (radians) --> np.degrees --> drop invalid --> Y (n, 3)
train_models()   YOUR CODE: for pitch, yaw, roll: model = Ridge/Lasso/Linear; model.fit(X, y); joblib.dump(...)

Webcam frame --> MediaPipe landmarks (fixed) --> predict_angles()   YOUR CODE: features -> models -> (pitch, yaw, roll)
             --> smoothing + centre calibration + mapping (fixed) --> virtual pointer
```

## Features

- Webcam view (mirror option, landmark overlay, face status, FPS) and a **Virtual Pointer** arena with trail and roll-rotated pointer.
- Choose **Linear**, **Ridge** or **Lasso** regression and `alpha`; training runs in a background thread with a busy indicator.
- Models are written to a staging folder and installed only if complete, so a failed training never destroys previously valid models.
- Set Center / Clear Center, Invert X / Y, Yaw and Pitch range, Smoothing, trail on/off.
- **Test on Image...** runs the pipeline on one image (a single image cannot show continuous pointer motion).
- Dark and light theme, resizable layout, settings restored between sessions, full-screen mode.
- Runs before you write any code and without a camera (the app then shows a clear message).

## Setup

Python 3.9 - 3.12 is recommended (MediaPipe wheels). Tested here with Python 3.12.

```
python -m venv venv
```

Linux / macOS: `source venv/bin/activate` - Windows PowerShell: `venv\Scripts\Activate.ps1`

```
pip install -r requirements.txt
python main.py
```

`assets/models/face_landmarker.task` and `assets/dataset/AFLW2000_landmarks.csv` are included locally; nothing is downloaded at runtime.

## Using the application

1. **Train:** the dataset (default AFLW2000) is pre-selected. Pick an algorithm and `alpha`, press **Train**. The status shows the active models and, if available, held-out MAE in degrees (error on unseen dataset samples - not a guarantee of pointer quality). Defaults: Ridge alpha 1, Lasso alpha 0.01 (Lasso is slower). Linear ignores alpha.
2. **Look at the pointer:** turn your head right and the pointer moves right; look up and it moves up; tilting your head rotates the pointer.
3. **Calibrate:** sit comfortably, look at the screen, press **Set Center**. It refuses to save when no valid pose is available. **Clear Center** returns to 0/0/0 degrees. The center is not persisted between sessions.
4. **Tune:** *Yaw Range* / *Pitch Range* = head angle that reaches the arena edge (smaller = more sensitive). *Smoothing* higher = steadier but slower (exponential moving average). Pointer movement stays inside the arena.
5. **Remove Trained Models** (with confirmation) deletes only the three model files and `metadata.json` from `assets/models/trained/`. **Reset to Defaults** resets only GUI/pointer settings.

If you change the algorithm, alpha or dataset after training, the status line tells you that the active models differ and that you must press Train again; predictions always come from the active models.

Shortcuts: `Space` pause/resume - `F11` full screen - `R` reset defaults - `Q` quit.

## Conventions (important)

| Item | Convention |
|---|---|
| Landmarks | 478 points, `(478, 2)` pixel coordinates; CSV columns interleaved `x_0, y_0, x_1, ...` |
| Features | `face_landmarker.landmarks_to_feature_vector()` - subtract landmark 99, divide by distance between landmarks 10 and 171, layout `[x..., y...]` (956 values). Use it in training **and** inference. |
| Angle units | CSV: **radians**. Models and GUI: **degrees**. Convert once with `np.degrees` in `train_models()`. |
| Output order | `(pitch, yaw, roll)` |
| Invalid labels | Samples with any `|angle| > 75 deg` (23 wrapped values in the supplied file) or non-finite values are skipped |
| Directions | positive yaw = head turned to the user's right, positive pitch = looking up, positive roll = tilt toward the user's left shoulder |
| Return values | `train_models` -> `{"success": bool, "message": str, "n_samples": int}`; `predict_angles` -> 3 floats or `None` (never invented numbers) |

Until you finish the TODOs, `train_models` reports "not implemented" and `predict_angles` returns `None`; the app shows "Models not trained yet" instead of failing.

## Dataset

See `assets/dataset/README.txt`. To build your own dataset from labeled images see `example_b_prepare_dataset.py`. Only `.csv` files are supported.

## Standalone examples (`assets/examples/`)

```
python assets/examples/example_a_extract_landmarks.py [image]     # landmarks -> annotated image
python assets/examples/example_b_prepare_dataset.py                # inspect the dataset (or: build IMAGES LABELS OUT)
python assets/examples/example_c_train_models.py Ridge 1.0         # train + save (Linear | Ridge | Lasso)
python assets/examples/example_d_predict_single_image.py [image]   # predict angles for one image
```
Outputs go to `assets/examples/output/`. Examples C and D are complete reference pipelines (instructors may remove them from the student copy; `instructor/` holds a full `regression.py` solution and must not be distributed).

## Repository structure

```
cvml-lab04-regression/
├── main.py              entry point
├── gui.py               window, themes, controls, settings, shortcuts
├── video_thread.py      camera/test image, landmarks, inference, FPS
├── training_thread.py   runs train_models() in the background
├── regression.py        STUDENT FILE
├── face_landmarker.py   MediaPipe wrapper + shared feature preprocessing
├── model_store.py       model paths, metadata, cached loading, safe install/delete
├── pointer.py           smoothing, calibration, angle -> pointer mapping
├── config.py            constants and defaults
├── requirements.txt
├── instructor/regression_solution.py
├── tests/test_core.py
└── assets/{dataset,models,examples}/
```

Run the tests: `python -m unittest discover -s tests -v` (they use the instructor solution; no camera or display needed).

## Limitations

Only the 2D landmark geometry is used, so estimates are approximate (typically a few degrees of MAE on held-out AFLW2000 samples) and weaker for large angles or partly hidden faces. Smoothing is per frame, not per second, so its feel depends on FPS. Models are pickled by joblib: load only model files you trust.
