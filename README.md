# Lab04_Regression

Welcome to the fourth assignment of the Computer Vision and Machine Learning lab course!

This project is a real-time PyQt6 application in which your **head controls a virtual pointer**. A fixed pipeline detects 478 facial landmarks; **regression models that you train** turn the landmarks into head angles (**pitch, yaw, roll**); the angles move the pointer. Your job is the machine-learning part: *training* the regressors and *predicting* with them.

## 🎯 Assignment Objectives

Complete two functions in `regression.py` (the **only** file you modify):

```python
def train_models(dataset_path: str, model_type: str, model_dir: str) -> tuple[bool, str]
def predict_angles(landmarks: np.ndarray, model_dir: str) -> tuple[float, float, float] | None
```

You will practice: separating features from targets, cleaning data, feature normalization, Linear / Ridge / Lasso regression with scikit-learn, a train/validation split and MAE, saving a model, and using exactly the same preprocessing at training and prediction time.

Do not modify `main.py`, `gui.py`, `video_thread.py`, `training_worker.py`, `annotation.py`, `landmarks.py`, `dataset.py`, `model_store.py`, `pointer.py`, `config.py`, `requirements.txt`, or the assets. You must be able to explain every line of your two functions in the oral assessment.

### Pipeline

```
camera / image / video ──► MediaPipe landmarks (fixed) ──► (478, 2) pixel points
                                                              │
        training:  dataset CSV ──► YOUR train_models()        │
                      load → split features/targets → degrees → clean → normalize_landmarks()
                      → split train/validation → fit pitch/yaw/roll regressors → MAE → save
                                                              ▼
        prediction:                                  YOUR predict_angles()
                      validate → load models (cached) → normalize_landmarks() → predict
                                                              │  (pitch, yaw, roll) in degrees
                                                              ▼
              fixed: calibration → smoothing → virtual pointer + camera overlay
```

The detailed step-by-step instructions are in the docstrings (TODO) of `regression.py`.

### Conventions (verified on the supplied dataset)

| Item | Convention |
|---|---|
| Landmarks | 478 points, shape `(478, 2)`, `(x, y)` pixels of the original un-mirrored image, truncated to integers |
| Features | `normalize_landmarks(points)` → 956 values `[x_0..x_477, y_0..y_477]`, centered on landmark 99, divided by the distance of landmarks 10 and 171 |
| Dataset angles | pitch, yaw, roll in **radians** |
| Model targets / predictions | **degrees**, order always `(pitch, yaw, roll)` |
| Signs | yaw > 0 = user turns to *their* right · pitch > 0 = looks up · roll > 0 = tilts to *their* left shoulder |
| Models | one `assets/models/head_pose_regressors.joblib` (3 estimators + metadata), written atomically |

Before you implement anything the application already runs: `train_models` reports "not implemented" and `predict_angles` returns `None`; the GUI shows this clearly and nothing crashes.

## 🛠️ Setup & Installation

Python 3.10 – 3.12 is recommended (the versions MediaPipe supports). Windows and Linux, CPU only.

```
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -r requirements.txt
python main.py
```

Required assets (already in the repository):

- `assets/models/face_landmarker.task` – MediaPipe landmark model (never deleted by the app)
- `assets/dataset/AFLW2000_landmarks.csv` – training data (format: `assets/dataset/README.md`)

MediaPipe installs `opencv-contrib-python`; `main.py` removes the Qt plugin path that this package sets on Linux so that it does not clash with PyQt6. If you still get a `xcb` plugin error on Linux, install `libxcb-cursor0`.

## 🎮 Using the Application

### Input source

- **Webcam** (default). If it cannot be opened the app keeps running ("Camera could not be opened") – change the camera index or load a file instead.
- **Open Image/Video…** loads a still image or a video file. A still image is analysed once (and again when you change a setting); a video plays with Pause/Resume and Restart Source.
- All three sources use the same processing and display. Switching sources resets the pointer and clears the center.

### Train a model

1. Choose a dataset with **Browse…** (default: the AFLW2000 CSV; CSV only).
2. Pick **Linear / Ridge / Lasso Regression**.
3. Press **Train**. Training runs in the background (a few seconds; Lasso is the slowest). Training never starts automatically.
4. On success the models are saved in `assets/models/` and used immediately; they are loaded automatically at the next launch. A failed training keeps the previous valid models.
5. **Delete Models…** (with confirmation) removes only the trained-model file, never `face_landmarker.task` or the dataset.

The status line under *Regression Model* shows whether usable models exist and their validation error (MAE in degrees for pitch / yaw / roll).

### Pointer

- **Set Center (`C`)**: current head pose = neutral. Refused when there is no valid face/prediction. **Clear Center** goes back to 0° (the dataset's frontal pose). The center is not saved between sessions.
- **Yaw / Pitch range**: head angle from the center that moves the pointer to the edge of the area.
- **Smoothing**: time constant of an exponential filter in ms (0 = off). Larger = steadier but laggier; it is independent of the frame rate.
- **Invert X / Y**, **Show pointer trail**, **Reset Pointer**. Roll rotates the pointer arrow.
- No face / invalid prediction: the pointer is **held** for 0.75 s, then shown frozen and dimmed ("No face - pointer frozen"). Old predictions are never reused, and when the face returns the pointer jumps instead of gliding from a stale position.

### Display

Mirror camera view (display only – the model always receives the original frame, otherwise yaw and roll would change sign), show landmarks, show angles. Dark/light theme, full screen, **Reset Defaults** (keeps trained models), settings are saved between sessions.

### Keyboard shortcuts

- `Space` – Pause / Resume (webcam and video)
- `C` – Set Center
- `F11` – Full Screen / Windowed
- `R` – Reset Defaults
- `Q` – Quit (if training is running you are asked to wait for it to finish)

## 📂 Repository Structure

```
cvml-lab04-regression/
├── main.py              # entry point
├── gui.py               # PyQt6 UI, themes, shortcuts, settings
├── video_thread.py      # webcam/image/video, landmarks, calls predict_angles, pointer
├── training_worker.py   # runs train_models in a background thread
├── regression.py        # ★ THE ONLY FILE STUDENTS MODIFY
├── landmarks.py         # MediaPipe wrapper + normalize_landmarks (shared preprocessing)
├── dataset.py           # safe CSV loading / validation
├── model_store.py       # atomic save, cached load, safe delete of the models
├── pointer.py           # angles → pointer (calibration, smoothing, hold)
├── annotation.py        # drawing on frames
├── config.py            # paths, constants, defaults
├── requirements.txt
├── tests/               # pytest tests of the fixed infrastructure
└── assets/
    ├── dataset/         # AFLW2000_landmarks.csv + README.md (format guide)
    ├── models/          # face_landmarker.task (+ your trained models)
    └── examples/        # standalone scripts 01–04 + image00013.jpg
```

## 🧪 Standalone Examples

Independent of the GUI; run from anywhere (outputs go to `assets/examples/output/`):

```
python assets/examples/01_detect_landmarks.py      # detect + draw landmarks on image00013.jpg
python assets/examples/02_build_dataset.py         # build a dataset CSV from images + angle labels
python assets/examples/03_train_regressor.py       # walkthrough of training (--model linear|ridge|lasso)
python assets/examples/04_predict_angles.py        # predict the three angles of an image
```

Examples 03–04 show the finished pipeline for reference; write your own `regression.py` rather than copying them – you will be asked to explain it. Use `02_build_dataset.py` (with your own images and labels) to create a custom dataset (see `assets/dataset/README.md`).

## About

Head-pose estimation drives gaze-free interfaces, driver monitoring and AR. Here you see the full chain in miniature: engineered features, a regularized linear model, and a real-time consumer of its predictions. Linear regression on 956 correlated coordinates overfits; compare it with Ridge and Lasso on the validation error and on how steady the pointer feels.
