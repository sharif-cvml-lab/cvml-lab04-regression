DATASET FORMAT  (assets/dataset/AFLW2000_landmarks.csv)

One row per labeled face image. Columns, in this order:

  filename, pitch, yaw, roll, x_0, y_0, x_1, y_1, ..., x_477, y_477

filename   Source image name (information only, not used for training).
pitch      Head pitch (nod up/down), yaw (turn left/right), roll (tilt sideways).
yaw        ANGLES ARE IN RADIANS in the supplied file. The application converts them to
roll       degrees once, in train_models(), and everything else uses degrees.
x_i, y_i   Pixel coordinates of MediaPipe Face Landmarker landmark i, i = 0..477 (478 landmarks).
           The columns are INTERLEAVED (x_0, y_0, x_1, y_1, ...). Read them in that order.

Notes
- The supplied file has 1755 rows. 23 of them contain wrapped/invalid angles (|angle| > 75 deg)
  and are skipped during training, leaving 1732 samples.
- Landmarks must come from the same extractor used by the app (face_landmarker.py with
  assets/models/face_landmarker.task). Another detector or landmark order is NOT compatible.
- To use your own data: create a CSV in this format (see assets/examples/example_b_prepare_dataset.py,
  "build" mode), press "Browse..." in the Training box, select the file, then press "Train".
  The app checks the header and tells you which columns are missing; invalid rows are skipped
  during training. Only .csv files are supported.
