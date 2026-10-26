# Dataset format

The application trains from a **CSV** file (UTF-8, comma separated, one sample per row). The supplied
`AFLW2000_landmarks.csv` has 1755 rows. Excel files are not supported - export to CSV.

## Columns

| Column(s) | Meaning |
|---|---|
| `filename` | Image name (informational only, may be omitted) |
| `pitch`, `yaw`, `roll` | Head angles in **radians** (verified on the supplied file) |
| `x_0`, `y_0`, `x_1`, `y_1`, ... `x_477`, `y_477` | 478 landmarks, **interleaved** x/y |

Column names matter, not their position, but the landmark columns must be exactly `x_0 ... x_477` and
`y_0 ... y_477` (478 landmarks - a different count is rejected). Extra columns are ignored.

## Angle convention (AFLW2000)

* **pitch** > 0: head looks up. **yaw** > 0: the person turns to *their own* right (nose moves to the image's left).
  **roll** > 0: the image-right eye is lower (head tilts to the person's left shoulder).
* The CSV stores radians; the application converts to **degrees** for training and prediction.
* Samples with any |angle| > 75° are discarded during training (the raw file contains a few impossible values).

## Landmark convention

* MediaPipe Face Landmarker order (the supplied `assets/models/face_landmarker.task`).
* Pixel coordinates of the original (never mirrored) image, origin top-left, x right, y down,
  truncated to integers. Values slightly outside the image are normal.
* Preprocessing (`landmarks.normalize_landmarks`): subtract landmark 99, divide by the distance between
  landmarks 10 and 171, layout `[x_0..x_477, y_0..y_477]`. **Training and inference must use the same
  function** - never preprocess differently in one of them.

## Bad rows

Rows with non-numeric, NaN or infinite values are skipped and reported. If more than half of the rows are
invalid, or fewer than 50 usable rows remain, training is rejected with an error message.

## Making your own dataset

Collect images with known pitch/yaw/roll (radians), put them in a labels CSV (`filename,pitch,yaw,roll`) and run
`python assets/examples/02_build_dataset.py --images DIR --labels labels.csv --output my_dataset.csv`.
Then choose `my_dataset.csv` with *Browse…* in the application.
