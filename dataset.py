"""Fixed infrastructure: safe loading and validation of the landmark dataset CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import ANGLE_NAMES, MIN_TRAIN_SAMPLES, NUM_LANDMARKS

MAX_INVALID_FRACTION = 0.5


class DatasetError(ValueError):
    """The dataset file is unusable. The message is meant to be shown to the user."""


def coordinate_columns() -> list[str]:
    """['x_0', 'y_0', 'x_1', 'y_1', ..., 'x_477', 'y_477'] - the CSV column order."""
    return [f"{axis}_{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y")]


def load_dataset_table(path) -> tuple[pd.DataFrame, str]:
    """Load and validate the dataset.

    Returns (table, report). The table has float64 columns in the order
    pitch, yaw, roll, x_0, y_0, ..., x_477, y_477 and contains only rows whose
    values are all finite. Angles are still in the units of the file (radians).
    Raises DatasetError with an actionable message for any fatal problem.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")
    if not path.is_file():
        raise DatasetError(f"Dataset path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise DatasetError(f"Only CSV datasets are supported (got '{path.suffix}').")
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise DatasetError("The dataset file is empty.") from None
    except pd.errors.ParserError as exc:
        raise DatasetError(f"The CSV has inconsistent rows and cannot be parsed: {exc}") from None
    except (UnicodeDecodeError, OSError) as exc:
        raise DatasetError(f"The dataset file cannot be read: {exc}") from None

    raw.columns = [str(c).strip() for c in raw.columns]
    x_columns = {c for c in raw.columns if c.startswith("x_")}
    if len(x_columns) > NUM_LANDMARKS:
        raise DatasetError(
            f"The dataset has {len(x_columns)} landmarks per row; this lab expects {NUM_LANDMARKS}."
        )
    wanted = list(ANGLE_NAMES) + coordinate_columns()
    missing = [c for c in wanted if c not in raw.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise DatasetError(
            f"The dataset is missing {len(missing)} required column(s): {shown}. "
            f"Expected pitch, yaw, roll and x_0, y_0 ... x_{NUM_LANDMARKS - 1}, y_{NUM_LANDMARKS - 1} "
            "(see assets/dataset/README.md)."
        )
    if len(raw) == 0:
        raise DatasetError("The dataset contains no rows.")

    numeric = raw[wanted].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    skipped = int((~valid).sum())
    if skipped > MAX_INVALID_FRACTION * len(raw):
        raise DatasetError(
            f"{skipped} of {len(raw)} rows contain non-numeric, NaN or infinite values. "
            "The file is probably malformed (wrong columns or delimiter?)."
        )
    table = numeric[valid].reset_index(drop=True)
    if len(table) < MIN_TRAIN_SAMPLES:
        raise DatasetError(
            f"Only {len(table)} usable rows; at least {MIN_TRAIN_SAMPLES} are required."
        )
    report = f"Loaded {len(table)} of {len(raw)} rows from {path.name}"
    if skipped:
        report += f" ({skipped} rows skipped: non-numeric, NaN or infinite values)"
    return table, report
