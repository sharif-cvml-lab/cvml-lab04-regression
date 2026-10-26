"""Fixed infrastructure: safe saving / loading / deleting of the trained models.

All three regressors are stored in ONE joblib file (assets/models/
head_pose_regressors.joblib) together with the metadata needed to verify that
inference uses the same conventions as training. The file is written to a
temporary name, re-loaded and validated, and only then atomically moved into
place, so a failed training run never damages a previous valid bundle.

Note: joblib files are pickles - only load files you created yourself.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from config import (
    ANGLE_NAMES,
    FEATURE_SPEC,
    MODEL_ANGLE_UNIT,
    MODEL_BUNDLE_FILENAME,
    MODEL_TYPES,
    NUM_FEATURES,
    NUM_LANDMARKS,
)

FORMAT_VERSION = 1
_TMP_PREFIX = "." + Path(MODEL_BUNDLE_FILENAME).stem + ".tmp-"


class ModelStoreError(RuntimeError):
    """Saving or loading the model bundle failed (message is user-readable)."""


@dataclass(frozen=True)
class ModelBundle:
    model_type: str
    models: dict  # {"pitch": est, "yaw": est, "roll": est}
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelStatus:
    valid: bool
    message: str
    model_type: str | None = None
    metadata: dict = field(default_factory=dict)


_cache_lock = threading.Lock()
_cache: dict = {"key": None, "bundle": None}


def bundle_path(model_dir) -> Path:
    return Path(model_dir) / MODEL_BUNDLE_FILENAME


def _validate(raw) -> ModelBundle:
    if not isinstance(raw, dict):
        raise ModelStoreError("model file has an unexpected structure")
    if raw.get("format_version") != FORMAT_VERSION:
        raise ModelStoreError("model file was written by an incompatible version")
    if raw.get("model_type") not in MODEL_TYPES:
        raise ModelStoreError("model file has an unknown regression type")
    if tuple(raw.get("angle_names", ())) != ANGLE_NAMES or raw.get("angle_unit") != MODEL_ANGLE_UNIT:
        raise ModelStoreError("model file uses different angle names or units")
    if raw.get("num_landmarks") != NUM_LANDMARKS or raw.get("feature_spec") != FEATURE_SPEC:
        raise ModelStoreError("model file was trained with different preprocessing")
    models = raw.get("models")
    if not isinstance(models, dict) or set(models) != set(ANGLE_NAMES):
        raise ModelStoreError("model file does not contain pitch, yaw and roll models")
    probe = np.zeros((1, NUM_FEATURES), dtype=np.float64)
    for name in ANGLE_NAMES:
        estimator = models[name]
        if not hasattr(estimator, "predict"):
            raise ModelStoreError(f"the {name} model is not a trained estimator")
        try:
            output = np.asarray(estimator.predict(probe), dtype=np.float64)
        except Exception as exc:
            raise ModelStoreError(f"the {name} model cannot predict ({exc})") from None
        if output.shape != (1,) or not np.all(np.isfinite(output)):
            raise ModelStoreError(f"the {name} model returned an invalid output")
    return ModelBundle(raw["model_type"], models, dict(raw.get("metadata", {})))


def _read(path: Path) -> ModelBundle:
    try:
        raw = joblib.load(path)
    except Exception as exc:
        raise ModelStoreError(f"model file is corrupted or unreadable ({type(exc).__name__})") from None
    return _validate(raw)


def save_model_bundle(model_dir, model_type: str, models: dict, metadata: dict | None = None) -> Path:
    """Validate, write (atomically) and re-load a bundle. Raises ModelStoreError."""
    directory = Path(model_dir)
    bundle = {
        "format_version": FORMAT_VERSION,
        "model_type": model_type,
        "angle_names": list(ANGLE_NAMES),
        "angle_unit": MODEL_ANGLE_UNIT,
        "num_landmarks": NUM_LANDMARKS,
        "feature_spec": FEATURE_SPEC,
        "models": models,
        "metadata": {**(metadata or {}), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    try:
        _validate(bundle)  # reject bad input before touching the disk
        directory.mkdir(parents=True, exist_ok=True)
    except ModelStoreError:
        raise
    except OSError as exc:
        raise ModelStoreError(f"cannot create the model directory: {exc}") from None
    final = bundle_path(directory)
    temporary = directory / f"{_TMP_PREFIX}{os.getpid()}-{time.time_ns()}.joblib"
    try:
        joblib.dump(bundle, temporary, compress=3)
        _read(temporary)  # verify that the written file can be loaded again
        os.replace(temporary, final)
    except ModelStoreError as exc:
        raise ModelStoreError(f"saved file failed verification: {exc}") from None
    except Exception as exc:
        raise ModelStoreError(f"could not write the model file: {exc}") from None
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    clear_cache()
    return final


def load_model_bundle(model_dir) -> ModelBundle:
    """Load (with caching) the bundle. The file is re-read only if it changed."""
    path = bundle_path(model_dir)
    try:
        info = path.stat()
    except OSError:
        clear_cache()
        raise ModelStoreError("no trained models found") from None
    key = (str(path), info.st_mtime_ns, info.st_size)
    with _cache_lock:
        if _cache["key"] == key and _cache["bundle"] is not None:
            return _cache["bundle"]
    bundle = _read(path)
    with _cache_lock:
        _cache["key"], _cache["bundle"] = key, bundle
    return bundle


def clear_cache() -> None:
    with _cache_lock:
        _cache["key"], _cache["bundle"] = None, None


def inspect_models(model_dir) -> ModelStatus:
    """Never raises: describes whether a usable bundle exists."""
    try:
        bundle = load_model_bundle(model_dir)
    except ModelStoreError as exc:
        return ModelStatus(False, str(exc))
    return ModelStatus(True, "ok", bundle.model_type, bundle.metadata)


def delete_model_bundle(model_dir) -> list[str]:
    """Delete ONLY the application's bundle file and its leftover temporary files."""
    directory = Path(model_dir)
    removed: list[str] = []
    if not directory.is_dir():
        return removed
    candidates = [directory / MODEL_BUNDLE_FILENAME]
    candidates += [p for p in directory.iterdir() if p.name.startswith(_TMP_PREFIX) and p.suffix == ".joblib"]
    for path in candidates:
        if path.parent == directory and path.is_file():
            try:
                path.unlink()
                removed.append(path.name)
            except OSError as exc:
                raise ModelStoreError(f"could not delete {path.name}: {exc}") from None
    clear_cache()
    return removed
