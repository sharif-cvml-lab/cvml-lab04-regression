"""Fixed infrastructure: runs the student's train_models() off the GUI thread."""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from config import MODEL_TYPES, MODELS_DIR
from model_store import clear_cache, inspect_models

try:
    from regression import train_models
    _IMPORT_ERROR = None
except Exception as exc:  # a syntax error in regression.py must not crash the app
    train_models = None
    _IMPORT_ERROR = f"regression.py could not be imported: {type(exc).__name__}: {exc}"
    traceback.print_exc()


class TrainingWorker(QThread):
    training_finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, dataset_path: str, model_type: str, parent=None):
        super().__init__(parent)
        self._dataset_path = dataset_path
        self._model_type = model_type

    def run(self) -> None:
        if train_models is None:
            self.training_finished.emit(False, _IMPORT_ERROR or "train_models is unavailable")
            return
        try:
            result = train_models(self._dataset_path, self._model_type, str(MODELS_DIR))
            if (not isinstance(result, tuple) or len(result) != 2
                    or not isinstance(result[0], (bool,)) or not isinstance(result[1], str)):
                self.training_finished.emit(
                    False, "train_models() must return a (bool, str) tuple.")
                return
            success, message = result
        except Exception as exc:  # student bugs must not kill the application
            traceback.print_exc()
            self.training_finished.emit(False, f"train_models() raised {type(exc).__name__}: {exc}")
            return
        if success:
            clear_cache()
            status = inspect_models(MODELS_DIR)
            if not status.valid:
                success, message = False, (
                    f"train_models() reported success but no valid model file was found ({status.message}). "
                    "Did you call save_model_bundle()?")
            elif status.model_type != self._model_type:
                success, message = False, (
                    f"Saved models are of type '{status.model_type}' but '{self._model_type}' was requested.")
        self.training_finished.emit(success, message)


assert MODEL_TYPES  # imported for clarity of the contract
