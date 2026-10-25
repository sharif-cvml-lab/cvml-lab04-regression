"""Fixed infrastructure: runs train_models() off the GUI thread."""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

import model_store
import regression


class TrainingThread(QThread):
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, dataset_path, model_type, alpha, parent=None):
        super().__init__(parent)
        self._dataset_path = str(dataset_path)
        self._model_type = model_type
        self._alpha = float(alpha)

    def run(self):
        try:
            result = model_store.train_and_install(
                regression.train_models, self._dataset_path, self._model_type, self._alpha
            )
        except Exception as exc:                      # bug in student code or a filesystem error
            traceback.print_exc()
            self.failed.emit(f"Training raised {type(exc).__name__}: {exc}")
            return
        if result.get("success"):
            self.succeeded.emit(result)
        else:
            self.failed.emit(str(result.get("message", "Training failed.")))
