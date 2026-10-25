"""Fixed infrastructure: camera / test-image acquisition, landmark extraction and inference.

One QThread owns the camera, the MediaPipe extractor and all per-frame work. It talks to the GUI
only through signals. The GUI acknowledges every frame (acknowledge()), so at most one frame is
ever waiting in the signal queue.
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

import config
import model_store
import regression
from config import FrameResult
from face_landmarker import FaceLandmarkExtractor, annotate_face_landmarks


def to_qimage(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage, object)     # annotated frame, FrameResult
    status_changed = pyqtSignal(str)             # camera / source status text
    error_occurred = pyqtSignal(str)             # non-fatal problem (deduplicated)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mirror = True
        self.show_landmarks = True
        self._stop_requested = False
        self._paused = False
        self._lock = threading.Lock()
        self._source = ("camera", config.DEFAULT_CAMERA_INDEX)
        self._source_dirty = True
        self._frame_pending = False
        self._last_error = ""
        # touched only by the worker thread:
        self._capture = None
        self._image = None
        self._extractor = None
        self._read_failures = 0
        self._times = deque(maxlen=30)

    # ---- called from the GUI thread --------------------------------------
    def set_camera(self, index):
        with self._lock:
            self._source = ("camera", int(index))
            self._source_dirty = True

    def set_image(self, path):
        with self._lock:
            self._source = ("image", str(path))
            self._source_dirty = True

    def set_paused(self, paused):
        self._paused = bool(paused)
        if not paused:
            with self._lock:
                self._read_failures = 0

    def acknowledge(self):
        self._frame_pending = False

    def stop(self):
        self._stop_requested = True

    # ---- worker thread ---------------------------------------------------
    def run(self):
        try:
            self._extractor = FaceLandmarkExtractor(config.LANDMARKER_PATH)
        except Exception as exc:
            self._report(f"Face landmarker could not be loaded from {config.LANDMARKER_PATH}: {exc}")
        try:
            while not self._stop_requested:
                try:
                    self._tick()
                except Exception as exc:               # never let one bad frame kill the thread
                    traceback.print_exc()
                    self._report(f"Frame processing error: {type(exc).__name__}: {exc}")
                    self.msleep(100)
        finally:
            self._release_camera()
            if self._extractor is not None:
                self._extractor.close()

    def _tick(self):
        with self._lock:
            dirty, source = self._source_dirty, self._source
            self._source_dirty = False
        if dirty:
            self._apply_source(source)

        if self._image is not None:
            if not self._paused:
                self._process_and_emit(self._image, source="image", mirror=False)
            self.msleep(config.IMAGE_REFRESH_MS)
            return
        if self._capture is None or self._paused:
            self.msleep(40)
            return

        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._read_failures += 1
            if self._read_failures >= config.CAMERA_MAX_READ_FAILURES:
                self._release_camera()
                self.status_changed.emit("Camera disconnected. Press 'Reconnect Camera' to try again.")
            self.msleep(15)
            return
        self._read_failures = 0
        self._process_and_emit(frame, source="camera", mirror=self.mirror)

    # ---- sources ---------------------------------------------------------
    def _apply_source(self, source):
        self._release_camera()
        self._image = None
        self._times.clear()
        kind, value = source
        if kind == "image":
            image = self._load_image(value)
            if image is None:
                self.status_changed.emit(f"Could not read image: {value}")
            else:
                self._image = image
                self.status_changed.emit(f"Test image: {value} (a single image cannot show continuous pointer motion)")
            return
        self._capture = self._open_camera(value)
        if self._capture is None:
            self.status_changed.emit(f"Camera {value} could not be opened. Check the camera index or connection.")
        else:
            self.status_changed.emit(f"Camera {value} running.")

    def _open_camera(self, index):
        try:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                capture.release()
                return None
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._read_failures = 0
            return capture
        except Exception as exc:
            self._report(f"Camera error: {exc}")
            return None

    def _release_camera(self):
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    @staticmethod
    def _load_image(path):
        try:
            data = np.fromfile(path, dtype=np.uint8)           # also works for non-ASCII paths
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None
        if image is None:
            return None
        longest = max(image.shape[:2])
        if longest > config.MAX_PROCESS_DIMENSION:
            scale = config.MAX_PROCESS_DIMENSION / longest
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return image

    # ---- per-frame work --------------------------------------------------
    def _process_and_emit(self, frame, source, mirror):
        if self._frame_pending:
            return                                           # GUI is still busy with the last frame
        started = time.perf_counter()

        landmarks = None
        if self._extractor is not None:
            landmarks = self._extractor.extract_landmarks(frame)

        shown = frame
        if landmarks is not None and self.show_landmarks:
            shown = annotate_face_landmarks(frame, landmarks, radius=1)
        if mirror:
            shown = cv2.flip(shown, 1)

        angles, status = None, "Face landmarker unavailable" if self._extractor is None else "No face detected"
        if landmarks is not None:
            angles, status = self._predict(landmarks)

        now = time.perf_counter()
        fps = 0.0
        if source == "camera":
            self._times.append(now)
            if len(self._times) >= 2 and self._times[-1] > self._times[0]:
                fps = (len(self._times) - 1) / (self._times[-1] - self._times[0])
        result = FrameResult(
            face_found=landmarks is not None, angles=angles, status=status, fps=fps,
            processing_ms=(now - started) * 1000.0, source=source,
        )
        self._frame_pending = True
        self.frame_ready.emit(to_qimage(shown), result)

    def _predict(self, landmarks):
        """Call the student's predict_angles() safely. Returns (angles or None, status text)."""
        try:
            output = regression.predict_angles(landmarks, str(config.TRAINED_MODELS_DIR))
        except Exception as exc:
            traceback.print_exc()
            self._report(f"predict_angles() raised {type(exc).__name__}: {exc}")
            return None, "predict_angles() raised an error (see terminal)"
        if output is None:
            if model_store.load_models(config.TRAINED_MODELS_DIR) is None:
                return None, "Models not trained yet"
            return None, "predict_angles() returned None (not implemented or invalid input)"
        try:
            pitch, yaw, roll = (float(value) for value in output)
        except (TypeError, ValueError):
            self._report("predict_angles() must return three numbers (pitch, yaw, roll) or None.")
            return None, "predict_angles() returned an invalid value"
        if not np.all(np.isfinite([pitch, yaw, roll])):
            return None, "predict_angles() returned a non-finite value"
        self._last_error = ""
        return (pitch, yaw, roll), "Pose estimated"

    def _report(self, message):
        if message != self._last_error:                       # no repeated identical messages
            self._last_error = message
            self.error_occurred.emit(message)
