from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from annotation import draw_landmarks, draw_overlay, make_placeholder
from config import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DEFAULT_CAMERA_INDEX,
    FRAME_TIMEOUT_SECONDS,
    LANDMARKER_MODEL_PATH,
    MAX_PROCESS_DIMENSION,
    MODELS_DIR,
    TARGET_FPS,
    VIDEO_EXTENSIONS,
    DisplaySettings,
    FrameResult,
    PointerSettings,
)
from landmarks import FaceLandmarkExtractor, LandmarkerError
from model_store import inspect_models
from pointer import PointerController, clean_angles

try:
    from regression import predict_angles
    _REGRESSION_ERROR = None
except Exception as exc:  # a broken regression.py must not crash the app
    predict_angles = None
    _REGRESSION_ERROR = f"regression.py could not be imported: {type(exc).__name__}: {exc}"
    traceback.print_exc()

IDLE_WAIT = 0.03


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def read_image_file(path: str):
    """Read an image (Unicode-safe). Returns a BGR array or None."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def prepare_image(image):
    """Return a contiguous 8-bit BGR image no larger than MAX_PROCESS_DIMENSION."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        return None
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > MAX_PROCESS_DIMENSION:
        scale = MAX_PROCESS_DIMENSION / longest
        image = cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


@dataclass
class _Analysis:
    landmarks: np.ndarray | None
    angles: tuple[float, float, float] | None
    message: str


class VideoThread(QThread):
    """Acquire frames (webcam / image / video), detect landmarks, call the student's
    predict_angles(), update the pointer and render the camera view.

    Frames are processed one at a time in this thread (no queue of stale frames).
    The mirror option only affects the picture that is shown; landmarks are always
    detected on the original, un-mirrored frame."""

    frame_ready = pyqtSignal(object, object)  # BGR ndarray, FrameResult
    source_status = pyqtSignal(str)
    source_error = pyqtSignal(str)  # modal-worthy problems (bad file...)
    notice = pyqtSignal(str)  # short non-modal messages
    source_changed = pyqtSignal(str, str)  # kind, description
    paused_changed = pyqtSignal(bool)
    calibration_changed = pyqtSignal(bool)

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._wake = threading.Event()
        self._in_flight = threading.Event()
        self._in_flight_since = 0.0

        # Shared with the GUI thread (always under the lock).
        self._display = DisplaySettings()
        self._commands: list[tuple] = []
        self._paused = False
        self._dirty = True

        # Owned by this thread.
        self._pointer = PointerController(PointerSettings())
        self._camera_index = camera_index
        self._kind = "webcam"
        self._cap = None
        self._video_ended = False
        self._video_period = 1.0 / TARGET_FPS
        self._scene = None
        self._scene_token = 0
        self._scene_message = "Starting camera..."
        self._last_status = ""
        self._extractor = None
        self._extractor_error = ""
        self._extractor_tried = False
        self._models_epoch = 0
        self._models_available = False
        self._analysis: _Analysis | None = None
        self._analysis_key = None
        self._frame_times: deque[float] = deque(maxlen=10)
        self._reported: set[str] = set()

    # ----- GUI-facing API (thread-safe) ------------------------------------
    def set_pointer_settings(self, settings: PointerSettings) -> None:
        self._post(("pointer_settings", settings))

    def set_display_settings(self, settings: DisplaySettings) -> None:
        with self._lock:
            self._display = settings
            self._dirty = True
        self._wake.set()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = bool(paused)
        self._wake.set()

    def set_camera_index(self, index: int) -> None:
        self._post(("camera_index", int(index)))

    def use_webcam(self) -> None:
        self._post(("webcam",))

    def open_file(self, path: str) -> None:
        self._post(("file", path))

    def restart_source(self) -> None:
        self._post(("restart",))

    def calibrate(self) -> None:
        self._post(("calibrate",))

    def clear_calibration(self) -> None:
        self._post(("clear_calibration",))

    def reset_pointer(self) -> None:
        self._post(("reset_pointer",))

    def models_changed(self) -> None:
        self._post(("models_changed",))

    def frame_consumed(self) -> None:
        """Called by the GUI after it displayed a frame (back-pressure)."""
        self._in_flight.clear()
        self._wake.set()

    def stop(self) -> None:
        self._running.clear()
        self._wake.set()
        if not self.wait(8000):
            print("Warning: the video thread did not stop within 8 s.")

    def _post(self, command: tuple) -> None:
        with self._lock:
            self._commands.append(command)
            self._dirty = True
        self._wake.set()

    # ----- helpers ----------------------------------------------------------
    def _report_once(self, message: str) -> None:
        if message not in self._reported:
            self._reported.add(message)
            print(message)

    def _emit_status(self, message: str) -> None:
        if message != self._last_status:
            self._last_status = message
            self.source_status.emit(message)

    def _set_scene_message(self, message: str) -> None:
        if message != self._scene_message:
            self._scene_message = message
            with self._lock:
                self._dirty = True

    def _set_paused_internal(self, paused: bool) -> None:
        with self._lock:
            changed = self._paused != paused
            self._paused = paused
        if changed:
            self.paused_changed.emit(paused)

    def _release_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _set_scene(self, frame) -> bool:
        prepared = prepare_image(frame)
        if prepared is None:
            return False
        self._scene = prepared
        self._scene_token += 1
        return True

    def _source_switched(self) -> None:
        """Reset state that belongs to the previous source."""
        self._pointer.reset()
        if self._pointer.calibrated:
            self._pointer.clear_calibration()
            self.calibration_changed.emit(False)
        self._analysis = None
        self._frame_times.clear()
        self._video_ended = False

    # ----- commands -----------------------------------------------------------
    def _apply_commands(self) -> None:
        with self._lock:
            commands, self._commands = self._commands, []
        for command in commands:
            try:
                self._run_command(command)
            except Exception as exc:
                self._report_once(traceback.format_exc())
                self.source_error.emit(f"Unexpected error: {exc}")

    def _run_command(self, command: tuple) -> None:
        name = command[0]
        if name == "pointer_settings":
            self._pointer.set_settings(command[1])
        elif name == "webcam":
            self._release_capture()
            self._kind = "webcam"
            self._scene = None
            self._scene_message = "Starting camera..."
            self._source_switched()
            self._set_paused_internal(False)
            self.source_changed.emit("webcam", f"Webcam {self._camera_index}")
        elif name == "camera_index":
            self._camera_index = command[1]
            if self._kind == "webcam":
                self._release_capture()
        elif name == "file":
            self._open_file(command[1])
        elif name == "restart":
            if self._kind == "webcam":
                self._release_capture()
                self._set_scene_message("Restarting camera...")
            elif self._kind == "video" and self._cap is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._video_ended = False
                self._set_paused_internal(False)
                self._emit_status("Video restarted")
        elif name == "calibrate":
            ok, message = self._pointer.calibrate()
            self.notice.emit(message)
            if ok:
                self.calibration_changed.emit(True)
        elif name == "clear_calibration":
            self._pointer.clear_calibration()
            self.calibration_changed.emit(False)
            self.notice.emit("Center cleared (0 degrees = frontal pose).")
        elif name == "reset_pointer":
            self._pointer.reset()
        elif name == "models_changed":
            self._models_epoch += 1
            self._models_available = inspect_models(MODELS_DIR).valid
            self._pointer.reset()

    def _open_file(self, path: str) -> None:
        extension = Path(path).suffix.lower()
        order = ("video", "image") if extension in VIDEO_EXTENSIONS else ("image", "video")
        name = Path(path).name
        for kind in order:
            if kind == "image":
                image = prepare_image(read_image_file(path))
                if image is None:
                    continue
                self._release_capture()
                self._kind = "image"
                self._scene = image
                self._scene_token += 1
                self._source_switched()
                self._set_paused_internal(False)
                self.source_changed.emit("image", f"Image: {name}")
                self._emit_status(f"Image loaded: {name}")
                return
            cap = cv2.VideoCapture(path)
            ok, frame = (cap.read() if cap.isOpened() else (False, None))
            if not ok or prepare_image(frame) is None:
                cap.release()
                continue
            self._release_capture()
            self._cap = cap
            self._kind = "video"
            fps = cap.get(cv2.CAP_PROP_FPS)
            self._video_period = 1.0 / fps if 1.0 < fps <= 60.0 else 1.0 / TARGET_FPS
            self._set_scene(frame)
            self._source_switched()
            self._set_paused_internal(False)
            self.source_changed.emit("video", f"Video: {name}")
            self._emit_status(f"Playing video: {name}")
            return
        self.source_error.emit(
            f"Could not open '{name}'. The file is missing, corrupted or not a supported "
            "image/video format.")

    # ----- acquisition ----------------------------------------------------------
    def _open_webcam(self) -> bool:
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            cap.release()
            self._emit_status(f"Camera could not be opened (index {self._camera_index}). Retrying...")
            self._set_scene_message("Camera could not be opened")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return True

    def _acquire_frame(self, paused: bool):
        """Return (got_new_frame, seconds_to_wait_before_next_loop)."""
        if self._kind == "image":
            return False, IDLE_WAIT
        if self._kind == "webcam":
            if paused:
                self._emit_status("Camera paused")
                return False, IDLE_WAIT
            if self._cap is None and not self._open_webcam():
                return False, 1.0
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._release_capture()
                self._emit_status("Camera frame read failed (disconnected?). Reconnecting...")
                self._set_scene_message("Camera frame unavailable. Reconnecting...")
                self._scene = None
                return False, 0.5
            if not self._set_scene(frame):
                return False, IDLE_WAIT
            self._emit_status(f"Camera {self._camera_index} connected")
            self._frame_times.append(time.perf_counter())
            return True, 1.0 / TARGET_FPS
        # video file
        if self._cap is None:
            return False, IDLE_WAIT
        if self._video_ended:
            if paused:
                return False, IDLE_WAIT
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._video_ended = False
        elif paused:
            self._emit_status("Video paused")
            return False, IDLE_WAIT
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._video_ended = True
            self._frame_times.clear()
            self._set_paused_internal(True)
            self._emit_status("Video ended - showing the last frame. Press Resume to replay.")
            return False, IDLE_WAIT
        if not self._set_scene(frame):
            return False, IDLE_WAIT
        self._emit_status("Playing video")
        self._frame_times.append(time.perf_counter())
        return True, self._video_period

    def _current_fps(self, paused: bool) -> float:
        if self._kind == "image" or paused or len(self._frame_times) < 2:
            return 0.0
        if time.perf_counter() - self._frame_times[-1] > 1.0:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    # ----- analysis ----------------------------------------------------------------
    def _ensure_extractor(self) -> None:
        if self._extractor is not None or self._extractor_tried:
            return
        self._extractor_tried = True
        try:
            self._extractor = FaceLandmarkExtractor(LANDMARKER_MODEL_PATH)
        except LandmarkerError as exc:
            self._extractor_error = str(exc).splitlines()[0]
            self._report_once(str(exc))
            self.source_error.emit(str(exc))

    def _predict(self, landmarks: np.ndarray):
        if predict_angles is None:
            return None, _REGRESSION_ERROR
        try:
            raw = predict_angles(landmarks.copy(), str(MODELS_DIR))
        except Exception as exc:  # student bugs must not kill the application
            self._report_once(traceback.format_exc())
            return None, f"predict_angles raised {type(exc).__name__}: {exc}"
        if raw is None:
            if not self._models_available:
                return None, "No trained models - train a model first (or finish regression.py)"
            return None, "predict_angles returned None (not implemented yet, or invalid input)"
        angles = clean_angles(raw) if isinstance(raw, (tuple, list, np.ndarray)) else None
        if angles is None:
            return None, "predict_angles returned an invalid result (need 3 finite angles in degrees)"
        return angles, ""

    def _analyze(self) -> _Analysis:
        key = (self._scene_token, self._models_epoch)
        if self._analysis is not None and self._analysis_key == key:
            return self._analysis
        self._ensure_extractor()
        if self._extractor is None:
            result = _Analysis(None, None, self._extractor_error or "Landmark model unavailable")
        else:
            try:
                landmarks = self._extractor.extract(self._scene)
            except LandmarkerError as exc:
                self._report_once(str(exc))
                landmarks, result = None, _Analysis(None, None, f"Landmark extraction failed: {exc}")
            else:
                if landmarks is None:
                    result = _Analysis(None, None, "No face detected")
                else:
                    angles, message = self._predict(landmarks)
                    result = _Analysis(landmarks, angles, message)
        self._analysis, self._analysis_key = result, key
        return result

    # ----- processing ----------------------------------------------------------------
    def _process(self, paused: bool, got_frame: bool):
        started = time.perf_counter()
        fps = self._current_fps(paused)
        with self._lock:
            display = self._display
        if self._scene is None:
            canvas = make_placeholder(CAPTURE_WIDTH, CAPTURE_HEIGHT, [self._scene_message])
            pointer_state = self._pointer.update(None)
            return canvas, FrameResult(
                kind=self._kind, pointer=pointer_state, models_available=self._models_available,
                status=self._scene_message)

        analysis = self._analyze()
        pointer_state = self._pointer.update(analysis.angles, snap=not got_frame)
        view = self._scene.copy()
        if display.show_landmarks and analysis.landmarks is not None:
            draw_landmarks(view, analysis.landmarks)
        if display.mirror:
            view = cv2.flip(view, 1)  # display only; inference used the original frame
        draw_overlay(view, analysis.landmarks is not None, analysis.angles,
                     display.show_angles, analysis.message)

        if analysis.angles is not None:
            status = "Tracking" + (" (centered)" if pointer_state.calibrated else "")
        else:
            status = analysis.message or "Waiting"
        return view, FrameResult(
            kind=self._kind,
            face_detected=analysis.landmarks is not None,
            angles=analysis.angles,
            pointer=pointer_state,
            models_available=self._models_available,
            fps=fps,
            process_ms=(time.perf_counter() - started) * 1000.0,
            status=status,
        )

    # ----- main loop ---------------------------------------------------------------
    def run(self) -> None:
        try:
            self._models_available = inspect_models(MODELS_DIR).valid
            while self._running.is_set():
                loop_started = time.perf_counter()
                wait = IDLE_WAIT
                try:
                    self._apply_commands()
                    with self._lock:
                        paused = self._paused
                    got_frame, wait = self._acquire_frame(paused)
                    with self._lock:
                        dirty, self._dirty = self._dirty, False
                    if got_frame or dirty:
                        if self._in_flight.is_set() and (
                                time.perf_counter() - self._in_flight_since < FRAME_TIMEOUT_SECONDS):
                            with self._lock:  # GUI still busy: drop this frame, retry later
                                self._dirty = True
                            wait = IDLE_WAIT
                        else:
                            canvas, result = self._process(paused, got_frame)
                            self._in_flight.set()
                            self._in_flight_since = time.perf_counter()
                            self.frame_ready.emit(canvas, result)
                except Exception as exc:  # never let the thread die
                    self._report_once(traceback.format_exc())
                    self._emit_status(f"Unexpected processing error: {exc}")
                    wait = 0.25
                elapsed = time.perf_counter() - loop_started
                self._wake.wait(timeout=max(0.005, wait - elapsed))
                self._wake.clear()
        finally:
            self._release_capture()
            if self._extractor is not None:
                self._extractor.close()
                self._extractor = None
            self.source_status.emit("Stopped")
