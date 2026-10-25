"""PyQt6 main window: training controls, camera view, virtual pointer view, settings."""
from __future__ import annotations

from collections import deque
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, QSettings, Qt
from PyQt6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPolygonF, QShortcut
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

import config
import model_store
from config import FrameResult
from pointer import PointerMapper, PointerSettings
from training_thread import TrainingThread
from video_thread import VideoThread

THEMES = {
    "dark": dict(bg="#1e2327", panel="#262c31", border="#3a434a", text="#e6e8ea", muted="#9aa5ad",
                 accent="#4da3ff", good="#5ac97a", warn="#f0b44c", bad="#ff6b6b", view="#14181b"),
    "light": dict(bg="#eef0f2", panel="#ffffff", border="#c9d0d6", text="#1f2328", muted="#5b6670",
                  accent="#0b6bcb", good="#1e8a45", warn="#a96a00", bad="#c62828", view="#f7f8f9"),
}


def build_stylesheet(c):
    return f"""
    QMainWindow, QWidget#panelRoot {{ background: {c['bg']}; }}
    QWidget {{ color: {c['text']}; font-size: 13px; }}
    QFrame#card, QGroupBox {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 8px; }}
    QGroupBox {{ margin-top: 14px; padding: 12px 8px 8px 8px; font-weight: bold; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; }}
    QLabel {{ background: transparent; border: none; }}
    QLabel#muted {{ color: {c['muted']}; }}
    QLabel#cardTitle {{ font-weight: bold; }}
    QPushButton {{ background: {c['bg']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 5px 10px; }}
    QPushButton:hover {{ border-color: {c['accent']}; }}
    QPushButton:disabled {{ color: {c['muted']}; }}
    QPushButton#primary {{ background: {c['accent']}; color: white; border: none; font-weight: bold; }}
    QPushButton#primary:disabled {{ background: {c['border']}; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {c['bg']}; border: 1px solid {c['border']}; border-radius: 5px; padding: 3px 6px; }}
    QProgressBar {{ border: 1px solid {c['border']}; border-radius: 5px; background: {c['bg']}; height: 8px; }}
    QProgressBar::chunk {{ background: {c['accent']}; }}
    QScrollArea {{ border: none; background: transparent; }}
    QStatusBar QLabel {{ padding: 0 8px; }}
    """


class CameraView(QWidget):
    """Shows the latest frame scaled to fit (aspect ratio preserved)."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self._image = None
        self._message = "Waiting for camera..."
        self._colors = THEMES["dark"]

    def set_theme(self, colors):
        self._colors = colors
        self.update()

    def set_image(self, image: QImage):
        self._image = image
        self.update()

    def set_message(self, text):
        self._image = None
        self._message = text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(self._colors["view"]))
        if self._image is None:
            painter.setPen(QColor(self._colors["muted"]))
            painter.drawText(self.rect().adjusted(16, 16, -16, -16),
                             Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value, self._message)
            return
        scaled = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRectF((self.width() - scaled.width()) / 2, (self.height() - scaled.height()) / 2,
                        scaled.width(), scaled.height())
        painter.drawImage(target, self._image)


class PointerView(QWidget):
    """Square arena with a centre, boundary, optional trail and a rotating pointer."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.show_trail = config.DEFAULT_SHOW_TRAIL
        self._trail = deque(maxlen=config.TRAIL_LENGTH)
        self._pos = (0.0, 0.0)
        self._rotation = 0.0
        self._active = False
        self._colors = THEMES["dark"]

    def set_theme(self, colors):
        self._colors = colors
        self.update()

    def set_state(self, position, rotation, active):
        self._pos, self._rotation, self._active = position, rotation, active
        if active:
            self._trail.append(position)
        self.update()

    def clear_trail(self):
        self._trail.clear()
        self.update()

    def paintEvent(self, event):
        c = self._colors
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(c["view"]))
        side = max(60, min(self.width(), self.height()) - 16)
        arena = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        painter.setPen(QPen(QColor(c["border"]), 2))
        painter.setBrush(QColor(c["panel"]))
        painter.drawRoundedRect(arena, 10, 10)

        grid = QPen(QColor(c["border"]), 1, Qt.PenStyle.DotLine)
        painter.setPen(grid)
        for fraction in (0.25, 0.5, 0.75):
            painter.drawLine(QPointF(arena.left() + fraction * side, arena.top()), QPointF(arena.left() + fraction * side, arena.bottom()))
            painter.drawLine(QPointF(arena.left(), arena.top() + fraction * side), QPointF(arena.right(), arena.top() + fraction * side))
        centre = arena.center()
        painter.setPen(QPen(QColor(c["muted"]), 2))
        painter.drawLine(QPointF(centre.x() - 10, centre.y()), QPointF(centre.x() + 10, centre.y()))
        painter.drawLine(QPointF(centre.x(), centre.y() - 10), QPointF(centre.x(), centre.y() + 10))

        radius = max(8.0, side * 0.035)
        half = side / 2 - radius - 4                                  # keeps the pointer inside the arena

        def to_point(position):
            return QPointF(centre.x() + position[0] * half, centre.y() + position[1] * half)

        if self.show_trail and len(self._trail) > 1:
            points = list(self._trail)
            for index in range(1, len(points)):
                color = QColor(c["accent"])
                color.setAlphaF(index / len(points) * 0.7)
                painter.setPen(QPen(color, 2))
                painter.drawLine(to_point(points[index - 1]), to_point(points[index]))

        tip = to_point(self._pos)
        painter.save()
        painter.translate(tip)
        painter.rotate(self._rotation)                                # clockwise degrees, from roll
        body = QColor(c["accent"] if self._active else c["muted"])
        painter.setPen(QPen(QColor(c["text"]), 1.5))
        painter.setBrush(body)
        painter.drawEllipse(QPointF(0, 0), radius, radius)
        painter.drawPolygon(QPolygonF([QPointF(0, -radius * 2.0), QPointF(-radius * 0.6, -radius * 1.05), QPointF(radius * 0.6, -radius * 1.05)]))
        painter.restore()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.resize(1280, 760)
        self.setMinimumSize(1000, 600)
        self.settings = QSettings(config.ORGANIZATION_NAME, config.APP_NAME)
        self.mapper = PointerMapper(PointerSettings())
        self.training = None
        self.dataset_ok = False
        self.theme = config.DEFAULT_THEME
        self._camera_note = ""

        self._build_ui()
        self._build_shortcuts()
        self._load_settings()
        self._apply_theme()
        self.refresh_model_status()

        self.video = VideoThread(self)
        self.video.mirror = self.mirror_check.isChecked()
        self.video.show_landmarks = self.landmarks_check.isChecked()
        self.video.set_camera(self.camera_spin.value())
        self.video.frame_ready.connect(self.on_frame)
        self.video.status_changed.connect(self.on_source_status)
        self.video.error_occurred.connect(self.on_video_error)
        self.video.start()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("panelRoot")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 6)
        layout.setSpacing(10)

        views = QHBoxLayout()
        views.setSpacing(10)
        self.camera_view = CameraView()
        self.pointer_view = PointerView()
        views.addWidget(self._card("Camera", self.camera_view, "camera_title"), 1)
        views.addWidget(self._card("Virtual Pointer", self.pointer_view, "pointer_title"), 1)
        layout.addLayout(views, 1)

        panel = QWidget()
        panel.setObjectName("panelRoot")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 6, 0)
        panel_layout.addWidget(self._training_group())
        panel_layout.addWidget(self._camera_group())
        panel_layout.addWidget(self._pointer_group())
        footer = QHBoxLayout()
        self.reset_button = QPushButton("Reset to Defaults (R)")
        self.reset_button.setToolTip("Resets GUI and pointer settings only. Trained models are kept.")
        self.reset_button.clicked.connect(self.reset_defaults)
        self.theme_button = QPushButton("Light / Dark")
        self.theme_button.clicked.connect(self.toggle_theme)
        footer.addWidget(self.reset_button)
        footer.addWidget(self.theme_button)
        panel_layout.addLayout(footer)
        panel_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setFixedWidth(380)
        layout.addWidget(scroll)

        self.status_label = QLabel("Ready")
        self.face_label = QLabel("Face: -")
        self.fps_label = QLabel("FPS: -")
        self.pose_label = QLabel("")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.pose_label)
        self.statusBar().addPermanentWidget(self.face_label)
        self.statusBar().addPermanentWidget(self.fps_label)

    def _card(self, title, widget, attr):
        card = QFrame()
        card.setObjectName("card")
        box = QVBoxLayout(card)
        box.setContentsMargins(8, 8, 8, 8)
        label = QLabel(title)
        label.setObjectName("cardTitle")
        setattr(self, attr, label)
        box.addWidget(label)
        box.addWidget(widget, 1)
        return card

    def _slider_row(self, form, text, low, high, tooltip, fmt):
        label = QLabel()
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setToolTip(tooltip)
        caption = QLabel(text)
        caption.setToolTip(tooltip)
        row = QHBoxLayout()
        row.addWidget(slider, 1)
        row.addWidget(label)
        form.addWidget(caption)
        form.addLayout(row)
        slider.valueChanged.connect(lambda value: label.setText(fmt.format(value)))
        label.setText(fmt.format(slider.value()))
        return slider, label

    def _training_group(self):
        group = QGroupBox("Training")
        box = QVBoxLayout(group)
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setReadOnly(True)
        browse = QPushButton("Browse...")
        browse.setToolTip("Select a dataset file (.csv) with pitch, yaw, roll and x_i / y_i landmark columns.")
        browse.clicked.connect(self.browse_dataset)
        row = QHBoxLayout()
        row.addWidget(self.dataset_edit, 1)
        row.addWidget(browse)
        box.addWidget(QLabel("Dataset file"))
        box.addLayout(row)
        self.dataset_info = QLabel()
        self.dataset_info.setWordWrap(True)
        box.addWidget(self.dataset_info)

        self.model_combo = QComboBox()
        self.model_combo.addItems(config.MODEL_TYPES)
        self.model_combo.setToolTip("Regression algorithm used for all three angles. Lasso is slower to train.")
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setDecimals(4)
        self.alpha_spin.setRange(config.ALPHA_MIN, config.ALPHA_MAX)
        self.alpha_spin.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.alpha_spin.setToolTip("Regularization strength (Ridge, Lasso). Larger = simpler model. Linear ignores it.")
        form = QHBoxLayout()
        form.addWidget(QLabel("Algorithm"))
        form.addWidget(self.model_combo, 1)
        form.addWidget(QLabel("alpha"))
        form.addWidget(self.alpha_spin)
        box.addLayout(form)
        self.model_combo.currentTextChanged.connect(self.on_model_type_changed)
        self.alpha_spin.valueChanged.connect(lambda _: self.refresh_model_status())

        self.train_button = QPushButton("Train")
        self.train_button.setObjectName("primary")
        self.train_button.clicked.connect(self.start_training)
        self.remove_button = QPushButton("Remove Trained Models")
        self.remove_button.setToolTip("Deletes only the trained model files. The dataset and face_landmarker.task are kept.")
        self.remove_button.clicked.connect(self.remove_models)
        buttons = QHBoxLayout()
        buttons.addWidget(self.train_button, 1)
        buttons.addWidget(self.remove_button, 1)
        box.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        box.addWidget(self.progress)
        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        box.addWidget(self.model_status)
        self.train_result = QLabel()
        self.train_result.setObjectName("muted")
        self.train_result.setWordWrap(True)
        box.addWidget(self.train_result)
        return group

    def _camera_group(self):
        group = QGroupBox("Camera")
        box = QVBoxLayout(group)
        self.mirror_check = QCheckBox("Mirror camera image")
        self.landmarks_check = QCheckBox("Show landmarks")
        self.mirror_check.toggled.connect(self.on_mirror_toggled)
        self.landmarks_check.toggled.connect(self.on_landmarks_toggled)
        box.addWidget(self.mirror_check)
        box.addWidget(self.landmarks_check)
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, config.MAX_CAMERA_INDEX)
        self.camera_spin.setToolTip("Camera index (0 = default camera).")
        reconnect = QPushButton("Reconnect Camera")
        reconnect.clicked.connect(self.use_webcam)
        row = QHBoxLayout()
        row.addWidget(QLabel("Camera index"))
        row.addWidget(self.camera_spin)
        row.addWidget(reconnect, 1)
        box.addLayout(row)
        self.pause_button = QPushButton("Pause (Space)")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self.set_paused)
        image_button = QPushButton("Test on Image...")
        image_button.setToolTip("Run landmarks + inference on one image. A single image cannot show continuous pointer motion.")
        image_button.clicked.connect(self.open_test_image)
        row2 = QHBoxLayout()
        row2.addWidget(self.pause_button, 1)
        row2.addWidget(image_button, 1)
        box.addLayout(row2)
        return group

    def _pointer_group(self):
        group = QGroupBox("Virtual Pointer")
        box = QVBoxLayout(group)
        self.set_center_button = QPushButton("Set Center")
        self.set_center_button.setToolTip("Use your current head pose as the neutral (centre) pose.")
        self.set_center_button.clicked.connect(self.set_center)
        self.clear_center_button = QPushButton("Clear Center")
        self.clear_center_button.setToolTip("Go back to the default neutral pose (0, 0, 0 degrees).")
        self.clear_center_button.clicked.connect(self.clear_center)
        row = QHBoxLayout()
        row.addWidget(self.set_center_button, 1)
        row.addWidget(self.clear_center_button, 1)
        box.addLayout(row)
        self.invert_x_check = QCheckBox("Invert X")
        self.invert_y_check = QCheckBox("Invert Y")
        self.trail_check = QCheckBox("Show pointer trail")
        checks = QHBoxLayout()
        checks.addWidget(self.invert_x_check)
        checks.addWidget(self.invert_y_check)
        box.addLayout(checks)
        box.addWidget(self.trail_check)
        self.yaw_slider, _ = self._slider_row(box, "Yaw Range (head turn that reaches the left/right edge)", *config.YAW_RANGE_LIMITS,
                                              "Smaller = the pointer moves more for the same head turn.", "+-{} deg")
        self.pitch_slider, _ = self._slider_row(box, "Pitch Range (head tilt that reaches the top/bottom edge)", *config.PITCH_RANGE_LIMITS,
                                                "Smaller = the pointer moves more for the same nod.", "+-{} deg")
        self.smooth_slider, _ = self._slider_row(box, "Smoothing (higher = steadier but slower)", *config.SMOOTHING_LIMITS,
                                                 "0 = raw estimates, 95 = very smooth, noticeably delayed.", "{} %")
        for widget in (self.invert_x_check, self.invert_y_check):
            widget.toggled.connect(self.on_pointer_settings_changed)
        for slider in (self.yaw_slider, self.pitch_slider, self.smooth_slider):
            slider.valueChanged.connect(self.on_pointer_settings_changed)
        self.trail_check.toggled.connect(self.on_trail_toggled)
        self.angles_label = QLabel("Pitch -   Yaw -   Roll -")
        self.center_label = QLabel("Center: default (0, 0, 0)")
        self.center_label.setObjectName("muted")
        box.addWidget(self.angles_label)
        box.addWidget(self.center_label)
        return group

    def _build_shortcuts(self):
        self._shortcuts = []
        actions = {"pause": self.toggle_pause, "fullscreen": self.toggle_fullscreen,
                   "reset": self.reset_defaults, "quit": self.close}
        for name, slot in actions.items():
            shortcut = QShortcut(QKeySequence(config.SHORTCUTS[name]), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

    # ------------------------------------------------------------ settings
    def _apply_defaults(self):
        self.theme = config.DEFAULT_THEME
        self._set_dataset(str(config.DEFAULT_DATASET_PATH))
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentText(config.DEFAULT_MODEL_TYPE)
        self.model_combo.blockSignals(False)
        self._set_alpha_for(config.DEFAULT_MODEL_TYPE)
        self.mirror_check.setChecked(config.DEFAULT_MIRROR)
        self.landmarks_check.setChecked(config.DEFAULT_SHOW_LANDMARKS)
        self.camera_spin.setValue(config.DEFAULT_CAMERA_INDEX)
        self.invert_x_check.setChecked(config.DEFAULT_INVERT_X)
        self.invert_y_check.setChecked(config.DEFAULT_INVERT_Y)
        self.trail_check.setChecked(config.DEFAULT_SHOW_TRAIL)
        self.yaw_slider.setValue(int(config.DEFAULT_YAW_RANGE))
        self.pitch_slider.setValue(int(config.DEFAULT_PITCH_RANGE))
        self.smooth_slider.setValue(config.DEFAULT_SMOOTHING)
        self.on_pointer_settings_changed()

    def _load_settings(self):
        self._apply_defaults()
        s = self.settings
        try:
            theme = str(s.value("theme", config.DEFAULT_THEME))
            self.theme = theme if theme in THEMES else config.DEFAULT_THEME
            dataset = str(s.value("dataset_path", str(config.DEFAULT_DATASET_PATH)))
            self._set_dataset(dataset if Path(dataset).is_file() else str(config.DEFAULT_DATASET_PATH))
            model_type = str(s.value("model_type", config.DEFAULT_MODEL_TYPE))
            if model_type in config.MODEL_TYPES:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentText(model_type)
                self.model_combo.blockSignals(False)
            self.alpha_spin.setValue(float(s.value("alpha", config.DEFAULT_ALPHA[self.model_combo.currentText()])))
            self.alpha_spin.setEnabled(self.model_combo.currentText() != "Linear")
            self.mirror_check.setChecked(s.value("mirror", config.DEFAULT_MIRROR, type=bool))
            self.landmarks_check.setChecked(s.value("landmarks", config.DEFAULT_SHOW_LANDMARKS, type=bool))
            self.camera_spin.setValue(int(s.value("camera_index", config.DEFAULT_CAMERA_INDEX)))
            self.invert_x_check.setChecked(s.value("invert_x", config.DEFAULT_INVERT_X, type=bool))
            self.invert_y_check.setChecked(s.value("invert_y", config.DEFAULT_INVERT_Y, type=bool))
            self.trail_check.setChecked(s.value("trail", config.DEFAULT_SHOW_TRAIL, type=bool))
            self.yaw_slider.setValue(int(s.value("yaw_range", config.DEFAULT_YAW_RANGE)))
            self.pitch_slider.setValue(int(s.value("pitch_range", config.DEFAULT_PITCH_RANGE)))
            self.smooth_slider.setValue(int(s.value("smoothing", config.DEFAULT_SMOOTHING)))
        except (TypeError, ValueError):
            self._apply_defaults()          # corrupted settings: fall back to defaults
        # The centre pose is deliberately not stored: it depends on today's camera and sitting position.

    def _save_settings(self):
        s = self.settings
        s.setValue("theme", self.theme)
        s.setValue("dataset_path", self.dataset_edit.text())
        s.setValue("model_type", self.model_combo.currentText())
        s.setValue("alpha", self.alpha_spin.value())
        s.setValue("mirror", self.mirror_check.isChecked())
        s.setValue("landmarks", self.landmarks_check.isChecked())
        s.setValue("camera_index", self.camera_spin.value())
        s.setValue("invert_x", self.invert_x_check.isChecked())
        s.setValue("invert_y", self.invert_y_check.isChecked())
        s.setValue("trail", self.trail_check.isChecked())
        s.setValue("yaw_range", self.yaw_slider.value())
        s.setValue("pitch_range", self.pitch_slider.value())
        s.setValue("smoothing", self.smooth_slider.value())

    def reset_defaults(self):
        self._apply_defaults()
        self._apply_theme()
        self.mapper.clear_center()
        self.mapper.reset_smoothing()
        self.pointer_view.clear_trail()
        self.refresh_model_status()
        self.set_status("Settings reset to defaults (trained models were not touched).")

    # ------------------------------------------------------------- theme
    def _apply_theme(self):
        colors = THEMES[self.theme]
        QApplication.instance().setStyleSheet(build_stylesheet(colors))
        self.camera_view.set_theme(colors)
        self.pointer_view.set_theme(colors)
        self.refresh_model_status()

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self._apply_theme()

    # -------------------------------------------------------- dataset/model
    def browse_dataset(self):
        start = str(Path(self.dataset_edit.text()).parent) if self.dataset_edit.text() else str(config.DATASET_DIR)
        path, _ = QFileDialog.getOpenFileName(self, "Select dataset file", start, "Dataset files (*.csv);;All files (*)")
        if path:
            self._set_dataset(path)
            self.refresh_model_status()

    def _set_dataset(self, path):
        self.dataset_edit.setText(path)
        self.dataset_edit.setToolTip(path)
        self.dataset_ok, message = model_store.inspect_dataset(path)
        colors = THEMES[self.theme]
        self.dataset_info.setText(message)
        self.dataset_info.setStyleSheet(f"color: {colors['good'] if self.dataset_ok else colors['bad']};")

    def _set_alpha_for(self, model_type):
        self.alpha_spin.blockSignals(True)
        self.alpha_spin.setValue(config.DEFAULT_ALPHA[model_type])
        self.alpha_spin.blockSignals(False)
        self.alpha_spin.setEnabled(model_type != "Linear")

    def on_model_type_changed(self, model_type):
        self._set_alpha_for(model_type)
        self.refresh_model_status()

    def refresh_model_status(self):
        colors = THEMES[self.theme]
        status, message, meta = model_store.check_models(config.TRAINED_MODELS_DIR)
        selected, alpha = self.model_combo.currentText(), self.alpha_spin.value()
        if status == "ok":
            alpha_text = "" if meta["model_type"] == "Linear" else f", alpha {meta['alpha']:g}"
            text = f"Active models: {meta['model_type']}{alpha_text} ({meta.get('n_samples', 0)} samples)."
            color = colors["good"]
            notes = []
            if meta["model_type"] != selected or (selected != "Linear" and abs(meta["alpha"] - alpha) > 1e-12):
                notes.append(f"The current selection ({selected}) differs from the active models. Press Train to apply it.")
            trained_on = meta.get("dataset_path", "")
            if trained_on and trained_on != str(Path(self.dataset_edit.text()).resolve()):
                notes.append("A different dataset is selected. Retrain to use it.")
            if notes:
                text += "\n" + "\n".join(notes)
                color = colors["warn"]
        elif status == "missing":
            text, color = "No trained models yet. Choose a dataset and algorithm, then press Train.", colors["warn"]
        else:
            text, color = f"Saved models are incompatible: {message} Train again.", colors["bad"]
        self.model_status.setText(text)
        self.model_status.setStyleSheet(f"color: {color};")
        busy = self.training is not None
        self.remove_button.setEnabled(status != "missing" and not busy)

    def start_training(self):
        if self.training is not None:
            return
        self._set_dataset(self.dataset_edit.text())         # re-check the file right now
        if not self.dataset_ok:
            QMessageBox.warning(self, "Dataset problem", self.dataset_info.text())
            return
        self.training = TrainingThread(self.dataset_edit.text(), self.model_combo.currentText(), self.alpha_spin.value(), self)
        self.training.succeeded.connect(self.on_training_succeeded)
        self.training.failed.connect(self.on_training_failed)
        self.training.finished.connect(self.on_training_finished)
        self._set_training_ui(True)
        self.train_result.setText("Training...")
        self.training.start()

    def _set_training_ui(self, busy):
        for widget in (self.train_button, self.remove_button, self.model_combo, self.alpha_spin):
            widget.setEnabled(not busy)
        if not busy:
            self.alpha_spin.setEnabled(self.model_combo.currentText() != "Linear")
        self.progress.setVisible(busy)

    def on_training_succeeded(self, result):
        extra = ""
        if result.get("val_mae_deg"):
            mae = result["val_mae_deg"]
            extra = (f"\nHeld-out MAE (deg): pitch {mae.get('pitch')}, yaw {mae.get('yaw')}, roll {mae.get('roll')}. "
                     "This is error on held-out dataset samples, not a guarantee of pointer quality.")
        self.train_result.setText(f"Training succeeded: {self.model_combo.currentText()}, {result.get('n_samples', '?')} samples used.\n"
                                  f"Saved to {result.get('model_dir', config.TRAINED_MODELS_DIR)}{extra}")
        self.mapper.reset_smoothing()
        self.set_status("Training finished.")

    def on_training_failed(self, message):
        self.train_result.setText(f"Training failed: {message}")
        self.set_status("Training failed. Previous models (if any) were kept.")
        QMessageBox.warning(self, "Training failed", message)

    def on_training_finished(self):
        worker, self.training = self.training, None
        if worker is not None:
            worker.deleteLater()
        self._set_training_ui(False)
        self.refresh_model_status()

    def remove_models(self):
        if self.training is not None:
            return
        answer = QMessageBox.question(
            self, "Remove trained models",
            "Delete the trained pitch/yaw/roll models?\nThe dataset and face_landmarker.task are not affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = model_store.delete_trained_models()
            self.train_result.setText(f"Removed {len(removed)} file(s).")
        except Exception as exc:
            QMessageBox.warning(self, "Could not remove models", str(exc))
        self.mapper.reset_smoothing()
        self.refresh_model_status()

    # ------------------------------------------------------------- camera
    def use_webcam(self):
        self.mapper.reset()
        self.pointer_view.clear_trail()
        self.video.set_camera(self.camera_spin.value())

    def open_test_image(self):
        exts = " ".join(f"*{e}" for e in config.IMAGE_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, "Select a test image", str(config.EXAMPLES_DIR), f"Images ({exts});;All files (*)")
        if path:
            self.mapper.reset()
            self.pointer_view.clear_trail()
            self.video.set_image(path)

    def set_paused(self, paused):
        self.video.set_paused(paused)
        self.pause_button.setText("Resume (Space)" if paused else "Pause (Space)")
        if not paused:
            self.mapper.reset_smoothing()
        self.set_status("Paused." if paused else "Resumed.")

    def toggle_pause(self):
        self.pause_button.toggle()

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # -------------------------------------------------------------- pointer
    def on_pointer_settings_changed(self, *_):
        settings = self.mapper.settings
        settings.yaw_range = float(self.yaw_slider.value())
        settings.pitch_range = float(self.pitch_slider.value())
        settings.smoothing = self.smooth_slider.value() / 100.0
        settings.invert_x = self.invert_x_check.isChecked()
        settings.invert_y = self.invert_y_check.isChecked()
        self.mapper.refresh()

    def on_mirror_toggled(self, on):
        if hasattr(self, "video"):                    # not yet created while settings are loading
            self.video.mirror = on

    def on_landmarks_toggled(self, on):
        if hasattr(self, "video"):
            self.video.show_landmarks = on

    def on_trail_toggled(self, on):
        self.pointer_view.show_trail = on
        self.pointer_view.clear_trail()

    def set_center(self):
        if self.mapper.set_center():
            pitch, yaw, roll = self.mapper.center
            self.center_label.setText(f"Center: set (pitch {pitch:.1f}, yaw {yaw:.1f}, roll {roll:.1f})")
            self.set_status("Center set from the current head pose.")
        else:
            self.set_status("Cannot set center: no valid head pose right now (no face, or models not ready).")

    def clear_center(self):
        self.mapper.clear_center()
        self.center_label.setText("Center: default (0, 0, 0)")
        self.set_status("Center cleared.")

    # ------------------------------------------------------ video callbacks
    def on_frame(self, image: QImage, result: FrameResult):
        try:
            self.camera_view.set_image(image)
            used = self.mapper.update(result.angles)
            self.pointer_view.set_state(self.mapper.position, self.mapper.rotation, used)
            if result.angles is not None:
                pitch, yaw, roll = result.angles
                self.angles_label.setText(f"Pitch {pitch:6.1f}   Yaw {yaw:6.1f}   Roll {roll:6.1f} deg")
            else:
                self.angles_label.setText("Pitch -   Yaw -   Roll -   (" + result.status + ")")
            self.face_label.setText("Face: detected" if result.face_found else "Face: not detected")
            self.fps_label.setText(f"FPS: {result.fps:.1f}  ({result.processing_ms:.0f} ms)" if result.source == "camera"
                                   else f"Image  ({result.processing_ms:.0f} ms)")
            self.pose_label.setText(result.status)
        finally:
            self.video.acknowledge()

    def on_source_status(self, message):
        self.set_status(message)
        if "could not" in message.lower() or "disconnected" in message.lower():
            self.camera_view.set_message(message + "\nTraining and the examples still work without a camera.")

    def on_video_error(self, message):
        self.set_status(message)

    def set_status(self, message):
        self.status_label.setText(message)

    # ---------------------------------------------------------------- close
    def closeEvent(self, event):
        if self.training is not None:
            answer = QMessageBox.question(self, "Training in progress", "Training is still running. Quit after it finishes?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.training.wait()
        for signal in (self.video.frame_ready, self.video.status_changed, self.video.error_occurred):
            try:
                signal.disconnect()
            except TypeError:
                pass
        self.video.stop()
        self.video.wait(5000)
        self._save_settings()
        event.accept()
