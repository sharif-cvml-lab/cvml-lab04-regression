from __future__ import annotations

from collections import deque

import cv2
from PyQt6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QShortcut,
)
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from config import (
    APP_NAME, DEFAULT_CAMERA_INDEX, DEFAULT_DATASET_PATH, DEFAULT_MIRROR, DEFAULT_MODEL_TYPE,
    DEFAULT_PITCH_RANGE, DEFAULT_SHOW_ANGLES, DEFAULT_SHOW_LANDMARKS, DEFAULT_SHOW_TRAIL,
    DEFAULT_SMOOTHING, DEFAULT_THEME, DEFAULT_YAW_RANGE, IMAGE_EXTENSIONS, MODEL_LABELS,
    MODEL_TYPES, MODELS_DIR, PITCH_RANGE_MAX, PITCH_RANGE_MIN, SHORTCUTS, SMOOTHING_MAX,
    SMOOTHING_MIN, TRAIL_LENGTH, VIDEO_EXTENSIONS, YAW_RANGE_MAX, YAW_RANGE_MIN, DisplaySettings,
    FrameResult, PointerSettings, validated_display_settings, validated_pointer_settings,
)
from model_store import ModelStoreError, bundle_path, delete_model_bundle, inspect_models
from pointer import HOLDING, LOST, TRACKING, WAITING, PointerState
from training_worker import TrainingWorker
from video_thread import VideoThread

DARK_STYLE = """
QMainWindow, QWidget {
    background: #11151b;
    color: #e7ebf0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QLabel { background: transparent; }
QFrame#Header, QFrame#SidebarCard, QFrame#VideoCard, QFrame#StatusBar {
    background: #171c23;
    border: 1px solid #28303a;
    border-radius: 12px;
}
QFrame#VideoCanvas {
    background: #0a0d11;
    border: 1px solid #303945;
    border-radius: 10px;
}
QLabel#Title { font-size: 22px; font-weight: 700; }
QLabel#Subtitle, QLabel#Muted { color: #8e98a6; }
QLabel#PanelTitle { font-size: 14px; font-weight: 700; }
QLabel#Metric { color: #9aa6b4; font-size: 12px; }
QLabel#MetricValue { font-size: 15px; font-weight: 700; }
QLabel#ValueChip {
    background: #202833;
    border: 1px solid #364250;
    border-radius: 6px;
    padding: 2px 8px;
    min-width: 42px;
}
QPushButton {
    background: #212833;
    color: #eef2f6;
    border: 1px solid #35404d;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background: #2a3441; border-color: #4e6175; }
QPushButton:pressed { background: #18202a; }
QPushButton#accent { background: #3a7d67; border-color: #4c9b80; font-weight: 700; }
QPushButton#accent:hover { background: #479276; }
QPushButton#danger { background: #322126; border-color: #5d333d; }
QCheckBox, QComboBox, QSpinBox {
    background: #1a2028;
    border: 1px solid #35404d;
    border-radius: 7px;
    padding: 6px;
}
QSpinBox:hover { border-color: #4e6175; }
QSlider::groove:horizontal { height: 5px; background: #2b333d; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #4d9d80; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #80c6ab;
    border: 1px solid #9ad9c0;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QGroupBox {
    border: 1px solid #28303a;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 700;
}
QGroupBox::title { left: 12px; padding: 0 6px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 10px; background: transparent; }
QScrollBar::handle:vertical { background: #313b47; border-radius: 5px; min-height: 30px; }

QLineEdit { background: #1a2028; border: 1px solid #35404d; border-radius: 7px; padding: 6px; }
QComboBox QAbstractItemView { background: #1a2028; selection-background-color: #3a7d67; }
QPushButton:disabled { color: #6b7684; background: #1a1f26; border-color: #2a323c; }
QProgressBar { background: #1a2028; border: 1px solid #35404d; border-radius: 6px; height: 10px; text-align: center; }
QProgressBar::chunk { background: #4d9d80; border-radius: 5px; }
"""

LIGHT_STYLE = """
QMainWindow, QWidget {
    background: #eef2f5;
    color: #1e2630;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QLabel { background: transparent; }
QFrame#Header, QFrame#SidebarCard, QFrame#VideoCard, QFrame#StatusBar {
    background: #ffffff;
    border: 1px solid #d4dbe2;
    border-radius: 12px;
}
QFrame#VideoCanvas {
    background: #f7f9fb;
    border: 1px solid #cbd4dd;
    border-radius: 10px;
}
QLabel#Title { font-size: 22px; font-weight: 700; }
QLabel#Subtitle, QLabel#Muted { color: #667485; }
QLabel#PanelTitle { font-size: 14px; font-weight: 700; }
QLabel#Metric { color: #657180; font-size: 12px; }
QLabel#MetricValue { font-size: 15px; font-weight: 700; }
QLabel#ValueChip {
    background: #f1f4f7;
    border: 1px solid #d2d9e0;
    border-radius: 6px;
    padding: 2px 8px;
    min-width: 42px;
}
QPushButton {
    background: #ffffff;
    color: #1e2630;
    border: 1px solid #cbd4dd;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background: #f3f6f8; border-color: #acbac7; }
QPushButton#accent { background: #3f806b; border-color: #3f806b; color: white; font-weight: 700; }
QPushButton#danger { background: #fff3f4; border-color: #e2b7bc; }
QCheckBox, QComboBox, QSpinBox {
    background: #ffffff;
    border: 1px solid #cbd4dd;
    border-radius: 7px;
    padding: 6px;
}
QSlider::groove:horizontal { height: 5px; background: #d9e0e6; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #4a9c80; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #4a9c80;
    border: 1px solid #3f806b;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QGroupBox {
    border: 1px solid #d4dbe2;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 700;
}
QGroupBox::title { left: 12px; padding: 0 6px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 10px; background: transparent; }
QScrollBar::handle:vertical { background: #c0cad3; border-radius: 5px; min-height: 30px; }

QLineEdit { background: #ffffff; border: 1px solid #cbd4dd; border-radius: 7px; padding: 6px; }
QPushButton:disabled { color: #9aa5b1; background: #eef1f4; border-color: #dbe1e7; }
QProgressBar { background: #ffffff; border: 1px solid #cbd4dd; border-radius: 6px; height: 10px; text-align: center; }
QProgressBar::chunk { background: #4a9c80; border-radius: 5px; }
"""

OK_COLOR = "color: #3fae7c;"
BAD_COLOR = "color: #e0707c;"


class VideoLabel(QLabel):
    """Scalable image view that preserves the source aspect ratio."""

    def __init__(self):
        super().__init__()
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_frame(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if not self._pixmap.isNull():
            self.setPixmap(self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))


class ValueSlider(QWidget):
    """Integer slider with a title and a chip that always shows the current value."""

    valueChanged = pyqtSignal(int)

    def __init__(self, title, minimum, maximum, value, formatter=str, parent=None):
        super().__init__(parent)
        self._formatter = formatter
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._on_value_changed)
        self.title_label = QLabel(title)
        self.title_label.setMinimumWidth(100)
        self.value_label = QLabel()
        self.value_label.setObjectName("ValueChip")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self._set_value_label(value)

    def _set_value_label(self, value: int) -> None:
        self.value_label.setText(self._formatter(value))

    def _on_value_changed(self, value: int) -> None:
        self._set_value_label(value)
        self.valueChanged.emit(value)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int, emit: bool = True) -> None:
        blocked = False
        if not emit:
            blocked = self.slider.blockSignals(True)
        self.slider.setValue(int(value))
        if not emit:
            self.slider.blockSignals(blocked)
        self._set_value_label(self.slider.value())


class PointerView(QWidget):
    """The virtual pointer area: x/y from yaw/pitch, rotation from roll."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._state = PointerState()
        self._trail: deque[tuple[float, float]] = deque(maxlen=TRAIL_LENGTH)
        self._show_trail = DEFAULT_SHOW_TRAIL
        self._dark = True

    def set_theme(self, dark: bool) -> None:
        self._dark = dark
        self.update()

    def set_show_trail(self, show: bool) -> None:
        self._show_trail = show
        if not show:
            self._trail.clear()
        self.update()

    def clear_trail(self) -> None:
        self._trail.clear()
        self.update()

    def set_state(self, state: PointerState) -> None:
        if state == self._state:
            return  # nothing changed: skip the repaint
        if state.status == TRACKING and self._show_trail:
            if not self._trail or (abs(self._trail[-1][0] - state.x) + abs(self._trail[-1][1] - state.y)) > 0.004:
                self._trail.append((state.x, state.y))
        self._state = state
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg, grid, text = (("#0a0d11", "#1f2832", "#8e98a6") if self._dark
                          else ("#f7f9fb", "#d9e0e6", "#667485"))
        p.fillRect(self.rect(), QColor(bg))
        side = max(20.0, min(self.width(), self.height()) - 24.0)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        half = side / 2.0
        area = QRectF(cx - half, cy - half, side, side)
        p.setPen(QPen(QColor(grid), 1))
        for k in (-0.5, 0.0, 0.5):
            p.drawLine(QPointF(cx + k * half, area.top()), QPointF(cx + k * half, area.bottom()))
            p.drawLine(QPointF(area.left(), cy + k * half), QPointF(area.right(), cy + k * half))
        p.setPen(QPen(QColor("#35404d" if self._dark else "#b5c0cb"), 2))
        p.drawRoundedRect(area, 10, 10)

        s = self._state
        active = s.status == TRACKING
        color = QColor("#4d9d80") if active else QColor("#8a8f98")

        def to_px(x: float, y: float) -> QPointF:
            return QPointF(cx + x * half, cy - y * half)

        if self._show_trail and len(self._trail) > 1:
            points = list(self._trail)
            for i in range(1, len(points)):
                c = QColor(color)
                c.setAlphaF(0.08 + 0.7 * i / len(points))
                p.setPen(QPen(c, 2 + 3 * i / len(points)))
                p.drawLine(to_px(*points[i - 1]), to_px(*points[i]))

        p.save()
        p.translate(to_px(s.x, s.y))
        p.rotate(-s.roll_deg)  # positive roll = counter-clockwise on screen
        arrow = QPolygonF([QPointF(0, -18), QPointF(12, 14), QPointF(0, 7), QPointF(-12, 14)])
        fill = QColor(color)
        if not active:
            fill.setAlphaF(0.45)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(QColor("#e7ebf0" if self._dark else "#1e2630"), 1.5))
        p.drawPolygon(arrow)
        p.restore()

        p.setPen(QColor(text))
        label = {TRACKING: "", HOLDING: "Face lost - holding position",
                 LOST: "No face - pointer frozen", WAITING: "Waiting for a valid prediction"}[s.status]
        if label:
            p.drawText(QRectF(area.left(), area.top() + 6, side, 22), Qt.AlignmentFlag.AlignCenter, label)
        p.drawText(QRectF(area.left() + 8, area.bottom() - 24, side - 16, 20),
                   Qt.AlignmentFlag.AlignLeft, f"x {s.x:+.2f}   y {s.y:+.2f}   roll {s.roll_deg:+.0f}°")
        p.end()


class HeadPointerWindow(QMainWindow):
    """Main UI: camera view, virtual pointer, training and pointer controls."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(QSize(1100, 680))
        self.resize(1500, 900)

        self.settings = QSettings()
        self.video_thread: VideoThread | None = None
        self.worker: TrainingWorker | None = None
        self._closing = False
        self._quit_after_training = False
        self._paused = False
        self._kind = "webcam"
        self.current_theme = self.settings.value("theme", DEFAULT_THEME, type=str)
        self._last_dir = self.settings.value("last_dir", "", type=str)

        self._build_ui()
        self._restore_settings()
        self._apply_theme()
        self._refresh_model_status()
        self._start_video_thread()

    # ------------------------------------------------------------------ UI
    def _card(self, title: str, widget: QWidget) -> QFrame:
        card = QFrame()
        card.setObjectName("VideoCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("PanelTitle")
        layout.addWidget(title_label)
        canvas = QFrame()
        canvas.setObjectName("VideoCanvas")
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(widget)
        layout.addWidget(canvas, 1)
        card.title_label = title_label  # type: ignore[attr-defined]
        return card

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(12)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 12, 18, 12)
        title_box = QVBoxLayout()
        title = QLabel("Head Pointer Studio")
        title.setObjectName("Title")
        subtitle = QLabel("Landmarks → Linear / Ridge / Lasso regression → pitch, yaw, roll → virtual pointer")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)
        self.theme_button = QPushButton("☼ Light Mode")
        self.theme_button.clicked.connect(self._toggle_theme)
        self.fullscreen_button = QPushButton("Full Screen")
        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        header_layout.addWidget(self.theme_button)
        header_layout.addWidget(self.fullscreen_button)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        views = QHBoxLayout()
        views.setSpacing(10)
        self.camera_label = VideoLabel()
        self.camera_label.setText("Starting...")
        self.pointer_view = PointerView()
        self.camera_card = self._card("Camera / Input View", self.camera_label)
        self.pointer_card = self._card("Virtual Pointer (not centered)", self.pointer_view)
        views.addWidget(self.camera_card, 1)
        views.addWidget(self.pointer_card, 1)
        left_layout.addLayout(views, 1)

        status = QFrame()
        status.setObjectName("StatusBar")
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(14, 8, 14, 8)
        text_box = QVBoxLayout()
        text_box.setSpacing(2)
        self.status_label = QLabel("Starting...")
        self.status_label.setObjectName("MetricValue")
        self.source_label = QLabel("")
        self.source_label.setObjectName("Muted")
        text_box.addWidget(self.status_label)
        text_box.addWidget(self.source_label)
        status_layout.addLayout(text_box, 1)
        self.metrics: dict[str, QLabel] = {}
        for key, initial in (("face", "Face: --"), ("pitch", "Pitch: --"), ("yaw", "Yaw: --"),
                             ("roll", "Roll: --"), ("fps", "FPS: --")):
            label = QLabel(initial)
            label.setObjectName("MetricValue")
            label.setMinimumWidth(92)
            status_layout.addSpacing(10)
            status_layout.addWidget(label)
            self.metrics[key] = label
        left_layout.addWidget(status)

        splitter.addWidget(left)
        splitter.addWidget(self._build_sidebar())
        splitter.setSizes([1120, 380])
        root.addWidget(splitter, 1)
        self.splitter = splitter
        self.setCentralWidget(central)
        self.statusBar().hide()
        self._build_shortcuts()

    def _button(self, text, slot, name=None) -> QPushButton:
        button = QPushButton(text)
        if name:
            button.setObjectName(name)
        button.clicked.connect(slot)
        return button

    def _note(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("Muted")
        label.setWordWrap(True)
        return label

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(10)

        # --- Input source ---------------------------------------------------
        box = QGroupBox("Input Source")
        v = QVBoxLayout(box)
        self.scene_label = self._note("Webcam")
        v.addWidget(self.scene_label)
        row = QHBoxLayout()
        row.addWidget(self._button("Use Webcam", self._use_webcam))
        row.addWidget(self._button("Open Image/Video…", self._open_file))
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Camera index"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        self.camera_spin.setValue(DEFAULT_CAMERA_INDEX)
        self.camera_spin.valueChanged.connect(self._camera_index_changed)
        row.addWidget(self.camera_spin)
        v.addLayout(row)
        row = QHBoxLayout()
        self.pause_button = self._button("Pause", self._toggle_pause)
        self.restart_button = self._button("Restart Source", self._restart_source)
        row.addWidget(self.pause_button)
        row.addWidget(self.restart_button)
        v.addLayout(row)
        layout.addWidget(box)

        # --- Training -----------------------------------------------------------
        box = QGroupBox("Regression Model")
        v = QVBoxLayout(box)
        self.model_status_label = self._note("")
        v.addWidget(self.model_status_label)
        row = QHBoxLayout()
        self.dataset_edit = QLineEdit(str(DEFAULT_DATASET_PATH))
        self.dataset_edit.setReadOnly(True)
        self.dataset_button = self._button("Browse…", self._choose_dataset)
        row.addWidget(self.dataset_edit, 1)
        row.addWidget(self.dataset_button)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Algorithm"))
        self.model_combo = QComboBox()
        for key in MODEL_TYPES:
            self.model_combo.addItem(MODEL_LABELS[key], key)
        row.addWidget(self.model_combo, 1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.train_button = self._button("Train", self._start_training, "accent")
        self.delete_button = self._button("Delete Models…", self._delete_models, "danger")
        row.addWidget(self.train_button)
        row.addWidget(self.delete_button)
        v.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        v.addWidget(self.progress)
        self.train_status = self._note("Training is explicit: press Train. Existing models stay valid until a new training succeeds.")
        v.addWidget(self.train_status)
        layout.addWidget(box)

        # --- Pointer ---------------------------------------------------------------
        box = QGroupBox("Pointer")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.center_button = self._button(f"Set Center ({SHORTCUTS['center']})", self._set_center, "accent")
        row.addWidget(self.center_button)
        row.addWidget(self._button("Clear Center", self._clear_center))
        v.addLayout(row)
        v.addWidget(self._button("Reset Pointer", self._reset_pointer))
        self.yaw_slider = ValueSlider("Yaw range", YAW_RANGE_MIN, YAW_RANGE_MAX, DEFAULT_YAW_RANGE,
                                      lambda x: f"±{x}°")
        self.pitch_slider = ValueSlider("Pitch range", PITCH_RANGE_MIN, PITCH_RANGE_MAX,
                                        DEFAULT_PITCH_RANGE, lambda x: f"±{x}°")
        self.smooth_slider = ValueSlider("Smoothing", SMOOTHING_MIN, SMOOTHING_MAX, DEFAULT_SMOOTHING,
                                         lambda x: "off" if x == 0 else f"{x} ms")
        for widget in (self.yaw_slider, self.pitch_slider, self.smooth_slider):
            v.addWidget(widget)
            widget.valueChanged.connect(self._pointer_settings_changed)
        v.addWidget(self._note(
            "Range = head angle (from the center) that moves the pointer to the edge. "
            "Smoothing = time constant of the filter: more = steadier but laggier."))
        self.invert_x = QCheckBox("Invert X")
        self.invert_y = QCheckBox("Invert Y")
        self.show_trail = QCheckBox("Show pointer trail")
        self.show_trail.setChecked(DEFAULT_SHOW_TRAIL)
        for widget in (self.invert_x, self.invert_y, self.show_trail):
            v.addWidget(widget)
            widget.toggled.connect(self._pointer_settings_changed)
        layout.addWidget(box)

        # --- Display -------------------------------------------------------------------
        box = QGroupBox("Display")
        v = QVBoxLayout(box)
        self.mirror_check = QCheckBox("Mirror camera view")
        self.landmarks_check = QCheckBox("Show facial landmarks")
        self.angles_check = QCheckBox("Show angles on the view")
        for widget, default in ((self.mirror_check, DEFAULT_MIRROR),
                                (self.landmarks_check, DEFAULT_SHOW_LANDMARKS),
                                (self.angles_check, DEFAULT_SHOW_ANGLES)):
            widget.setChecked(default)
            widget.toggled.connect(self._display_settings_changed)
            v.addWidget(widget)
        v.addWidget(self._note("Mirroring only changes the picture you see; the model always receives the original frame."))
        layout.addWidget(box)

        row = QHBoxLayout()
        row.addWidget(self._button("Reset Defaults", self._reset_defaults, "accent"))
        row.addWidget(self._button("Quit", self.close, "danger"))
        layout.addLayout(row)
        layout.addWidget(self._note(
            "Shortcuts: Space = pause/resume · C = set center · F11 = full screen · R = reset · Q = quit"))
        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence(SHORTCUTS["pause"]), self, activated=self._toggle_pause)
        QShortcut(QKeySequence(SHORTCUTS["fullscreen"]), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence(SHORTCUTS["reset"]), self, activated=self._reset_defaults)
        QShortcut(QKeySequence(SHORTCUTS["quit"]), self, activated=self.close)
        QShortcut(QKeySequence(SHORTCUTS["center"]), self, activated=self._set_center)

    # ------------------------------------------------------------ video thread
    def _start_video_thread(self) -> None:
        self._stop_video_thread()
        thread = VideoThread(self.camera_spin.value(), self)
        thread.set_pointer_settings(self._pointer_settings())
        thread.set_display_settings(self._display_settings())
        thread.frame_ready.connect(self._update_frame)
        thread.source_status.connect(self._on_source_status)
        thread.notice.connect(self._on_source_status)
        thread.source_error.connect(self._show_error)
        thread.source_changed.connect(self._source_changed)
        thread.paused_changed.connect(self._set_pause_ui)
        thread.calibration_changed.connect(self._calibration_changed)
        self.video_thread = thread
        thread.models_changed()
        thread.start()

    def _stop_video_thread(self) -> None:
        if self.video_thread is not None:
            thread, self.video_thread = self.video_thread, None
            thread.stop()

    # ----------------------------------------------------------------- actions
    def _pointer_settings(self) -> PointerSettings:
        return validated_pointer_settings(
            self.yaw_slider.value(), self.pitch_slider.value(), self.smooth_slider.value(),
            self.invert_x.isChecked(), self.invert_y.isChecked(), self.show_trail.isChecked())

    def _display_settings(self) -> DisplaySettings:
        return validated_display_settings(
            self.mirror_check.isChecked(), self.landmarks_check.isChecked(), self.angles_check.isChecked())

    def _pointer_settings_changed(self, *_args) -> None:
        settings = self._pointer_settings()
        self.pointer_view.set_show_trail(settings.show_trail)
        if self.video_thread is not None:
            self.video_thread.set_pointer_settings(settings)

    def _display_settings_changed(self, *_args) -> None:
        if self.video_thread is not None:
            self.video_thread.set_display_settings(self._display_settings())

    def _remember_directory(self, path: str) -> None:
        normalized = path.replace("\\", "/")
        if "/" in normalized:
            self._last_dir = normalized.rsplit("/", 1)[0]

    def _use_webcam(self) -> None:
        if self.video_thread is not None:
            self.video_thread.use_webcam()

    def _open_file(self) -> None:
        images = " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)
        videos = " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image or video", self._last_dir,
            f"Images and videos ({images} {videos});;Images ({images});;Videos ({videos});;All files (*)")
        if path and self.video_thread is not None:
            self._remember_directory(path)
            self.video_thread.open_file(path)

    def _camera_index_changed(self, _value: int) -> None:
        if self.video_thread is not None:
            self.video_thread.set_camera_index(self.camera_spin.value())

    def _restart_source(self) -> None:
        if self.video_thread is not None:
            self.video_thread.restart_source()

    def _toggle_pause(self) -> None:
        if self.video_thread is None or self._kind == "image":
            return
        paused = not self._paused
        self.video_thread.set_paused(paused)
        self._set_pause_ui(paused)

    def _set_pause_ui(self, paused: bool) -> None:
        self._paused = paused
        self.pause_button.setText("Resume" if paused else "Pause")

    def _set_center(self) -> None:
        if self.video_thread is not None:
            self.video_thread.calibrate()

    def _clear_center(self) -> None:
        if self.video_thread is not None:
            self.video_thread.clear_calibration()

    def _reset_pointer(self) -> None:
        self.pointer_view.clear_trail()
        if self.video_thread is not None:
            self.video_thread.reset_pointer()

    def _calibration_changed(self, calibrated: bool) -> None:
        self.pointer_card.title_label.setText(
            "Virtual Pointer (centered)" if calibrated else "Virtual Pointer (not centered)")

    def _on_source_status(self, message: str) -> None:
        if self.source_label.text() != message:
            self.source_label.setText(message)

    def _source_changed(self, kind: str, description: str) -> None:
        self._kind = kind
        self.scene_label.setText(description)
        self.pause_button.setEnabled(kind != "image")
        self.restart_button.setEnabled(kind != "image")
        self.pointer_view.clear_trail()

    def _show_error(self, message: str) -> None:
        if not self._closing:
            QMessageBox.warning(self, APP_NAME, message)

    # ---------------------------------------------------------------- training
    @property
    def training_active(self) -> bool:
        return self.worker is not None

    def _choose_dataset(self) -> None:
        if self.training_active:
            return
        start = self.dataset_edit.text() or self._last_dir
        path, _ = QFileDialog.getOpenFileName(self, "Select dataset (CSV)", start,
                                              "CSV files (*.csv);;All files (*)")
        if path:
            self._remember_directory(path)
            self.dataset_edit.setText(path)

    def _set_training_ui(self, busy: bool) -> None:
        for widget in (self.train_button, self.dataset_button, self.model_combo, self.delete_button):
            widget.setEnabled(not busy)
        self.progress.setVisible(busy)
        if not busy:
            self.delete_button.setEnabled(bundle_path(MODELS_DIR).exists())

    def _start_training(self) -> None:
        if self.training_active:
            return
        path = self.dataset_edit.text().strip()
        model_type = self.model_combo.currentData()
        worker = TrainingWorker(path, model_type, self)
        worker.training_finished.connect(self._training_finished)
        self.worker = worker
        self._set_training_ui(True)
        self.train_status.setText(f"Training {MODEL_LABELS[model_type]} on {path} ...")
        self.train_status.setStyleSheet("")
        worker.start()

    def _training_finished(self, success: bool, message: str) -> None:
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self._set_training_ui(False)
        self._refresh_model_status()
        if success and self.video_thread is not None:
            self.video_thread.models_changed()
        self.train_status.setText(("Training finished. " if success else "Training failed: ") + message)
        self.train_status.setStyleSheet(OK_COLOR if success else BAD_COLOR)
        if self._quit_after_training:
            self._quit_after_training = False
            self.close()
            return
        if not success and not self._closing:
            QMessageBox.warning(self, APP_NAME, f"Training failed:\n\n{message}")

    def _delete_models(self) -> None:
        if self.training_active:
            return
        answer = QMessageBox.question(
            self, APP_NAME,
            "Delete the trained regression models?\n\nOnly the saved model file is removed "
            "(face_landmarker.task and the dataset are not touched). You will have to train again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = delete_model_bundle(MODELS_DIR)
        except ModelStoreError as exc:
            self._show_error(str(exc))
            return
        self._refresh_model_status()
        if self.video_thread is not None:
            self.video_thread.models_changed()
        self.train_status.setStyleSheet("")
        self.train_status.setText("Trained models deleted." if removed else "There were no trained models.")

    def _refresh_model_status(self) -> None:
        status = inspect_models(MODELS_DIR)
        if status.valid:
            meta = status.metadata
            mae = meta.get("validation_mae_deg") or {}
            text = f"✔ Trained: {MODEL_LABELS.get(status.model_type, status.model_type)}"
            if "num_samples" in meta:
                text += f" · {meta['num_samples']} samples"
            if mae:
                text += " · MAE " + " / ".join(f"{mae.get(k, float('nan')):.1f}°" for k in ("pitch", "yaw", "roll"))
            self.model_status_label.setStyleSheet(OK_COLOR)
        elif bundle_path(MODELS_DIR).exists():
            text = f"✖ Saved models unusable ({status.message}). Delete and train again."
            self.model_status_label.setStyleSheet(BAD_COLOR)
        else:
            text = "No trained models - press Train to enable angle prediction."
            self.model_status_label.setStyleSheet("")
        self.model_status_label.setText(text)
        self.delete_button.setEnabled(bundle_path(MODELS_DIR).exists() and not self.training_active)

    # ------------------------------------------------------------------- defaults
    def _reset_defaults(self) -> None:
        for slider, value in ((self.yaw_slider, DEFAULT_YAW_RANGE), (self.pitch_slider, DEFAULT_PITCH_RANGE),
                              (self.smooth_slider, DEFAULT_SMOOTHING)):
            slider.setValue(value, emit=False)
        for check, value in ((self.invert_x, False), (self.invert_y, False), (self.show_trail, DEFAULT_SHOW_TRAIL),
                             (self.mirror_check, DEFAULT_MIRROR), (self.landmarks_check, DEFAULT_SHOW_LANDMARKS),
                             (self.angles_check, DEFAULT_SHOW_ANGLES)):
            check.blockSignals(True)
            check.setChecked(value)
            check.blockSignals(False)
        self.camera_spin.blockSignals(True)
        self.camera_spin.setValue(DEFAULT_CAMERA_INDEX)
        self.camera_spin.blockSignals(False)
        if not self.training_active:
            self.model_combo.setCurrentIndex(self.model_combo.findData(DEFAULT_MODEL_TYPE))
            self.dataset_edit.setText(str(DEFAULT_DATASET_PATH))
        self.current_theme = DEFAULT_THEME
        self._apply_theme()
        self.pointer_view.set_show_trail(DEFAULT_SHOW_TRAIL)
        self.pointer_view.clear_trail()
        if self.video_thread is not None:
            self.video_thread.set_pointer_settings(self._pointer_settings())
            self.video_thread.set_display_settings(self._display_settings())
            self.video_thread.set_camera_index(DEFAULT_CAMERA_INDEX)
            self.video_thread.clear_calibration()
            self.video_thread.reset_pointer()
        self._save_settings()
        self.source_label.setText("Default settings restored (trained models are kept)")

    # -------------------------------------------------------------------- theme
    def _toggle_theme(self) -> None:
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self._apply_theme()

    def _apply_theme(self) -> None:
        dark = self.current_theme == "dark"
        self.setStyleSheet(DARK_STYLE if dark else LIGHT_STYLE)
        self.theme_button.setText("☼ Light Mode" if dark else "◐ Dark Mode")
        self.pointer_view.set_theme(dark)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_button.setText("Full Screen")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("Exit Full Screen")

    # ----------------------------------------------------------------- display
    @staticmethod
    def _to_qimage(frame) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        return QImage(rgb.data, width, height, width * channels, QImage.Format.Format_RGB888).copy()

    def _set_metric(self, key: str, text: str, style: str = "") -> None:
        label = self.metrics[key]
        if label.text() != text:
            label.setText(text)
        if label.styleSheet() != style:
            label.setStyleSheet(style)

    def _update_frame(self, canvas, result: FrameResult) -> None:
        if self._closing:
            return
        self.camera_label.set_frame(self._to_qimage(canvas))
        if result.pointer is not None:
            self.pointer_view.set_state(result.pointer)
        if self.status_label.text() != result.status:
            self.status_label.setText(result.status or "Ready")
        self._set_metric("face", "Face: yes" if result.face_detected else "Face: no",
                         OK_COLOR if result.face_detected else "")
        names = ("pitch", "yaw", "roll")
        for i, name in enumerate(names):
            value = f"{result.angles[i]:+.1f}°" if result.angles else "--"
            self._set_metric(name, f"{name.capitalize()}: {value}")
        self._set_metric("fps", f"FPS: {result.fps:.1f}" if result.fps > 0 else "FPS: --")
        if self.video_thread is not None:
            self.video_thread.frame_consumed()

    # ----------------------------------------------------------------- settings
    def _restore_settings(self) -> None:
        s = self.settings
        try:
            self.camera_spin.setValue(s.value("camera_index", DEFAULT_CAMERA_INDEX, type=int))
            pointer = validated_pointer_settings(
                s.value("pointer/yaw_range", DEFAULT_YAW_RANGE), s.value("pointer/pitch_range", DEFAULT_PITCH_RANGE),
                s.value("pointer/smoothing_ms", DEFAULT_SMOOTHING), s.value("pointer/invert_x", False),
                s.value("pointer/invert_y", False), s.value("pointer/show_trail", DEFAULT_SHOW_TRAIL))
            display = validated_display_settings(
                s.value("display/mirror", DEFAULT_MIRROR), s.value("display/landmarks", DEFAULT_SHOW_LANDMARKS),
                s.value("display/angles", DEFAULT_SHOW_ANGLES))
        except (TypeError, ValueError):
            pointer, display = validated_pointer_settings(), validated_display_settings()
            self.source_label.setText("Saved settings were invalid - defaults restored")
        self.yaw_slider.setValue(int(pointer.yaw_range), emit=False)
        self.pitch_slider.setValue(int(pointer.pitch_range), emit=False)
        self.smooth_slider.setValue(pointer.smoothing_ms, emit=False)
        for check, value in ((self.invert_x, pointer.invert_x), (self.invert_y, pointer.invert_y),
                             (self.show_trail, pointer.show_trail), (self.mirror_check, display.mirror),
                             (self.landmarks_check, display.show_landmarks), (self.angles_check, display.show_angles)):
            check.blockSignals(True)
            check.setChecked(value)
            check.blockSignals(False)
        self.pointer_view.set_show_trail(pointer.show_trail)
        dataset = str(s.value("dataset_path", str(DEFAULT_DATASET_PATH), type=str)) or str(DEFAULT_DATASET_PATH)
        self.dataset_edit.setText(dataset)
        index = self.model_combo.findData(str(s.value("model_type", DEFAULT_MODEL_TYPE, type=str)))
        self.model_combo.setCurrentIndex(index if index >= 0 else self.model_combo.findData(DEFAULT_MODEL_TYPE))
        if self.current_theme not in ("dark", "light"):
            self.current_theme = DEFAULT_THEME
        geometry, splitter_state = s.value("window/geometry"), s.value("window/splitter")
        if geometry:
            self.restoreGeometry(geometry)
        if splitter_state:
            self.splitter.restoreState(splitter_state)

    def _save_settings(self) -> None:
        s, p, d = self.settings, self._pointer_settings(), self._display_settings()
        s.setValue("theme", self.current_theme)
        s.setValue("camera_index", self.camera_spin.value())
        s.setValue("pointer/yaw_range", p.yaw_range)
        s.setValue("pointer/pitch_range", p.pitch_range)
        s.setValue("pointer/smoothing_ms", p.smoothing_ms)
        s.setValue("pointer/invert_x", p.invert_x)
        s.setValue("pointer/invert_y", p.invert_y)
        s.setValue("pointer/show_trail", p.show_trail)
        s.setValue("display/mirror", d.mirror)
        s.setValue("display/landmarks", d.show_landmarks)
        s.setValue("display/angles", d.show_angles)
        s.setValue("dataset_path", self.dataset_edit.text())
        s.setValue("model_type", self.model_combo.currentData())
        s.setValue("last_dir", self._last_dir)
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/splitter", self.splitter.saveState())
        s.sync()

    def closeEvent(self, event) -> None:
        if self.training_active:
            if self._quit_after_training:
                event.ignore()
                return
            answer = QMessageBox.question(
                self, APP_NAME,
                "Training is still running and cannot be interrupted safely.\n\n"
                "Wait for it to finish and then quit? (Existing models are never damaged.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if answer == QMessageBox.StandardButton.Yes:
                self._quit_after_training = True
                self.train_status.setText("Finishing training - the application will close automatically...")
            event.ignore()
            return
        self._closing = True
        self._save_settings()
        self._stop_video_thread()
        event.accept()
