"""Fixed infrastructure: turns head-pose angles into a smoothed pointer position.

Conventions (angles in degrees, verified on the AFLW2000 labels):
    positive yaw   = head turned toward the user's own right   -> pointer moves right
    positive pitch = head tilted up                            -> pointer moves up
    positive roll  = head tilted toward the user's own left    -> pointer rotates counter-clockwise
Smoothing is an exponential moving average: smoothed += (1 - s) * (new - smoothed), where s is
the Smoothing setting (0 = raw, 0.95 = very steady but slow).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import config


@dataclass
class PointerSettings:
    yaw_range: float = config.DEFAULT_YAW_RANGE
    pitch_range: float = config.DEFAULT_PITCH_RANGE
    smoothing: float = config.DEFAULT_SMOOTHING / 100.0
    invert_x: bool = config.DEFAULT_INVERT_X
    invert_y: bool = config.DEFAULT_INVERT_Y


def _clip(value, low, high):
    return max(low, min(high, value))


class PointerMapper:
    def __init__(self, settings=None):
        self.settings = settings or PointerSettings()
        self.center = (0.0, 0.0, 0.0)          # (pitch, yaw, roll) neutral pose
        self.center_is_set = False
        self._smoothed = None                  # (pitch, yaw, roll) or None
        self._missed = 0
        self._position = (0.0, 0.0)            # x, y in [-1, 1]; +x right, +y DOWN (screen)
        self._rotation = 0.0                   # degrees, clockwise on screen

    # -- state -------------------------------------------------------------
    @property
    def position(self):
        return self._position

    @property
    def rotation(self):
        return self._rotation

    @property
    def smoothed_angles(self):
        return self._smoothed

    def reset_smoothing(self):
        self._smoothed = None
        self._missed = 0

    def reset(self):
        self.reset_smoothing()
        self._position = (0.0, 0.0)
        self._rotation = 0.0

    # -- calibration -------------------------------------------------------
    def set_center(self):
        """Use the current pose as neutral. Returns False if there is no valid current pose."""
        if self._smoothed is None or self._missed > 0:
            return False
        self.center = tuple(self._smoothed)
        self.center_is_set = True
        self._recompute()
        return True

    def clear_center(self):
        self.center = (0.0, 0.0, 0.0)
        self.center_is_set = False
        self._recompute()

    # -- per-frame update --------------------------------------------------
    def update(self, angles):
        """angles = (pitch, yaw, roll) in degrees, or None when no valid pose. Returns True if used."""
        if angles is None or len(angles) != 3 or not all(math.isfinite(a) for a in angles):
            self._missed += 1
            if self._missed >= config.FACE_LOST_RESET_FRAMES:
                self._smoothed = None
            return False

        self._missed = 0
        if self._smoothed is None:
            self._smoothed = tuple(float(a) for a in angles)
        else:
            keep = _clip(self.settings.smoothing, 0.0, 0.95)
            self._smoothed = tuple(old + (1.0 - keep) * (new - old) for old, new in zip(self._smoothed, angles))
        self._recompute()
        return True

    def refresh(self):
        """Re-apply the current settings (call after a range or invert change)."""
        self._recompute()

    def _recompute(self):
        if self._smoothed is None:
            return
        pitch, yaw, roll = (value - ref for value, ref in zip(self._smoothed, self.center))
        x = yaw / max(self.settings.yaw_range, 1e-6)
        y_up = pitch / max(self.settings.pitch_range, 1e-6)
        if self.settings.invert_x:
            x = -x
        if self.settings.invert_y:
            y_up = -y_up
        self._position = (_clip(x, -1.0, 1.0), _clip(-y_up, -1.0, 1.0))
        self._rotation = _clip(-roll, -90.0, 90.0)
