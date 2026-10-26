"""Fixed infrastructure: maps head angles to a virtual pointer (no Qt, easy to test).

Conventions (angles in degrees, dataset convention - verified on the AFLW2000 data):
  yaw   > 0 : the user turns their head to THEIR right  -> pointer moves right
  pitch > 0 : the user looks up                          -> pointer moves up
  roll  > 0 : the user tilts the head to THEIR left shoulder -> pointer rotates
              counter-clockwise on screen (like a mirror)
Pointer position: x in [-1, 1] (right = +), y in [-1, 1] (up = +).
Smoothing: exponential moving average with a time constant (ms), so the
behaviour does not depend on the frame rate.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from config import (
    HOLD_SECONDS,
    MAX_PLAUSIBLE_ANGLE_DEG,
    MAX_ROLL_DISPLAY_DEG,
    PointerSettings,
)

TRACKING = "tracking"  # a valid prediction for the current frame
HOLDING = "holding"  # no valid prediction for a moment: pointer frozen
LOST = "lost"  # no valid prediction for a while: pointer frozen, shown as lost
WAITING = "waiting"  # nothing valid received yet


@dataclass(frozen=True)
class PointerState:
    x: float = 0.0
    y: float = 0.0
    roll_deg: float = 0.0  # calibrated head roll (dataset convention)
    status: str = WAITING
    calibrated: bool = False


def clean_angles(angles) -> tuple[float, float, float] | None:
    """Return (pitch, yaw, roll) as floats, or None if anything is invalid."""
    try:
        values = tuple(float(a) for a in angles)
    except (TypeError, ValueError):
        return None
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        return None
    if any(abs(v) > MAX_PLAUSIBLE_ANGLE_DEG for v in values):
        return None
    return values


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class PointerController:
    def __init__(self, settings: PointerSettings | None = None):
        self.settings = settings or PointerSettings()
        self._center: tuple[float, float, float] | None = None
        self._last_angles: tuple[float, float, float] | None = None
        self._x = self._y = self._roll = 0.0
        self._last_valid_time: float | None = None
        self._last_update_time: float | None = None
        self._status = WAITING
        self._snap_next = True

    def set_settings(self, settings: PointerSettings) -> None:
        self.settings = settings

    @property
    def calibrated(self) -> bool:
        return self._center is not None

    def calibrate(self) -> tuple[bool, str]:
        """Use the current pose as the neutral pose. Refused without a live valid face."""
        if self._status != TRACKING or self._last_angles is None:
            return False, "Cannot set the center: no valid face/prediction right now."
        self._center = self._last_angles
        return True, "Center set to the current head pose."

    def clear_calibration(self) -> None:
        self._center = None

    def reset(self) -> None:
        """Put the pointer back in the middle; the next valid frame jumps (no glide)."""
        self._x = self._y = self._roll = 0.0
        self._snap_next = True

    def _state(self) -> PointerState:
        return PointerState(self._x, self._y, self._roll, self._status, self.calibrated)

    def update(self, angles, now: float | None = None, snap: bool = False) -> PointerState:
        """Feed the prediction of the CURRENT frame (or None). Returns the new state."""
        now = time.monotonic() if now is None else now
        valid = clean_angles(angles) if angles is not None else None
        if valid is None:
            self._last_angles = None
            if self._last_valid_time is None:
                self._status = WAITING
            elif now - self._last_valid_time <= HOLD_SECONDS:
                self._status = HOLDING
            else:
                self._status = LOST
                self._snap_next = True  # do not glide from a stale position
            self._last_update_time = now
            return self._state()

        pitch0, yaw0, roll0 = self._center if self._center else (0.0, 0.0, 0.0)
        pitch, yaw, roll = valid
        s = self.settings
        target_x = _clamp((yaw - yaw0) / s.yaw_range, 1.0)
        target_y = _clamp((pitch - pitch0) / s.pitch_range, 1.0)
        if s.invert_x:
            target_x = -target_x
        if s.invert_y:
            target_y = -target_y
        target_roll = _clamp(roll - roll0, MAX_ROLL_DISPLAY_DEG)

        tau = s.smoothing_ms / 1000.0
        dt = 0.0 if self._last_update_time is None else min(max(now - self._last_update_time, 0.0), 0.5)
        if snap or self._snap_next or tau <= 0.0:
            alpha = 1.0
        else:
            alpha = 1.0 - math.exp(-dt / tau)
        self._x += alpha * (target_x - self._x)
        self._y += alpha * (target_y - self._y)
        self._roll += alpha * (target_roll - self._roll)

        self._snap_next = False
        self._last_angles = valid
        self._last_valid_time = now
        self._last_update_time = now
        self._status = TRACKING
        return self._state()
