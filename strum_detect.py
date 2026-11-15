"""Strum detection from palm motion.
Palm centre -> smoothed velocity history -> classifier (rules OR trained model)
-> StrokeGate (debounce + cooldown) -> StrumEvent."""
import logging
import os
from collections import deque
from dataclasses import dataclass
import numpy as np
import config
from smoothing import OneEuro

log = logging.getLogger(__name__)
NONE, DOWN, UP = 0, 1, 2
WINDOW = 6                          # velocity samples per feature vector (12 numbers)
PALM_IDS = [0, 5, 9, 13, 17]        # wrist + finger bases = stable palm centre
ASPECT = config.FRAME_WIDTH / config.FRAME_HEIGHT   # so x and y speeds use the same units


@dataclass
class StrumEvent:
    direction: str      # "down" | "up"
    velocity: float     # 0.3 .. 1.0


class PalmMotion:
    def __init__(self):
        self.fx, self.fy = OneEuro(), OneEuro()
        self.reset()

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self._last = None
        self.hist = deque(maxlen=WINDOW)

    def update(self, pts, t):
        """pts: (21,2) normalized landmarks. Returns a 12-number feature vector
        (vx, vy per frame, oldest first) or None while warming up."""
        c = pts[PALM_IDS].mean(axis=0)
        x, y = self.fx(float(c[0]) * ASPECT, t), self.fy(float(c[1]), t)
        if self._last is not None and t > self._last[2]:
            dt = t - self._last[2]
            vx = float(np.clip((x - self._last[0]) / dt, -8, 8))
            vy = float(np.clip((y - self._last[1]) / dt, -8, 8))
            self.hist.append((vx, vy))
        self._last = (x, y, t)
        if len(self.hist) < WINDOW:
            return None
        return np.array(self.hist, dtype=np.float32).ravel()


def rule_classify(feat, speed=None):
    """Fixed-rule classifier: fast, mostly vertical palm motion = a strum."""
    speed = config.STROKE_SPEED if speed is None else speed
    vx, vy = (feat[-2] + feat[-4]) / 2, (feat[-1] + feat[-3]) / 2
    if abs(vy) < speed or abs(vy) < 1.2 * abs(vx):
        return NONE
    return DOWN if vy > 0 else UP


def label_frame(feat, session):
    """Training label for one frame. session: 'down' | 'up' | 'idle'."""
    if session == "idle":
        return NONE
    cls = rule_classify(feat, config.STROKE_SPEED * 0.8)
    return cls if cls == {"down": DOWN, "up": UP}[session] else NONE


class ModelClassifier:
    def __init__(self, model):
        self.model = model

    def __call__(self, feat):
        return int(self.model.predict(feat.reshape(1, -1))[0])


def load_classifier():
    """Return (classify_function, name). Falls back to rules if no model."""
    if config.USE_MODEL and os.path.exists(config.MODEL_PATH):
        try:
            import joblib
            return ModelClassifier(joblib.load(config.MODEL_PATH)), "trained model"
        except Exception as e:
            log.warning("Could not load %s (%s); using rules.", config.MODEL_PATH, e)
    return rule_classify, "rules"


class StrokeGate:
    """Turns a per-frame class stream into single strum events."""

    def __init__(self, classify):
        self.classify = classify
        self.reset()

    def reset(self):
        self._cand, self._run = NONE, 0
        self._active = NONE
        self._last_event = -1e9

    def update(self, feat, t):
        cls = self.classify(feat)
        self._run = self._run + 1 if cls == self._cand else 1
        self._cand = cls
        if cls == NONE:
            if self._run >= 2:
                self._active = NONE          # stroke finished
            return None
        if (cls != self._active and self._run >= config.CONFIRM_FRAMES
                and t - self._last_event >= config.STROKE_COOLDOWN):
            self._active, self._last_event = cls, t
            peak = float(np.abs(feat[1::2][-3:]).max())
            span = config.MAX_STROKE_SPEED - config.STROKE_SPEED
            vel = 0.3 + 0.7 * min(1.0, max(0.0, (peak - config.STROKE_SPEED) / span))
            return StrumEvent("down" if cls == DOWN else "up", vel)
        return None
