"""Hand detection (MediaPipe) and per-finger state (extended / moving)."""
import cv2
import numpy as np
import mediapipe as mp
import config

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
TIPS = (4, 8, 12, 16, 20)
PIPS = (3, 6, 10, 14, 18)   # thumb uses its IP joint (3)
WRIST, PINKY_MCP = 0, 17


class Hand:
    """One detected hand. pts is a (21, 2) array of normalized x, y."""

    def __init__(self, label, pts):
        self.label = label
        self.pts = pts
        self.extended = self._extended_fingers()

    def _extended_fingers(self):
        p = self.pts
        d = lambda a, b: float(np.linalg.norm(p[a] - p[b]))
        out = {"Thumb": d(4, PINKY_MCP) > d(3, PINKY_MCP)}
        for name, tip, pip in zip(FINGERS[1:], TIPS[1:], PIPS[1:]):
            # Rotation-proof: a straight finger's tip is farther from the wrist
            # than its middle joint; a curled finger's tip folds back.
            out[name] = d(tip, WRIST) > d(pip, WRIST)
        return out

    @property
    def finger_count(self):
        return sum(self.extended.values())

    def tip(self, name):
        return tuple(self.pts[TIPS[FINGERS.index(name)]])


class HandTracker:
    def __init__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_HANDS,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr):
        """Return a dict {'Left': Hand, 'Right': Hand} for hands in the frame.
        The frame must already be mirrored (Camera does this)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return {}
        found = []
        for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
            pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
            found.append((hd.classification[0].label, pts))
        # If MediaPipe gives both hands the same label, split by screen position.
        if len(found) == 2 and found[0][0] == found[1][0]:
            found.sort(key=lambda f: f[1][WRIST][0])
            found = [("Left", found[0][1]), ("Right", found[1][1])]
        return {label: Hand(label, pts) for label, pts in found}

    def close(self):
        self._hands.close()


class FingerMotion:
    """Works out which fingertips are moving, by comparing to the last frame."""

    def __init__(self):
        self._prev = {}   # (label, finger) -> (x, y, t)

    def update(self, hands, t):
        """Return {label: {finger: speed}} with speed in frame-widths/second."""
        speeds, seen = {}, set()
        for label, hand in hands.items():
            speeds[label] = {}
            for f in FINGERS:
                x, y = hand.tip(f)
                key = (label, f)
                seen.add(key)
                prev = self._prev.get(key)
                if prev is not None and t > prev[2]:
                    speeds[label][f] = float(np.hypot(x - prev[0], y - prev[1]) / (t - prev[2]))
                else:
                    speeds[label][f] = 0.0
                self._prev[key] = (x, y, t)
        for key in list(self._prev):
            if key not in seen:
                del self._prev[key]
        return speeds

    @staticmethod
    def moving(speeds_for_hand):
        return [f for f, s in speeds_for_hand.items() if s >= config.MOVING_SPEED]


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    px = (hand.pts * (w, h)).astype(int)
    for a, b in mp.solutions.hands.HAND_CONNECTIONS:
        cv2.line(frame, tuple(px[a]), tuple(px[b]), (0, 200, 0), 2)
    for i, (x, y) in enumerate(px):
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255) if i in TIPS else (255, 255, 255), -1)
