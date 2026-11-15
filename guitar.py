"""Guitar simulation: chords from raised fingers, notes from strings crossed.
Pure Python (no camera/audio), so it is easy to test."""
from dataclasses import dataclass
import config

STRING_NAMES = ("E", "A", "D", "G", "B", "e")        # low -> high (top -> bottom on screen)
OPEN_MIDI = (40, 45, 50, 55, 59, 64)                 # E2 A2 D3 G3 B3 E4

# Fret per string (low E -> high e); -1 = muted string.
CHORDS = {
    "Em": (0, 2, 2, 0, 0, 0),
    "G":  (3, 2, 0, 0, 0, 3),
    "C":  (-1, 3, 2, 0, 1, 0),
    "D":  (-1, -1, 0, 2, 3, 2),
    "Am": (-1, 0, 2, 2, 1, 0),
    "A":  (-1, 0, 2, 2, 2, 0),
}
CHORD_ORDER = tuple(CHORDS)          # raised-finger count 0..5 -> chord


def chord_for_count(n):
    return CHORD_ORDER[max(0, min(n, len(CHORD_ORDER) - 1))]


def string_note(chord, string):
    """MIDI note for a string in a chord, or None if that string is muted."""
    fret = CHORDS[chord][string]
    return None if fret < 0 else OPEN_MIDI[string] + fret


@dataclass
class Pluck:
    string: int
    velocity: float   # 0..1
    finger: str


class Plucker:
    """Virtual strings are horizontal lines inside the strum zone.
    A fingertip crossing a line fast enough plucks that string."""

    def __init__(self, zone=None, n=None):
        self.zone = zone or config.STRUM_ZONE
        self.n = n or config.NUM_STRINGS
        self._prev = {}                       # finger -> (x, y, t)
        self._last = [-1e9] * self.n          # last pluck time per string

    def string_y(self, i):
        _, y0, _, y1 = self.zone
        return y0 + (y1 - y0) * i / (self.n - 1)

    def reset(self):
        self._prev.clear()

    def update(self, tips, t):
        """tips: {finger_name: (x, y)} normalized. Returns a list of Pluck."""
        x0, _, x1, _ = self.zone
        plucks = []
        for finger in list(self._prev):
            if finger not in tips:
                del self._prev[finger]
        for finger, (x, y) in tips.items():
            prev = self._prev.get(finger)
            self._prev[finger] = (x, y, t)
            if prev is None or not (x0 <= x <= x1) or t <= prev[2]:
                continue
            py, dt = prev[1], t - prev[2]
            speed = abs(y - py) / dt
            if speed < config.MIN_PLUCK_SPEED:
                continue
            lo, hi = (py, y) if py < y else (y, py)
            order = range(self.n) if y > py else range(self.n - 1, -1, -1)
            for i in order:
                if lo < self.string_y(i) <= hi and t - self._last[i] >= config.PLUCK_COOLDOWN:
                    span = config.MAX_PLUCK_SPEED - config.MIN_PLUCK_SPEED
                    vel = 0.2 + 0.8 * min(1.0, (speed - config.MIN_PLUCK_SPEED) / span)
                    self._last[i] = t
                    plucks.append(Pluck(i, vel, finger))
        return plucks


def strum_notes(chord, direction):
    """MIDI notes of a chord in strum order (down = low E first, up = high e first)."""
    order = range(6) if direction == "down" else range(5, -1, -1)
    return [string_note(chord, i) for i in order if string_note(chord, i) is not None]


class ChordStabilizer:
    """Only changes chord once the finger count is stable for a few frames,
    so a single misread frame does not play the wrong chord."""

    def __init__(self, confirm=None):
        self.confirm = confirm or config.CHORD_CONFIRM_FRAMES
        self.chord = CHORD_ORDER[0]
        self._cand, self._run = None, 0

    def update(self, count):
        if count is not None:
            self._run = self._run + 1 if count == self._cand else 1
            self._cand = count
            if self._run >= self.confirm:
                self.chord = chord_for_count(count)
        return self.chord
