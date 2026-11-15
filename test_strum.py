import numpy as np
from strum_detect import PalmMotion, StrokeGate, rule_classify, label_frame, ModelClassifier
from guitar import ChordStabilizer, strum_notes

FPS = 30


def trajectory(segments, noise=0.0, seed=1):
    """segments: [(seconds, dx, dy)] smooth moves. Returns [(t, x, y)]."""
    rng = np.random.default_rng(seed)
    x, y, t, out = 0.6, 0.5, 0.0, []
    for dur, dx, dy in segments:
        n = max(1, int(dur * FPS))
        for k in range(1, n + 1):
            s = k / n
            s = s * s * (3 - 2 * s)                       # smoothstep
            out.append((t + k / FPS, x + dx * s + rng.normal(0, noise), y + dy * s + rng.normal(0, noise)))
        x, y, t = x + dx, y + dy, t + dur
    return out


def run(traj, classify=rule_classify):
    palm, gate, events = PalmMotion(), StrokeGate(classify), []
    for t, x, y in traj:
        feat = palm.update(np.tile([x / 1.3333, y], (21, 1)).astype(np.float32), t)
        if feat is not None:
            ev = gate.update(feat, t)
            if ev:
                events.append(ev.direction)
    return events


HOLD = (0.5, 0, 0)
DOWN = (0.18, 0, 0.35)
UP = (0.18, 0, -0.35)
SLOW_BACK = (0.8, 0, -0.35)


def test_single_downstroke():
    assert run(trajectory([HOLD, DOWN, HOLD])) == ["down"]


def test_repeated_downstrokes_with_slow_return():
    assert run(trajectory([HOLD, DOWN, SLOW_BACK] * 3 + [HOLD])) == ["down"] * 3


def test_down_then_fast_up():
    assert run(trajectory([HOLD, DOWN, (0.1, 0, 0), UP, HOLD])) == ["down", "up"]


def test_jitter_and_drift_make_no_sound():
    assert run(trajectory([(3.0, 0, 0)], noise=0.004)) == []
    assert run(trajectory([(2.0, 0, 0.3)], noise=0.003)) == []        # slow drift
    assert run(trajectory([HOLD, (0.18, 0.4, 0), HOLD])) == []        # fast sideways


def test_soft_strum_quieter_than_hard():
    def vel(dur):
        palm, gate = PalmMotion(), StrokeGate(rule_classify)
        for t, x, y in trajectory([HOLD, (dur, 0, 0.35), HOLD]):
            f = palm.update(np.tile([x / 1.3333, y], (21, 1)).astype(np.float32), t)
            ev = gate.update(f, t) if f is not None else None
            if ev:
                return ev.velocity
    assert vel(0.12) > vel(0.3)


def test_model_pipeline_trains_and_detects():
    from train_strum_model import train
    X, y = [], []
    for session, segs in (("down", [HOLD, DOWN, SLOW_BACK] * 6), ("up", [HOLD, UP, (0.8, 0, 0.35)] * 6),
                          ("idle", [(4, 0, 0.1), (0.5, 0.2, 0), (4, 0, 0)])):
        palm = PalmMotion()
        for t, x, yy in trajectory(segs, noise=0.003):
            f = palm.update(np.tile([x / 1.3333, yy], (21, 1)).astype(np.float32), t)
            if f is not None:
                X.append(f)
                y.append(label_frame(f, session))
    model, acc, _ = train(np.array(X), np.array(y))
    assert acc > 0.9
    assert run(trajectory([HOLD, DOWN, SLOW_BACK, HOLD]), ModelClassifier(model)) == ["down"]


def test_chord_stabilizer_ignores_single_bad_frame():
    s = ChordStabilizer(confirm=4)
    for _ in range(5):
        s.update(1)
    assert s.chord == "G"
    s.update(3)                       # one misread frame
    assert s.chord == "G"
    for _ in range(4):
        s.update(3)
    assert s.chord == "D"


def test_strum_order():
    assert strum_notes("Em", "down")[0] == 40 and strum_notes("Em", "up")[0] == 64
    assert len(strum_notes("C", "down")) == 5          # low E muted
