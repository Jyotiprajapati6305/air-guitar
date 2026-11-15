"""Builds strum_model.npz from SIMULATED strums (varied speed, direction, camera FPS,
jitter) plus non-strum motion. Run `python make_default_model.py` to regenerate it.
Recording your own strums with record_strums.py will fit you better."""
import numpy as np
import config
from strum_detect import PalmMotion, StrokeGate, NumpyMLP, label_frame, NONE
from train_strum_model import train


def simulate(segments, rng, fps=30.0, noise=0.003):
    """segments: [(seconds, dx, dy)] smooth moves -> [(t, x, y)] with jittery frame timing."""
    x, y, t, out = 0.6, 0.5, 0.0, []
    for dur, dx, dy in segments:
        x0, y0, t0, tc = x, y, t, 0.0
        while tc < dur:
            tc += max(0.004, rng.normal(1 / fps, 0.12 / fps))
            s = min(1.0, tc / dur)
            s = s * s * (3 - 2 * s)
            out.append((t0 + tc, x0 + dx * s + rng.normal(0, noise), y0 + dy * s + rng.normal(0, noise)))
        x, y, t = x0 + dx, y0 + dy, t0 + tc
    return out


def stroke_session(rng, kind):
    """Strokes of random speed/length, each followed by a slow return."""
    sign = 1 if kind == "down" else -1
    segs = []
    for _ in range(rng.integers(4, 9)):
        segs.append((rng.uniform(0.15, 0.9), rng.normal(0, 0.02), rng.normal(0, 0.02)))          # pause
        amp = rng.uniform(0.18, 0.5)
        segs.append((rng.uniform(0.08, 0.30), rng.normal(0, 0.08), sign * amp))                  # stroke
        segs.append((rng.uniform(0.5, 1.3), rng.normal(0, 0.05), -sign * amp))                   # slow return
    return segs


def idle_session(rng):
    """Motion that must NOT strum: slow drifts, fast sideways moves, hovering, small wobble."""
    segs = []
    for _ in range(rng.integers(5, 10)):
        pick = rng.integers(0, 4)
        if pick == 0:
            segs.append((rng.uniform(1, 3), rng.normal(0, 0.1), rng.normal(0, 0.1)))      # slow drift
        elif pick == 1:
            segs.append((rng.uniform(0.12, 0.3), rng.uniform(-0.5, 0.5), rng.normal(0, 0.03)))  # sideways
        elif pick == 2:
            segs.append((rng.uniform(0.5, 1.5), 0.0, 0.0))                                # hover
        else:
            segs.append((rng.uniform(0.2, 0.5), rng.normal(0, 0.03), rng.uniform(-0.12, 0.12)))  # wobble
    return segs


def features(traj):
    palm, out = PalmMotion(), []
    for t, x, y in traj:
        f = palm.update(np.tile([x / 1.3333, y], (21, 1)).astype(np.float32), t)
        out.append(f)
    return out


def build_dataset(rng, per_class=70):
    X, y = [], []
    for session in ("down", "up", "idle"):
        for _ in range(per_class):
            segs = idle_session(rng) if session == "idle" else stroke_session(rng, session)
            traj = simulate(segs, rng, fps=rng.uniform(22, 60), noise=rng.uniform(0.001, 0.006))
            for f in features(traj):
                if f is not None:
                    X.append(f)
                    y.append(label_frame(f, session))
    return np.array(X), np.array(y)


def event_check(model, rng, n=60):
    """Fresh sequences with a known number of strokes: strokes caught vs false strums."""
    caught = expected = wrong = 0
    for i in range(n):
        kind = "down" if i % 2 == 0 else "up"
        segs = stroke_session(rng, kind)
        expected += sum(1 for k, sg in enumerate(segs) if k % 3 == 1)
        traj = simulate(segs, rng, fps=rng.uniform(22, 60), noise=rng.uniform(0.001, 0.006))
        gate = StrokeGate(model)
        for (t, x, y), f in zip(traj, features(traj)):
            ev = gate.update(f, t) if f is not None else None
            if ev:
                caught += ev.direction == kind
                wrong += ev.direction != kind
    return caught, expected, wrong


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X, y = build_dataset(rng)
    print("Frames  idle/down/up:", np.bincount(y, minlength=3).tolist())
    model, acc, cm = train(X, y)
    print(f"Frame accuracy on held-out simulated data: {acc:.1%}\n{cm}")
    caught, expected, wrong = event_check(model, np.random.default_rng(7))
    print(f"Fresh strum sequences: caught {caught}/{expected} strokes, {wrong} wrong-direction/false strums")
    model.save(config.MODEL_PATH)
    print("Saved", config.MODEL_PATH)
