"""AIR GUITAR.

Left hand : raise 0-5 fingers to choose a chord.
Right hand: sweep your hand DOWN or UP to strum (STRUM_MODE='gesture'),
            or cross the on-screen strings with your fingertips (STRUM_MODE='strings').
Keys      : 1-6 force a chord, 0 = back to hand control, Q/ESC = quit.
"""
import logging
import sys
import time
import cv2
import config
from camera import Camera, CameraError
from hand_tracker import HandTracker, FingerMotion, draw_hand
from guitar import (CHORD_ORDER, CHORDS, STRING_NAMES, ChordStabilizer, Plucker,
                    string_note, strum_notes)
from strum_detect import PalmMotion, StrokeGate, load_classifier
from audio import AudioEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
WIN = "AIR GUITAR"


def draw_strings(frame, plucker, flash, t):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = plucker.zone
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (90, 90, 90), 1)
    for i in range(plucker.n):
        y = int(plucker.string_y(i) * h)
        hot = t - flash[i] < 0.15
        color = (0, 255, 255) if hot else (230, 230, 230)
        cv2.line(frame, (int(x0 * w), y), (int(x1 * w), y), color, 4 - i // 2 + (2 if hot else 0))
        cv2.putText(frame, STRING_NAMES[i], (int(x0 * w) - 18, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    gesture = config.STRUM_MODE == "gesture"
    cam, tracker, audio = Camera(), None, None
    motion, plucker, palm = FingerMotion(), Plucker(), PalmMotion()
    classify, det_name = load_classifier()
    gate, stabilizer = StrokeGate(classify), ChordStabilizer()
    try:
        cam.open()
        audio = AudioEngine()
        audio.warm_up({string_note(c, s) for c in CHORDS for s in range(6)} - {None})
        tracker = HandTracker()
    except CameraError as e:
        print("\nCAMERA ERROR:\n" + str(e))
        return 1

    manual, last_ev = None, None
    flash = [-1e9] * config.NUM_STRINGS
    fps, last = 0.0, time.perf_counter()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(t - last, 1e-6)
            last = t

            hands = tracker.process(frame)
            speeds = motion.update(hands, t)

            ch = hands.get(config.CHORD_HAND)
            chord = stabilizer.update(ch.finger_count if ch else None)
            if manual:
                chord = manual

            sh = hands.get(config.STRUM_HAND)
            if sh and gesture:
                feat = palm.update(sh.pts, t)
                ev = gate.update(feat, t) if feat is not None else None
                if ev and ev.direction in config.STRUM_DIRECTIONS:
                    audio.strum(strum_notes(chord, ev.direction), ev.velocity)
                    last_ev = (ev, t)
            elif sh:
                tips = {f: sh.tip(f) for f in ("Index", "Middle", "Ring", "Pinky")}
                for p in plucker.update(tips, t):
                    note = string_note(chord, p.string)
                    flash[p.string] = t
                    if note is not None:
                        audio.play(note, p.velocity)
            else:
                plucker.reset()
                palm.reset()
                gate.reset()

            # ---- drawing ----
            for hand in hands.values():
                draw_hand(frame, hand)
            h, w = frame.shape[:2]
            if not gesture:
                draw_strings(frame, plucker, flash, t)
            elif last_ev and t - last_ev[1] < 0.4:
                ev = last_ev[0]
                cv2.putText(frame, ev.direction.upper(), (w - 170, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
                cv2.rectangle(frame, (w - 170, 85), (w - 170 + int(150 * ev.velocity), 100), (0, 255, 255), -1)
            cv2.putText(frame, f"Chord: {chord}" + ("  (manual)" if manual else ""),
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"{fps:.0f} FPS | {config.STRUM_MODE} | {det_name}", (w - 330, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y = 62
            for label, hand in hands.items():
                up = [f for f, e in hand.extended.items() if e]
                mv = FingerMotion.moving(speeds[label])
                cv2.putText(frame, f"{label}: up={','.join(up) or '-'}  moving={','.join(mv) or '-'}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 22
            cv2.putText(frame, "1-6 chord | 0 hand control | Q/ESC quit", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(WIN, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("6"):
                manual = CHORD_ORDER[key - ord("1")]
            elif key == ord("0"):
                manual = None
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
    except CameraError as e:
        print("\nCAMERA ERROR:\n" + str(e))
        return 1
    finally:
        if tracker:
            tracker.close()
        if audio:
            audio.close()
        cam.release()
        cv2.destroyAllWindows()
    print("Air Guitar closed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
