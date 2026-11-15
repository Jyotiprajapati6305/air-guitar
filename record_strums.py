"""Record YOUR strums to train the strum model.

Keys:  1 = DOWN session   2 = UP session   3 = IDLE session
       SPACE = start/stop recording        Q/ESC = quit
How to record (about 1-2 minutes each is plenty):
  DOWN: strum down, return your hand SLOWLY, repeat. Vary speed (soft and hard).
  UP:   strum up, return slowly, repeat.
  IDLE: do NOT strum: wiggle your hand, move it slowly, change chords, hold still.
"""
import csv
import time
import cv2
import config
from camera import Camera, CameraError
from hand_tracker import HandTracker, draw_hand
from strum_detect import PalmMotion, label_frame

SESSIONS = {ord("1"): "down", ord("2"): "up", ord("3"): "idle"}


def main():
    cam, tracker, palm = Camera(), None, PalmMotion()
    try:
        cam.open()
        tracker = HandTracker()
    except CameraError as e:
        print(e)
        return 1
    session, recording, rows, counts = "down", False, [], {"down": 0, "up": 0, "idle": 0}
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            t = time.perf_counter()
            hand = tracker.process(frame).get(config.STRUM_HAND)
            if hand:
                draw_hand(frame, hand)
                feat = palm.update(hand.pts, t)
                if recording and feat is not None:
                    rows.append([*map(float, feat), label_frame(feat, session)])
                    counts[session] += 1
            else:
                palm.reset()
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(frame, f"Session: {session.upper()}  {'REC' if recording else 'paused'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"frames  down:{counts['down']} up:{counts['up']} idle:{counts['idle']}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, "1/2/3 session | SPACE rec | Q quit+save", (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Record strums", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in SESSIONS:
                session, recording = SESSIONS[key], False
            elif key == 32:
                recording = not recording
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        if rows:
            with open(config.DATA_PATH, "a", newline="") as f:
                csv.writer(f).writerows(rows)
            print(f"Saved {len(rows)} rows to {config.DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
