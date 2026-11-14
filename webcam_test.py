"""Phase 1: verify the webcam works. Press Q or ESC to quit."""
import logging
import sys
import time
import cv2
from camera import Camera, CameraError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    cam = Camera()
    try:
        cam.open()
    except CameraError as e:
        print("\nCAMERA ERROR:\n" + str(e))
        return 1

    last, fps = time.perf_counter(), 0.0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue  # temporary failure, try again
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            h, w = frame.shape[:2]
            cv2.putText(frame, f"{w}x{h}  ~{fps:.0f} FPS (measured)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q or ESC to quit", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("AIR GUITAR - webcam test", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty("AIR GUITAR - webcam test", cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
    except CameraError as e:
        print("\nCAMERA ERROR:\n" + str(e))
        return 1
    finally:
        cam.release()
        cv2.destroyAllWindows()
    print("Webcam test finished cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
