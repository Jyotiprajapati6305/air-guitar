"""Prints installed library versions and checks the MediaPipe hands API exists."""
import sys
print("Python:", sys.version.split()[0])
for name in ("cv2", "numpy", "mediapipe", "pygame"):
    try:
        mod = __import__(name)
        print(f"{name:10s} OK  {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"{name:10s} FAILED: {e}")
try:
    import mediapipe as mp
    mp.solutions.hands.Hands
    print("MediaPipe Hands API: OK")
except Exception as e:
    print("MediaPipe Hands API: FAILED:", e)
