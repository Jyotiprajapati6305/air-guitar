"""Central settings for AIR GUITAR. Phase 1 only needs the camera settings."""

CAMERA_INDEX = 0          # 0 = default webcam. Try 1 or 2 for external cameras.
FRAME_WIDTH = 640         # 640x480 is plenty for hand tracking and keeps FPS high
FRAME_HEIGHT = 480
MIRROR_VIEW = True        # flip horizontally so moving right looks like moving right
MAX_FAILED_FRAMES = 30    # consecutive bad reads before we give up
