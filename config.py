"""Central settings for AIR GUITAR."""

# ---- Phase 1: camera ----
CAMERA_INDEX = 0          # 0 = default webcam. Try 1 or 2 for external cameras.
FRAME_WIDTH = 640         # 640x480 is plenty for hand tracking and keeps FPS high
FRAME_HEIGHT = 480
MIRROR_VIEW = True        # flip horizontally so moving right looks like moving right
MAX_FAILED_FRAMES = 30    # consecutive bad reads before we give up

# ---- Phase 2: hand detection ----
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 0      # 0 = fastest, 1 = more accurate

# ---- Phase 3: finger tracking ----
MOVING_SPEED = 0.35       # fingertip speed (frame-widths per second) counted as "moving"

# ---- Phase 4: guitar simulation ----
# Because the view is mirrored, "Left" = your real left hand.
CHORD_HAND = "Left"       # number of raised fingers (0-5) picks the chord
STRUM_HAND = "Right"      # fingertips sweeping across the strings play notes
NUM_STRINGS = 6
# Strum zone as fractions of the frame: (x_min, y_min, x_max, y_max)
STRUM_ZONE = (0.50, 0.20, 0.98, 0.85)
MIN_PLUCK_SPEED = 0.50    # slower than this (frame-heights/sec) = no sound
MAX_PLUCK_SPEED = 3.50    # this fast or faster = full volume
PLUCK_COOLDOWN = 0.07     # seconds before the same string can sound again

# ---- Phase 5: audio ----
SAMPLE_RATE = 44100
NOTE_SECONDS = 2.5
MASTER_VOLUME = 0.8

# ---- Phase 6: accurate strumming ----
STRUM_MODE = "gesture"        # "gesture" = whole-hand stroke strums the chord
                              # "strings" = old mode, fingertips cross virtual strings
SMOOTH_MIN_CUTOFF = 1.2       # lower = smoother but laggier
SMOOTH_BETA = 2.0             # higher = less lag during fast movement
STROKE_SPEED = 0.9            # palm speed (frame-heights/sec) that counts as a strum
MAX_STROKE_SPEED = 4.0        # this fast = full volume
STROKE_COOLDOWN = 0.12        # seconds between strums
CONFIRM_FRAMES = 2            # frames in a row needed to confirm a strum
STRUM_DIRECTIONS = ("down", "up")   # use ("down",) if fast hand returns cause extra upstrokes
STRUM_STAGGER = 0.012         # seconds between strings, like a real pick
CHORD_CONFIRM_FRAMES = 4      # finger count must be stable this long before chord changes
USE_MODEL = True              # use strum_model.joblib if it exists
MODEL_PATH = "strum_model.joblib"
DATA_PATH = "strum_data.csv"
