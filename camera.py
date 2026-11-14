"""Camera module: opens the webcam, reads frames, releases it safely."""
import logging
import cv2
import config

log = logging.getLogger(__name__)


class CameraError(Exception):
    """Raised with a human-readable message when the camera cannot be used."""


class Camera:
    def __init__(self, index=config.CAMERA_INDEX,
                 width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT,
                 mirror=config.MIRROR_VIEW):
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror
        self._cap = None
        self._failed = 0

    def open(self):
        # CAP_DSHOW opens much faster and more reliably on Windows.
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self.release()
            raise CameraError(
                f"Could not open camera index {self.index}.\n"
                "- Close other apps using the webcam (Zoom, Teams, Camera app).\n"
                "- Windows Settings > Privacy & security > Camera: allow desktop apps.\n"
                "- Try another camera index (set CAMERA_INDEX in config.py to 1 or 2)."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._failed = 0
        log.info("Camera %d opened", self.index)

    def read(self):
        """Return a BGR frame, or None for a temporary failure.
        Raises CameraError after too many consecutive failures."""
        if self._cap is None:
            raise CameraError("Camera is not open.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._failed += 1
            if self._failed >= config.MAX_FAILED_FRAMES:
                raise CameraError("Camera stopped delivering frames (unplugged or in use?).")
            return None
        self._failed = 0
        return cv2.flip(frame, 1) if self.mirror else frame

    @property
    def is_open(self):
        return self._cap is not None and self._cap.isOpened()

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Camera released")
