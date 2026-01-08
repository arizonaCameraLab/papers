import time  # Used for waiting during camera setup
import traceback, logging  # For exception tracebacks

# logging.basicConfig(level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from queue import Queue, Empty, Full  # Thread-safe queues
import cv2  # OpenCV for image processing
from vmbpy import VmbSystem, VmbCameraError, Frame, FrameStatus  # Vimba camera API
from config import opencv_display_format  # Desired display pixel format for OpenCV
from threading import Lock  # For thread-safe access to shared resources
from PyQt5.QtCore import QObject, pyqtSignal
import contextlib  # suppress context manager
from global_state import state  # Global state dictionary


# Detect whether the connected camera supports color output formats.
def is_color_camera(cam) -> bool:
    """
    Detect whether the connected camera supports color output formats.
    It checks the 'PixelFormat' feature for any entry containing 'RGB'.
    Returns True if color, False if monochrome.
    """
    try:
        feat = cam.get_feature_by_name("PixelFormat")
        entries = feat.get_available_entries()
        for entry in entries:
            if not entry.is_available():
                continue  # skip unavailable enum entries
            name, _ = entry.as_tuple()  # (string_name, integer_value)
            if "RG" in name.upper():
                print(f"[is_color_camera] Detected color format option: {name}")
                return True
        return False
    except Exception as e:
        print(f"[is_color_camera] Detection failed: {e}")
        return False


# Retrieve a camera object, by ID if provided, or the first available
def get_camera(camera_id=None):
    vmb = (
        VmbSystem.get_instance()
    )  # Get singleton instance of Vimba system (persistent)
    if camera_id:
        try:
            return vmb.get_camera_by_id(camera_id)  # Return camera with given ID
        except VmbCameraError as e:
            raise Exception(f"Failed to access Camera '{camera_id}': {e}")
    else:
        cams = vmb.get_all_cameras()  # Get list of all available cameras
        if not cams:
            raise Exception("No Cameras accessible.")
        return cams[0]  # Return the first available camera


# Configure basic camera features (disable auto exposure, white balance, etc.)
def setup_camera(cam: Frame):
    try:
        cam.ExposureAuto.set("Off")  # Disable auto exposure
    except Exception:
        pass
    try:
        cam.BalanceWhiteAuto.set("Off")  # Disable auto white balance
    except Exception:
        pass
    try:
        # Optimize packet size for GigE Vision cameras
        stream = cam.get_streams()[0]
        stream.GVSPAdjustPacketSize.run()
        while not stream.GVSPAdjustPacketSize.is_done():
            pass
    except Exception:
        pass


# Ensure the camera pixel format is compatible with OpenCV
def setup_pixel_format(cam: Frame):
    cam_formats = cam.get_pixel_formats()  # Get all supported formats
    from vmbpy import MONO_PIXEL_FORMATS  # Valid monochrome formats

    cam_mono_formats = [fmt for fmt in cam_formats if fmt in MONO_PIXEL_FORMATS]
    convertible_mono_formats = tuple(
        f
        for f in cam_mono_formats
        if opencv_display_format in f.get_convertible_formats()
    )

    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)  # Use desired format directly
    elif convertible_mono_formats:
        cam.set_pixel_format(convertible_mono_formats[0])  # Use convertible format
    else:
        raise Exception("Camera does not support an OpenCV compatible format.")


# Frame handler for receiving and storing the latest image from camera
class Handler(QObject):
    frameset_done = pyqtSignal()

    def __init__(self, maxlen=10, frames_per=10):
        super().__init__()
        self.cv_queue = Queue(maxlen)
        self.np_queue = Queue(maxlen)
        self.frames_per = frames_per
        self._cnt = 0
        self._lock = Lock()

    def __call__(self, cam, stream, frame):
        if frame.get_status() == FrameStatus.Complete:
            with self._lock:
                pix_fmt = frame.get_pixel_format()
                fmt_name = str(pix_fmt).upper()

                try:
                    # Access raw numpy array from frame
                    np_img = frame.as_numpy_ndarray().copy()
                    cv_img = np_img  # default assignment

                    # --------------------------------------------
                    # Auto-detect and handle format dynamically
                    # --------------------------------------------
                    if "BAYER" in fmt_name:
                        # --- Color (Bayer mosaic) ---
                        if "RG" in fmt_name:
                            cv_img = cv2.cvtColor(np_img, cv2.COLOR_BAYER_RG2RGB)
                        elif "BG" in fmt_name:
                            cv_img = cv2.cvtColor(np_img, cv2.COLOR_BAYER_BG2RGB)
                        elif "GR" in fmt_name:
                            cv_img = cv2.cvtColor(np_img, cv2.COLOR_BAYER_GR2RGB)
                        elif "GB" in fmt_name:
                            cv_img = cv2.cvtColor(np_img, cv2.COLOR_BAYER_GB2RGB)
                        else:
                            # Unknown Bayer type — use fallback
                            cv_img = np_img
                    elif "RGB" in fmt_name:
                        # --- True RGB formats ---
                        cv_img = np_img
                    elif "BGR" in fmt_name:
                        # --- Already BGR format ---
                        cv_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                    else:
                        # --- Monochrome path ---
                        if pix_fmt != opencv_display_format:
                            try:
                                display = frame.convert_pixel_format(
                                    opencv_display_format
                                )
                                cv_img = display.as_opencv_image().copy()
                            except Exception:
                                cv_img = np_img
                        else:
                            cv_img = np_img

                except Exception as e:
                    print(f"[Handler] Frame conversion failed ({fmt_name}): {e}")
                    cv_img = frame.as_opencv_image().copy()

                # ---- push into queues (drop oldest when full) ----
                for q, data in ((self.cv_queue, cv_img), (self.np_queue, cv_img)):
                    try:
                        q.put_nowait(data)
                    except Full:
                        q.get_nowait()  # drop oldest
                        q.put_nowait(data)

                # ---- frame-set counter ----
                self._cnt += 1
                if self._cnt >= self.frames_per:
                    self._cnt = 0
                    emit_needed = True
                else:
                    emit_needed = False

            # —— leave critical section —— #
            if emit_needed:
                self.frameset_done.emit()

        # Requeue the frame for continuous capture
        cam.queue_frame(frame)

    # blocking get（支持可选超时，兼容 viewer 的 timeout 调用）
    # 兼容旧的 get_cv_image() 调用：仅支持关键字参数
    def get_cv_image(self, *, timeout=None):
        return self.cv_queue.get(timeout=timeout)

    def get_np_image(self, timeout=None):
        return self.np_queue.get(timeout=timeout)
