import time  # Used for adding delays between stop retries
from config import (
    SLIDER_MAX,
    MIN_EXPOSURE,
    MAX_EXPOSURE,
)  # Slider range and exposure bounds
from vmbpy import VmbCameraError  # Specific camera error type from Vimba SDK


# Convert a horizontal slider value into an exposure time using a logarithmic scale
def slider_to_exposure(val):
    # Compute the normalized position (0.0–1.0)
    ratio = val / SLIDER_MAX
    # Map the ratio onto the [MIN_EXPOSURE, MAX_EXPOSURE] range logarithmically
    return float(MIN_EXPOSURE * ((MAX_EXPOSURE / MIN_EXPOSURE) ** ratio))


# Safely stop the camera streaming, retrying on transient errors
def safe_stop_camera(cam):
    try:
        if not cam.is_streaming():
            return True  # Already stopped
    except Exception as e:
        print(f"[safe_stop_camera] unexpected exception: {e}")

    for _ in range(5):  # Attempt up to 5 times
        try:
            # Try to stop the stream
            cam.stop_streaming()
            return True  # Success
        except (VmbCameraError, RuntimeError) as e:
            print(f"[safe_stop_camera] attempt failed with: {e}")
            # Known recoverable errors: wait a bit then retry
            time.sleep(0.1)
        except Exception as e:
            print(f"[safe_stop_camera] unexpected exception: {e}")
            # Catch-all for any other unexpected exception, then retry
            time.sleep(0.1)
    # If both attempts failed, indicate failure
    return False
