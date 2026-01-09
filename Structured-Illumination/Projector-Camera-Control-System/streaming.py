from threading import Thread  # For running streaming actions in a background thread
from global_state import (
    state,
    stream_lock,
)  # Import shared streaming state and thread lock
import traceback  # For printing exception tracebacks
from utils import safe_stop_camera  # Utility to stop camera safely with retries
from config import default_projector_config


# default buffer: burst_count during projector calibration + 2
DEFAULT_BUFFER = default_projector_config["burst_count"] + 2


def start_streaming(cam, handler, buffer_count: int = DEFAULT_BUFFER):
    """Start the camera streaming in a separate thread."""
    buffer_count = int(buffer_count)  # Ensure buffer_count is an integer

    def _start():
        # Ensure only one thread toggles the stream at a time
        with stream_lock:
            # If not currently streaming, attempt to start
            if not state["streaming"]:
                try:
                    # Begin camera capture with the provided handler and buffer
                    cam.start_streaming(handler=handler, buffer_count=buffer_count)
                    # Update global state flags
                    state["streaming"] = True
                    state["paused"] = False
                except Exception:
                    # Print any errors to the console for debugging
                    traceback.print_exc()

    # Launch the _start function as a daemon thread
    Thread(target=_start, daemon=True).start()

def stop_streaming(cam):
    """Stop the camera streaming synchronously (inside VmbSystem context)."""
    with stream_lock:
        if not state["streaming"]:
            return
        try:
            success = safe_stop_camera(cam)
            if success:
                state["streaming"] = False
                state["paused"] = True
            else:
                print("safe_stop_camera failed — streaming state unchanged")
        except Exception:
            traceback.print_exc()
