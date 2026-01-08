from threading import Lock

# Global streaming state shared across modules
state = {
    "streaming": False,
    "paused": False,
    "color_camera": False,  # new flag: True if camera supports RGB output
}

# Thread-safe lock for camera access
stream_lock = Lock()
