import sys  # Provides access to command-line arguments
import signal  # Allows setting up signal handlers (e.g., for graceful shutdown)
from PyQt5.QtWidgets import QApplication  # Main Qt application class
from vmbpy import VmbSystem  # Entry point to the Allied Vision Vimba SDKv
from vmbpy import PixelFormat  # Pixel format definitions from Vimba SDK

# Camera control and initialization utilities
from camera_utils import (
    get_camera,
    setup_camera,
    setup_pixel_format,
    Handler,
    is_color_camera,
)
from camera_feature_manager import CameraFeatureManager  # For applying feature settings
from config import (
    features_to_update,
    output_file,
    default_projector_config,
    opencv_display_format,
)  # Global config values
from global_state import state  # Streaming state dictionary
from utils import safe_stop_camera  # Utility function for stopping the camera
from viewer import CameraViewer  # Main UI class
from streaming import (
    start_streaming,
    stop_streaming,
)  # Threaded stream control functions

from PyQt5.QtWidgets import QApplication


def main():
    # Get camera ID from command-line arguments (optional)
    camera_id = sys.argv[1] if len(sys.argv) > 1 else None

    # Create Qt application
    app = QApplication(sys.argv)

    # Handle Ctrl+C to gracefully exit
    signal.signal(signal.SIGINT, lambda *args: app.quit())

    # Open the Vimba system and initialize the selected camera
    with VmbSystem.get_instance() as vmb, get_camera(camera_id) as cam:
        setup_camera(cam)  # Apply basic settings (e.g., disable auto exposure)
        setup_pixel_format(cam)  # Ensure format is OpenCV-compatible
        handler = Handler()  # Thread-safe handler for receiving images

        # Detect whether this camera is color or monochrome 
        color_mode = is_color_camera(cam)
        print(f"Detected camera type: {'Color' if color_mode else 'Monochrome'}")

        # store it in global state
        state["color_camera"] = color_mode

        if color_mode:
            # Update display format dynamically for color cameras
            try:
                cam.set_pixel_format(PixelFormat.BayerRG8)
                print("[setup_pixel_format] Set to BayerRG8 for color display.")
            except Exception as e:
                print(f"[setup_pixel_format] Failed to set BayerRG8: {e}")
        else:
            cam.set_pixel_format(PixelFormat.Mono8)

        handler.moveToThread(QApplication.instance().thread())

        # Attempt to load the current exposure time from the camera
        try:
            feat = cam.get_feature_by_name("ExposureTime")
            current = float(feat.get())
            features_to_update["ExposureTime"] = (
                current  # Update config with current value
            )
        # If the feature is not available, use the default from the config
        except:
            current = features_to_update["ExposureTime"]  # Use fallback from config
        try:
            # Apply initial feature settings from the global configuration
            CameraFeatureManager(cam).update_features(features_to_update, output_file)
        except Exception as e:
            print(f"Error applying camera features: {e}")
        exposure_val = [current]  # Use list for mutable value reference

        # Wrap stream control functions:
        # default buffer_count = burst_count + 2
        default_bc = default_projector_config["burst_count"] + 2
        start_fn = lambda bc=default_bc: start_streaming(cam, handler, buffer_count=bc)
        stop_fn = lambda: stop_streaming(cam)

        # Create and show the main viewer window
        viewer = CameraViewer(cam, handler, feat, exposure_val, start_fn, stop_fn)
        # viewer = CameraViewer(
        #     cam, handler, feat, exposure_val, start_fn, stop_fn, is_color=color_mode
        # )

        viewer.show()

        # Start the camera stream on launch
        start_fn()

        # Enter Qt main event loop
        sys.exit(app.exec_())


# If this file is executed directly, run the main function
if __name__ == "__main__":
    main()
