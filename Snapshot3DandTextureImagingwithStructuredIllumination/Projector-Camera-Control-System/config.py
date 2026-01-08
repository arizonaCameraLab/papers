# Global Configuration

from vmbpy import PixelFormat  # Pixel format definitions from Vimba SDK
import os
from PyQt5.QtWidgets import QApplication

# === Camera and Exposure Settings ===
opencv_display_format = PixelFormat.Mono8  # Use Mono8 for OpenCV display compatibility

MIN_EXPOSURE = 12  # Minimum exposure time in microseconds
MAX_EXPOSURE = 9999995  # Maximum exposure time in microseconds
SLIDER_MAX = 1000  # Max value of the exposure slider in UI

# === Default Camera Feature Dictionary ===
# These values are applied to the camera when initialized or reconfigured
features_to_update = {
    "ExposureAuto": "Off",  # Disable automatic exposure
    "ExposureTime": 1000,  # Initial exposure time in microseconds
    "GainAuto": "Off",  # Disable automatic gain control
    "Gain": 0.0,  # Initial gain
    # --- ROI offsets & binning (added) ---
    "BinningHorizontalMode": "Sum",  # Enum: "Average" / "Sum"
    "BinningHorizontal": 3,  # int >= 1
    "BinningVerticalMode": "Sum",  # Enum: "Average" / "Sum"
    "BinningVertical": 3,  # int >= 1
    "BinningSelector": "Digital",  # Enum: e.g. "Digital"
    "Height": 928,  # Image height (sensor ROI)
    "OffsetY": 42,  # Vertical ROI offset (px)
    "Width": 928,  # Image width (sensor ROI)
    "OffsetX": 204,  # Horizontal ROI offset (px)
    "IntensityControllerTarget": 50.0,  # Target brightness for auto functions
    "BlackLevel": 0,  # Black level adjustment
    "Gamma": 1.0,  # Gamma correction value
    "PixelFormat": "BayerRG8",  # Pixel format (must match opencv_display_format)
    "BalanceRatioSelector": "Red",  # Select red channel for white balance
    "BalanceRatio": 3.31,  # Initial white balance ratio for red channel
    "BalanceRatioSelector": "Blue",  # Select blue channel for white balance
    "BalanceRatio": 1.34,  # Initial white balance ratio for blue channel
    "BalanceWhiteAuto": "Off",  # Disable auto white balance
    "DeviceLinkThroughputLimitMode": "Off",  # Disable bandwidth throttling
    "DeviceLinkThroughputLimit": 450000000,  # Bandwidth limit in bits per second (if enabled)
    "AcquisitionFrameRateEnable": True,  # Enable manual frame rate setting
    "AcquisitionFrameRate": 30.0,  # Desired frame rate in FPS
    "SensorShutterMode": "RollingShutter",  # Use rolling shutter
    "ConvolutionMode": "Off",  # Disable hardware convolution
    "ContrastEnable": False,  # Disable hardware contrast enhancement
}

output_file = "camera_features.txt"  # File to save updated camera feature settings

# === Default Projector Calibration Configuration ===
# These values control how the projector behaves during structured light capture
default_projector_config = {
    "burst_count": 1,  # Number of frames to average or store per projection
    "pattern_folder": os.path.join(
        os.getcwd(),
        "gray_code_patterns",
    ),  # Folder where projection patterns are stored
    "screen_number": 1,  # Monitor index for full-screen projection
    "projector_width": 768,  # Resolution width of the projector
    "projector_height": 768,  # Resolution height of the projector
    "gray_code_step": 2,  # pixel binning of the projector
    "default_synth_bit0": True,  # Whether to synthesize bit0 pattern by default
    "inverse_pattern": True,  # Whether to invert projection patterns
    "pattern_background": 0,  # Background gray level (0–255)
    # Default folder for calibration captures; auto-clean on startup
    "capture_save_folder": os.path.join(os.getcwd(), "calibration_capture"),
}

try:
    app = QApplication.instance() or QApplication([])
    screens = app.screens()
    idx = default_projector_config.get("screen_number", 0)
    if 0 <= idx < len(screens):
        geom = screens[idx].geometry()
        default_projector_config["projector_width"] = geom.width()
        default_projector_config["projector_height"] = geom.height()
except Exception:
    print("Error initializing projector width and height. Using defaults.")
    # Fallback to default values if screen information is unavailable
    pass
# === Other Global Settings ===
DEBOUNCE_TIME = 0.5  # Debounce delay for 'p' keypress (seconds)
