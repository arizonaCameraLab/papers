# -*- coding: utf-8 -*-

import os
import shutil
import contextlib
import sys
import math
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QColor, QKeyEvent, QMouseEvent
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

# use a list to hold references to live viewers, preventing garbage collection
_live_viewers = []


def prepare_save_folder(folder):
    """
    Remove all existing files and directories in the folder.
    Create the folder if it does not exist.
    """
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        os.makedirs(folder)


def generate_gray_code_patterns_opencv(
    width,
    height,
    save_folder,
    inverse=False,
    step=1,
    synth_bit0=False,
):
    """
    Generate horizontal & vertical Gray Code patterns (with optional inverse),
    plus white/black images, using OpenCV's structured_light.GrayCodePattern.

    If synth_bit0=True:
        Generate shifted versions of the safe low-frequency patterns
        (pattern_shifted_0/_1) that can later be subtracted to synthesize
        Gray bit0 pattern without directly projecting high-frequency stripes.

    Args:
        width, height (int): final image resolution
        save_folder (str): directory to save outputs
        inverse (bool): whether to generate inverse frames
        step (int): scale factor for upsampling code elements
        synth_bit0 (bool): whether to synthesize shifted safe bit0 patterns
    """
    # ---- Step 0: Prepare output folder ----
    prepare_save_folder(save_folder)
    file_counter = 0

    # ---- Step 1: Compute GrayCode grid size ----
    gc_h = int((height - 1) / step) + 1
    gc_w = int((width - 1) / step) + 1

    # ---- Step 2: Generate GrayCode patterns ----
    graycode = cv2.structured_light.GrayCodePattern.create(gc_w, gc_h)
    _, patterns = graycode.generate()
    num_hor_bits = int(np.ceil(np.log2(gc_w)))
    num_ver_bits = int(np.ceil(np.log2(gc_h)))
    print(f"hor_bits={num_hor_bits}, ver_bits={num_ver_bits}")

    # ---- Step 3: Expand patterns if step > 1 ----
    def expand(pat):
        big = np.repeat(np.repeat(pat, step, axis=0), step, axis=1)
        return big[:height, :width].astype(np.uint8)

    # ---- Step 4: Save horizontal patterns ----
    for i in range(num_hor_bits):
        p = expand(patterns[2 * i])
        cv2.imwrite(os.path.join(save_folder, f"HorGrayCode_{i+1:02d}.png"), p)
        file_counter += 1
        if inverse:
            ip = expand(patterns[2 * i + 1])
            cv2.imwrite(
                os.path.join(save_folder, f"HorGrayCode_{i+1:02d}_Inverse.png"), ip
            )
            file_counter += 1

    # ---- Step 5: Save vertical patterns ----
    offset = 2 * num_hor_bits
    for j in range(num_ver_bits):
        p = expand(patterns[offset + 2 * j])
        cv2.imwrite(os.path.join(save_folder, f"VerGrayCode_{j+1:02d}.png"), p)
        file_counter += 1
        if inverse:
            ip = expand(patterns[offset + 2 * j + 1])
            cv2.imwrite(
                os.path.join(save_folder, f"VerGrayCode_{j+1:02d}_Inverse.png"), ip
            )
            file_counter += 1

    # ---- Step 6: Synthesized safe bit0 shifted patterns ----
    if synth_bit0:
        print("Synthesizing safe bit0 shifted patterns...")

        cell = max(1, int(step))  # shift relative to code-cell width

        # --- Horizontal (x-direction) ---
        hor_bit1_idx = (num_hor_bits - 1 - 1) * 2  # use bit1 (next-to-LSB)
        pat_ref_h = expand(patterns[hor_bit1_idx])
        pat_ref_h_inv = expand(patterns[hor_bit1_idx + 1]) if inverse else None

        pattern_h_0 = np.roll(pat_ref_h, shift=-cell, axis=1)
        pattern_h_1 = np.roll(pat_ref_h, shift=+cell, axis=1)
        cv2.imwrite(os.path.join(save_folder, "HorBit0_Shifted_0.png"), pattern_h_0)
        cv2.imwrite(os.path.join(save_folder, "HorBit0_Shifted_1.png"), pattern_h_1)
        file_counter += 2

        if inverse and pat_ref_h_inv is not None:
            pattern_h_0i = np.roll(pat_ref_h_inv, shift=-3 * cell, axis=1)
            pattern_h_1i = np.roll(pat_ref_h_inv, shift=-1 * cell, axis=1)
            cv2.imwrite(
                os.path.join(save_folder, "HorBit0_Shifted_0_Inverse.png"), pattern_h_0i
            )
            cv2.imwrite(
                os.path.join(save_folder, "HorBit0_Shifted_1_Inverse.png"), pattern_h_1i
            )
            file_counter += 2

        # --- Vertical (y-direction) ---
        ver_bit1_idx = (num_ver_bits - 1 - 1) * 2 + offset
        pat_ref_v = expand(patterns[ver_bit1_idx])
        pat_ref_v_inv = expand(patterns[ver_bit1_idx + 1]) if inverse else None

        pattern_v_0 = np.roll(pat_ref_v, shift=-cell, axis=0)
        pattern_v_1 = np.roll(pat_ref_v, shift=+cell, axis=0)
        cv2.imwrite(os.path.join(save_folder, "VerBit0_Shifted_0.png"), pattern_v_0)
        cv2.imwrite(os.path.join(save_folder, "VerBit0_Shifted_1.png"), pattern_v_1)
        file_counter += 2

        if inverse and pat_ref_v_inv is not None:
            pattern_v_0i = np.roll(pat_ref_v_inv, shift=-3 * cell, axis=0)
            pattern_v_1i = np.roll(pat_ref_v_inv, shift=-1 * cell, axis=0)
            cv2.imwrite(
                os.path.join(save_folder, "VerBit0_Shifted_0_Inverse.png"), pattern_v_0i
            )
            cv2.imwrite(
                os.path.join(save_folder, "VerBit0_Shifted_1_Inverse.png"), pattern_v_1i
            )
            file_counter += 2

    # ---- Step 7: White/black patterns ----
    white = np.ones((height, width), dtype=np.uint8) * 255
    black = np.zeros((height, width), dtype=np.uint8)
    cv2.imwrite(os.path.join(save_folder, "WhitePattern.png"), white)
    cv2.imwrite(os.path.join(save_folder, "BlackPattern.png"), black)
    file_counter += 2

    print(f"Generated {file_counter} images in {save_folder}")


def generate_gray_code_patterns_3dscanning(
    width, height, save_folder, inverse=True, *, step=1, bit_order="MSB"
):
    """
    only for the software "3d Scanning"
    Generate horizontal & vertical Gray Code patterns (with optional inverse),
    plus white/black images, using OpenCV's structured_light.GrayCodePattern.

    Filenames:
      HorGrayCode_01.png … HorGrayCode_NN.png
      HorGrayCode_01_Inverse.png … (if inverse=True)
      VerGrayCode_01.png … VerGrayCode_MM.png
      VerGrayCode_01_Inverse.png … (if inverse=True)
      WhitePattern.png, BlackPattern.png

    Parameters
    ----------
    width, height : int
        Native projector resolution (px).
    save_folder : str
        Where to put the PNGs; will be emptied/created.
    inverse : bool
        Save inverse patterns as well.
    step : int, optional
        Logic-pixel pitch (default 1 → 最细条纹=1 px).
    bit_order : {"MSB","LSB"}
        MSB means saving from most significant bit to least significant bit.
        LSB means saving from least significant bit to most significant bit.  
    """
    # Prepare output folder
    prepare_save_folder(save_folder)

    # Determine bit count and generated square size
    nbits = math.ceil(math.log2(max(width, height)))
    gen_size = 1 << nbits
    print(f"Gray code size: {gen_size}x{gen_size} ({nbits} bits)")
    print(f"Gray code step: {step} px")
    print(f"Gray code size: {width}x{height} ({width/step} bits)")
    print(f"Gray code order: {bit_order.upper()}")

    # Generate full-square Gray code (with built-in inverse interleaved)
    pattern = cv2.structured_light_GrayCodePattern.create(gen_size, gen_size)
    ok, imgs = pattern.generate()
    if not ok:
        raise RuntimeError("OpenCV failed to generate GrayCodePattern")
    # imgs: [bit0, inv0, bit1, inv1, ..., bitN-1, invN-1]

    # Split original and inverse planes
    orig_planes = imgs[0::2]
    inv_planes = imgs[1::2]

    # Order bits for saving
    indices = list(range(nbits))
    if bit_order.upper() == "LSB":
        indices = indices[::-1]

    # Helper to upscale and crop
    def _process_plane(plane):
        # upscale by step in both dims
        if step != 1:
            plane = np.repeat(np.repeat(plane, step, axis=1), step, axis=0)
        # calculate margins (center-crop)
        h0, w0 = plane.shape
        top = (h0 - height) // 2
        left = (w0 - width) // 2
        return plane[top : top + height, left : left + width]

    # Save horizontal patterns (vertical stripes encode X)
    for idx, b in enumerate(indices, start=1):
        img_f = _process_plane(orig_planes[b])
        fname = f"HorGrayCode_{idx:02d}.png"
        cv2.imwrite(os.path.join(save_folder, fname), img_f)
        if inverse:
            img_inv = _process_plane(inv_planes[b])
            fname_inv = f"HorGrayCode_{idx:02d}_Inverse.png"
            cv2.imwrite(os.path.join(save_folder, fname_inv), img_inv)

    # Save vertical patterns (horizontal stripes encode Y)
    for idx, b in enumerate(indices, start=1):
        # transpose to get horizontal stripes from column code
        img_f = _process_plane(orig_planes[b].T)
        fname = f"VerGrayCode_{idx:02d}.png"
        cv2.imwrite(os.path.join(save_folder, fname), img_f)
        if inverse:
            img_inv = _process_plane(inv_planes[b].T)
            fname_inv = f"VerGrayCode_{idx:02d}_Inverse.png"
            cv2.imwrite(os.path.join(save_folder, fname_inv), img_inv)

    # Save reference white/black
    white = np.full((height, width), 255, np.uint8)
    black = np.zeros((height, width), np.uint8)
    cv2.imwrite(os.path.join(save_folder, "WhitePattern.png"), white)
    cv2.imwrite(os.path.join(save_folder, "BlackPattern.png"), black)
    print(f"Generated {len(os.listdir(save_folder))} images in {save_folder}")


class ImageViewer(QLabel):
    # send its path to receiver, when a new pattern is fully drawn
    image_changed = pyqtSignal(str)

    def __init__(
        self,
        image_paths,
        screen_geometry,
        loop=True,
        scale=False,
        bg_gray=0,
        mode="preview",
        capture_condition_fn=None,
    ):
        """
        :param image_paths: List of image file paths.
        :param screen_geometry: Target screen geometry from QScreen.geometry().
        :param loop: Whether to loop images (only applicable in preview mode).
        :param scale: Whether to stretch the image to fill the screen (True: scale while keeping aspect ratio; False: display at actual size in center).
        :param bg_gray: Background gray value (used for areas not covered by the image).
        :param mode: "preview" or "capture".
        :param capture_condition_fn: When mode is "capture", a callback function that determines if a condition is met
                                     to switch to the next image. It should return True to trigger switching.
        """
        super().__init__()
        self.image_paths = image_paths
        self.current_index = 0
        self.loop = loop
        self.scale = scale
        self.bg_gray = bg_gray
        self.mode = mode
        self.capture_condition_fn = capture_condition_fn

        self.screen_geometry = screen_geometry

        # Set up a full screen, frameless window
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setGeometry(screen_geometry)
        self.move(screen_geometry.x(), screen_geometry.y())
        self.setAlignment(Qt.AlignCenter)
        self.showFullScreen()
        # make Qt auto-delete the object on close to prevent GPU/memory leaks
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.show_image()

        # In capture mode, start a timer to check the capture condition
        if self.mode == "capture" and self.capture_condition_fn is not None:
            self.timer = QTimer()
            self.timer.timeout.connect(self.check_capture_condition)
            self.timer.start(50)  # Check every 50 ms

    def show_image(self):
        """
        Display the current image.
        """
        if 0 <= self.current_index < len(self.image_paths):
            original_pix = QPixmap(self.image_paths[self.current_index])
            if self.scale:
                scaled = original_pix.scaled(
                    self.screen_geometry.width(),
                    self.screen_geometry.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.setPixmap(scaled)
            else:
                bg = QPixmap(
                    self.screen_geometry.width(), self.screen_geometry.height()
                )
                bg.fill(QColor(self.bg_gray, self.bg_gray, self.bg_gray))
                x_offset = (bg.width() - original_pix.width()) // 2
                y_offset = (bg.height() - original_pix.height()) // 2
                painter = QPainter(bg)
                painter.drawPixmap(x_offset, y_offset, original_pix)
                painter.end()
                self.setPixmap(bg)

            # emit signal with current image path
            self.image_changed.emit(self.image_paths[self.current_index])

    def advance_image(self):
        """
        Advance to the next image. If the last image is reached:
          - In capture mode, stop the timer and switch to preview mode.
          - In preview mode, loop to the first image if looping is enabled.
        """
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_image()
        else:
            if self.mode == "capture":
                # In capture mode, when reaching the last image, stop capture and switch to preview mode.
                self.timer.stop()
                self.mode = "preview"
                print("Capture mode ended, switching to preview mode.")
            elif self.loop:
                self.current_index = 0
                self.show_image()

    def check_capture_condition(self):
        """
        In capture mode, this timer callback checks the capture condition.
        When the condition is met, switch to the next image.
        """
        try:
            if self.capture_condition_fn and self.capture_condition_fn():
                self.advance_image()
        except Exception as e:
            print(f"Error in capture condition callback: {e}")

    def keyPressEvent(self, event: QKeyEvent):
        """
        Process keyboard events only in preview mode:
         - Space key to advance the image.
         - Esc key to exit.
        """
        if self.mode != "preview":
            # Do not respond to manual input in capture mode.
            return

        if event.key() == Qt.Key_Space:
            self.advance_image()
        elif event.key() == Qt.Key_Escape:
            self.close()

    def mousePressEvent(self, event: QMouseEvent):
        """
        Process left mouse click events only in preview mode to advance the image.
        """
        if self.mode != "preview":
            return

        if event.button() == Qt.LeftButton:
            self.advance_image()

    def closeEvent(self, event):
        super().closeEvent(event)
        with contextlib.suppress(ValueError):
            if self in _live_viewers:
                _live_viewers.remove(self)


def launch_viewer(
    image_folder,
    target_screen=0,
    loop=True,
    scale=False,
    bg_gray=0,
    mode="preview",
    capture_condition_fn=None,
):
    """
    Launch the image viewer.

    :param image_folder: Folder containing the images.
    :param target_screen: Index of the screen to display the images.
    :param loop: Whether to loop the images (only applicable in preview mode).
    :param scale: Whether to scale the image to fill the screen.
    :param bg_gray: Background gray value.
    :param mode: "preview" or "capture". In capture mode, the viewer advances images according to capture_condition_fn.
    :param capture_condition_fn: Callback function for capture mode. Should return True when the condition is met to switch images.
    """
    supported_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
    # custom sort: White → Black → HorGrayCode → VerGrayCode → others
    raw_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in supported_exts
    ]

    def pattern_key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "WhitePattern":
            return (0, 0)
        if name == "BlackPattern":
            return (1, 0)
        if name.startswith("HorGrayCode_"):
            idx = int(name.split("_")[1])
            return (2, idx)
        if name.startswith("VerGrayCode_"):
            idx = int(name.split("_")[1])
            return (3, idx)
        # others go last, sorted by name
        return (4, name)

    image_files = sorted(raw_files, key=pattern_key)
    if not image_files:
        print("No images found in", image_folder)
        return

    # if there is no existing QApplication, create one and enter event loop at the end
    app = QApplication.instance()
    own_app = False
    if app is None:
        app = QApplication(sys.argv)
        own_app = True

    screens = app.screens()
    if target_screen >= len(screens):  # reset to primary screen when out of range
        print(f"Screen index {target_screen} out of range, use 0")
        target_screen = 0

    screen_geom = screens[target_screen].geometry()
    viewer = ImageViewer(
        image_files,
        screen_geom,
        loop=loop,
        scale=scale,
        bg_gray=bg_gray,
        mode=mode,
        capture_condition_fn=capture_condition_fn,
    )
    viewer.show()

    # put the viewer into the global list to keep a reference and prevent garbage collection
    _live_viewers.append(viewer)

    # only if we created the QApplication instance ourselves, enter exec_()
    if own_app:
        sys.exit(app.exec_())


if __name__ == "__main__":
    # if run this script as main, launch in preview mode
    DEFAULT_SAVE_FOLDER = os.path.join(os.getcwd(), "CodePatterns")
    WIDTH = 1280
    HEIGHT = 720
    BG_GRAY = 0
    SCALE_TO_FULLSCREEN = False
    INVERSE_IMAGES = True
    DEFAULT_SCREEN_NUMBER = 1
    LOOP = True
    DISPLAY_MODE = "preview"

    generate_gray_code_patterns_opencv(
        WIDTH,
        HEIGHT,
        DEFAULT_SAVE_FOLDER,
        inverse=INVERSE_IMAGES,
    )
