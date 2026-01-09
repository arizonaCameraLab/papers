from PyQt5.QtWidgets import (
    QSizePolicy,
    QWidget,  # Base class for all UI panels
    QSpinBox,  # for integer ranges
    QVBoxLayout,  # Vertical layout manager
    QHBoxLayout,  # Horizontal layout manager
    QFormLayout,  # Form layout for label-field pairs
    QLabel,  # Display text labels
    QPushButton,  # Clickable button widget
    QRadioButton,  # Single-option radio button
    QButtonGroup,  # Group for radio buttons
    QComboBox,  # Dropdown selection widget
    QCheckBox,  # Checkbox widget
    QLineEdit,  # Single-line text input
    QMessageBox,  # Dialog for messages/warnings
    QFileDialog,  # File/folder selection dialog
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QEventLoop, QMetaObject
from config import (
    default_projector_config,
    MIN_EXPOSURE,
    MAX_EXPOSURE,
    SLIDER_MAX,
)  # Import global config
from camera_feature_manager import CameraFeatureManager  # Manage camera features
from math import log  # Math for slider conversion
from PyQt5.QtWidgets import QApplication  # Access top-level widgets
import traceback  # For printing exception tracebacks
from project_pattern import (
    generate_gray_code_patterns_opencv as generate_gray_code_patterns,
    launch_viewer,
)  # Pattern generation and viewer launch
from project_pattern import ImageViewer
import time, sys, os, shutil, numpy as np
from PyQt5.QtWidgets import QInputDialog
from global_state import state # Global state dictionary
from queue import Empty
from PyQt5.QtCore import Qt
import threading
from PyQt5 import sip
import cv2
import numpy as np
from tqdm.auto import tqdm  # For progress bar in average calculation


# Convert an exposure time into a slider position on a logarithmic scale
def exposure_to_slider(exposure: int) -> int:
    exposure = max(exposure, MIN_EXPOSURE)  # prevent ≤0 cases
    ratio = log(exposure / MIN_EXPOSURE) / log(MAX_EXPOSURE / MIN_EXPOSURE)
    return int(ratio * SLIDER_MAX)


# Build the panel embedding camera feature controls
# manager: CameraFeatureManager instance
# features_dict: dict of feature names to default values
# exposure_feature: camera exposure feature object
# exposure_val: single-element list storing current exposure
# output_file: path to save feature logs
# paused_callback: function to check if camera is paused
# start_callback / stop_callback: functions to control streaming
def create_feature_config_panel(
    manager,
    features_dict,
    exposure_feature,
    exposure_val,
    output_file,
    paused_callback,
    start_callback,
    stop_callback,
):
    panel = QWidget()  # Container widget for panel
    layout = QVBoxLayout(panel)  # Vertical layout for all controls

    # Top row: Start and Pause buttons
    top_layout = QHBoxLayout()
    start_btn = QPushButton("Start Streaming")
    pause_btn = QPushButton("Pause Streaming")
    top_layout.addWidget(start_btn)
    top_layout.addWidget(pause_btn)
    # two buttons share horizontal space equally, keep default height
    top_layout.setStretch(0, 1)
    top_layout.setStretch(1, 1)
    layout.addLayout(top_layout)

    # Form layout for individual camera features
    form_layout = QFormLayout()
    layout.addLayout(form_layout)
    widget_vars = {}  # Store references to created widgets

    # Define choice lists for specific features
    auto_choices = ["Off", "Once", "Continuous"]
    dl_tlm_choices = ["Off", "On"]
    conv_choices = ["Off", "Sharpness", "CustomConvolution", "AdaptiveNoiseSuppression"]
    pixel_format_choices = ["Mono8", "Mono10", "Mono10p", "Mono12", "Mono12p"]
    # New: binning-related enum choices (constrained)
    binning_selector_choices = ["Digital"]  # only Digital selectable
    binning_mode_choices = ["Average", "Sum"]

    # For each feature, create an appropriate input widget
    for feat_name, default_val in features_dict.items():
        try:
            # Try reading the current feature value from camera
            current_val = manager.camera.get_feature_by_name(feat_name).get()
        except Exception:
            current_val = default_val  # Fallback to default

        # Auto-exposure / auto-gain: use radio buttons
        if feat_name in ["ExposureAuto", "GainAuto", "BalanceWhiteAuto"]:
            group = QWidget()
            hl = QHBoxLayout(group)
            btn_group = QButtonGroup(group)
            idx = (
                auto_choices.index(str(current_val))
                if str(current_val) in auto_choices
                else 0
            )
            for i, ch in enumerate(auto_choices):
                rb = QRadioButton(ch)
                rb.setChecked(i == idx)
                btn_group.addButton(rb)
                hl.addWidget(rb)
            widget_vars[feat_name] = ("radio", btn_group)
            form_layout.addRow(QLabel(feat_name), group)

        # Device link throughput limit mode: radio buttons
        elif feat_name == "DeviceLinkThroughputLimitMode":
            group = QWidget()
            hl = QHBoxLayout(group)
            btn_group = QButtonGroup(group)
            idx = (
                dl_tlm_choices.index(str(current_val))
                if str(current_val) in dl_tlm_choices
                else 0
            )
            for i, ch in enumerate(dl_tlm_choices):
                rb = QRadioButton(ch)
                rb.setChecked(i == idx)
                btn_group.addButton(rb)
                hl.addWidget(rb)
            widget_vars[feat_name] = ("radio", btn_group)
            form_layout.addRow(QLabel(feat_name), group)

        # Convolution mode: dropdown list
        elif feat_name == "ConvolutionMode":
            combo = QComboBox()
            combo.addItems(conv_choices)
            if str(current_val) in conv_choices:
                combo.setCurrentText(str(current_val))
            widget_vars[feat_name] = ("dropdown", combo)
            form_layout.addRow(QLabel(feat_name), combo)

        # Pixel format: dropdown list
        elif feat_name == "PixelFormat":
            combo = QComboBox()
            try:
                feat = manager.camera.get_feature_by_name("PixelFormat")
                entries = feat.get_available_entries()
                format_names = []
                for entry in entries:
                    if not entry.is_available():
                        continue
                    name, _ = entry.as_tuple()
                    format_names.append(name)
                combo.addItems(format_names)
                if str(current_val) in format_names:
                    combo.setCurrentText(str(current_val))
                else:
                    combo.setCurrentIndex(0)
                print(f"[UI] PixelFormat options: {format_names}")
            except Exception as e:
                print(f"[UI] Could not query PixelFormat entries: {e}")
                combo.addItems(["Mono8"])  # Fallback list

            widget_vars[feat_name] = ("dropdown", combo)
            form_layout.addRow(QLabel(feat_name), combo)

        # Binning enums and integer ranges
        elif feat_name == "BinningSelector":
            # Only "Digital" is allowed
            combo = QComboBox()
            combo.addItems(binning_selector_choices)
            # if current camera value isn't Digital, just force Digital
            try:
                if str(current_val) in binning_selector_choices:
                    combo.setCurrentText(str(current_val))
                else:
                    combo.setCurrentText("Digital")
            except Exception:
                combo.setCurrentText("Digital")
            widget_vars[feat_name] = ("dropdown", combo)
            form_layout.addRow(QLabel(feat_name), combo)

        elif feat_name in ["BinningHorizontalMode", "BinningVerticalMode"]:
            combo = QComboBox()
            combo.addItems(binning_mode_choices)
            if str(current_val) in binning_mode_choices:
                combo.setCurrentText(str(current_val))
            widget_vars[feat_name] = ("dropdown", combo)
            form_layout.addRow(QLabel(feat_name), combo)

        elif feat_name in ["BinningHorizontal", "BinningVertical"]:
            # Must be integer in [1, 8]
            spin = QSpinBox()
            spin.setRange(1, 8)
            try:
                iv = int(current_val)
            except Exception:
                iv = (
                    int(default_val)
                    if isinstance(default_val, (int, float, str))
                    else 1
                )
            spin.setValue(min(8, max(1, iv)))
            widget_vars[feat_name] = ("spin", spin)
            form_layout.addRow(QLabel(feat_name), spin)

        # Boolean features: checkbox
        elif isinstance(current_val, bool):
            chk = QCheckBox()
            chk.setChecked(current_val)
            widget_vars[feat_name] = ("checkbox", chk)
            form_layout.addRow(QLabel(feat_name), chk)

        # Other numeric/string: text entry
        else:
            le = QLineEdit(str(current_val))
            widget_vars[feat_name] = ("entry", le)
            form_layout.addRow(QLabel(feat_name), le)

    # Function to update UI controls after applying new settings
    def refresh_controls(updated: dict):
        for fn, (wtype, w) in widget_vars.items():
            if fn in updated:
                newv = updated[fn].get("Current Value", updated[fn])
                if wtype == "radio":
                    for b in w.buttons():
                        if b.text() == str(newv):
                            b.setChecked(True)
                elif wtype == "dropdown":
                    w.setCurrentText(str(newv))
                elif wtype == "checkbox":
                    w.setChecked(bool(newv))
                elif wtype == "spin":
                    try:
                        w.setValue(int(newv))
                    except Exception:
                        pass
                else:
                    w.setText(str(newv))

    # Handler for the Apply button: validate, gather inputs, update features
    def apply_changes():
        if not paused_callback():
            QMessageBox.warning(panel, "Warning", "Camera must be paused.")
            return
        newf = {}
        for fn, (wtype, w) in widget_vars.items():
            try:
                if wtype == "radio":
                    cb = w.checkedButton()
                    val = cb.text() if cb else ""
                elif wtype == "dropdown":
                    val = w.currentText()
                elif wtype == "checkbox":
                    val = w.isChecked()
                elif wtype == "spin":
                    # QSpinBox returns int and is already range-clamped
                    val = int(w.value())
                else:
                    txt = w.text()
                    try:
                        val = float(txt) if "." in txt else int(txt)
                    except:
                        val = txt
                newf[fn] = val
            except Exception:
                traceback.print_exc()

        # Attempt to update camera features and refresh UI
        try:
            updated, cnt = manager.update_features(newf, output_file)
            QMessageBox.information(panel, "Results", f"Updated {cnt} features.")
            refresh_controls(updated)
            # Also update exposure slider if changed
            ne = float(manager.camera.get_feature_by_name("ExposureTime").get())
            exposure_val[0] = ne
            for win in QApplication.topLevelWidgets():
                if hasattr(win, "slider") and hasattr(win, "exposure_val"):
                    win.slider.setValue(exposure_to_slider(ne))
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(panel, "Error", f"Apply failed: {e}")

    # Create and add the Apply button to the layout
    apply_btn = QPushButton("Apply")
    apply_btn.clicked.connect(apply_changes)
    layout.addWidget(apply_btn)

    # clicked(bool) will carry a checked parameter; use lambda to discard it
    start_btn.clicked.connect(lambda _=False: start_callback())
    pause_btn.clicked.connect(lambda _=False: stop_callback())

    return panel, widget_vars


def create_projector_calibration_panel(
    handler,
    start_stream_fn,
    stop_stream_fn,
):
    """
    Build the projector-calibration panel, including:
     - Burst Count
     - Pattern folder
     - **Save folder**
     - Generate / Preview / Projection Capture / Restart buttons
    """
    panel = QWidget()
    panel.session_count = 1  # initialize the first session
    panel.is_capturing = False  # whether capturing is in progress
    panel._start_stream_fn = start_stream_fn  # remember start function
    panel._stop_stream_fn = stop_stream_fn  # (optional, fallback)
    vlayout = QVBoxLayout(panel)
    form = QFormLayout()
    vlayout.addLayout(form)
    # vars = {}
    widgets = {}

    # pattern type dropdown selection
    pattern_type_combo = QComboBox()
    # Customized Pattern
    pattern_type_combo.addItems(["Gray Code", "Fringe Pattern", "Customized Pattern"])
    form.addRow(QLabel("Pattern Type"), pattern_type_combo)
    widgets["pattern_type"] = pattern_type_combo

    # Fringe parameters input boxes (initially hidden)
    period_label = QLabel("Fringe Periods (px)")
    fringe_period_box = QLineEdit()
    fringe_period_box.setPlaceholderText("e.g. 20,40,60")
    form.addRow(period_label, fringe_period_box)
    widgets["fringe_periods"] = fringe_period_box

    phase_label = QLabel("Phase Shifts")
    fringe_phase_box = QLineEdit()
    fringe_phase_box.setPlaceholderText(">=3")
    form.addRow(phase_label, fringe_phase_box)
    widgets["phase_shifts"] = fringe_phase_box

    for w in (period_label, fringe_period_box, phase_label, fringe_phase_box):
        w.hide()
    # for w in (
    #     inverse_label,
    #     widgets["inverse_pattern"],
    #     gray_step_label,
    #     widgets["gray_code_step"],
    # ):
    #     w.hide()

    # according to dropdown selection, show/hide relevant controls
    def on_pattern_type_changed(text):
        is_fringe = text == "Fringe Pattern"
        is_custom = text == "Customized Pattern"

        # if it's Fringe Pattern, show fringe controls
        for w in (period_label, fringe_period_box, phase_label, fringe_phase_box):
            w.setVisible(is_fringe)

        # if it's Gray Code, show gray code controls; hide all for Customized
        show_gray = text == "Gray Code"
        for w in (
            inverse_label,
            widgets["inverse_pattern"],
            gray_step_label,
            widgets["gray_code_step"],
        ):
            w.setVisible(show_gray)

        # when fringe / gray code, both groups are not visible; otherwise visible as above

    pattern_type_combo.currentTextChanged.connect(on_pattern_type_changed)

    # Burst Count
    widgets["burst_count"] = QLineEdit(str(default_projector_config["burst_count"]))
    form.addRow(QLabel("Burst Count"), widgets["burst_count"])

    # Pattern folder
    hl = QHBoxLayout()
    widgets["pattern_folder"] = QLineEdit(default_projector_config["pattern_folder"])
    btn_browse = QPushButton("Browse…")
    hl.addWidget(widgets["pattern_folder"])
    hl.addWidget(btn_browse)
    form.addRow(QLabel("Pattern Folder"), hl)
    btn_browse.clicked.connect(lambda: _on_browse(panel, widgets))

    # default program dir / calibration_capture, if exists, clear all contents; else create it
    hl2 = QHBoxLayout()
    # get default path from config
    save_dir = default_projector_config["capture_save_folder"]
    # if folder exists, clear all contents; else create it
    if os.path.exists(save_dir):
        pass
        # for fn in os.listdir(save_dir):
        #     path = os.path.join(save_dir, fn)
        #     try:
        #         if os.path.isdir(path):
        #             shutil.rmtree(path)
        #         else:
        #             os.remove(path)
        #     except Exception:
        #         pass
    else:
        os.makedirs(save_dir, exist_ok=True)
    # set the default path
    widgets["save_folder"] = QLineEdit(save_dir)
    btn_save_browse = QPushButton("Browse…")
    hl2.addWidget(widgets["save_folder"])
    hl2.addWidget(btn_save_browse)
    form.addRow(QLabel("Capture Save Folder"), hl2)
    btn_save_browse.clicked.connect(lambda: _on_browse_save(panel, widgets))

    # other parameters：screen, width, height, background
    for key in [
        "screen_number",
        "projector_width",
        "projector_height",
        "pattern_background",
    ]:
        widgets[key] = QLineEdit(str(default_projector_config[key]))
        form.addRow(QLabel(key.replace("_", " ").title()), widgets[key])

    gray_step_label = QLabel("Gray Code Step")
    widgets["gray_code_step"] = QLineEdit(
        str(default_projector_config["gray_code_step"])
    )
    form.addRow(gray_step_label, widgets["gray_code_step"])

    # inverse / loop / scale
    widgets["inverse_pattern"] = QCheckBox()
    widgets["inverse_pattern"].setChecked(default_projector_config["inverse_pattern"])
    inverse_label = QLabel("Inverse Pattern")
    form.addRow(inverse_label, widgets["inverse_pattern"])
    widgets["inverse_pattern"].setVisible(True)

    # Synth Bit0 checkbox
    widgets["synth_bit0"] = QCheckBox()
    widgets["synth_bit0"].setChecked(
        default_projector_config.get("default_synth_bit0", False)
    )  # default unchecked
    form.addRow(QLabel("Synthesize Bit0 (Shifted Safe Pattern)"), widgets["synth_bit0"])

    widgets["loop"] = QCheckBox()
    widgets["loop"].setChecked(True)
    form.addRow(QLabel("Loop Projection"), widgets["loop"])

    widgets["scale"] = QCheckBox()
    widgets["scale"].setChecked(False)
    form.addRow(QLabel("Scale to Fullscreen"), widgets["scale"])

    # button layout: two rows, two columns, equal width
    btn_gen = QPushButton("Generate Pattern")
    btn_prev = QPushButton("Projector Preview")
    btn_cap = QPushButton("Projection Capture")
    btn_restart = QPushButton("Restart Capture")
  
    # first row: two buttons share horizontal space equally, keep default height
    row1 = QHBoxLayout()
    row1.addWidget(btn_gen)
    row1.addWidget(btn_prev)
    row1.setStretch(0, 1)
    row1.setStretch(1, 1)
    vlayout.addLayout(row1)

    # second row: two buttons share horizontal space equally, keep default height
    row2 = QHBoxLayout()
    row2.addWidget(btn_cap)
    row2.addWidget(btn_restart)
    row2.setStretch(0, 1)
    row2.setStretch(1, 1)
    vlayout.addLayout(row2)

    # button: average npy and save 
    btn_avg = QPushButton("Average Captures")
    btn_avg.setEnabled(False)
    panel.avg_btn = btn_avg
    btn_avg.clicked.connect(lambda: _on_average(panel, widgets))

    # third row: button + stretch
    row3 = QHBoxLayout()
    # button takes 1 part, blank takes 1 part
    row3.addWidget(btn_avg, 1)
    row3.addStretch(1)
    vlayout.addLayout(row3)

    # mounting button handlers
    btn_gen.clicked.connect(lambda: _on_generate(widgets, panel))
    btn_prev.clicked.connect(lambda: _on_preview(widgets, panel))
    btn_cap.clicked.connect(
        lambda: _on_capture(
            widgets,
            panel,
            handler,
            start_stream_fn,
            stop_stream_fn,
        )
    )
    btn_restart.clicked.connect(lambda: _on_restart(panel, widgets))

    return panel, widgets, {}


# auxiliary functions for button handlers
def _on_browse_save(panel, widgets):
    folder = QFileDialog.getExistingDirectory(panel, "Select Save Folder")
    if folder:
        widgets["save_folder"].setText(folder)


def _on_restart(panel, widgets):
    # pop input dialog to set next session_count
    num, ok = QInputDialog.getInt(
        panel,
        "Restart Capture",
        "Start from count:",
        panel.session_count,
        1,
        9999,
    )
    if ok:
        panel.session_count = num
        # remove existing folder if exists
        save_dir = widgets["save_folder"].text()
        target = os.path.join(save_dir, str(num))
        if os.path.isdir(target):
            shutil.rmtree(target)
        QMessageBox.information(
            panel, "Restarted", f"Next capture will start from {num}"
        )


def _on_browse(panel, widgets):
    folder = QFileDialog.getExistingDirectory(panel, "Select Pattern Folder")
    if folder:
        widgets["pattern_folder"].setText(folder)


def _on_generate(widgets, panel):
    try:
        folder = widgets["pattern_folder"].text()
        width = int(widgets["projector_width"].text())
        height = int(widgets["projector_height"].text())
        step = int(widgets["gray_code_step"].text())
        inverse = widgets["inverse_pattern"].isChecked()
        synth_bit0 = widgets["synth_bit0"].isChecked()
        generate_gray_code_patterns(
            width, height, folder, inverse, step, synth_bit0=synth_bit0
        )
        QMessageBox.information(panel, "Success", f"Patterns generated in {folder}")
    except Exception as e:
        QMessageBox.critical(panel, "Error", f"Pattern generation failed: {e}")


def _on_preview(widgets, panel):
    try:
        pattern_type = widgets["pattern_type"].currentText()
        # if it's Customized Pattern, directly launch viewer with all images in folder
        if pattern_type == "Customized Pattern":
            image_folder = widgets["pattern_folder"].text()
            launch_viewer(
                image_folder=image_folder,
                target_screen=int(widgets["screen_number"].text()),
                loop=widgets["loop"].isChecked(),
                scale=widgets["scale"].isChecked(),
                bg_gray=int(widgets["pattern_background"].text()),
                mode="preview",
                capture_condition_fn=None,
            )
        else:
            # keep the original preview logic for Gray Code / Fringe
            launch_viewer(
                image_folder=widgets["pattern_folder"].text(),
                target_screen=int(widgets["screen_number"].text()),
                loop=widgets["loop"].isChecked(),
                scale=widgets["scale"].isChecked(),
                bg_gray=int(widgets["pattern_background"].text()),
                mode="preview",
                capture_condition_fn=None,
            )
    except Exception as e:
        QMessageBox.critical(panel, "Error", f"Preview failed: {e}")
    # try:
    #     launch_viewer(
    #         image_folder=widgets["pattern_folder"].text(),
    #         target_screen=int(widgets["screen_number"].text()),
    #         loop=widgets["loop"].isChecked(),
    #         scale=widgets["scale"].isChecked(),
    #         bg_gray=int(widgets["pattern_background"].text()),
    #         mode="preview",
    #         capture_condition_fn=None,
    #     )
    # except Exception as e:
    #     QMessageBox.critical(panel, "Error", f"Preview failed: {e}")


class CaptureThread(QThread):
    """
    Background thread to run structured‑light capture without blocking GUI.
    """

    # request main thread to show pattern
    show_pattern = pyqtSignal(str)

    FRAME_TIMEOUT = 5.0  # seconds to wait for each frame
    # original signals
    finished = pyqtSignal(int, int)
    error = pyqtSignal(str)

    ## tell main thread to show/hide pattern
    # show_pattern = pyqtSignal(str)
    # hide_pattern = pyqtSignal()

    def __init__(self, widgets, handler, start_fn, stop_fn, session_idx, parent=None):
        super().__init__(parent)
        self.widgets = widgets
        self.handler = handler
        self.start_fn = start_fn
        self.stop_fn = stop_fn
        self.session_idx = session_idx

    def run(self):
        try:
            # prepare save folder directory
            save_root = self.widgets["save_folder"].text().strip()
            if not save_root:
                self.error.emit("Please select Save Folder before capture.")
                return
            session_dir = os.path.join(save_root, str(self.session_idx))
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
                os.makedirs(session_dir, exist_ok=True)
            except OSError as e:
                self.error.emit(
                    f"Failed to create session directory {session_dir}: {e}"
                )
                return

            # read parameters
            try:
                frames_per = int(self.widgets["burst_count"].text())
            except ValueError:
                self.error.emit("Invalid Burst Count value.")
                return

            patt_folder = self.widgets["pattern_folder"].text()
            try:
                # choose sorting method based on Pattern Type
                ptype = self.widgets["pattern_type"].currentText()
                all_pngs = [
                    os.path.join(patt_folder, f)
                    for f in os.listdir(patt_folder)
                    if f.lower().endswith(".png")
                ]
                if ptype == "Customized Pattern":
                    # customized mode: sort by filename (or other simple sorting)
                    patterns = sorted(all_pngs)
                else:
                    # gray code / fringe mode: use original pattern_key sorting
                    patterns = sorted(all_pngs, key=self.pattern_key)
            except OSError as e:
                self.error.emit(f"Cannot read Pattern Folder {patt_folder}: {e}")
                return
            if not patterns:
                self.error.emit("Pattern Folder must contain at least one PNG image.")
                return

            # capture each pattern in sequence: Stop → Flush → Show → Start → Grab → Stop
            for pat_i, img_path in enumerate(patterns):
                # stop streaming first to avoid old frames coming in
                try:
                    self.stop_fn()
                except Exception:
                    pass
                t0 = time.time()
                while state["streaming"] and time.time() - t0 < 2:
                    time.sleep(0.02)

                # clear any old frames in the queue
                _flush_queues(self.handler)

                # project pattern (must be excuted in GUI thread) and wait until it's really drawn
                # request main thread to show pattern and wait for image_changed signal
                done_evt = threading.Event()

                def _ready(_):
                    done_evt.set()

                # let main thread create / switch ImageViewer
                self.show_pattern.emit(img_path)

                # Wait for ImageViewer to emit the `image_changed` signal
                # Wait until the main thread has created a new viewer that has not been destroyed
                while True:
                    viewer = self.widgets.get("_pat_viewer")
                    if viewer is not None and not sip.isdeleted(viewer):
                        break
                    time.sleep(0.01)
                viewer.image_changed.connect(_ready)
                done_evt.wait(0.2)  # wait no more than 200 ms
                viewer.image_changed.disconnect(_ready)
                viewer = self.widgets.get("_pat_viewer")  # viewer created in main thread
                # Wait for the `image_changed` signal (up to 100 ms)
                image_ready = threading.Event()

                def _ready(path):
                    image_ready.set()

                viewer.image_changed.connect(_ready)
                image_ready.wait(0.1)
                viewer.image_changed.disconnect(_ready)
                time.sleep(0.1)  # wait for drawing settle

                # Restart streaming with the minimum buffer size
                try:
                    self.start_fn(frames_per + 2)
                except TypeError:  # compatible with old signature
                    self.start_fn()
                t0 = time.time()
                while not state["streaming"] and time.time() - t0 < 2:
                    time.sleep(0.02)
                if not state["streaming"]:
                    self.error.emit("Timeout while starting camera streaming.")
                    return

                # Create a subdirectory for this pattern; clear it if it already exists
                # subdir = os.path.join(session_dir, f"{pat_i:02d}")
                # if os.path.isdir(subdir):
                #     try:
                #         shutil.rmtree(subdir)
                #     except Exception as e:
                #         self.error.emit(
                #             f"Failed to clear existing folder {subdir}: {e}"
                #         )
                #         self.stop_fn()
                #         return
                # try:
                #     os.makedirs(subdir, exist_ok=True)
                # except OSError as e:
                #     self.error.emit(f"Cannot create folder {subdir}: {e}")
                #     self.stop_fn()
                #     return

                # Capture `frames_per` frames
                for frame_i in range(frames_per):
                    try:
                        np_frame = self.handler.np_queue.get(timeout=self.FRAME_TIMEOUT)
                    except Empty:
                        self.error.emit(
                            f"Timeout ({self.FRAME_TIMEOUT}s) waiting for frame {frame_i} of pattern {pat_i}"
                        )
                        self.stop_fn()
                        return

                    # If this is a "Customized Pattern", use the original filename as the prefix;
                    # otherwise, continue using the numeric index
                    pattern_name = os.path.splitext(os.path.basename(img_path))[0]
                    if ptype == "Customized Pattern":
                        save_name = f"{pattern_name}_projection_{frame_i:02d}.png"
                    else:
                        if (
                            pattern_name.startswith("VerGrayCode_")
                            or pattern_name.startswith("HorGrayCode_")
                            or pattern_name in ["WhitePattern", "BlackPattern"]
                        ):
                            save_name = f"{pat_i:02d}_{frame_i:02d}.png"
                        else:
                            save_name = f"{pattern_name}_{frame_i:02d}.png"
                    try:
                        cv2.imwrite(
                            os.path.join(session_dir, save_name),
                            np_frame.astype(np.uint8),
                        )
                    except Exception as e:
                        self.error.emit(f"Failed saving frame {frame_i}: {e}")
                        self.stop_fn()
                        return
                # Stop streaming immediately after capture and clear the queue again
                self.stop_fn()
                _flush_queues(self.handler)

            # all done
            self.finished.emit(len(patterns), frames_per)

        finally:
            # Ensure the camera is stopped on all execution paths
            try:
                self.stop_fn()
            except Exception:
                pass

    def pattern_key(self, path):
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
        # Place other files at the end and sort them again by name
        return (4, name)


def _on_capture(
    widgets,
    panel,
    handler,
    start_stream_fn,
    stop_stream_fn,
):
    panel.calib_widgets = widgets
    # Concurrency safety check
    if getattr(panel, "is_capturing", False):
        QMessageBox.warning(panel, "Warning", "A capture is already in progress.")
        return
    if not state["streaming"]:
        QMessageBox.warning(
            panel, "Warning", "Camera must be streaming before capture."
        )
        return
    panel.is_capturing = True

    # Write the "Burst Count" into the handler
    # try:
    #     handler.frames_per = int(widgets["burst_count"].text())
    # except ValueError:
    #     QMessageBox.warning(panel, "Error", "Invalid Burst Count value.")
    #     return
    try:
        new_fp = int(widgets["burst_count"].text())
        # Update parameters and counters using the same lock
        with handler._lock:
            handler.frames_per = new_fp
            handler._cnt = 0
    except ValueError:
        QMessageBox.warning(panel, "Error", "Invalid Burst Count value.")
        return

    # Let CaptureThread exclusively control pattern switching. The GUI side no longer
    # listens for `frameset_done`, to avoid a race condition where the screen switches
    # patterns while the thread is still saving the previous frames.

    # disable button
    for btn in panel.findChildren(QPushButton):
        if btn.text() in (
            "Generate Pattern",
            "Projector Preview",
            "Projection Capture",
        ):
            btn.setEnabled(False)

    # Create and start the thread
    thread = CaptureThread(
        widgets, handler, start_stream_fn, stop_stream_fn, panel.session_count
    )
    panel._capture_thread = thread  # keep a reference

    # Deliver the signal to the GUI thread to run `_on_show_pattern`
    thread.show_pattern.connect(lambda img: _on_show_pattern(widgets, img))

    # Original: completion and error signals
    thread.finished.connect(lambda p, f: _on_capture_done(panel, p, f))
    # Capture the widgets in the closure as well
    thread.error.connect(lambda msg: _on_capture_error(panel, widgets, msg))

    thread.start()


def _on_capture_done(panel, patterns_count, frames_per):
    panel.is_capturing = False
    panel.session_count += 1
    # Display WhitePattern.png
    white = os.path.join(
        panel.calib_widgets["pattern_folder"].text(), "WhitePattern.png"
    )
    if os.path.isfile(white):
        _on_show_pattern(panel.calib_widgets, white)
    QMessageBox.information(
        panel,
        f"Session #{panel.session_count - 1} is done!",
        f"Captured {patterns_count} patterns × {frames_per} frames.",
    )
    # restart streaming
    try:
        panel._start_stream_fn(frames_per + 2)  # optional buffer size
        # Wait for the streaming state to become True (up to 2 seconds)
        t0 = time.time()
        while not state["streaming"] and time.time() - t0 < 2:
            time.sleep(0.02)
    except Exception as e:
        print(f"[Warn] Failed to restart streaming: {e}")

    for btn in panel.findChildren(QPushButton):
        if btn.text() in (
            "Generate Pattern",
            "Projector Preview",
            "Projection Capture",
        ):
            btn.setEnabled(True)

    # activate "Average Captures" button
    getattr(panel, "avg_btn", QPushButton()).setEnabled(True)

    del panel._capture_thread


def _on_capture_error(panel, widgets, msg):
    """
    Error handler for capture thread.

    :param panel: The calibration panel widget
    :param widgets: The dict of calibration controls passed into _on_capture
    :param msg: The error message emitted by the thread
    """
    # first close residual pattern viewer
    _on_hide_pattern(widgets)
    # if it's a capture error, show white pattern
    white = os.path.join(widgets["pattern_folder"].text(), "WhitePattern.png")
    if os.path.isfile(white):
        _on_show_pattern(widgets, white)

    # reset is_capturing flag
    panel.is_capturing = False

    # recover buttons
    for btn in panel.findChildren(QPushButton):
        if btn.text() in (
            "Generate Pattern",
            "Projector Preview",
            "Projection Capture",
        ):
            btn.setEnabled(True)

    # pop up error message
    QMessageBox.critical(panel, "Error", msg)
    
    # delete the thread reference to avoid memory leak
    if hasattr(panel, "_capture_thread"):
        del panel._capture_thread


def _on_show_pattern(widgets, img_path):
    """ safely create or reuse an ImageViewer to show a single pattern image in the main thread. """
    # QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    # screen_idx = int(widgets["screen_number"].text())
    # screen_geom = app.screens()[screen_idx].geometry()
    screen_idx = int(widgets["screen_number"].text())
    screens = app.screens()
    if screen_idx >= len(screens):  # return to primary screen if out of range
        screen_idx = 0
    screen_geom = screens[screen_idx].geometry()

    # if it is not created yet, create a new one; otherwise update image_paths                
    viewer = widgets.get("_pat_viewer")

    # if not existing viewer or closed by Esc and C++ object deleted, then recreate
    if viewer is None or sip.isdeleted(viewer) or not viewer.isVisible():
        viewer = ImageViewer(
            [img_path],
            screen_geom,
            loop=False,
            scale=widgets["scale"].isChecked(),
            bg_gray=int(widgets["pattern_background"].text()),
            mode="preview",
            capture_condition_fn=None,
        )
        widgets["_pat_viewer"] = viewer
    else:
        viewer.image_paths = [img_path]
        viewer.current_index = 0
        viewer.show_image()

    viewer.show()
    return viewer  # return current viewer for further use


def _on_hide_pattern(widgets):
    """ Close the pattern viewer in the main thread. """
    viewer = widgets.get("_pat_viewer")
    if viewer and not sip.isdeleted(viewer):
        viewer.close()
    # no matter viewer existed or not, set to None to avoid dangling pointer
    widgets["_pat_viewer"] = None


def _advance_pattern(widgets):
    """Advance to the next pattern in the list when frameset_done, or show white pattern if done."""
    patterns = widgets.get("_patterns", [])
    if not patterns:
        return
    viewer = widgets.get("_pat_viewer")
    if not viewer:
        return

    widgets["_pat_idx"] += 1
    if widgets["_pat_idx"] < len(patterns):
        next_img = patterns[widgets["_pat_idx"]]
        _on_show_pattern(widgets, next_img)
    else:
        # all done → white pattern
        white = os.path.join(widgets["pattern_folder"].text(), "WhitePattern.png")
        if os.path.isfile(white):
            _on_show_pattern(widgets, white)
        else:
            _on_hide_pattern(widgets)


# Helper: flush handler queues
def _flush_queues(handler):
    for q in (handler.cv_queue, handler.np_queue):
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass


def _on_average(panel, widgets):
    """iterate through all session/bit directories, compute average and save as .npy and .png"""
    save_root = widgets["save_folder"].text().strip()
    if not save_root or not os.path.isdir(save_root):
        QMessageBox.warning(
            panel, "Warning", "Please complete a Projection Capture first and ensure the Save Folder is valid."
        )
        return
    for session in tqdm(os.listdir(save_root), desc="Sessions"):
        session_dir = os.path.join(save_root, session)
        if not os.path.isdir(session_dir):
            continue
        for bit in tqdm(
            os.listdir(session_dir), desc=f"Session {session}", leave=False
        ):
            bit_dir = os.path.join(session_dir, bit)
            if not os.path.isdir(bit_dir):
                continue
            # read all .npy files (excluding existing average files)
            files = [
                f
                for f in os.listdir(bit_dir)
                if f.endswith(".npy") and not f.endswith("_avrg.npy")
            ]
            if not files:
                continue
            arrs = [np.load(os.path.join(bit_dir, f)).astype(float) for f in files]
            avg = sum(arrs) / len(arrs)
            out_npy = os.path.join(bit_dir, f"{session}_{bit}_avrg.npy")
            np.save(out_npy, avg.astype(float))
            # save as 8-bit PNG
            img8 = np.clip(avg, 0, 255).astype(np.uint8)
            out_png = os.path.join(bit_dir, f"{session}_{bit}_avrg.png")
            cv2.imwrite(out_png, img8)
    QMessageBox.information(panel, "Completed", "All captures averaged and saved.")
