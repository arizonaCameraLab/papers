from PyQt5.QtWidgets import (
    QSizePolicy,
    QWidget,  # Base class for all UI panels
    QSpinBox,  # <-- NEW: for integer ranges
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
from global_state import state  # 新增
from queue import Empty
from PyQt5.QtCore import Qt
import threading
from PyQt5 import sip
import cv2
import numpy as np
from tqdm.auto import tqdm  # For progress bar in average calculation


# Convert an exposure time into a slider position on a logarithmic scale
def exposure_to_slider(exposure: int) -> int:
    exposure = max(exposure, MIN_EXPOSURE)  # 防止 ≤0
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
    # 两个按钮平分水平空间，保持默认高度
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
    # --- New: binning-related enum choices (constrained) ---
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

        # --- New: Binning enums and integer ranges ---
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

    # clicked(bool) 会携带一个 checked 参数；用 lambda 把它丢掉
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
    panel.session_count = 1  # 初始化第 1 次
    panel.is_capturing = False  # <— 新增：当前没有正在采集
    panel._start_stream_fn = start_stream_fn  # ➊ 记住启动函数
    panel._stop_stream_fn = stop_stream_fn  #    （可选，备用）
    vlayout = QVBoxLayout(panel)
    form = QFormLayout()
    vlayout.addLayout(form)
    # vars = {}
    widgets = {}

    # === Pattern Type 下拉框 ===
    pattern_type_combo = QComboBox()
    # 新增 “Customized Pattern” 选项
    pattern_type_combo.addItems(["Gray Code", "Fringe Pattern", "Customized Pattern"])
    form.addRow(QLabel("Pattern Type"), pattern_type_combo)
    widgets["pattern_type"] = pattern_type_combo

    # === Fringe 参数输入框（初始隐藏） ===
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

    # 根据下拉选项动态显示/隐藏
    def on_pattern_type_changed(text):
        is_fringe = text == "Fringe Pattern"
        is_custom = text == "Customized Pattern"

        # —— 如果是 Fringe，只显示 Fringe 相关控件 —— #
        for w in (period_label, fringe_period_box, phase_label, fringe_phase_box):
            w.setVisible(is_fringe)

        # —— 如果是 Gray Code，则显示 Gray Code 相关控件；如果是 Custom 则都隐藏 —— #
        show_gray = text == "Gray Code"
        for w in (
            inverse_label,
            widgets["inverse_pattern"],
            gray_step_label,
            widgets["gray_code_step"],
        ):
            w.setVisible(show_gray)

        # （Customized Pattern 时，两组都隐藏；Fringe/Gray 则如上分别可见）

    pattern_type_combo.currentTextChanged.connect(on_pattern_type_changed)

    # === Burst Count ===
    widgets["burst_count"] = QLineEdit(str(default_projector_config["burst_count"]))
    form.addRow(QLabel("Burst Count"), widgets["burst_count"])

    # === Pattern folder ===
    hl = QHBoxLayout()
    widgets["pattern_folder"] = QLineEdit(default_projector_config["pattern_folder"])
    btn_browse = QPushButton("Browse…")
    hl.addWidget(widgets["pattern_folder"])
    hl.addWidget(btn_browse)
    form.addRow(QLabel("Pattern Folder"), hl)
    btn_browse.clicked.connect(lambda: _on_browse(panel, widgets))

    # === Save folder 新增：默认 program_dir/calibration_capture，若已存在则清空 ===
    hl2 = QHBoxLayout()
    # 从 config 拿默认路径
    save_dir = default_projector_config["capture_save_folder"]
    # 如果存在，就删除所有子文件／目录；否则创建它
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
    # 设置成默认路径
    widgets["save_folder"] = QLineEdit(save_dir)
    btn_save_browse = QPushButton("Browse…")
    hl2.addWidget(widgets["save_folder"])
    hl2.addWidget(btn_save_browse)
    form.addRow(QLabel("Capture Save Folder"), hl2)
    btn_save_browse.clicked.connect(lambda: _on_browse_save(panel, widgets))

    # === 其它参数：screen, width, height, background ===
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

    # --- NEW: Synth Bit0 checkbox ---
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

    # === 按钮组：两行两列，横向等宽自适应 ===
    btn_gen = QPushButton("Generate Pattern")
    btn_prev = QPushButton("Projector Preview")
    btn_cap = QPushButton("Projection Capture")
    btn_restart = QPushButton("Restart Capture")
    # 第一行：两个按钮平分水平空间，保持默认高度
    row1 = QHBoxLayout()
    row1.addWidget(btn_gen)
    row1.addWidget(btn_prev)
    row1.setStretch(0, 1)
    row1.setStretch(1, 1)
    vlayout.addLayout(row1)

    # 第二行：两个按钮平分水平空间，保持默认高度
    row2 = QHBoxLayout()
    row2.addWidget(btn_cap)
    row2.addWidget(btn_restart)
    row2.setStretch(0, 1)
    row2.setStretch(1, 1)
    vlayout.addLayout(row2)

    # —— 新增：平均 NPY 并保存按钮 —— #
    btn_avg = QPushButton("Average Captures")
    btn_avg.setEnabled(False)
    panel.avg_btn = btn_avg
    btn_avg.clicked.connect(lambda: _on_average(panel, widgets))

    # —— 新增：单独一行，按钮占一半宽度 —— #
    row3 = QHBoxLayout()
    # 按钮占 1 份，空白占 1 份
    row3.addWidget(btn_avg, 1)
    row3.addStretch(1)
    vlayout.addLayout(row3)

    # === 挂载回调 ===
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


# —— 辅助函数，实现按钮逻辑 —— #
def _on_browse_save(panel, widgets):
    folder = QFileDialog.getExistingDirectory(panel, "Select Save Folder")
    if folder:
        widgets["save_folder"].setText(folder)


def _on_restart(panel, widgets):
    # 弹出输入框设定下次 session_count
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
        # 如果对应文件夹已存在，先清空
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
        # —— 如果是 “Customized Pattern”，直接投影用户指定文件夹里的所有图片 —— #
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
            # —— 保留原来的 Gray Code / Fringe 预览逻辑 —— #
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

    # --- 新增：跨线程请求主线程显示图案 ---
    show_pattern = pyqtSignal(str)

    FRAME_TIMEOUT = 5.0  # seconds to wait for each frame
    # 原有信号
    finished = pyqtSignal(int, int)
    error = pyqtSignal(str)
    # # 新增：告诉主线程显示哪张 pattern／隐藏 pattern
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
            # ------------------------------------------------------------------
            # 0. 准备保存目录
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # 1. 读取参数
            # ------------------------------------------------------------------

            try:
                frames_per = int(self.widgets["burst_count"].text())
            except ValueError:
                self.error.emit("Invalid Burst Count value.")
                return

            patt_folder = self.widgets["pattern_folder"].text()
            try:
                # —— 根据期望的 Pattern Type 选择排序方式 —— #
                ptype = self.widgets["pattern_type"].currentText()
                all_pngs = [
                    os.path.join(patt_folder, f)
                    for f in os.listdir(patt_folder)
                    if f.lower().endswith(".png")
                ]
                if ptype == "Customized Pattern":
                    # 自定义模式：按文件名排序（或其他简单排序）
                    patterns = sorted(all_pngs)
                else:
                    # Gray Code / Fringe 模式：仍然使用原来的 pattern_key 排序
                    patterns = sorted(all_pngs, key=self.pattern_key)
            except OSError as e:
                self.error.emit(f"Cannot read Pattern Folder {patt_folder}: {e}")
                return
            if not patterns:
                self.error.emit("Pattern Folder must contain at least one PNG image.")
                return

            # ------------------------------------------------------------------
            # 2. 逐‑pattern 采集：Stop → Flush → Show → Start → Grab → Stop
            # ------------------------------------------------------------------
            for pat_i, img_path in enumerate(patterns):
                # 2‑1 先停流，保证无旧帧继续灌入
                try:
                    self.stop_fn()
                except Exception:
                    pass
                t0 = time.time()
                while state["streaming"] and time.time() - t0 < 2:
                    time.sleep(0.02)

                # 2‑2 清空 handler 队列
                _flush_queues(self.handler)

                # 2‑3 投屏（必须在 GUI 线程执行）并等待 ImageViewer 真正完成绘制
                # ── 通过信号请求主线程显示图案，并阻塞等待它真正绘制完 ── #
                done_evt = threading.Event()

                def _ready(_):
                    done_evt.set()

                # 1) 让主线程创建 / 切换 ImageViewer
                self.show_pattern.emit(img_path)

                # 2) 等到 ImageViewer 反馈 image_changed
                # 等到主线程创建了一个未被删除的新 viewer
                while True:
                    viewer = self.widgets.get("_pat_viewer")
                    if viewer is not None and not sip.isdeleted(viewer):
                        break
                    time.sleep(0.01)
                viewer.image_changed.connect(_ready)
                done_evt.wait(0.2)  # 最多等 200 ms
                viewer.image_changed.disconnect(_ready)
                viewer = self.widgets.get("_pat_viewer")  # 主线程创建好的 viewer
                # 等待 image_changed 信号（最长 100ms）
                image_ready = threading.Event()

                def _ready(path):
                    image_ready.set()

                viewer.image_changed.connect(_ready)
                image_ready.wait(0.1)
                viewer.image_changed.disconnect(_ready)
                time.sleep(0.1)  # 等待绘制完成

                # 2‑4 重新开流，使用最小 buffer
                try:
                    self.start_fn(frames_per + 2)
                except TypeError:  # 兼容旧签名
                    self.start_fn()
                t0 = time.time()
                while not state["streaming"] and time.time() - t0 < 2:
                    time.sleep(0.02)
                if not state["streaming"]:
                    self.error.emit("Timeout while starting camera streaming.")
                    return

                # # 2-5 为该图案创建子目录，如果已存在则清空
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

                # 2‑6 抓取 frames_per 帧
                for frame_i in range(frames_per):
                    try:
                        np_frame = self.handler.np_queue.get(timeout=self.FRAME_TIMEOUT)
                    except Empty:
                        self.error.emit(
                            f"Timeout ({self.FRAME_TIMEOUT}s) waiting for frame {frame_i} of pattern {pat_i}"
                        )
                        self.stop_fn()
                        return
                    # —— 如果是 “Customized Pattern”，就用原始文件名作为前缀，否则沿用数字索引 —— #
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
                # 2‑7 采完立即停流并再次清空队列
                self.stop_fn()
                _flush_queues(self.handler)

            # ------------------------------------------------------------------
            # 3. 全部完成
            # ------------------------------------------------------------------
            self.finished.emit(len(patterns), frames_per)

        finally:
            # 确保相机在所有路径上都被停掉
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
        # 其它文件放最后，按名字再排一次
        return (4, name)


def _on_capture(
    widgets,
    panel,
    handler,
    start_stream_fn,
    stop_stream_fn,
):
    panel.calib_widgets = widgets
    # 并发安全检查
    if getattr(panel, "is_capturing", False):
        QMessageBox.warning(panel, "Warning", "A capture is already in progress.")
        return
    if not state["streaming"]:
        QMessageBox.warning(
            panel, "Warning", "Camera must be streaming before capture."
        )
        return
    panel.is_capturing = True

    # ★★ 把“Burst Count”写进 handler ★★
    # try:
    #     handler.frames_per = int(widgets["burst_count"].text())
    # except ValueError:
    #     QMessageBox.warning(panel, "Error", "Invalid Burst Count value.")
    #     return
    try:
        new_fp = int(widgets["burst_count"].text())
        # 使用同一把锁更新参数与计数
        with handler._lock:
            handler.frames_per = new_fp
            handler._cnt = 0
    except ValueError:
        QMessageBox.warning(panel, "Error", "Invalid Burst Count value.")
        return

    # 让 CaptureThread 独占图案切换，GUI 端不再监听 frameset_done，
    # 避免“屏幕先切图、线程还在保存旧图”的竞态

    # 禁用按钮
    for btn in panel.findChildren(QPushButton):
        if btn.text() in (
            "Generate Pattern",
            "Projector Preview",
            "Projection Capture",
        ):
            btn.setEnabled(False)

    # 创建并启动线程
    thread = CaptureThread(
        widgets, handler, start_stream_fn, stop_stream_fn, panel.session_count
    )
    panel._capture_thread = thread  # 保留引用

    # 让信号跑到 GUI 线程执行 _on_show_pattern
    thread.show_pattern.connect(lambda img: _on_show_pattern(widgets, img))

    # 原有：完成和错误信号
    thread.finished.connect(lambda p, f: _on_capture_done(panel, p, f))
    # 把 widgets 也闭包进来
    thread.error.connect(lambda msg: _on_capture_error(panel, widgets, msg))

    thread.start()


def _on_capture_done(panel, patterns_count, frames_per):
    panel.is_capturing = False
    panel.session_count += 1
    # ➜ 新增：显示 WhitePattern.png
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
    # 重启 streaming
    try:
        panel._start_stream_fn(frames_per + 2)  # optional buffer size
        # 等待 streaming 状态变为 True，最多等 2 秒
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

    # 激活“Average Captures”按钮
    getattr(panel, "avg_btn", QPushButton()).setEnabled(True)

    del panel._capture_thread


def _on_capture_error(panel, widgets, msg):
    """
    Error handler for capture thread.

    :param panel: The calibration panel widget
    :param widgets: The dict of calibration controls passed into _on_capture
    :param msg: The error message emitted by the thread
    """
    # 1. 先关闭残留的 pattern viewer
    _on_hide_pattern(widgets)
    # 1.1 如果是捕获失败，显示白色图案
    white = os.path.join(widgets["pattern_folder"].text(), "WhitePattern.png")
    if os.path.isfile(white):
        _on_show_pattern(widgets, white)

    # 2. 重置标志
    panel.is_capturing = False

    # 3. 恢复按钮（根据按钮文本匹配）
    for btn in panel.findChildren(QPushButton):
        if btn.text() in (
            "Generate Pattern",
            "Projector Preview",
            "Projection Capture",
        ):
            btn.setEnabled(True)

    # 4. 弹出错误提示框
    QMessageBox.critical(panel, "Error", msg)

    # 5. 删除线程引用，避免内存泄漏
    if hasattr(panel, "_capture_thread"):
        del panel._capture_thread


def _on_show_pattern(widgets, img_path):
    """在主线程里安全创建／复用 ImageViewer 并显示单张 pattern"""
    # 获取或创建 QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    # 读取屏幕几号
    # screen_idx = int(widgets["screen_number"].text())
    # screen_geom = app.screens()[screen_idx].geometry()
    screen_idx = int(widgets["screen_number"].text())
    screens = app.screens()
    if screen_idx >= len(screens):  # 越界回退主屏
        screen_idx = 0
    screen_geom = screens[screen_idx].geometry()

    # 如果之前没创建，就新建；否则更新 image_paths                   # ⬅︎ 新增
    viewer = widgets.get("_pat_viewer")

    # 如果不存在，或已被 Esc 关闭而实际 C++ 对象已删除，就重新创建
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
    return viewer  # 返回当前 viewer 供后续使用


def _on_hide_pattern(widgets):
    """在主线程里关闭 pattern viewer"""
    viewer = widgets.get("_pat_viewer")
    if viewer and not sip.isdeleted(viewer):
        viewer.close()
    # 无论是否已删除，都把字典项设成 None，避免残留野指针
    widgets["_pat_viewer"] = None


def _advance_pattern(widgets):
    """收到 frameset_done 后切到下一张；全部结束后转白场"""
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
        # 全部完成 → 白场
        white = os.path.join(widgets["pattern_folder"].text(), "WhitePattern.png")
        if os.path.isfile(white):
            _on_show_pattern(widgets, white)
        else:
            _on_hide_pattern(widgets)


# ================================================================
# Helper: 清空 Handler 队列
# ================================================================
def _flush_queues(handler):
    for q in (handler.cv_queue, handler.np_queue):
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass


def _on_average(panel, widgets):
    """遍历所有 session/bit 目录，计算平均并保存为 .npy 和 .png"""
    save_root = widgets["save_folder"].text().strip()
    if not save_root or not os.path.isdir(save_root):
        QMessageBox.warning(
            panel, "Warning", "请先完成一次 Projection Capture 并确保保存目录有效。"
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
            # 读取所有 .npy（排除已有平均文件）
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
            # 保存为 8‑bit PNG
            img8 = np.clip(avg, 0, 255).astype(np.uint8)
            out_png = os.path.join(bit_dir, f"{session}_{bit}_avrg.png")
            cv2.imwrite(out_png, img8)
    QMessageBox.information(panel, "Completed", "All captures averaged and saved.")
