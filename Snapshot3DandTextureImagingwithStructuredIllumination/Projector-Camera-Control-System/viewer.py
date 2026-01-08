import os  # For file path operations
import time  # For timing and FPS calculation
import traceback  # To print exception tracebacks
import cv2  # OpenCV for saving images
from datetime import datetime  # For timestamping saved frames
import subprocess  # For running external commands (e.g., to open file dialogs)
import platform  # To check the operating system type

# PyQt5 widgets for building the UI
from PyQt5.QtWidgets import (
    QWidget,  # Base class for all UI containers
    QVBoxLayout,  # Vertical layout manager
    QHBoxLayout,  # Horizontal layout manager
    QLabel,  # Simple text label
    QSizePolicy,  # Size policy for widgets
    QSlider,  # Slider widget for exposure control
    QFileDialog,  # File/folder selection dialog
    QPushButton,  # Button widget
    QGraphicsView,  # View for rendering QGraphicsScene
    QGraphicsScene,  # Scene holding graphics items
    QGraphicsLineItem,  # Line item for drawing lines
    QGraphicsPixmapItem,  # Pixmap item for images
    QGraphicsEllipseItem,  # Ellipse item for handles
    QGraphicsRectItem,  # Rect item for bounding boxes
    QOpenGLWidget,  # OpenGL viewport for smooth rendering
    QShortcut,  # Keyboard shortcut helper
    QMessageBox,  # Message box for alerts
)
from PyQt5.QtCore import (
    Qt,
    QTimer,
    pyqtSignal,
    QRectF,
    QPointF,
    QEvent,
)  # Core Qt classes
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QKeySequence,
    QColor,
    QPen,
)  # GUI-related classes

from config import SLIDER_MAX, output_file, features_to_update  # Global configuration
from dialogs import (
    create_feature_config_panel,  # Builds the camera feature settings panel
    create_projector_calibration_panel,  # Builds the projector calibration panel
    exposure_to_slider,  # Converts exposure to slider position
)
from camera_feature_manager import CameraFeatureManager  # Applies feature updates
from global_state import state  # Shared streaming state
from utils import slider_to_exposure  # Converts slider value to exposure time
from queue import Empty
import numpy as np  # NumPy for image handling
from scipy.ndimage import map_coordinates  # For image profile calculations
from plot_1d_profile import (
    ProfilePlotWindow,
)  # Profile plot window for displaying intensity profiles


class ImageView(QGraphicsView):
    # 鼠标移动时发射 (x,y,gray)
    mouse_position_signal = pyqtSignal(int, int, int)
    # 拖动结束后发射 (p1, p2, profile_values)
    profile_updated = pyqtSignal(tuple, tuple, np.ndarray)

    # ----------------------------------------------------------
    # 一个 1×1 的隐藏矩形，用来作为 event-filter 载体
    # ----------------------------------------------------------
    class _HandleEventFilter(QGraphicsRectItem):
        def __init__(self, view: "ImageView"):
            super().__init__(-1, -1, 1, 1)  # tiny & invisible
            self._view = view
            self.setVisible(False)

        # 这里把所有事件再转给 ImageView 做原本的统一处理
        def sceneEventFilter(self, watched, event):
            return self._view._process_handle_event(watched, event)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        # OpenGL 渲染，加速平移缩放
        self.setViewport(QOpenGLWidget())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        # Scene + 主 Pixmap
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.original_image = None  # 保存灰度图 (numpy array)

        # Profile 模式相关
        self.profile_mode = False  # 是否处于“画线”模式
        self.profile_points = []  # [(x1,y1),(x2,y2)]

        # 实际画红线
        self.profile_line = QGraphicsLineItem()
        self.profile_line.setPen(QPen(QColor("red"), 3.0))
        self._scene.addItem(self.profile_line)
        self.profile_line.setVisible(False)
        # 创建并添加隐藏过滤器
        self._handle_filter = ImageView._HandleEventFilter(self)
        self._scene.addItem(self._handle_filter)

        # 起点/终点/中点（3 个小圆）
        x, y, w, h = (-4, -4, 8, 8)
        scale = 2
        self.handle_start = QGraphicsEllipseItem(
            x * scale, y * scale, w * scale, h * scale
        )
        self.handle_end = QGraphicsEllipseItem(
            x * scale, y * scale, w * scale, h * scale
        )
        self.handle_mid = QGraphicsEllipseItem(
            x * scale, y * scale, w * scale, h * scale
        )
        for h in (self.handle_start, self.handle_end, self.handle_mid):
            h.setBrush(QColor("red"))
            h.setZValue(10)
            h.setVisible(False)
            h.setAcceptHoverEvents(True)  # 接收 hover 事件
            # h.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)

            self._scene.addItem(h)  # ① 先入 scene
            h.installSceneEventFilter(self._handle_filter)  # ② 再装过滤器

        # （可留可删）scene-level 过滤器对拖动画面平移等仍然有效
        # self._scene.installEventFilter(self)

        # 拖动相关状态
        self._drag_handle = None  # 当前正在拖动的 handle
        self._drag_start_pos = None  # 按下时的 scene 坐标

    def enable_profile_mode(self, enable: bool):
        """开启/关闭 Profile 模式（点击两点画线、拖动圆点调整）"""
        self.profile_mode = enable
        self.profile_points.clear()
        self.profile_line.setVisible(False)
        for h in (self.handle_start, self.handle_end, self.handle_mid):
            h.setVisible(False)

    def mousePressEvent(self, event):
        """
        在 Profile 模式下，点击一次记录一个点，记录两个点后画线。
        否则仍保持默认的 Scene 平移逻辑。
        """
        if self.profile_mode and self.original_image is not None:
            pt = self.mapToScene(event.pos())
            x, y = int(pt.x()), int(pt.y())
            h, w = self.original_image.shape
            if 0 <= x < w and 0 <= y < h:
                self.profile_points.append((x, y))
                if len(self.profile_points) == 2:
                    # 两点齐全后，调用统一画线函数
                    self.set_profile_line(*self.profile_points)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """
        拖动圆点时，鼠标释放后发射 profile_updated 信号。
        """
        if self.profile_mode and self._drag_handle:
            self._drag_handle = None
            if len(self.profile_points) == 2:
                self.profile_updated.emit(
                    *self.profile_points, self.get_profile_values(*self.profile_points)
                )
        super().mouseReleaseEvent(event)

    def set_profile_line(self, p1, p2):
        """
        根据两个端点 (p1,p2) 更新红线与三个圆点的位置。
        """
        self.profile_points = [p1, p2]
        self.profile_line.setLine(p1[0], p1[1], p2[0], p2[1])
        self.profile_line.setVisible(True)
        self._update_handles(p1, p2)

        # ⇩ 新增：立即通知 ProfilePlotWindow 画曲线
        self.profile_updated.emit(p1, p2, self.get_profile_values(p1, p2))

    def _update_handles(self, p1, p2):
        """
        放置 start, end, mid 三个圆点到对应位置。
        """
        self.handle_start.setPos(QPointF(*p1))
        self.handle_end.setPos(QPointF(*p2))
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        self.handle_mid.setPos(QPointF(*mid))
        for h in (self.handle_start, self.handle_end, self.handle_mid):
            h.setVisible(True)

    # 被 _HandleEventFilter 调用的真正处理函数
    def _process_handle_event(self, obj, event):
        """
        统一拦截 handle_start、handle_end、handle_mid 这三个小圆抛出的
        QGraphicsSceneHoverEvent / QGraphicsSceneMouseEvent，并作以下处理：
         - 鼠标按下：记录当前拖动的 handle 与它的 scenePos
         - 鼠标释放：取消拖动，并在 p1/p2 均存在时发射 profile_updated
         - 悬停进入：切换为十字光标 + 高亮黄色圆
         - 悬停离开：还原普通箭头 + 红色圆
         - 拖动中：计算位移增量 (dx, dy)，更新 p1/p2 并调用 set_profile_line()，
                   以实现“端点拖动时线段同步更新”、“中点拖动时整体平移”
        """
        # 仅处理这三个 “小圆”
        if obj in (self.handle_start, self.handle_end, self.handle_mid):
            # ——— 鼠标按下 ———
            if event.type() == QEvent.GraphicsSceneMousePress:
                # 记录被按下的 handle 以及按下时对应的 scene 坐标
                self._drag_handle = obj
                self._drag_start_pos = event.scenePos()
                self.setDragMode(QGraphicsView.NoDrag)  # 关闭整幅图平移
                return True

            # ——— 鼠标释放 ———
            elif event.type() == QEvent.GraphicsSceneMouseRelease:
                # 结束拖动；如果已有两点，就发射 profile_updated
                self._drag_handle = None
                self.setDragMode(QGraphicsView.ScrollHandDrag)  # 恢复平移
                if len(self.profile_points) == 2:
                    self.profile_updated.emit(
                        *self.profile_points,
                        self.get_profile_values(*self.profile_points),
                    )
                return True

            # ——— 悬停进入 ———
            elif event.type() == QEvent.GraphicsSceneHoverEnter:
                # 十字光标 + 高亮黄色
                obj.setCursor(Qt.CrossCursor)
                obj.setBrush(QColor("yellow"))
                return True

            # ——— 悬停离开 ———
            elif event.type() == QEvent.GraphicsSceneHoverLeave:
                # 普通箭头 + 还原红色
                obj.setCursor(Qt.ArrowCursor)
                obj.setBrush(QColor("red"))
                return True

            # ——— 拖动中不断移动 ———
            elif event.type() == QEvent.GraphicsSceneMouseMove and self._drag_handle:
                # 新的 scene 坐标，并裁剪到图像边界
                new_scene = event.scenePos()
                x = max(0, min(new_scene.x(), self.original_image.shape[1] - 1))
                y = max(0, min(new_scene.y(), self.original_image.shape[0] - 1))
                new_scene = QPointF(x, y)

                # 计算增量 = 当前 scenePos - 上一次记录的 scenePos
                dx = new_scene.x() - self._drag_start_pos.x()
                dy = new_scene.y() - self._drag_start_pos.y()
                # 更新 _drag_start_pos 供下次使用
                self._drag_start_pos = new_scene

                # 根据当前拖动的是哪个 handle，分别计算新的 p1, p2
                if self._drag_handle == self.handle_start:
                    p1 = (x, y)
                    p2 = self.profile_points[1]
                elif self._drag_handle == self.handle_end:
                    p1 = self.profile_points[0]
                    p2 = (x, y)
                else:  # self._drag_handle == self.handle_mid
                    px0, py0 = self.profile_points[0]
                    px1, py1 = self.profile_points[1]
                    p1 = (
                        max(0, min(px0 + dx, self.original_image.shape[1] - 1)),
                        max(0, min(py0 + dy, self.original_image.shape[0] - 1)),
                    )
                    p2 = (
                        max(0, min(px1 + dx, self.original_image.shape[1] - 1)),
                        max(0, min(py1 + dy, self.original_image.shape[0] - 1)),
                    )

                # 只有当新旧端点确实变化时，才重新 set_profile_line
                if [p1, p2] != self.profile_points:
                    self.set_profile_line(p1, p2)
                return True

        return False  # 表示此过滤器未处理，让 Qt 继续分发

    # def get_profile_values(self, p1, p2):
    #     """
    #     计算线段上各像素的灰度值（双线性插值）。
    #     """
    #     if self.original_image is None:
    #         return np.array([])

    #     length = max(2, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
    #     x_vals = np.linspace(p1[0], p2[0], length)
    #     y_vals = np.linspace(p1[1], p2[1], length)
    #     coords = np.vstack((y_vals, x_vals))
    #     values = map_coordinates(
    #         self.original_image, coords, order=1, mode="reflect"
    #     )
    #     return values

    def get_profile_values(self, p1, p2):
        """
        使用 OpenCV 的 cv2.remap 实现双线性插值，获取线段上的灰度值。
        如果图像不是灰度图，则自动转换为灰度后再采样。
        """
        if self.original_image is None:
            return np.array([])

        # 自动转换为灰度图（如果是 BGR 或 RGB）
        img = self.original_image
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # 计算线段长度与插值坐标
        length = max(2, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
        x_vals = np.linspace(p1[0], p2[0], length).astype(np.float32)
        y_vals = np.linspace(p1[1], p2[1], length).astype(np.float32)

        map_x = x_vals.reshape(1, -1)
        map_y = y_vals.reshape(1, -1)

        # 使用双线性插值获取灰度值
        profile = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return profile[0]

    def set_image(self, img):
        """
        Update the displayed frame. Supports both monochrome and color cameras.
        """
        # print(img.shape)
        if img is None:
            return

        # Handle different image shapes
        if img.ndim == 3:
            h, w, c = img.shape
            # Color image (H, W, 3)
            # If the camera provides RGB format, convert to BGR for OpenCV consistency
            if c == 3:
                # Make sure to convert RGB→BGR for correct display
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bytes_per_line = 3 * w
                qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
            else:
                # Monochrome image with 3D shape (e.g., single-channel expanded)
                img = img[:, :, 0]
                qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # Normal grayscale image (H, W)
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)

        self.original_image = img
        pix = QPixmap.fromImage(qimg)
        self._pixmap_item.setPixmap(pix)
        self._scene.setSceneRect(QRectF(qimg.rect()))

        # If a profile line exists, update it
        if self.profile_line.isVisible() and len(self.profile_points) == 2:
            p1, p2 = self.profile_points
            self.profile_updated.emit(p1, p2, self.get_profile_values(p1, p2))

    def mouseMoveEvent(self, event):
        """
        除了拖动圆点时更新线段外，还要在鼠标移动时不断发射
        (x,y,intensity)，以更新状态栏灰度值显示。
        """
        if self.profile_mode and self._drag_handle:
            # （如果拖动时也想在此同步更新线段，可以放这段，但已经在 eventFilter 中处理）
            pass

        super().mouseMoveEvent(event)

        if self.original_image is not None:
            pt = self.mapToScene(event.pos())
            x, y = int(pt.x()), int(pt.y())
            shape = self.original_image.shape
            if len(shape) == 2:
                h, w = shape
            else:
                h, w = shape[:2]
            if 0 <= x < w and 0 <= y < h:
                # get pixel value(s)
                val = self.original_image[y, x]
                if val.ndim == 0:
                    val_disp = (int(val),) * 3
                elif val.size == 3:
                    val_disp = tuple(map(int, val))
                else:
                    val_disp = (int(val), int(val), int(val))
                # update status bar, etc.
                self.mouse_position_signal.emit(x, y, val_disp[0])

    def wheelEvent(self, event):
        """
        缩放视图：滚轮缩放，居中在鼠标位置。
        """
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


class CameraViewer(QWidget):
    def __init__(
        self,
        cam,
        handler,
        exposure_feature,
        exposure_val,
        start_stream_fn,
        stop_stream_fn,
    ):
        super().__init__()
        # Store references to camera and control functions
        self.cam = cam
        self.handler = handler
        self.exposure_feature = exposure_feature
        self.exposure_val = exposure_val
        self.start_stream_fn = start_stream_fn
        self.stop_stream_fn = stop_stream_fn

        # Variables for tracking FPS
        self.last_frame = None
        self.fps = 0
        self.frame_count = 0
        self.prev_time = time.time()

        self.init_ui()  # Build the UI layout

        # Timer to periodically update the displayed image
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(30)  # 30 ms interval

        # Keyboard shortcuts: Q to quit, S to save frame
        QShortcut(QKeySequence("q"), self, self.close)
        QShortcut(QKeySequence("s"), self, self.save_frame)

    def init_ui(self):
        # Window title and initial size
        self.setWindowTitle("Projector-Camera Calibration System")
        self.resize(1000, 600)

        # Image display area
        self.view = ImageView()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Exposure slider setup
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, SLIDER_MAX)
        # Initialize slider to current exposure
        self.slider.setValue(exposure_to_slider(int(self.exposure_val[0])))
        self.slider.valueChanged.connect(self.on_slider_changed)

        # Status label showing position, gray value, FPS, state, exposure
        init_state = "Streaming" if state["streaming"] else "Paused"
        self.status_label = QLabel(
            f"Position: (-, -);  Value: -;  FPS: 0.0;  State: {init_state};  Exposure: {int(self.exposure_val[0])}"
        )

        # Save path controls
        self.save_path = os.getcwd()
        self.save_path_label = QLabel(f"Frame Save Path: {self.save_path}")
        self.save_button = QPushButton("Save Frame")
        self.change_path_button = QPushButton("Change Frame Save Path")
        self.profile_btn = QPushButton("Plot 1D Profile")
        self.profile_btn.clicked.connect(self.show_profile_plot)
        self.save_button.clicked.connect(self.save_frame)
        self.change_path_button.clicked.connect(self.change_save_path)
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.clicked.connect(self.open_save_folder)

        # Create embedded panels for feature config and calibration
        self.config_panel, self.config_vars = create_feature_config_panel(
            CameraFeatureManager(self.cam),
            features_to_update,
            self.exposure_feature,
            self.exposure_val,
            output_file,
            lambda: state["paused"],
            self.start_stream_fn,
            self.stop_stream_fn,
        )

        # self.calib_panel, self.calib_vars, self.calib_state = (
        #     create_projector_calibration_panel(
        #         lambda: state["paused"],
        #         lambda: self.handler.get_cv_image() is not None,  # 或者你自己的帧触发条件
        #     )
        # )
        self.calib_panel, self.calib_vars, self.calib_state = (
            create_projector_calibration_panel(
                self.handler,
                self.start_stream_fn,
                self.stop_stream_fn,  # 或者你自己的帧触发条件
            )
        )

        # Layout: left=feature panel, center=view, right=calib panel
        main_h = QHBoxLayout()
        main_h.addWidget(self.config_panel)
        main_h.addWidget(self.view, stretch=1)
        main_h.addWidget(self.calib_panel)

        # Bottom controls: status, slider, save buttons
        bottom_v = QVBoxLayout()
        bottom_v.addWidget(self.status_label)
        bottom_v.addWidget(self.slider)
        btn_h = QHBoxLayout()
        btn_h.addWidget(self.save_button)
        btn_h.addWidget(self.change_path_button)
        btn_h.addWidget(self.open_folder_button)
        btn_h.addWidget(self.profile_btn)
        bottom_v.addLayout(btn_h)
        bottom_v.addWidget(self.save_path_label)

        # Combine top and bottom in main layout
        layout = QVBoxLayout(self)
        layout.addLayout(main_h, stretch=1)
        layout.addLayout(bottom_v)

        # Connect hover signal to update status bar
        self.view.mouse_position_signal.connect(self.update_status)

    def show_profile_plot(self):
        if not hasattr(self, "profile_window") or self.profile_window is None:
            self.profile_window = ProfilePlotWindow(self.view)
            self.profile_window.show()
            self.profile_window.raise_()
            # Connect the window's close event to clean up
            # the profile_window attribute
            self.profile_window.destroyed.connect(
                lambda: setattr(self, "profile_window", None)
            )
        else:
            self.profile_window.raise_()

    def update_image(self):
        # Only update if streaming is active
        if not state["streaming"]:
            return
        try:
            try:
                frame = self.handler.get_cv_image(timeout=0.0)
            except Empty:
                return
            self.last_frame = frame
            if frame is not None:
                self.view.set_image(frame)  # Display it
                self.frame_count += 1
                now = time.time()
                # Update FPS every second
                if now - self.prev_time >= 1.0:
                    self.fps = self.frame_count / (now - self.prev_time)
                    self.prev_time = now
                    self.frame_count = 0
                    # 每秒刷新一次状态栏中的 FPS
                    status = "Streaming" if state["streaming"] else "Paused"
                    self.status_label.setText(
                        f"Position: (-, -);  Value: -;  FPS: {self.fps:.1f};  "
                        + f"State: {status};  Exposure: {int(self.exposure_val[0])}"
                    )
        except Exception:
            traceback.print_exc()

    # def update_status(self, x, y, value):
    #     """
    #     Update status bar text depending on image type (color or grayscale).
    #     - For grayscale, repeats (v, v, v)
    #     - For color, shows (R, G, B)
    #     """
    #     # Handle missing pixel info
    #     if value is None:
    #         value_text = "(-, -, -)"
    #     else:
    #         if isinstance(value, (list, tuple, np.ndarray)):
    #             # Likely color image
    #             vals = [int(v) for v in value[:3]]  # ensure (R,G,B)
    #             value_text = f"({vals[0]}, {vals[1]}, {vals[2]})"
    #         else:
    #             # Single grayscale value
    #             v = int(value)
    #             value_text = f"({v}, {v}, {v})"

    #     state_str = "Streaming" if state["streaming"] else "Paused"
    #     fps_text = getattr(self, "fps", 0.0)
    #     exposure_text = (
    #         int(self.exposure_val[0]) if hasattr(self, "exposure_val") else 0
    #     )

    #     self.status_label.setText(
    #         f"Position: ({x}, {y});  Value: {value_text};  "
    #         f"FPS: {fps_text:.1f};  State: {state_str};  Exposure: {exposure_text}"
    #     )


    def update_status(self, x, y, value):
        """
        Update status bar text depending on image type (color or grayscale).
        For grayscale, repeats (v, v, v).
        For color, shows actual (R, G, B) values.
        """
        if self.view.original_image is None:
            return

        img = self.view.original_image
        h, w = img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        # --- Extract pixel value ---
        if img.ndim == 2:
            # Grayscale → repeat 3 times
            v = int(img[y, x])
            value_text = f"({v}, {v}, {v})"
        elif img.ndim == 3 and img.shape[2] >= 3:
            # Color → assume BGR (OpenCV default)
            b, g, r = img[y, x][:3]
            value_text = f"({int(r)}, {int(g)}, {int(b)})"
        else:
            value_text = "(-, -, -)"

        # --- Compose status text ---
        fps_text = getattr(self, "fps", 0.0)
        exposure_text = int(self.exposure_val[0]) if hasattr(self, "exposure_val") else "-"
        state_str = "Streaming" if state.get("streaming", False) else "Paused"

        self.status_label.setText(
            f"Position: ({x}, {y});  Value: {value_text};  "
            f"FPS: {fps_text:.1f};  State: {state_str};  Exposure: {exposure_text}"
        )

    def on_slider_changed(self, val):
        # When slider moves, set new exposure on camera
        try:
            new_exp = slider_to_exposure(val)
            self.exposure_feature.set(new_exp)
            actual = float(self.exposure_feature.get())
            self.exposure_val[0] = actual
        except Exception:
            traceback.print_exc()

    # def save_frame(self):
    #     # Save the current frame as a PNG with timestamp
    #     if self.last_frame is None:
    #         return
    #     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     fn = os.path.join(self.save_path, f"frame_{ts}.png")
    #     try:
    #         success = cv2.imwrite(fn, self.last_frame)
    #         if success:
    #             print(f"Frame saved to {fn}")
    #         else:
    #             print(f"Failed to save frame to {fn}")
    #     except Exception as e:
    #         print(f"Error saving frame to {fn}: {e}")
    def save_frame(self):
        """Save the current frame as a PNG with timestamp."""
        if self.last_frame is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = os.path.join(self.save_path, f"frame_{ts}.png")
        try:
            success = cv2.imwrite(fn, self.last_frame)
            if success:
                print(f"Frame saved to {fn}")
                QMessageBox.information(self, "Saved", f"Frame saved to:\n{fn}")
            else:
                print(f"Failed to save frame to {fn}")
                QMessageBox.warning(self, "Failed", f"Failed to save frame to:\n{fn}")
        except Exception as e:
            print(f"Error saving frame to {fn}: {e}")
            QMessageBox.critical(self, "Error", f"Error saving frame:\n{e}")

    def change_save_path(self):
        # Open a dialog to select a new save directory
        path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if path:
            self.save_path = path
            self.save_path_label.setText(f"Save Path: {path}")

    def closeEvent(self, event):
        # When the window closes, stop streaming and quit
        self.timer.stop()
        self.stop_stream_fn()
        event.accept()

    def open_save_folder(self):
        """Open the folder where frames are saved."""
        try:
            if platform.system() == "Windows":
                os.startfile(self.save_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", self.save_path])
            else:  # Linux and others
                subprocess.run(["xdg-open", self.save_path])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open folder:\n{e}")
