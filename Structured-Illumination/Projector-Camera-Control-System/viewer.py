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
    # emit (x, y, gray_value) when mouse moves
    mouse_position_signal = pyqtSignal(int, int, int)
    # emit (point1, point2, profile_values) after dragging ends
    profile_updated = pyqtSignal(tuple, tuple, np.ndarray)

    # a 1x1 invisible rectangle to serve as event-filter carrier
    class _HandleEventFilter(QGraphicsRectItem):
        def __init__(self, view: "ImageView"):
            super().__init__(-1, -1, 1, 1)  # tiny & invisible
            self._view = view
            self.setVisible(False)

        # send all events back again to ImageView for unified processing
        def sceneEventFilter(self, watched, event):
            return self._view._process_handle_event(watched, event)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        # OpenGL rendering for smoother panning and zooming
        self.setViewport(QOpenGLWidget())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        # Scene + main Pixmap
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.original_image = None  # save gray image (numpy array)

        # Profile mode state
        self.profile_mode = False  # whether in profile mode
        self.profile_points = []  # [(x1,y1),(x2,y2)]

        # draw the red line
        self.profile_line = QGraphicsLineItem()
        self.profile_line.setPen(QPen(QColor("red"), 3.0))
        self._scene.addItem(self.profile_line)
        self.profile_line.setVisible(False)
        # create and add the hidden event filter
        self._handle_filter = ImageView._HandleEventFilter(self)
        self._scene.addItem(self._handle_filter)

        # create the three draggable handles:
        # starting point, ending point, midpoint (3 small circles)
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
            h.setAcceptHoverEvents(True)  # receive hover events
            # h.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)

            self._scene.addItem(h)  # first add to scene
            h.installSceneEventFilter(self._handle_filter)  # then install filter

        # (optionally) install scene-level event filter
        # self._scene.installEventFilter(self)

        # dragging handle state
        self._drag_handle = None  # handle being dragged
        self._drag_start_pos = None  # coords of mouse press in scene

    def enable_profile_mode(self, enable: bool):
        """Enable or disable Profile mode (click two points to draw line, drag handles to adjust)."""
        self.profile_mode = enable
        self.profile_points.clear()
        self.profile_line.setVisible(False)
        for h in (self.handle_start, self.handle_end, self.handle_mid):
            h.setVisible(False)

    def mousePressEvent(self, event):
        """
        In Profile mode, click to record points; after two points, draw line.
        Otherwise, keep default panning.
        """
        if self.profile_mode and self.original_image is not None:
            pt = self.mapToScene(event.pos())
            x, y = int(pt.x()), int(pt.y())
            h, w = self.original_image.shape
            if 0 <= x < w and 0 <= y < h:
                self.profile_points.append((x, y))
                if len(self.profile_points) == 2:
                    # call set_profile_line to draw the line when both points are ready
                    self.set_profile_line(*self.profile_points)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """
        In Profile mode, drag handles to adjust line.
        After releasing, emit profile_updated.
        """
        if self.profile_mode and self._drag_handle:
            self._drag_handle = None
            if len(self.profile_points) == 2:
                self.profile_updated.emit(
                    *self.profile_points, self.get_profile_values(*self.profile_points)
                )
        super().mouseReleaseEvent(event)

    def set_profile_line(self, p1, p2):
        """Set the profile line between two points and update handle positions."""
        self.profile_points = [p1, p2]
        self.profile_line.setLine(p1[0], p1[1], p2[0], p2[1])
        self.profile_line.setVisible(True)
        self._update_handles(p1, p2)

        # notice ProfilePlotWindow to plot the profile immediately
        self.profile_updated.emit(p1, p2, self.get_profile_values(p1, p2))

    def _update_handles(self, p1, p2):
        """Place the start, end, and mid handles at the correct positions."""
        self.handle_start.setPos(QPointF(*p1))
        self.handle_end.setPos(QPointF(*p2))
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        self.handle_mid.setPos(QPointF(*mid))
        for h in (self.handle_start, self.handle_end, self.handle_mid):
            h.setVisible(True)

    # the function that actually processes events from the three handles, called by _HandleEventFilter
    def _process_handle_event(self, obj, event):
        """
        unified intercept QGraphicsSceneHoverEvent / QGraphicsSceneMouseEvents
        from the three handles: handle_start, handle_end, handle_mid, and process them as follows:
            - Mouse Press: record the handle being dragged and its scenePos
            - Mouse Release: end dragging, and emit profile_updated if both p1/p2 exist
            - Hover Enter: change to cross cursor + highlight yellow circle
            - Hover Leave: restore normal arrow cursor + red circle
            - During Dragging: calculate displacement (dx, dy), update p1/p2 and call set_profile_line(),
                               to achieve "line updates when endpoints are dragged" and "overall translation when midpoint is dragged"
        """

        # only process these three "handles"
        if obj in (self.handle_start, self.handle_end, self.handle_mid):
            # mouse press event
            if event.type() == QEvent.GraphicsSceneMousePress:
                # record the handle being pressed and its scenePos
                self._drag_handle = obj
                self._drag_start_pos = event.scenePos()
                self.setDragMode(QGraphicsView.NoDrag)  # close panning of whole image
                return True

            # mouse release event
            elif event.type() == QEvent.GraphicsSceneMouseRelease:
                # end dragging; if both points exist, emit profile_updated
                self._drag_handle = None
                self.setDragMode(QGraphicsView.ScrollHandDrag)  # resume panning
                if len(self.profile_points) == 2:
                    self.profile_updated.emit(
                        *self.profile_points,
                        self.get_profile_values(*self.profile_points),
                    )
                return True

            # hover enter event
            elif event.type() == QEvent.GraphicsSceneHoverEnter:
                # change cursor to crosshair + highlight yellow circle
                obj.setCursor(Qt.CrossCursor)
                obj.setBrush(QColor("yellow"))
                return True

            # hover leave event
            elif event.type() == QEvent.GraphicsSceneHoverLeave:
                # normal arrow cursor + red circle
                obj.setCursor(Qt.ArrowCursor)
                obj.setBrush(QColor("red"))
                return True

            # mouse move event (during dragging)
            elif event.type() == QEvent.GraphicsSceneMouseMove and self._drag_handle:
                # new scenePos, clipped to image boundaries
                new_scene = event.scenePos()
                x = max(0, min(new_scene.x(), self.original_image.shape[1] - 1))
                y = max(0, min(new_scene.y(), self.original_image.shape[0] - 1))
                new_scene = QPointF(x, y)

                # calculate displacement = current scenePos - previous scenePos
                dx = new_scene.x() - self._drag_start_pos.x()
                dy = new_scene.y() - self._drag_start_pos.y()
                # update _drag_start_pos for next use
                self._drag_start_pos = new_scene

                # according to which handle is being dragged, calculate new p1, p2
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

                # only call set_profile_line if [p1, p2] actually changed
                if [p1, p2] != self.profile_points:
                    self.set_profile_line(p1, p2)
                return True

        return False  # means not handled, let Qt continue dispatching

    # def get_profile_values(self, p1, p2):
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
        use OpenCV's cv2.remap to perform bilinear interpolation and get gray values along the line segment.
        if the image is not grayscale, automatically convert to grayscale before sampling.
        """
        if self.original_image is None:
            return np.array([])

        # automatically convert to grayscale if the image is color
        img = self.original_image
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # calculate line segment length and interpolation coordinates
        length = max(2, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
        x_vals = np.linspace(p1[0], p2[0], length).astype(np.float32)
        y_vals = np.linspace(p1[1], p2[1], length).astype(np.float32)

        map_x = x_vals.reshape(1, -1)
        map_y = y_vals.reshape(1, -1)

        # use cv2.remap for bilinear interpolation to get gray values
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
        In addition to updating the line when dragging the handles,
        also emit (x, y, intensity) on mouse move to update status bar gray value display.
        """
        if self.profile_mode and self._drag_handle:
            # if you want to update the line segment during dragging, put it here, but it's already handled in eventFilter
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
        Zoom the view: mouse wheel zooms in/out centered at mouse position.
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
        self.setWindowTitle("Projector-Camera Control System")
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
        #         lambda: self.handler.get_cv_image() is not None,  # or your own frame trigger condition
        #     )
        # )
        self.calib_panel, self.calib_vars, self.calib_state = (
            create_projector_calibration_panel(
                self.handler,
                self.start_stream_fn,
                self.stop_stream_fn,  # or your own frame trigger condition
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
                    # Also update status bar with new FPS
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
        exposure_text = (
            int(self.exposure_val[0]) if hasattr(self, "exposure_val") else "-"
        )
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
