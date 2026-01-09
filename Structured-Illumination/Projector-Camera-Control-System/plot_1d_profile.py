# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from viewer import ImageView  # only for type hints

plt.style.use("default")
plt.rcParams["figure.figsize"] = [4, 4]
plt.rcParams.update({"font.size": 10})
# plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["image.cmap"] = "gray"


class ProfilePlotWindow(QWidget):
    def __init__(self, view):
        super().__init__()  # type: ignore
        self.view: "ImageView" = view
        self.setWindowTitle("Profile Plot")
        self.init_ui()

        self.view.enable_profile_mode(True)  # make sure profile mode is enabled immediately
        self.view.profile_updated.connect(self.update_plot)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Plot
        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        # Coordinate input
        coord_layout = QHBoxLayout()
        self.p1_input = QLineEdit()
        self.p2_input = QLineEdit()
        coord_layout.addWidget(QLabel("P1:"))
        coord_layout.addWidget(self.p1_input)
        coord_layout.addWidget(QLabel("P2:"))
        coord_layout.addWidget(self.p2_input)
        layout.addLayout(coord_layout)

        # Apply & Clear buttons
        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_coords)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_profile)

        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(clear_btn)
        layout.addLayout(btn_layout)

    def update_plot(self, p1, p2, profile_values: np.ndarray):
        self.p1_input.setText(f"{p1[0]},{p1[1]}")
        self.p2_input.setText(f"{p2[0]},{p2[1]}")
        self.ax.clear()
        self.ax.plot(profile_values, color="black", linewidth=1.5)

        # add title and labels
        self.ax.set_title("1D Intensity Profile")
        self.ax.set_xlabel("Distance (pixels)")
        self.ax.set_ylabel("Gray Level")

        # show grid
        self.ax.grid(True, linestyle="--", alpha=0.5)

        self.canvas.draw()

    def apply_coords(self):
        try:
            x1, y1 = map(float, self.p1_input.text().split(","))
            x2, y2 = map(float, self.p2_input.text().split(","))
            self.view.set_profile_line((x1, y1), (x2, y2))
        except Exception as e:
            print(f"Invalid coordinates: {e}")

    def closeEvent(self, event):
        """exit profile mode, when closing the Profile window."""
        self.view.enable_profile_mode(False)
        self.deleteLater()
        event.accept()

    def clear_profile(self):
        """Clear profile line and plot."""
        self.view.enable_profile_mode(True)  # keep in profile mode
        self.view.profile_line.setVisible(False)
        for h in [
            self.view.handle_start,
            self.view.handle_end,
            self.view.handle_mid,
        ]:
            h.setVisible(False)
        self.view.profile_points.clear()

        self.ax.clear()
        self.canvas.draw()

        self.p1_input.clear()
        self.p2_input.clear()
