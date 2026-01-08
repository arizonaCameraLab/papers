import cv2
import numpy as np
import os, sys
import shutil
import glob
import re
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import colorsys
import copy
from matplotlib.gridspec import GridSpec
from pathlib import Path
import open3d as o3d
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay


# --- Helper function to extract numbers from filename for sorting ---
def extract_number(filename):
    """
    Extracts the first sequence of digits found in the filename's base name.
    Used as a key for natural sorting.
    Returns the integer value, or 0 if no digits are found.
    """
    # Get only the filename part (e.g., "img10.jpg" from "/path/to/img10.jpg")
    basename = os.path.basename(filename)
    # Search for the first occurrence of one or more digits (\d+)
    match = re.search(r"\d+", basename)
    if match:
        try:
            # Convert the found digits to an integer and return it
            return int(match.group(0))
        except ValueError:
            # This should generally not happen with \d+, but handle just in case
            print(
                f"Warning: Could not convert extracted digits '{match.group(0)}' from {basename} to int."
            )
            return 0  # Fallback value
    else:
        # If no digits are found in the filename, return 0
        # Files without numbers will typically be sorted first
        return 0


# --- End of extract_number function ---


def load_gray_float32(filename, stretch=True, threshold=False):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if stretch:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    if threshold:
        _, img = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)
    return img


# Function to process bit 0 numeric files
def process_bit0_numeric_files(root_dir, pairs, bit0_synthetic_flag=True):

    # List all files in the root directory
    files = os.listdir(root_dir)

    number_files = []
    number_pattern = re.compile(
        r"^(\d+)_\d+\.png$"
    )  # match files like "00_00.png", "01_00.png", etc.

    for f in files:
        m = number_pattern.match(f)
        if m:
            num = int(m.group(1))
            number_files.append((num, f))

    if not number_files:
        print("No number-starting files found.")
        return

    # sort by number
    number_files.sort(key=lambda x: x[0])

    # skip files with number < 2
    filtered = [(n, f) for (n, f) in number_files if n >= 2]

    if len(filtered) < 4:
        raise RuntimeError("Need at least 4 numeric files (>=02).")

    # divide into two groups
    total = len(filtered)
    half = total // 2

    groupA = filtered[:half]
    groupB = filtered[half:]

    # last two from each group
    last_A = groupA[-2:]
    last_B = groupB[-2:]
    target_files = last_A + last_B  # 4 files in total

    print("Files selected for replacement:")
    for _, fname in target_files:
        print("  ", fname)

    # create org_bit0 directory
    org_dir = os.path.join(root_dir, "org_bit0")
    os.makedirs(org_dir, exist_ok=True)

    # move files to org_bit0
    for _, fname in target_files:
        src = os.path.join(root_dir, fname)
        dst = os.path.join(org_dir, fname)
        print(f"Moving {src} -> {dst}")
        shutil.move(src, dst)

    # read the 8 pattern images
    patterns = {}
    for key_prepare in [
        "HorBit0_Shifted_0",
        "HorBit0_Shifted_1",
        "HorBit0_Shifted_0_Inverse",
        "HorBit0_Shifted_1_Inverse",
        "VerBit0_Shifted_0",
        "VerBit0_Shifted_1",
        "VerBit0_Shifted_0_Inverse",
        "VerBit0_Shifted_1_Inverse",
    ]:
        pattern_file = os.path.join(root_dir, f"{key_prepare}_00.png")
        patterns[key_prepare] = load_gray_float32(pattern_file)

    # generate and save the abs diff images
    for pair, (_, out_filename) in zip(pairs, target_files):
        key1, key2 = pair
        img1 = patterns[key1]
        img2 = patterns[key2]
        diff = np.abs(img1 - img2)

        # normalize to 0–255
        diff = diff / np.max(diff) * 255
        diff = diff.astype(np.uint8)

        save_path = os.path.join(root_dir, out_filename)
        print(f"Writing new diff image: {save_path}")
        cv2.imwrite(save_path, diff)

    print("\n✓ Processing complete.")


def is_graycode_main_file(fname):
    """Only keep NN_00.png, ignore HorBit0*, VerBit0*, SynthBit* etc."""
    base = os.path.basename(fname)
    # must start with digits and end with "_00.png"
    return base.split("_")[0].isdigit() and base.endswith("_00.png")


def loadCameraParam(json_file):
    with open(json_file, "r") as f:
        param_data = json.load(f)
        P = param_data["camera_matrix"]
        d = param_data["dist_coeffs"]
        return np.array(P).reshape([3, 3]), np.array(d)


def printNumpyWithIndent(tar, indentchar):
    print(indentchar + str(tar).replace("\n", "\n" + indentchar))


def generate_colors(n):
    """Generate n distinct HSL colors (returns Plotly-compatible hex)."""
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.7)
        colors.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")
    return colors


# Generate small plane corners for each checkerboard
def create_plane(origin, R, w, h, scale=1.0):
    """Create (x,y,z) corners of a checkerboard plane in world coords."""
    # Define board corners in local coordinates
    local_pts = (
        np.array(
            [
                [0, 0, 0],
                [w, 0, 0],
                [w, h, 0],
                [0, h, 0],
                [0, 0, 0],
            ]
        )
        * scale
    )
    world_pts = (R @ local_pts.T).T + origin
    return world_pts


def plot_checkerboard_poses_plotly(
    board_origins,
    board_rotations,
    plane_colors,
    w,
    h,
    square_size,
    create_plane=create_plane,
    cam_position=None,
    proj_position=None,
    plot_camera=False,
    plot_projector=False,
    camera_name="Camera",
    projector_name="Projector",
    camera_color="magenta",
    projector_color="orange",
):
    """
    Plot checkerboard planes and normals in a world coordinate system using Plotly.

    Parameters
    ----------
    board_origins : (N,3) array-like
        Each board origin in world coordinates.
    board_rotations : list or (N,3,3) array-like
        Each board rotation matrix (board local -> world).
    plane_colors : list
        Colors for each board plane.
    w, h : float
        Plane width and height in board local coordinates.
    square_size : float
        Square size (mm). Used only for scaling axes and normal length.
    create_plane : callable
        Function create_plane(origin, R, w, h, scale=1.0) -> (M,3) points in world coords.
    cam_position : array-like or None
        Camera position in world coordinates, shape (3,).
    proj_position : array-like or None
        Projector position in world coordinates, shape (3,).
    plot_camera : bool
        If True, plot camera marker (requires cam_position).
    plot_projector : bool
        If True, plot projector marker (requires proj_position).
    camera_name, projector_name : str
        Labels shown on the plot for camera and projector.
    camera_color, projector_color : str
        Marker colors for camera and projector.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The created Plotly figure.
    """

    # Build the Plotly figure
    fig = go.Figure()

    for i, (O, Rw, color) in enumerate(
        zip(board_origins, board_rotations, plane_colors)
    ):
        plane_pts = create_plane(O, Rw, w, h)
        x, y, z = plane_pts[:-1, 0], plane_pts[:-1, 1], plane_pts[:-1, 2]

        # main visible surface (keep legend)
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=0.4,
                name=f"Board {i}",
                showlegend=True,
                hovertext=f"Board {i}",
                hoverinfo="text",
            )
        )

        # center marker (no legend)
        fig.add_trace(
            go.Scatter3d(
                x=[O[0]],
                y=[O[1]],
                z=[O[2]],
                mode="markers+text",
                text=[f"{i}"],
                marker=dict(size=3, color=color),
                showlegend=False,
            )
        )

        # outline (no legend)
        fig.add_trace(
            go.Scatter3d(
                x=list(x) + [x[0]],
                y=list(y) + [y[0]],
                z=list(z) + [z[0]],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )

    # Camera
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=[cam_position[0]],
    #         y=[cam_position[1]],
    #         z=[cam_position[2]],
    #         mode="markers+text",
    #         text=["Camera"],
    #         marker=dict(size=6, color="magenta"),
    #         name="Camera",
    #     )
    # )
    # # Projector
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=[proj_position[0]],
    #         y=[proj_position[1]],
    #         z=[proj_position[2]],
    #         mode="markers+text",
    #         text=["Projector"],
    #         marker=dict(size=6, color="orange"),
    #         name="Projector",
    #     )
    # )

    # Optional Camera / Projector (same as commented section, but controllable)
    if plot_camera:
        if cam_position is None:
            raise ValueError("plot_camera=True requires cam_position (shape (3,)).")
        fig.add_trace(
            go.Scatter3d(
                x=[cam_position[0]],
                y=[cam_position[1]],
                z=[cam_position[2]],
                mode="markers+text",
                text=[camera_name],
                marker=dict(size=6, color=camera_color),
                name=camera_name,
            )
        )

    if plot_projector:
        if proj_position is None:
            raise ValueError("plot_projector=True requires proj_position (shape (3,)).")
        fig.add_trace(
            go.Scatter3d(
                x=[proj_position[0]],
                y=[proj_position[1]],
                z=[proj_position[2]],
                mode="markers+text",
                text=[projector_name],
                marker=dict(size=6, color=projector_color),
                name=projector_name,
            )
        )

    # World origin and axes (with arrowheads)
    axis_len = square_size * 3
    axes_colors = ["red", "green", "blue"]
    axes_name = ["X", "Y", "Z"]
    axes_dirs = np.eye(3) * axis_len

    for c, color, axis_name in zip(axes_dirs, axes_colors, axes_name):
        # Draw main axis line (hide from legend to reduce clutter)
        fig.add_trace(
            go.Scatter3d(
                x=[0, c[0]],
                y=[0, c[1]],
                z=[0, c[2]],
                mode="lines",
                line=dict(color=color, width=6),
                name=f"World {axis_name} axis",
                showlegend=True,  # <- set True if you want them in legend
            )
        )

        # Add arrowhead cone (never in legend)
        fig.add_trace(
            go.Cone(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                u=[c[0] / axis_len],
                v=[c[1] / axis_len],
                w=[c[2] / axis_len],
                sizemode="absolute",
                sizeref=axis_len * 0.15,
                anchor="tail",
                showscale=False,
                colorscale=[[0, color], [1, color]],
            )
        )

        # Add text label at arrow tip
        fig.add_trace(
            go.Scatter3d(
                x=[c[0] * 1.05],
                y=[c[1] * 1.05],
                z=[c[2] * 1.05],
                mode="text",
                text=[axis_name],
                textposition="top center",
                textfont=dict(color=color, size=14),
                showlegend=False,
            )
        )

    # Compute checkerboard normal vectors (world coordinates)
    board_normals = []
    for Rw in board_rotations:
        n = Rw[:, 2]  # local +Z in world
        board_normals.append(n / np.linalg.norm(n))
    board_normals = np.array(board_normals)

    # Add normal vectors (each normal gets its own legend item)
    normal_scale = square_size * 2.5
    for i, (O, n_vec) in enumerate(zip(board_origins, board_normals)):
        tip = O + n_vec * normal_scale

        # Line segment for normal (legend ON for each)
        fig.add_trace(
            go.Scatter3d(
                x=[O[0], tip[0]],
                y=[O[1], tip[1]],
                z=[O[2], tip[2]],
                mode="lines",
                line=dict(color="cyan", width=3),
                showlegend=True,
                name=f"Normal {i}",  # or f"Board {i} Normal"
            )
        )

        # Arrowhead for normal (no legend)
        fig.add_trace(
            go.Cone(
                x=[tip[0]],
                y=[tip[1]],
                z=[tip[2]],
                u=[n_vec[0]],
                v=[n_vec[1]],
                w=[n_vec[2]],
                sizemode="absolute",
                sizeref=normal_scale * 0.2,
                anchor="tail",
                showscale=False,
                colorscale=[[0, "cyan"], [1, "cyan"]],
            )
        )

    # Layout
    fig.update_layout(
        title="Checkerboard Poses in World Coordinates (Camera–Projector System)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        width=1600,
        height=800,
        legend=dict(x=0, y=1),
    )

    # Identify all Mesh3d traces that are boards
    board_trace_indices = [
        i for i, t in enumerate(fig.data) if isinstance(t, go.Mesh3d)
    ]

    # Add two buttons: Hide all / Show all boards (still works)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=1.1,
                buttons=[
                    dict(
                        label="Hide All Boards",
                        method="update",
                        args=[
                            {
                                "visible": [
                                    False if i in board_trace_indices else True
                                    for i in range(len(fig.data))
                                ]
                            },
                            {"title": "All boards hidden"},
                        ],
                    ),
                    dict(
                        label="Show All Boards",
                        method="update",
                        args=[
                            {
                                "visible": [
                                    True if i in board_trace_indices else True
                                    for i in range(len(fig.data))
                                ]
                            },
                            {"title": "All boards visible"},
                        ],
                    ),
                ],
                showactive=False,
            )
        ]
    )

    fig.show()
    return fig, board_normals, board_trace_indices


def compute_plane_equation(origin, normal):
    """
    Compute plane equation coefficients A, B, C, D for:
        Ax + By + Cz + D = 0
    """
    n = normal / np.linalg.norm(normal)
    A, B, C = n
    D = -np.dot(n, origin)
    return A, B, C, D


def fit_plane_pca(points):
    """
    PCA-based plane fitting.
    points: (N, 3)
    Returns: normal, centroid
    """
    centroid = np.mean(points, axis=0)
    X = points - centroid
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    return normal, centroid


def normal_to_rotation(n):
    """
    Construct orthonormal basis (x_axis, y_axis, n) from unit normal n.
    Ensures numerical stability.
    """
    n = n / np.linalg.norm(n)

    # pick an arbitrary vector not parallel to n
    if abs(n[2]) < 0.9:
        t = np.array([0, 0, 1])
    else:
        t = np.array([0, 1, 0])

    x_axis = np.cross(t, n)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(n, x_axis)
    return np.stack([x_axis, y_axis, n], axis=1)


def plot_selected_and_pca_planes_plotly(
    selected_indices,
    board_origins,
    board_rotations,
    board_normals,
    w,
    h,
    square_size,
    pca_origins,
    pca_normals,
    A,
    B,
    C,
    D,
):
    """
    Visualize:
      1) Selected individual checkerboard planes + their normal vectors
      2) PCA-averaged planes + their normal vectors
      3) World coordinate axes (RGB)

    Parameters
    ----------
    selected_indices : iterable of int
        Indices of checkerboards to plot from (board_origins, board_rotations, board_normals).

    board_origins : array-like, shape (N, 3)
        World coordinates of each checkerboard origin (one origin per calibration view).

    board_rotations : array-like, shape (N, 3, 3) or list of (3, 3)
        Rotation matrices for each checkerboard, mapping:
            X_world = R_world * X_local + origin_world
        i.e. checkerboard local coordinates -> world coordinates.

    board_normals : array-like, shape (N, 3)
        Unit normal vectors of each checkerboard plane in world coordinates.
        Typically computed from board_rotations as the 3rd column: R_world[:, 2].

    w : float
        Plane width in checkerboard local coordinates (same unit as square_size, usually mm).

    h : float
        Plane height in checkerboard local coordinates (same unit as square_size, usually mm).

    square_size : float
        Checkerboard square size (mm). Used only for setting visualization scales
        (normal length, axis length, arrow sizes).

    pca_origins : array-like, shape (G, 3)
        (Optional / currently unused) PCA group origins. Kept as input to match your original
        code structure. In the current implementation, `avg_o` from this array is not used
        when building the averaged plane.

    pca_normals : array-like, shape (G, 3)
        Unit normal vectors (PCA-averaged) for each group of planes.

    A, B, C, D : float
        Plane parameters for the averaged plane center calculation.
        The plane equation is assumed:
            A*x + B*y + C*z + D = 0
        This is used to choose `plane_center`:
            if |C| > 1e-8: plane_center = (0, 0, -D/C)
            else: plane_center = foot of perpendicular from origin to the plane.

    utils : module/object
        Utility provider that must implement:
          - create_plane(origin, R, w, h) -> (5,3) array of plane corner vertices in world coords
          - normal_to_rotation(n) -> (3,3) rotation matrix whose 3rd column aligns with n

    Returns
    -------
    fig2 : plotly.graph_objects.Figure
        The Plotly figure containing all traces. The function also calls `fig2.show()`.
    """

    fig2 = go.Figure()

    # =====================================================
    # Selected individual boards + normals
    # =====================================================
    for i in selected_indices:
        O = board_origins[i]
        Rw = board_rotations[i]
        n_vec = board_normals[i]

        # Create plane
        plane_pts = create_plane(O, Rw, w, h)
        x, y, z = plane_pts[:-1, 0], plane_pts[:-1, 1], plane_pts[:-1, 2]

        # Plane surface (legend ON)
        fig2.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=f"rgb({100+i*20}, {80+i*30}, {150+i*10})",
                opacity=0.4,
                name=f"Board {i}",
                # legendgroup="Boards",
                showlegend=True,
            )
        )

        # Outline (legend OFF)
        fig2.add_trace(
            go.Scatter3d(
                x=list(x) + [x[0]],
                y=list(y) + [y[0]],
                z=list(z) + [z[0]],
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False,
                hoverinfo="none",
            )
        )

        # Normal vector (legend ON)
        tip = O + n_vec * square_size * 2.5
        fig2.add_trace(
            go.Scatter3d(
                x=[O[0], tip[0]],
                y=[O[1], tip[1]],
                z=[O[2], tip[2]],
                mode="lines",
                line=dict(color="cyan", width=3),
                name=f"Normal {i}",
                # legendgroup="Normals",
                showlegend=True,
            )
        )

        # Cone arrowhead (legend OFF)
        fig2.add_trace(
            go.Cone(
                x=[tip[0]],
                y=[tip[1]],
                z=[tip[2]],
                u=[n_vec[0]],
                v=[n_vec[1]],
                w=[n_vec[2]],
                sizemode="absolute",
                sizeref=square_size * 0.6,
                anchor="tail",
                showscale=False,
                colorscale=[[0, "cyan"], [1, "cyan"]],
                # legendgroup="Normals",
                showlegend=False,
            )
        )

    # =====================================================
    # PCA Averaged Planes + Normals
    # =====================================================
    normal_scale = square_size * 3  # length of visualization normal vectors

    for g_idx, (avg_o, avg_n) in enumerate(zip(pca_origins, pca_normals)):

        # 1) use PCA normal to build local coordinate system
        Rz = normal_to_rotation(avg_n)  # 3x3, align with third column avg_n

        # 2) change plane center to (0,0,-D/C)
        if abs(C) > 1e-8:
            plane_center = np.array([0.0, 0.0, -D / C], dtype=float)
        else:
            # C approach 0 case: use foot of perpendicular from origin to plane as center
            n_plane = np.array([A, B, C], dtype=float)
            plane_center = -D * n_plane / np.dot(n_plane, n_plane)

        # 3) Move lower left (0,0,0) to plane_center - (w/2, h/2, 0) in local coords
        plane_origin = plane_center - Rz @ np.array([w / 2, h / 2, 0.0])

        # 4) Generate four vertices (+ back to start to close)
        plane_pts = create_plane(plane_origin, Rz, w, h)
        x, y, z = plane_pts[:-1, 0], plane_pts[:-1, 1], plane_pts[:-1, 2]

        # plane surface (legend ON)
        fig2.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="yellow",
                opacity=0.3,
                name=f"PCA Avg Plane {g_idx}",
                showlegend=True,
            )
        )

        # plane outline (legend OFF)
        fig2.add_trace(
            go.Scatter3d(
                x=list(x) + [x[0]],
                y=list(y) + [y[0]],
                z=list(z) + [z[0]],
                mode="lines",
                line=dict(color="black", width=5),
                showlegend=False,
            )
        )

        # normal vector (legend ON)
        tip = plane_center + avg_n * normal_scale
        fig2.add_trace(
            go.Scatter3d(
                x=[plane_center[0], tip[0]],
                y=[plane_center[1], tip[1]],
                z=[plane_center[2], tip[2]],
                mode="lines",
                line=dict(color="yellow", width=5),
                name=f"PCA Avg Normal {g_idx}",
                showlegend=True,
            )
        )

        fig2.add_trace(
            go.Cone(
                x=[tip[0]],
                y=[tip[1]],
                z=[tip[2]],
                u=[avg_n[0]],
                v=[avg_n[1]],
                w=[avg_n[2]],
                sizemode="absolute",
                sizeref=normal_scale * 0.25,
                anchor="tail",
                colorscale=[[0, "yellow"], [1, "yellow"]],
                showscale=False,
                showlegend=False,
            )
        )

    # =====================================================
    # World coordinate axes (RGB)
    # =====================================================
    axis_len = square_size * 5
    axes_dirs = np.eye(3) * axis_len
    axes_colors = ["red", "green", "blue"]
    axes_names = ["X axis", "Y axis", "Z axis"]

    for c, color, axis_name in zip(axes_dirs, axes_colors, axes_names):
        fig2.add_trace(
            go.Scatter3d(
                x=[0, c[0]],
                y=[0, c[1]],
                z=[0, c[2]],
                mode="lines",
                line=dict(color=color, width=6),
                name=axis_name,
                # legendgroup="Axes",
                showlegend=True,
            )
        )

        fig2.add_trace(
            go.Cone(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                u=[c[0] / axis_len],
                v=[c[1] / axis_len],
                w=[c[2] / axis_len],
                sizemode="absolute",
                sizeref=axis_len * 0.2,
                anchor="tail",
                colorscale=[[0, color], [1, color]],
                showscale=False,
                showlegend=False,
            )
        )

    # =====================================================
    # update Layout
    # =====================================================
    fig2.update_layout(
        title="Selected & Averaged Checkerboard Planes (with Normals)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            xaxis=dict(backgroundcolor="white", gridcolor="lightgray"),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgray"),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgray"),
        ),
        width=1200,
        height=950,
        legend=dict(
            x=0,
            y=1,
            bgcolor="rgba(255,255,255,0.7)",
            font=dict(size=13),
        ),
    )

    fig2.show()
    return fig2


# -------------------------------
# 2. Triangulate in Camera Frame
#   ⚠️ Use normalized coords + [R|t]
# -------------------------------
def triangulate_cam_frame(
    projector_coords, mask_valid, K_c, dist_c, K_p, dist_p, R_cp, t_cp
):
    """
    projector_coords: (H, W, 2)  -> (u_p, v_p)
    mask_valid:       (H, W)
    return:
        points_cam:   (H, W, 3)  camera coordinates
    """

    H, W, _ = projector_coords.shape

    # pixel grid (camera pixel coords)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    cam_pix = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float32)
    proj_pix = projector_coords.reshape(-1, 1, 2).astype(np.float32)

    mask_flat = mask_valid.reshape(-1) > 0

    # --- normalized coords: (N, 1, 2) (undistort + K^-1)---
    cam_norm = cv2.undistortPoints(cam_pix, K_c, dist_c)  # (N,1,2)
    proj_norm = cv2.undistortPoints(proj_pix, K_p, dist_p)  # (N,1,2)

    cam_norm = cam_norm.reshape(-1, 2).T[:, mask_flat]  # (2, N_valid)
    proj_norm = proj_norm.reshape(-1, 2).T[:, mask_flat]  # (2, N_valid)

    # --- projection matrices using [R|t] ---
    P1 = np.hstack(
        [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)]
    )  # camera
    P2 = np.hstack([R_cp.astype(np.float32), t_cp.astype(np.float32)])  # projector

    # --- trangualation in homogeneous coordinates ---
    X_h = cv2.triangulatePoints(P1, P2, cam_norm, proj_norm)  # (4, N)
    X = X_h[:3, :] / X_h[3:, :]  # (3, N)

    # --- refill to (H*W, 3) with NaN ---
    pts = np.zeros((H * W, 3), dtype=np.float32)
    pts[:] = np.nan
    pts[mask_flat] = X.T

    return pts.reshape(H, W, 3)


# -------------------------------
# 3. Camera → World
#    X_world = R_cw * X_cam + C_cam_world
# -------------------------------
def cam_to_world(points_cam, mask_valid, R_cam_to_world, cam_pos_world):
    """
    points_cam:   (H, W, 3)
    mask_valid:   (H, W)
    R_cam_to_world: 3x3
    cam_pos_world:  3x1  (camera_position_world)
    return:
        points_world: (H, W, 3)
    """
    H, W, _ = points_cam.shape
    pts = points_cam.reshape(-1, 3)
    mask = (mask_valid.reshape(-1) > 0) & np.isfinite(pts).all(axis=1)

    pts_valid = pts[mask]  # (N,3)
    Xw_valid = (R_cam_to_world @ pts_valid.T).T + cam_pos_world.T  # (N,3)

    out = np.zeros((H * W, 3), dtype=np.float32)
    out[:] = np.nan
    out[mask] = Xw_valid

    return out.reshape(H, W, 3)


def inpaint_height_map_cv(points, valid_mask, inpaint_radius=3.0, method="telea"):
    """
    Inpaint NaN values in the Z channel (height) of a structured-light point cloud
    using OpenCV's cv.inpaint. Only fills pixels where:
        valid_mask == True AND Z is NaN.
    Other pixels are preserved.

    Parameters
    ----------
    points : np.ndarray, shape (H, W, 3)
        3D points in camera or world coordinates. points[..., 2] may contain NaNs.

    valid_mask : np.ndarray, shape (H, W)
        0/1 or bool mask. Inside this region we want a continuous surface.
        Outside this region, Z will remain NaN.

    inpaint_radius : float, optional (default=3.0)
        Radius (in pixels) of the circular neighborhood used by cv.inpaint.
        Typical values:
            2.0 ~ 4.0 : small holes, detailed preservation
            4.0 ~ 6.0 : larger holes, smoother fill

    method : {"telea", "ns"}, optional (default="telea")
        Inpainting algorithm:
            - "telea" -> cv.INPAINT_TELEA (fast, good for small gaps)
            - "ns"    -> cv.INPAINT_NS    (Navier-Stokes based, a bit smoother)

    Returns
    -------
    points_filled : np.ndarray, shape (H, W, 3)
        Copy of input points with Z-channel inpainted inside valid_mask.
        Outside valid_mask, Z remains NaN.
    """

    assert points.ndim == 3 and points.shape[2] == 3, "points must be (H, W, 3)"

    H, W, _ = points.shape
    Z = points[..., 2].astype(np.float32)
    vm = valid_mask.astype(bool)

    # Pixels where we actually want to fill: inside vm AND Z is NaN
    nan_mask = vm & ~np.isfinite(Z)
    if not nan_mask.any():
        print("[cv.inpaint] No NaNs to fill; returning original points.")
        return points.copy()

    # Build inpaint mask: 255 = to be inpainted, 0 = keep as is
    inpaint_mask = np.zeros((H, W), dtype=np.uint8)
    inpaint_mask[nan_mask] = 255

    # cv.inpaint does not like NaNs → replace NaNs with a neutral value
    Z_src = Z.copy()
    # Use mean of finite values inside valid_mask as neutral value
    finite_inside = vm & np.isfinite(Z_src)
    z_mean = float(Z_src[finite_inside].mean()) if finite_inside.any() else 0.0
    Z_src[~np.isfinite(Z_src)] = z_mean
    # Outside valid_mask we don't care; set to mean as well
    Z_src[~vm] = z_mean

    # Choose OpenCV flag
    if method.lower() == "telea":
        flag = cv2.INPAINT_TELEA
    else:
        flag = cv2.INPAINT_NS

    # OpenCV supports 32F single-channel inpaint
    Z_inpainted = cv2.inpaint(Z_src, inpaint_mask, inpaint_radius, flag)

    # Build output Z: only overwrite NaN pixels we wanted to fill
    Z_out = Z.copy()
    Z_out[nan_mask] = Z_inpainted[nan_mask]
    # Ensure outside valid_mask remains NaN
    Z_out[~vm] = np.nan

    # Combine back into (H, W, 3)
    points_filled = points.copy()
    points_filled[..., 2] = Z_out

    print("Z before fill:", np.nanmin(points[..., 2]), np.nanmax(points[..., 2]))
    print(
        "Z after fill:",
        np.nanmin(points_filled[..., 2]),
        np.nanmax(points_filled[..., 2]),
    )
    return points_filled


def median_filter_in_mask(points_filled, valid_mask, ksize=3):
    """
    Median filter Z-channel inside valid_mask.
    Mask-outside pixels are set to +inf, so they never affect the median.
    After filtering, outside mask → restored to NaN.
    If a valid pixel becomes +inf (rare), restore original Z.

    Parameters
    ----------
    points_filled : (H, W, 3)
    valid_mask : (H, W), bool
    ksize : odd int

    Returns
    -------
    points_smoothed : (H, W, 3)
    height_map_smooth : (H, W)
    """

    Z = points_filled[..., 2]
    vm = valid_mask.astype(bool)

    # save original Z
    Z_original = Z.copy()

    # 1) Set pixels outside mask to +inf
    Z_work = Z.copy()
    Z_work[~vm] = np.inf

    # 2) median filter (OpenCV supports inf)
    Z_med = cv2.medianBlur(Z_work.astype(np.float32), ksize)

    # 3) Restore outside mask to NaN
    Z_med[~vm] = np.nan

    # 4) If any valid pixel became inf due to sorting, restore original value
    bad_inside = vm & np.isinf(Z_med)
    Z_med[bad_inside] = Z_original[bad_inside]

    # 5) Combine results
    points_smoothed = points_filled.copy()
    points_smoothed[..., 2] = Z_med

    print(
        "Z before median filter:",
        np.nanmin(points_filled[..., 2]),
        np.nanmax(points_filled[..., 2]),
    )
    print(
        "Z after median filter:",
        np.nanmin(points_smoothed[..., 2]),
        np.nanmax(points_smoothed[..., 2]),
    )
    return points_smoothed


def numpy_to_o3d_pointcloud(points, valid_mask=None, rgb_image=None, gray_image=None):
    """
    Convert a dense structured-light point cloud stored as a NumPy array
    into an Open3D PointCloud object. Supports optional validity mask and
    per-pixel RGB or grayscale color.

    Parameters
    ----------
    points : np.ndarray, shape (H, W, 3)
        The 3D point cloud in either camera or world coordinates.
        This array is *pixel-aligned* with the camera image: the point at
        (y, x) corresponds exactly to the pixel (x, y) in the camera frame.
        The unit is typically millimeters in your structured-light setup.

        points[...,0] = X (world or camera)
        points[...,1] = Y
        points[...,2] = Z  ← often used as depth/height

    valid_mask : np.ndarray, shape (H, W), optional
        Binary mask indicating which points are valid.
        If provided, only locations where mask != 0 will be kept.
        This is typically the shadow mask or decoding validity mask
        computed from structured-light Gray code.
        If None, all finite (non-NaN) points are used.

    rgb_image : np.ndarray, shape (H, W, 3), optional
        Optional per-pixel RGB image in uint8 format.
        Used to colorize the point cloud.
        Must be pixel-aligned with `points`.

        Example: camera captured color image.

    gray_image : np.ndarray, shape (H, W), optional
        Optional single-channel grayscale image used as color.
        If supplied, it will be expanded to RGB by repeating the channel.

    Returns
    -------
    pcd : open3d.geometry.PointCloud
        The resulting Open3D point cloud object, containing:
            - pcd.points : Nx3 array of 3D coordinates
            - pcd.colors : Nx3 array of RGB values in [0,1] (if provided)

    Notes
    -----
    - Any NaN or infinite point will be removed.
    - If both rgb_image and gray_image are None, the point cloud will have no color.
    - This function reshapes the (H,W,3) structured-light output into (N,3),
      where N is the number of valid 3D points.
    """

    H, W, _ = points.shape
    pts = points.reshape(-1, 3)  # flatten to (H*W, 3)

    # ----------------------------------------------------------
    # 1. Construct validity mask
    # ----------------------------------------------------------
    if valid_mask is not None:
        # mask > 0 means valid
        mask = valid_mask.reshape(-1) > 0
    else:
        mask = np.ones((H * W,), dtype=bool)

    # Remove NaN or inf points
    mask &= np.isfinite(pts).all(axis=1)

    pts = pts[mask]  # (N,3) valid points

    # ----------------------------------------------------------
    # 2. Create Open3D point cloud
    # ----------------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    # ----------------------------------------------------------
    # 3. Handle colors (optional)
    # ----------------------------------------------------------
    if rgb_image is not None:
        # reshape and pick only valid pixels
        rgb = rgb_image.reshape(-1, 3)[mask].astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    elif gray_image is not None:
        g = gray_image.reshape(-1)[mask].astype(np.float32) / 255.0
        # replicate to R=G=B
        rgb = np.stack([g, g, g], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    # If no color, the point cloud remains uncolored.

    return pcd


def preprocess_pointcloud(
    pcd,
    voxel_size=2.0,
    stat_nb_neighbors=30,
    stat_std_ratio=2.0,
    rad_nb_points=None,
    rad_radius=None,
    estimate_normals=True,
    normal_radius=10.0,
    normal_max_nn=30,
    camera_location=None,
):
    """
    Preprocess an Open3D point cloud with voxel downsampling, statistical denoising,
    optional radius-based outlier removal, and optional normal estimation.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Input point cloud, typically dense and noisy before filtering.

    voxel_size : float, optional (default=2.0)
        Size of the voxel grid in the same unit as the point coordinates.
        Each voxel (cube) merges all points inside it into a single representative point.
        Effects:
            - Larger voxel_size → stronger downsampling → smoother + fewer points.
            - Smaller voxel_size → preserves details but retains noise.
        Typical values:
            - 0.5–1.0 mm → high-detail objects
            - 1.0–2.0 mm → flat surfaces (e.g., calibration board or whiteboard)

    stat_nb_neighbors : int, optional (default=30)
        For Statistical Outlier Removal (SOR):
        Number of nearest neighbors used to compute the mean distance of a point
        to its local neighborhood.
        A larger value makes the filter more robust but slower.
        Typical:
            - 20–40 for structured-light dense point clouds.

    stat_std_ratio : float, optional (default=2.0)
        Threshold in the SOR filter.
        A point is considered an outlier if:
            distance > (mean + stat_std_ratio * std_dev)
        Lower value → more aggressive filtering (may remove edges).
        Higher value → looser filtering.
        Typical:
            - 1.5–2.5

    rad_nb_points : int or None, optional (default=None)
        For Radius Outlier Removal:
        Minimum number of neighbors required inside the spherical radius `rad_radius`.
        If None, radius removal is skipped.
        Use this to remove isolated points that statistical filter might miss.

    rad_radius : float or None, optional (default=None)
        Radius (in same unit as point cloud) used by radius outlier removal.
        Larger radius → smoother filtering.
        Typical for structured-light scanning:
            - 3–6 mm radius
            - 10–20 neighbors

    estimate_normals : bool, optional (default=True)
        If True, estimate surface normals after denoising/downsampling.
        Required for:
            - Surface reconstruction (Poisson, BPA)
            - Mesh generation
            - Smoothing
            - Normal-based shading

    normal_radius : float, optional (default=10.0)
        Search radius (in same unit as points) for finding neighbors
        when estimating normals.
        Larger radius → smoother normals (good for flat planes).
        Smaller radius → more detailed normals but noisier.

    normal_max_nn : int, optional (default=30)
        Maximum number of neighbors used for normal estimation.
        Acts as a safety limit in case the radius includes too many points.

    camera_location : array-like shape (3,), optional (default=None)
        If provided, orient the estimated normals to consistently point
        toward the given camera location.
        Useful for structured-light systems where the normal direction
        should face the camera for consistent shading & meshing.

    Returns
    -------
    pcd : open3d.geometry.PointCloud
        The cleaned, optionally downsampled, denoised, and normal-estimated point cloud.
    """

    print("Input point cloud:", pcd)
    print("Original number of points:", np.asarray(pcd.points).shape[0])

    # --------------------------------------------------
    # 1. VOXEL DOWNSAMPLING
    # --------------------------------------------------
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"[Voxel Downsample] voxel_size={voxel_size}")
        print("  #points:", np.asarray(pcd.points).shape[0])

    # --------------------------------------------------
    # 2. STATISTICAL OUTLIER REMOVAL (SOR)
    # --------------------------------------------------
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=stat_nb_neighbors, std_ratio=stat_std_ratio
    )
    print(
        f"[Statistical Outlier Removal] neighbors={stat_nb_neighbors}, std_ratio={stat_std_ratio}"
    )
    print("  #points:", np.asarray(pcd.points).shape[0])

    # --------------------------------------------------
    # 3. RADIUS OUTLIER REMOVAL (optional)
    # --------------------------------------------------
    if rad_nb_points is not None and rad_radius is not None:
        pcd, ind = pcd.remove_radius_outlier(nb_points=rad_nb_points, radius=rad_radius)
        print(
            f"[Radius Outlier Removal] radius={rad_radius}, nb_points={rad_nb_points}"
        )
        print("  #points:", np.asarray(pcd.points).shape[0])

    # --------------------------------------------------
    # 4. NORMAL ESTIMATION (optional)
    # --------------------------------------------------
    if estimate_normals:
        print("[Normal Estimation]")
        print(f"  search radius={normal_radius}, max_nn={normal_max_nn}")

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn
            )
        )

        if camera_location is not None:
            print("  Orienting normals toward camera:", camera_location)
            pcd.orient_normals_towards_camera_location(camera_location)

    return pcd


def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])


def save_pcd(pcd, filename="preprocessed.ply"):
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    print("Saved:", filename)


def build_preprocessed_pointcloud_from_grid(
    points,
    valid_mask=None,
    rgb_image=None,
    gray_image=None,
    # cv.inpaint parameters:
    inpaint_radius=3.0,
    method="telea",
    # preprocess_pointcloud parameters:
    voxel_size=2.0,
    stat_nb_neighbors=30,
    stat_std_ratio=2.0,
    rad_nb_points=None,
    rad_radius=None,
    estimate_normals=True,
    normal_radius=10.0,
    normal_max_nn=30,
    camera_location=None,
):
    """
    High-level pipeline:
      1) Use 2D RBF to fill NaNs in Z (height) inside valid_mask on the (H, W, 3) grid.
      2) Convert the filled grid to an Open3D PointCloud via numpy_to_o3d_pointcloud.
      3) Run preprocess_pointcloud(...) to denoise & smooth in 3D.

    Parameters
    ----------
    points : np.ndarray, shape (H, W, 3)
        Raw triangulated 3D points in camera/world coordinates.
        points[...,2] may contain NaNs.

    valid_mask : np.ndarray, shape (H, W), optional
        0/1 mask from decoding. Inside this mask we want a smooth surface.
        Outside this mask, points are discarded from the Open3D cloud and
        Z will remain NaN in the filled grid.

    rgb_image : np.ndarray, shape (H, W, 3), optional
        Optional RGB image (uint8) used for coloring the point cloud.

    gray_image : np.ndarray, shape (H, W), optional
        Optional grayscale image used for coloring the point cloud.

    inpaint_radius : float, optional (default=3.0)
        Radius (in pixels) used by cv.inpaint to fill NaNs in Z.

    method : {"telea", "ns"}, optional (default="telea")
        Inpainting algorithm for cv.inpaint.

    Other parameters:
        Passed directly to preprocess_pointcloud(...)

    Returns
    -------
    pcd_clean : open3d.geometry.PointCloud
        Denoised and smoothed Open3D point cloud (after RBF filling).

    points_filled : np.ndarray, shape (H, W, 3)
        The (H, W, 3) 3D grid after RBF filling of Z inside valid_mask.
        Still pixel-aligned with the original camera image.
    """
    # If no valid_mask is given, treat all finite points as valid region.
    if valid_mask is None:
        finite_mask = np.isfinite(points).all(axis=2)
        valid_mask = finite_mask.astype(np.uint8)

    # 1) Fill NaNs in Z inside valid region using RBF
    points_filled = inpaint_height_map_cv(
        points=points,
        valid_mask=valid_mask,
        inpaint_radius=inpaint_radius,
        method=method,
    )

    # 2) Convert to Open3D point cloud (uses valid_mask & removes any residual NaNs)
    pcd = numpy_to_o3d_pointcloud(
        points_filled,
        valid_mask=valid_mask,
        rgb_image=rgb_image,
        gray_image=gray_image,
    )

    # 3) Preprocess in Open3D
    pcd_clean = preprocess_pointcloud(
        pcd,
        voxel_size=voxel_size,
        stat_nb_neighbors=stat_nb_neighbors,
        stat_std_ratio=stat_std_ratio,
        rad_nb_points=rad_nb_points,
        rad_radius=rad_radius,
        estimate_normals=estimate_normals,
        normal_radius=normal_radius,
        normal_max_nn=normal_max_nn,
        camera_location=camera_location,
    )

    return pcd_clean, points_filled


def points_to_heightmap(
    points_xyz, x_grid, y_grid, fill_holes=True, max_fill_dist=None
):
    """
    points_xyz: (N,3) array, columns [x,y,z]
    x_grid: (W,) or (H,W)  (meshgrid后也行)
    y_grid: (H,) or (H,W)
    returns: Z (H,W), mask_valid (H,W)
    """
    pts = np.asarray(points_xyz)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Ensure meshgrid
    if x_grid.ndim == 1 and y_grid.ndim == 1:
        X, Y = np.meshgrid(x_grid, y_grid)
    else:
        X, Y = x_grid, y_grid

    # main interpolation with linear method
    Z = griddata((x, y), z, (X, Y), method="linear")
    mask_valid = ~np.isnan(Z)

    if not fill_holes:
        return Z, mask_valid

    # fill holes using nearest neighbor interpolation
    Z_nn = griddata((x, y), z, (X, Y), method="nearest")
    hole = ~mask_valid

    if max_fill_dist is None:
        Z[hole] = Z_nn[hole]
        return Z, ~np.isnan(Z)

    # Use cKDTree to find nearest data point distances for hole pixels
    tree = cKDTree(np.c_[x, y])
    d, _ = tree.query(np.c_[X[hole], Y[hole]], k=1)
    ok = d <= max_fill_dist

    Zi = Z.copy()
    Zi[hole] = np.nan
    Zi[np.where(hole)[0][ok], np.where(hole)[1][ok]] = Z_nn[hole][ok]  # 这行对索引较绕

    # More robust way: use flat indexing
    Zi = Z.copy()
    flat = hole.ravel()
    Z_flat = Zi.ravel()
    Znn_flat = Z_nn.ravel()
    d_full = np.full(flat.shape, np.inf)
    d_full[flat] = d
    fill_idx = flat & (d_full <= max_fill_dist)
    Z_flat[fill_idx] = Znn_flat[fill_idx]
    Zi = Z_flat.reshape(Z.shape)

    return Zi, ~np.isnan(Zi)


# class ZofXY_TIN_GapAware:
#     def __init__(
#         self,
#         xyz_valid: np.ndarray,
#         fill_value=np.nan,
#         max_edge=None,
#         max_area=None,
#         max_circumradius=None,
#     ):
#         """
#         xyz_valid: (K,3) with finite x,y,z only
#         max_edge:        reject triangles if any XY edge > max_edge
#         max_area:        reject triangles if triangle area in XY > max_area
#         max_circumradius:reject triangles if circumradius in XY > max_circumradius
#         """
#         xyz_valid = np.asarray(xyz_valid, dtype=np.float64)
#         if xyz_valid.ndim != 2 or xyz_valid.shape[1] != 3:
#             raise ValueError("xyz_valid must be (K,3).")
#         self.fill_value = fill_value
#         self.xy = xyz_valid[:, :2]
#         self.z = xyz_valid[:, 2]

#         self.tri = Delaunay(self.xy)

#         self.bad_simplex = None
#         if (
#             (max_edge is not None)
#             or (max_area is not None)
#             or (max_circumradius is not None)
#         ):
#             self.bad_simplex = self._compute_bad_simplices(
#                 max_edge=max_edge, max_area=max_area, max_circumradius=max_circumradius
#             )

#     @staticmethod
#     def _tri_area_2d(p0, p1, p2):
#         # area = 0.5 * |(p1-p0) x (p2-p0)| in 2D (scalar cross)
#         v1 = p1 - p0
#         v2 = p2 - p0
#         cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
#         return 0.5 * np.abs(cross)

#     def _compute_bad_simplices(
#         self, max_edge=None, max_area=None, max_circumradius=None
#     ):
#         simp = self.tri.simplices  # (S,3)
#         p0 = self.xy[simp[:, 0]]
#         p1 = self.xy[simp[:, 1]]
#         p2 = self.xy[simp[:, 2]]

#         bad = np.zeros((simp.shape[0],), dtype=bool)

#         # Edge lengths
#         if max_edge is not None:
#             e01 = np.linalg.norm(p0 - p1, axis=1)
#             e12 = np.linalg.norm(p1 - p2, axis=1)
#             e20 = np.linalg.norm(p2 - p0, axis=1)
#             bad |= (e01 > max_edge) | (e12 > max_edge) | (e20 > max_edge)

#         # Triangle area in XY
#         if max_area is not None:
#             A = self._tri_area_2d(p0, p1, p2)
#             bad |= A > max_area

#         # Circumradius R = abc / (4A)
#         if max_circumradius is not None:
#             a = np.linalg.norm(p1 - p2, axis=1)
#             b = np.linalg.norm(p2 - p0, axis=1)
#             c = np.linalg.norm(p0 - p1, axis=1)
#             A = self._tri_area_2d(p0, p1, p2)
#             eps = 1e-12
#             R = (a * b * c) / (4.0 * np.maximum(A, eps))
#             bad |= R > max_circumradius

#         return bad

#     def __call__(self, X, Y=None):
#         if Y is None:
#             xy = np.asarray(X, dtype=np.float64)
#             if xy.shape[-1] != 2:
#                 raise ValueError("If Y is None, X must have shape (...,2).")
#             orig_shape = xy.shape[:-1]
#             xy_flat = xy.reshape(-1, 2)
#             z_flat = self._interp_flat(xy_flat)
#             return z_flat.reshape(orig_shape)

#         X = np.asarray(X, dtype=np.float64)
#         Y = np.asarray(Y, dtype=np.float64)
#         if X.shape != Y.shape:
#             raise ValueError("X and Y must have the same shape.")
#         xy_flat = np.stack([X.ravel(), Y.ravel()], axis=1)
#         z_flat = self._interp_flat(xy_flat)
#         return z_flat.reshape(X.shape)

#     def _interp_flat(self, xy_flat):
#         simp = self.tri.find_simplex(xy_flat)
#         out = np.full((xy_flat.shape[0],), self.fill_value, dtype=np.float64)

#         inside = simp >= 0
#         if not np.any(inside):
#             return out

#         idx_inside = np.where(inside)[0]

#         if self.bad_simplex is None:
#             idx_ok = idx_inside
#         else:
#             ok_tri = ~self.bad_simplex[simp[idx_inside]]
#             idx_ok = idx_inside[ok_tri]

#         if idx_ok.size == 0:
#             return out

#         simp_ok = simp[idx_ok]
#         q = xy_flat[idx_ok]

#         T = self.tri.transform[simp_ok, :2]
#         r = q - self.tri.transform[simp_ok, 2]
#         b = np.einsum("mij,mj->mi", T, r)
#         w0 = b[:, 0]
#         w1 = b[:, 1]
#         w2 = 1.0 - w0 - w1

#         verts = self.tri.simplices[simp_ok]
#         z0 = self.z[verts[:, 0]]
#         z1 = self.z[verts[:, 1]]
#         z2 = self.z[verts[:, 2]]

#         out[idx_ok] = w0 * z0 + w1 * z1 + w2 * z2
#         return out


class TINInterpXY:
    """
    Piecewise-linear interpolation over Delaunay triangulation in XY.

    - Input vertices: xyz_valid (K,3) finite only
    - Optional per-vertex features: feat_valid (K,) or (K,C) (e.g., intensity, RGB)
    - Query: meshgrid X,Y or xy points (...,2)
    - Gap-aware rejection: max_edge / max_area / max_circumradius in XY
    """

    def __init__(
        self,
        xyz_valid,
        feat_valid=None,
        fill_value=np.nan,
        max_edge=None,
        max_area=None,
        max_circumradius=None,
    ):
        xyz_valid = np.asarray(xyz_valid, dtype=np.float64)
        if xyz_valid.ndim != 2 or xyz_valid.shape[1] != 3:
            raise ValueError("xyz_valid must be (K,3).")
        self.fill_value = fill_value

        self.xy = xyz_valid[:, :2]
        self.z = xyz_valid[:, 2]

        # features
        self.feat = None
        if feat_valid is not None:
            feat_valid = np.asarray(feat_valid)
            if feat_valid.shape[0] != xyz_valid.shape[0]:
                raise ValueError("feat_valid length must match xyz_valid.")
            if feat_valid.ndim == 1:
                feat_valid = feat_valid[:, None]  # (K,1)
            self.feat = feat_valid.astype(np.float64, copy=False)  # (K,C)

        # triangulation
        self.tri = Delaunay(self.xy)

        # mark bad triangles
        self.bad_simplex = None
        if (
            (max_edge is not None)
            or (max_area is not None)
            or (max_circumradius is not None)
        ):
            self.bad_simplex = self._compute_bad_simplices(
                max_edge=max_edge, max_area=max_area, max_circumradius=max_circumradius
            )

    @staticmethod
    def _tri_area_2d(p0, p1, p2):
        v1 = p1 - p0
        v2 = p2 - p0
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def _compute_bad_simplices(
        self, max_edge=None, max_area=None, max_circumradius=None
    ):
        simp = self.tri.simplices  # (S,3)
        p0 = self.xy[simp[:, 0]]
        p1 = self.xy[simp[:, 1]]
        p2 = self.xy[simp[:, 2]]

        bad = np.zeros((simp.shape[0],), dtype=bool)

        if max_edge is not None:
            e01 = np.linalg.norm(p0 - p1, axis=1)
            e12 = np.linalg.norm(p1 - p2, axis=1)
            e20 = np.linalg.norm(p2 - p0, axis=1)
            bad |= (e01 > max_edge) | (e12 > max_edge) | (e20 > max_edge)

        if max_area is not None:
            A = self._tri_area_2d(p0, p1, p2)
            bad |= A > max_area

        if max_circumradius is not None:
            a = np.linalg.norm(p1 - p2, axis=1)
            b = np.linalg.norm(p2 - p0, axis=1)
            c = np.linalg.norm(p0 - p1, axis=1)
            A = self._tri_area_2d(p0, p1, p2)
            eps = 1e-12
            R = (a * b * c) / (4.0 * np.maximum(A, eps))
            bad |= R > max_circumradius

        return bad  # (S,)

    def __call__(self, X, Y=None, return_feat=False):
        """
        If Y is provided: X,Y are meshgrid arrays -> returns Z (and F if return_feat)
        If Y is None: X is (...,2) xy array -> returns z (...), (and f (...,C) if return_feat)
        """
        if Y is None:
            xy = np.asarray(X, dtype=np.float64)
            if xy.shape[-1] != 2:
                raise ValueError("If Y is None, X must have shape (...,2).")
            orig_shape = xy.shape[:-1]
            xy_flat = xy.reshape(-1, 2)
            z_flat, f_flat = self._interp_flat(xy_flat, return_feat=return_feat)
            z = z_flat.reshape(orig_shape)
            if not return_feat:
                return z
            f = f_flat.reshape(orig_shape + (f_flat.shape[-1],))
            return z, f

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape.")
        xy_flat = np.stack([X.ravel(), Y.ravel()], axis=1)
        z_flat, f_flat = self._interp_flat(xy_flat, return_feat=return_feat)
        Z = z_flat.reshape(X.shape)
        if not return_feat:
            return Z
        F = f_flat.reshape(X.shape + (f_flat.shape[-1],))
        return Z, F

    def _interp_flat(self, xy_flat, return_feat=False):
        simp = self.tri.find_simplex(xy_flat)

        z_out = np.full((xy_flat.shape[0],), self.fill_value, dtype=np.float64)

        if return_feat:
            if self.feat is None:
                raise ValueError("return_feat=True but feat_valid was not provided.")
            C = self.feat.shape[1]
            f_out = np.full((xy_flat.shape[0], C), self.fill_value, dtype=np.float64)
        else:
            f_out = None

        inside = simp >= 0
        if not np.any(inside):
            return z_out, f_out

        idx_inside = np.where(inside)[0]

        # reject bad triangles if enabled
        if self.bad_simplex is None:
            idx_ok = idx_inside
        else:
            ok_tri = ~self.bad_simplex[simp[idx_inside]]
            idx_ok = idx_inside[ok_tri]

        if idx_ok.size == 0:
            return z_out, f_out

        simp_ok = simp[idx_ok]
        q = xy_flat[idx_ok]

        T = self.tri.transform[simp_ok, :2]
        r = q - self.tri.transform[simp_ok, 2]
        b = np.einsum("mij,mj->mi", T, r)
        w0 = b[:, 0]
        w1 = b[:, 1]
        w2 = 1.0 - w0 - w1

        verts = self.tri.simplices[simp_ok]
        z0 = self.z[verts[:, 0]]
        z1 = self.z[verts[:, 1]]
        z2 = self.z[verts[:, 2]]
        z_out[idx_ok] = w0 * z0 + w1 * z1 + w2 * z2

        if return_feat:
            f0 = self.feat[verts[:, 0]]  # (M,C)
            f1 = self.feat[verts[:, 1]]
            f2 = self.feat[verts[:, 2]]
            # broadcast weights to (M,1)
            ww0 = w0[:, None]
            ww1 = w1[:, None]
            ww2 = w2[:, None]
            f_out[idx_ok] = ww0 * f0 + ww1 * f1 + ww2 * f2

        return z_out, f_out


def json_loader(file_path: str) -> dict:
    """
    Load calibration data from a JSON file.
    Convert each value to a NumPy array or scalar if it contains only one element.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    result = {}
    for key, value in data.items():
        arr = np.array(value)
        # if array has only one element, return scalar
        result[key] = arr.item() if arr.size == 1 else arr

    return result


# -------------------------------
# Load calibration JSON
# -------------------------------
def load_calibration(json_path):
    data = json_loader(json_path)

    K_c = data["camera_intrinsic"]
    K_p = data["projector_intrinsic"]

    dist_c = data["camera_distortion"]
    dist_p = data["projector_distortion"]

    R_cp = data["R_camera_to_projector"]
    t_cp = data["t_camera_to_projector"].reshape(3, 1)

    # NEW WORLD COORDINATE (checkerboard-centered)
    R_world_to_cam = data["R_world_to_camera"]
    R_cam_to_world = R_world_to_cam.T

    # t_new(3,1)
    t_new = data["t_new"].reshape(3, 1)

    # derived:
    t_cam_to_world = -R_cam_to_world @ t_new

    return (
        K_c,
        dist_c,
        K_p,
        dist_p,
        R_cp,
        t_cp,
        R_world_to_cam,
        R_cam_to_world,
        t_new,
        t_cam_to_world,
    )


def save_pointcloud_ply(
    filename,
    points,  # (H, W, 3) float32
    valid_mask=None,  # (H, W) uint8 or bool, optional
    rgb_image=None,  # (H, W, 3) uint8, optional
    gray_image=None,  # (H, W) uint8, optional
    image_value_scale=1.0,
):
    """
    Save point cloud to ASCII PLY format.
    Supports:
        - XYZ only
        - XYZ + RGB
        - XYZ + gray
        - valid_mask (invalid points removed)
    """

    H, W, _ = points.shape
    pts = points.reshape(-1, 3)

    # ------------------------------
    # Build mask
    # ------------------------------
    mask = np.ones((H * W,), dtype=bool)

    # mask from user
    if valid_mask is not None:
        mask &= valid_mask.reshape(-1) > 0

    # mask NaN points
    mask &= np.isfinite(pts).all(axis=1)

    pts_valid = pts[mask]  # (N,3)

    # ------------------------------
    # Handle colors
    # ------------------------------
    colors = None

    if rgb_image is not None:
        rgb = rgb_image.reshape(-1, 3) / image_value_scale * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        colors = rgb[mask]  # (N,3)

    elif gray_image is not None:
        g = gray_image.reshape(-1) / image_value_scale * 255.0
        g = np.clip(g, 0, 255).astype(np.uint8)
        g = g[mask]
        colors = np.stack([g, g, g], axis=1)
    # ------------------------------
    # Write PLY
    # ------------------------------
    with open(filename, "w") as f:

        # header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts_valid.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        # data
        if colors is None:
            # XYZ only
            for x, y, z in pts_valid:
                f.write(f"{x} {y} {z}\n")

        else:
            # XYZ + RGB
            for (x, y, z), (r, g, b) in zip(pts_valid, colors):
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

    print(f"Saved PLY: {filename}  (points = {pts_valid.shape[0]})")


def plot_pointcloud_plotly(
    points_3d,  # (H, W, 3) or (N,3)
    valid_mask=None,  # (H,W) or (N,) optional
    rgb=None,  # (H,W,3) or (N,3) optional, uint8
    gray=None,  # (H,W) or (N,) optional, uint8
    title="Point Cloud",
    plot_width=800,
    plot_height=600,
):
    # --------------------------------------------
    # Reshape point cloud to (N,3)
    # --------------------------------------------
    if points_3d.ndim == 3:
        H, W, _ = points_3d.shape
        pts = points_3d.reshape(-1, 3)
    else:
        pts = points_3d

    N = pts.shape[0]

    # --------------------------------------------
    # Mask
    # --------------------------------------------
    mask = np.ones((N,), dtype=bool)

    if valid_mask is not None:
        mask &= valid_mask.reshape(-1) > 0

    mask &= np.isfinite(pts).all(axis=1)

    pts_valid = pts[mask]

    # --------------------------------------------
    # Colors
    # --------------------------------------------
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)[mask]
        colors = (
            "rgb("
            + rgb[:, 0].astype(str)
            + ","
            + rgb[:, 1].astype(str)
            + ","
            + rgb[:, 2].astype(str)
            + ")"
        )

    elif gray is not None:
        g = gray.reshape(-1)[mask]
        colors = (
            "rgb(" + g.astype(str) + "," + g.astype(str) + "," + g.astype(str) + ")"
        )

    else:
        colors = "rgb(0.1216, 0.4667, 0.7059)"  # or a list same length

    # --------------------------------------------
    # Plotly scatter3d
    # --------------------------------------------
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts_valid[:, 0],
                y=pts_valid[:, 1],
                z=pts_valid[:, 2],
                mode="markers",
                marker=dict(size=1, color=colors, opacity=0.5),  # can be a list
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X [mm]",
            yaxis_title="Y [mm]",
            zaxis_title="Z [mm]",
            aspectmode="data",  # keep XYZ aspect ratio
        ),
        width=plot_width,
        height=plot_height,
    )

    fig.show()


def show_image_and_heightmap(
    image,
    point_cloud,
    mask_valid=None,
    vmin=None,
    vmax=None,
    titles=[None, None],
    extent=[None, None],
    units=["pixel", "pixel"],
    origin="upper",
):
    """
    Display the camera grayscale image alongside the height map derived from a point cloud.

    Parameters
    ----------
    image : (H, W, 1 or 3) array
        Grayscale or RGB camera image (uint8 or float32).
    point_cloud : (H, W, 3) array
        Per-pixel 3D point cloud in world coordinates.
    mask_valid : (H, W) array, optional
        Binary mask of valid 3D points (0 or 255 or 0/1).
        Invalid pixels will be shown as NaN in the height map.
    vmin, vmax : float, optional
        Display range for the height-map colorbar.
    titles : list of str, optional
        Titles for the two subplots. If None, default titles are used.
    """

    if titles is None:
        titles = [
            "Camera Image (Reflectance)",
            "Height Map (z in World Coordinates)",
        ]

    # -----------------------------
    # Extract height (world Z)
    # -----------------------------
    if point_cloud.ndim != 3 or point_cloud.shape[2] != 3:
        height_map = point_cloud.copy()
    else:
        height_map = point_cloud[..., 2].copy()

    if mask_valid is not None:
        height_map[mask_valid == 0] = np.nan

    # -----------------------------
    # Figure layout (KEY CHANGE)
    # -----------------------------
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)  # colorbar 单独一列

    ax_img = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    cax_h = fig.add_subplot(gs[0, 2])

    # -----------------------------
    # Left: image
    # -----------------------------
    ax_img.set_title(titles[0])

    if image.dtype == np.uint8:
        if extent[0] is None:
            im_img = ax_img.imshow(image, cmap="gray", vmin=0, vmax=255, origin=origin)
        else:
            im_img = ax_img.imshow(
                image, cmap="gray", vmin=0, vmax=255, extent=extent[0], origin=origin
            )
    else:
        if extent[0] is None:
            im_img = ax_img.imshow(
                image, cmap="gray", vmin=0.0, vmax=1.0, origin=origin
            )
        else:
            im_img = ax_img.imshow(
                image, cmap="gray", vmin=0.0, vmax=1.0, extent=extent[0], origin=origin
            )
    if units[0] is None:
        ax_img.set_xlabel("u [pixel]")
        ax_img.set_ylabel("v [pixel]")
    else:
        ax_img.set_xlabel(f"x [{units[0]}]")
        ax_img.set_ylabel(f"y [{units[0]}]")
    ax_img.set_aspect("equal")

    # colorbar for reflectance is optional, so commented out
    # fig.colorbar(im_img, ax=ax_img, fraction=0.046)

    # -----------------------------
    # Right: height map
    # -----------------------------
    ax_h.set_title(titles[1])
    if extent[1] is None:
        im_h = ax_h.imshow(
            height_map,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            origin=origin,
        )
    else:
        im_h = ax_h.imshow(
            height_map,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            extent=extent[1],
            origin=origin,
        )

    if units[1] is None:
        ax_h.set_xlabel("u [pixel]")
        ax_h.set_ylabel("v [pixel]")
    else:
        ax_h.set_xlabel(f"x [{units[1]}]")
        ax_h.set_ylabel(f"y [{units[1]}]")
    ax_h.set_aspect("equal")

    # Height colorbar
    fig.colorbar(im_h, cax=cax_h, label="z [mm]")

    plt.show()

    return height_map


def create_rect_mask(shape, top_left, bottom_right):
    """
    fill a rectangle area with 1s in a zero matrix.

    Args:
        shape: tuple (H, W) matrix shape
        top_left: tuple (x1, y1) upper left corner coordinates (column, row)
        bottom_right: tuple (x2, y2) lower right corner coordinates (column, row)
        mask: (H, W) numpy array
    Returns:
        mask: (H, W) numpy array with rectangle area filled with 1s
    """
    H, W = shape
    x1, y1 = top_left  # x corresponds to column, y to row
    x2, y2 = bottom_right

    # create zero matrix
    mask = np.zeros((H, W), dtype=np.uint8)  # or np.float32

    # core step: slice assignment
    # Note 1: numpy indexing order is [row, column] i.e. [y, x]
    # Note 2: Python slicing excludes the end index, so we add +1 to include bottom-right corner
    # Note 3: Use np.clip to prevent out-of-bounds errors

    r_start = np.clip(y1, 0, H)
    r_end = np.clip(y2 + 1, 0, H)
    c_start = np.clip(x1, 0, W)
    c_end = np.clip(x2 + 1, 0, W)

    mask[r_start:r_end, c_start:c_end] = 1

    return mask


# bit0 gray code synthetic processing
def process_bit0_numeric_files_3d_recon(root_dir, pairs, bit0_synthetic_flag=True):

    # 1. List all files starting with numbers in the current directory
    files = os.listdir(root_dir)

    number_files = []
    number_pattern = re.compile(
        r"^(\d+)_\d+\.png$"
    )  # match files like "00_00.png", "01_00.png", etc.

    for f in files:
        m = number_pattern.match(f)
        if m:
            num = int(m.group(1))
            number_files.append((num, f))

    if not number_files:
        print("No number-starting files found.")
        return

    # sort by number
    number_files.sort(key=lambda x: x[0])

    # 2. skip 00 and 01
    filtered = [(n, f) for (n, f) in number_files if n >= 2]

    if len(filtered) < 4:
        raise RuntimeError("Need at least 4 numeric files (>=02).")

    # 3. Split into two equal groups
    total = len(filtered)
    half = total // 2

    groupA = filtered[:half]
    groupB = filtered[half:]

    # 4. last two files from each group
    last_A = groupA[-2:]
    last_B = groupB[-2:]
    target_files = last_A + last_B  # 共 4 个

    print("Files selected for replacement:")
    for _, fname in target_files:
        print("  ", fname)

    # 5. Create org_bit0 directory
    org_dir = os.path.join(root_dir, "org_bit0")
    os.makedirs(org_dir, exist_ok=True)

    # 6. Move original files to org_bit0
    for _, fname in target_files:
        src = os.path.join(root_dir, fname)
        dst = os.path.join(org_dir, fname)
        print(f"Moving {src} -> {dst}")
        shutil.move(src, dst)

    # 7. read pattern pairs
    patterns = {}
    for key_prepare in [
        "HorBit0_Shifted_0",
        "HorBit0_Shifted_1",
        "HorBit0_Shifted_0_Inverse",
        "HorBit0_Shifted_1_Inverse",
        "VerBit0_Shifted_0",
        "VerBit0_Shifted_1",
        "VerBit0_Shifted_0_Inverse",
        "VerBit0_Shifted_1_Inverse",
    ]:
        pattern_file = os.path.join(root_dir, f"{key_prepare}_00.png")
        patterns[key_prepare] = load_gray_float32(pattern_file)

    # 8. Generate abs diff images (in the same order as recorded)
    for pair, (_, out_filename) in zip(pairs, target_files):
        key1, key2 = pair
        img1 = patterns[key1]
        img2 = patterns[key2]
        diff = np.abs(img1 - img2)

        # normalize to 0–255
        diff = diff / np.max(diff) * 255
        diff = diff.astype(np.uint8)

        save_path = os.path.join(root_dir, out_filename)
        print(f"Writing new diff image: {save_path}")
        cv2.imwrite(save_path, diff)

    print("✓ Processing complete.\n")


def flatten_and_filter_nan(pointcloud_nm3: np.ndarray):
    """
    pointcloud_nm3: (N, M, 3) with NaNs in invalid regions
    returns xyz_valid: (P, 3) finite points only
    """
    xyz = pointcloud_nm3.reshape(-1, 3)
    good = np.isfinite(xyz).all(axis=1)  # require x,y,z all finite
    xyz_valid = xyz[good]
    return xyz_valid


def o3d_statistical_outlier_removal(xyz, nb_neighbors=30, std_ratio=2.0):
    """
    nb_neighbors: number of neighbors used for mean distance estimation
    std_ratio: lower -> more aggressive removal
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd_f, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    xyz_f = np.asarray(pcd_f.points)
    return xyz_f, ind  # ind are indices of kept points (w.r.t input xyz)


def estimate_spacing(xyz, sample=200000):
    """
    Estimate typical point spacing by median nearest-neighbor distance.
    """
    if xyz.shape[0] > sample:
        idx = np.random.choice(xyz.shape[0], sample, replace=False)
        xyz_use = xyz[idx]
    else:
        xyz_use = xyz

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_use))
    d = np.asarray(pcd.compute_nearest_neighbor_distance())
    return float(np.median(d)), float(np.mean(d))


def o3d_radius_outlier_removal(xyz, radius, nb_points=6):
    """
    radius: neighborhood radius
    nb_points: minimum neighbors within radius to be considered inlier
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd_f, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return np.asarray(pcd_f.points), ind


def aggregate_by_xy_bins(xyz, dx, dy=None, agg="median"):
    """
    Aggregate points that fall into the same (x,y) bin.
    - xyz: (P,3)
    - dx, dy: bin size in same unit as x,y (e.g. mm)
    - agg: "median" | "mean" | "min" | "max"

    Returns:
      xyz_agg: (Q,3) with representative x,y (bin center) and aggregated z
    """
    if dy is None:
        dy = dx
    xyz = np.asarray(xyz, dtype=np.float64)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    xmin, ymin = x.min(), y.min()
    ix = np.floor((x - xmin) / dx).astype(np.int64)
    iy = np.floor((y - ymin) / dy).astype(np.int64)

    # Create a single key for each bin
    # Use max_iy+1 stride to avoid collisions
    stride = iy.max() + 1
    key = ix * stride + iy

    order = np.argsort(key)
    key_s = key[order]
    x_s, y_s, z_s = x[order], y[order], z[order]

    # Find group boundaries
    starts = np.r_[0, np.where(key_s[1:] != key_s[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(key_s)]

    # Representative (x,y): use bin center for stability
    ix_s = ix[order]
    iy_s = iy[order]
    ix_rep = ix_s[starts]
    iy_rep = iy_s[starts]
    x_rep = xmin + (ix_rep + 0.5) * dx
    y_rep = ymin + (iy_rep + 0.5) * dy

    # Aggregate z
    z_rep = np.empty(len(starts), dtype=np.float64)
    for gi, (s, e) in enumerate(zip(starts, ends)):
        zz = z_s[s:e]
        if agg == "median":
            z_rep[gi] = np.median(zz)
        elif agg == "mean":
            z_rep[gi] = np.mean(zz)
        elif agg == "min":
            z_rep[gi] = np.min(zz)
        elif agg == "max":
            z_rep[gi] = np.max(zz)
        else:
            raise ValueError("agg must be one of: median/mean/min/max")

    xyz_agg = np.stack([x_rep, y_rep, z_rep], axis=1)
    return xyz_agg


def flatten_and_filter_nan_with_index(points_hw3: np.ndarray):
    """
    points_hw3: (H,W,3) with NaNs in invalid regions
    returns:
      xyz_valid: (P,3) finite points
      valid_idx: (P,) indices into the flattened (H*W,) array
      H, W: original shape
    """
    H, W, _ = points_hw3.shape
    xyz = points_hw3.reshape(-1, 3)  # (H*W,3)
    good = np.isfinite(xyz).all(axis=1)
    valid_idx = np.where(good)[0]
    xyz_valid = xyz[valid_idx]
    return xyz_valid, valid_idx, H, W
