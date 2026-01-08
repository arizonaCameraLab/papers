# pcs_setup_new_utils_ver5.py
# Compatible Blender 5.X
from __future__ import annotations

import os, sys, math
from dataclasses import dataclass
from typing import Dict, Tuple

import bpy
import numpy as np
import mathutils


def force_all_image_textures_nearest_interpolation(node_tree):
    """Force all image texture nodes in a node_tree to use 'Closest' interpolation."""
    try:
        nodes = getattr(node_tree, "nodes", [])
        for n in nodes:
            # ShaderNodeTexImage is the image texture node type
            if getattr(n, "bl_idname", "") == "ShaderNodeTexImage":
                if hasattr(n, "interpolation"):
                    n.interpolation = "Closest"
    except Exception as _e:
        print("[warn] Could not force image texture interpolation:", _e)


# ============================
# Data containers
# ============================


@dataclass
class CameraParams:
    pos_w_m: np.ndarray  # (3,), camera position [m]
    R_world_to_cam: np.ndarray  # (3,3), world → camera
    intrinsics: np.ndarray  # (3,3), OpenCV K
    pixel_pitch_mm: float  # [mm]
    res_x: int
    res_y: int


@dataclass
class ProjectorParams:
    pos_w_m: np.ndarray  # (3,), projector position [m]
    R_world_to_proj: np.ndarray  # (3,3), world → projector
    intrinsics: np.ndarray  # (3,3), OpenCV K
    pixel_pitch_mm: float  # [mm]
    panel_res_x: int
    panel_res_y: int
    power_w: float = 100.0


@dataclass
class ObjectParams:
    width_m: float  # plane width used for X/Y scale (square)
    max_height_m: float  # displacement strength
    subdivisions: int  # base mesh subdivisions (we still use Subsurf)


@dataclass
class IOParams:
    script_dir: str
    height_map_filename: str
    pattern_filename: str
    render_output_filename: str = "render_output.tiff"

    @property
    def height_map_path(self) -> str:
        return os.path.join(self.script_dir, self.height_map_filename)

    @property
    def pattern_path(self) -> str:
        return os.path.join(self.script_dir, self.pattern_filename)

    @property
    def render_output_path(self) -> str:
        return os.path.join(self.script_dir, self.render_output_filename)


# ============================
# Utility helpers
# ============================


def opencv_extrinsics_to_blender_matrix(
    R_wc: np.ndarray,
    cam_pos_world_m: np.ndarray,
    apply_cv2_to_blender_axes: bool = True,
) -> mathutils.Matrix:
    """Build a Blender `matrix_world` from OpenCV extrinsics.

    OpenCV: camera looks toward +Z; Blender camera looks toward −Z.
    We optionally apply diag([1, −1, −1]) to align conventions.
    """
    R_cw = R_wc.T
    if apply_cv2_to_blender_axes:
        R_cv2blender = np.diag([1.0, -1.0, -1.0])
        R_blender = R_cw @ R_cv2blender
    else:
        R_blender = R_cw

    M = np.eye(4)
    M[:3, :3] = R_blender
    M[:3, 3] = cam_pos_world_m
    return mathutils.Matrix(M)


def load_image_file(
    path: str,
    colorspace: str = "Non-Color",
    use_view_as_render: bool = False,
    seam_margin: int = 5,
) -> bpy.types.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = bpy.data.images.load(path)
    image.colorspace_settings.name = colorspace
    image.use_view_as_render = use_view_as_render
    image.seam_margin = seam_margin
    return image


def purge_orphans() -> None:
    bpy.ops.outliner.orphans_purge(do_recursive=True)


def clear_scene() -> None:
    """Delete all objects/collections and purge orphans."""
    for obj in bpy.data.objects:
        obj.hide_set(False)
        obj.hide_select = False
        obj.hide_viewport = False

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)

    for datablock in (
        bpy.data.meshes,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.materials,
        bpy.data.images,
    ):
        for block in list(datablock):
            if block.users == 0:
                datablock.remove(block)

    purge_orphans()


def setup_render(
    scene: bpy.types.Scene, res_x: int, res_y: int, output_path: str
) -> None:
    scene.render.engine = "CYCLES"

    # Cycles device (safe access)
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        scene.cycles.device = (
            "GPU" if getattr(prefs, "compute_device_type", "NONE") != "NONE" else "CPU"
        )
    except Exception:
        scene.cycles.device = "CPU"
        
    # print(dir(scene.cycles))

    # scene.cycles.feature_set = "EXPERIMENTAL"
    # scene.cycles.use_adaptive_subdivision = True

    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "TIFF"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.tiff_codec = "NONE"
    scene.render.filepath = output_path

    scene.frame_start = 1
    scene.frame_end = 1

    # Pure black background via nodes (defensive checks)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    nt = scene.world.node_tree
    force_all_image_textures_nearest_interpolation(nt)
    if nt and "Background" in nt.nodes:
        nt.nodes["Background"].inputs[1].default_value = 0.0
    # Match original script: sample world as light (harmless when background is black)
    try:
        if scene.world and hasattr(scene.world, "cycles"):
            scene.world.cycles.sample_as_light = True
    except Exception:
        pass


def set_blender_camera_from_opencv(
    camera_obj: bpy.types.Object,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    pixel_pitch_mm: float,
) -> None:
    """Apply OpenCV intrinsics to Blender camera or Projector-like object."""
    name = camera_obj.name
    if name == "Camera":
        data = camera_obj.data
        data.sensor_width = pixel_pitch_mm * img_w
        data.sensor_height = pixel_pitch_mm * img_h
        data.sensor_fit = "HORIZONTAL"
        data.lens_unit = "MILLIMETERS"
        data.lens = (fx) * pixel_pitch_mm
        print(data.lens)
        # Match original behavior: use rounded pixel offsets for principal point
        shift_x_pix = float((cx - 0.5 * img_w))
        shift_y_pix = float((cy - 0.5 * img_h))
        data.shift_x = -shift_x_pix / img_w
        data.shift_y = shift_y_pix / img_h
    elif name == "Projector":
        proj_lens_mm = (fx) * pixel_pitch_mm
        print(proj_lens_mm)
        panel_width_mm = img_w * pixel_pitch_mm
        camera_obj.data.sensor_fit = "HORIZONTAL"
        try:
            camera_obj.proj_settings.throw_ratio = proj_lens_mm / panel_width_mm
            # Match original behavior: use rounded pixel offsets, scaled to percent
            h_shift_pix = float((cx - 0.5 * img_w))
            v_shift_pix = float((cy - 0.5 * img_h))
            camera_obj.proj_settings.h_shift = -h_shift_pix / img_w * 100.0
            camera_obj.proj_settings.v_shift = v_shift_pix / img_h * 100.0
        except AttributeError:
            raise RuntimeError(
                "Projector add‑on 'proj_settings' not found on object; check the add‑on is enabled."
            )


def create_displaced_plane(
    obj_params: ObjectParams, height_image: bpy.types.Image
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    plane = bpy.context.active_object

    # Square plane scaled by width
    plane.scale = (obj_params.width_m, obj_params.width_m, 1.0)

    # Edit ops
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.faces_shade_smooth()

    subsurf = plane.modifiers.new(name="Subsurf", type="SUBSURF")
    subsurf.levels = 8
    subsurf.render_levels = 8
    subsurf.subdivision_type = "SIMPLE"
    subsurf.use_adaptive_subdivision = True
    
    # print(dir(subsurf))
    # bpy.context.scene.cycles.use_adaptive_subdivision = True

    bpy.ops.uv.unwrap(method="ANGLE_BASED")
    bpy.ops.object.mode_set(mode="OBJECT")

    # Displace via legacy texture (keeps parity with your pipeline)
    tex = bpy.data.textures.new("HeightMapTex", type="IMAGE")
    tex.image = height_image

    disp = plane.modifiers.new("Displace", type="DISPLACE")
    disp.texture = tex
    disp.texture_coords = "UV"
    disp.strength = obj_params.max_height_m
    disp.mid_level = 0.0

    # Apply (bake) modifiers
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.select_all(action="DESELECT")
    plane.select_set(True)

    for mod in list(plane.modifiers):
        if mod.type in {"SUBSURF", "DISPLACE"}:
            bpy.ops.object.modifier_apply(modifier=mod.name)

    # Simple white Lambert
    mat = create_lambert_material()
    plane.data.materials.clear()
    plane.data.materials.append(mat)

    # Visibility toggles for speed (Blender 4.x properties)
    for attr in (
        "visible_diffuse",
        "visible_glossy",
        "visible_transmission",
        "visible_volume_scatter",
    ):
        if hasattr(plane, attr):
            setattr(plane, attr, False)

    return plane


def create_lambert_material(name: str = "LambertianWhite") -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (1, 1, 1, 1)
    out = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(diffuse.outputs["BSDF"], out.inputs["Surface"])
    return mat


def create_projector(
    proj: ProjectorParams, pattern_image: bpy.types.Image
) -> bpy.types.Object:
    # Create and reset
    bpy.ops.projector.create()
    projector = bpy.context.object
    projector.name = "Projector"
    projector.location = (0.0, 0.0, 0.0)
    projector.rotation_euler = (0.0, 0.0, 0.0)
    projector.scale = (1.0, 1.0, 1.0)

    # Pose from OpenCV extrinsics
    projector.matrix_world = opencv_extrinsics_to_blender_matrix(
        proj.R_world_to_proj, proj.pos_w_m
    )

    # Intrinsics
    fx, fy = proj.intrinsics[0, 0], proj.intrinsics[1, 1]
    cx, cy = proj.intrinsics[0, 2], proj.intrinsics[1, 2]
    set_blender_camera_from_opencv(
        projector,
        fx,
        fy,
        cx,
        cy,
        proj.panel_res_x,
        proj.panel_res_y,
        proj.pixel_pitch_mm,
    )

    # Assign texture to the projector's light node
    light_obj = bpy.data.objects.get("Projector.Spot")
    if not light_obj or light_obj.type != "LIGHT":
        raise RuntimeError("Projector light 'Projector.Spot' not found or not a LIGHT")

    # Find first image texture node and assign
    nt = light_obj.data.node_tree
    force_all_image_textures_nearest_interpolation(nt)
    if nt:
        for node in nt.nodes:
            if getattr(node, "type", "") == "TEX_IMAGE":
                node.image = pattern_image
                break
        # Ensure nearest neighbor sampling like the original
        try:
            nt.nodes["Image Texture"].interpolation = "Closest"
        except KeyError:
            pass
    # Fallback: also set on the default 'Spot' light datablock if present
    try:
        spot_data = bpy.data.lights.get("Spot")
        if (
            spot_data
            and spot_data.node_tree
            and "Image Texture" in spot_data.node_tree.nodes
        ):
            spot_data.node_tree.nodes["Image Texture"].interpolation = "Closest"
    except Exception:
        pass

    # Power & texture
    try:
        projector.proj_settings.power = float(proj.power_w)
        projector.proj_settings.projected_texture = "custom_texture"
        projector.proj_settings.use_custom_texture_res = True
        projector.proj_settings.show_pixel_grid = False
    except AttributeError:
        raise RuntimeError(
            "Projector add‑on 'proj_settings' not found on object; check the add‑on is enabled."
        )

    # Some useful viewport toggles
    if hasattr(projector.data, "show_limits"):
        projector.data.show_limits = True
    if hasattr(projector.data, "display_size"):
        projector.data.display_size = 0.1

    return projector


def create_camera(cam: CameraParams) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.active_object

    camera.matrix_world = opencv_extrinsics_to_blender_matrix(
        cam.R_world_to_cam, cam.pos_w_m
    )

    fx, fy = cam.intrinsics[0, 0], cam.intrinsics[1, 1]
    cx, cy = cam.intrinsics[0, 2], cam.intrinsics[1, 2]
    set_blender_camera_from_opencv(
        camera, fx, fy, cx, cy, cam.res_x, cam.res_y, cam.pixel_pitch_mm
    )

    if hasattr(camera.data, "display_size"):
        camera.data.display_size = 0.1
    if hasattr(camera.data, "show_name"):
        camera.data.show_name = False
    if hasattr(camera.data, "show_limits"):
        camera.data.show_limits = True

    bpy.context.scene.camera = camera
    return camera
