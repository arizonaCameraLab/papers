# pcs_setup_new_main_ver5.py
# Compatible with Blender 5.x

import os, sys, math, json, importlib
import bpy
import numpy as np

module_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
print(module_dir)
if module_dir and module_dir not in sys.path:
    sys.path.append(module_dir)

import pcs_setup_new_utils_ver5
importlib.reload(pcs_setup_new_utils_ver5)
from pcs_setup_new_utils_ver5 import *  # "from ... import ..." cannot import the symbol name starting with "_"

# ============================
# Main
# ============================


def main() -> None:
    # 1) Clear scene
    clear_scene()

    # 2) Define parameters (kept consistent with your previous script)
    json_path = os.path.join(module_dir, "11072025_final_calibration_summary.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Calibration JSON not found: {json_path}\n" f"module_dir = {module_dir}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        calib_data = json.load(f)

    IMG_W, IMG_H = 928, 928
    PIX_PITCH_MM = 0.00185 * 3  # 1.85 µm * 3
    camera_intrinsics = np.array(calib_data["camera_intrinsic"], dtype=np.float64)
    R_wc = np.array(calib_data["R_world_to_camera"], dtype=np.float64)
    cam_pos_w_m = np.array(calib_data["camera_position_world"], dtype=np.float64) * 1e-3
    camp = CameraParams(
        pos_w_m=cam_pos_w_m,
        R_world_to_cam=R_wc,
        intrinsics=camera_intrinsics,
        pixel_pitch_mm=PIX_PITCH_MM,
        res_x=IMG_W,
        res_y=IMG_H,
    )

    PROJ_W, PROJ_H = 512, 512
    PROJ_PITCH_MM = PIX_PITCH_MM
    projector_intrinsics = np.array(calib_data["projector_intrinsic"], dtype=np.float64)
    R_wp = np.array(calib_data["R_world_to_projector"], dtype=np.float64)
    proj_pos_w_m = (
        np.array(calib_data["projector_position_world"], dtype=np.float64) * 1e-3
    )
    projp = ProjectorParams(
        pos_w_m=proj_pos_w_m,
        R_world_to_proj=R_wp,
        intrinsics=projector_intrinsics,
        pixel_pitch_mm=PROJ_PITCH_MM,
        panel_res_x=PROJ_W,
        panel_res_y=PROJ_H,
        power_w=50,
    )

    io = IOParams(
        script_dir=(
            os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        ),
        height_map_filename="heightmap0.png",
        pattern_filename="pattern_16.png",
        render_output_filename = 'render_output.tiff',
    )

    objp = ObjectParams(width_m=1.0, max_height_m=0.08, subdivisions=64)

    # 3) Load images
    pattern_image = load_image_file(
        io.pattern_path, colorspace="sRGB", use_view_as_render=False
    )
    height_image = load_image_file(io.height_map_path, colorspace="Non-Color")

    # 4) Geometry & materials
    plane = create_displaced_plane(objp, height_image)

    # 5) Projector
    projector = create_projector(projp, pattern_image)

    # 6) Camera
    camera = create_camera(camp)

    # 7) Render settings
    setup_render(
        bpy.context.scene,
        res_x=camp.res_x,
        res_y=camp.res_y,
        output_path=io.render_output_path,
    )

    # 8) execute render and save output
    scene = bpy.context.scene
    out_dir = os.path.dirname(scene.render.filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print(f"[Render] Image saved to: {scene.render.filepath}")


if __name__ == "__main__":
    main()
    # Match original script behavior: save the .blend automatically
    if bpy.data.filepath:
        bpy.ops.wm.save_mainfile()
