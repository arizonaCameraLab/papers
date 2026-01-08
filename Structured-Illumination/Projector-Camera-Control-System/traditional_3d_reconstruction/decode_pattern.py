import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm


def decode_graycode(folder, proj_w=768, proj_h=768, black_thr=75, white_thr=10):
    """
    Decode Gray Code patterns to get projector pixel coordinates for each camera pixel.
    reference: https://docs.opencv.org/4.12.0/dc/da9/tutorial_decode_graycode_pattern.html
    Args:
        folder: contains 00_00.png, 01_00.png, ...
        proj_w, proj_h: projector resolution
        black_thr: black threshold
        white_thr: white threshold
    Returns:
        projector_coords: (H, W, 2) float array storing projector pixel coords (u, v) for each camera pixel
        mask_valid: (H, W) binary array, 255 for valid pixels, 0 for invalid pixels
    """

    folder = Path(folder)

    # -------------------------------
    # read white / black
    # -------------------------------
    white_img = cv2.imread(str(folder / "00_00.png"), cv2.IMREAD_GRAYSCALE)
    black_img = cv2.imread(str(folder / "01_00.png"), cv2.IMREAD_GRAYSCALE)

    if white_img is None:
        raise FileNotFoundError("white not found!")
    if black_img is None:
        raise FileNotFoundError("black not found!")

    H, W = white_img.shape
    print(f"Camera image size: {W} x {H}")
    print(f"Projector panel size: {proj_w} x {proj_h}")

    # -------------------------------
    # create GrayCodePattern
    # -------------------------------
    gray = cv2.structured_light.GrayCodePattern.create(proj_w, proj_h)
    gray.setBlackThreshold(black_thr)  # set black threshold
    gray.setWhiteThreshold(white_thr)  # set white threshold

    # amount must be larger than num_patterns
    num_patterns = gray.getNumberOfPatternImages()
    print(f"GrayCodePattern expects: {num_patterns} pattern images")

    # -------------------------------
    # Read all Gray Code pattern images
    # -------------------------------
    pattern_imgs = []  # list of pattern images
    for i in range(num_patterns):  # read pattern images
        filename = f"{i+2:02d}_00.png"  # 02_00.png .. 41_00.png
        img = cv2.imread(str(folder / filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(filename)
        if img.shape != (H, W):  # shape check
            raise ValueError("Image size mismatch")
        pattern_imgs.append(img)

    # -------------------------------
    # Call getProjPixel for each pixel
    # -------------------------------
    projector_coords = np.zeros(
        (H, W, 2), dtype=np.float32
    )  # (u,v) for each camera pixel
    projector_coords = np.full((H, W, 2), np.nan, np.float32)
    mask_valid = np.zeros((H, W), dtype=np.bool)  # valid mask

    # total = H * W
    # cnt = 0

    print("Start decoding each pixel...")

    for y in tqdm(range(H)):  # for each row
        for x in range(W):  # for each column

            if int(white_img[y, x]) - int(black_img[y, x]) <= black_thr:
                continue  # not a valid pixel, skip this position

            error, uv = gray.getProjPixel(pattern_imgs, x, y)

            projector_coords[y, x, 0] = uv[0]  # camera pixel coord u
            projector_coords[y, x, 1] = uv[1]  # camera pixel coord v

            if not error:  # valid pixel
                mask_valid[y, x] = True  # valid pixel

            # cnt += 1
            # if cnt % 200000 == 0:
            #     print(f"Progress: {cnt}/{total}")

    print("Decoding finished.")

    return projector_coords, mask_valid


def decode_graycode_refined(
    folder,
    proj_w=768,
    proj_h=768,
    black_thr=40,
    white_thr=5,
    patch_half=3,  # patch size = 2*patch_half + 1
    min_valid_pts=10,  # minimum pts for homography
):
    """
    Decode Gray Code patterns with local homography refinement.

    reference: D. Moreno and G. Taubin, "Simple, Accurate, and Robust Projector-Camera Calibration,"
    2012 Second International Conference on 3D Imaging, Modeling, Processing, Visualization & Transmission,
    Zurich, Switzerland, 2012, pp. 464-471, doi: 10.1109/3DIMPVT.2012.77.

    Args:
        folder: folder containing pattern images
        proj_w, proj_h: projector resolution
        black_thr: black threshold
        white_thr: white threshold
        patch_half: half size of local patch for homography
        min_valid_pts: minimum valid points in patch to compute homography

    Returns:
        proj_refined: (H, W, 2) float array storing refined projector pixel
    """
    folder = Path(folder)

    # -------------------------------
    # read white / black
    # -------------------------------
    white_img = cv2.imread(str(folder / "00_00.png"), 0)
    black_img = cv2.imread(str(folder / "01_00.png"), 0)

    H, W = white_img.shape
    print(f"Camera image size: {W} x {H}")
    print(f"Projector panel size: {proj_w} x {proj_h}")

    # -------------------------------
    # create GrayCodePattern
    # -------------------------------
    gray = cv2.structured_light.GrayCodePattern.create(proj_w, proj_h)
    gray.setBlackThreshold(black_thr)
    gray.setWhiteThreshold(white_thr)

    num_patterns = gray.getNumberOfPatternImages()
    print("Gray code patterns =", num_patterns)

    # -------------------------------
    # read pattern images
    # -------------------------------
    pattern_imgs = []
    for i in range(num_patterns):
        fname = f"{i+2:02d}_00.png"
        img = cv2.imread(str(folder / fname), 0)
        if img is None:
            raise FileNotFoundError(fname)
        pattern_imgs.append(img)

    # -------------------------------
    # first pass: raw decode
    # -------------------------------
    proj_raw = np.full((H, W, 2), np.nan, np.float32)
    valid_raw = np.zeros((H, W), np.bool)

    print("Decoding raw Gray code...")
    for y in tqdm(range(H)):
        for x in range(W):

            if int(white_img[y, x]) - int(black_img[y, x]) <= black_thr:
                continue  # not a valid pixel, skip this position

            err, uv = gray.getProjPixel(pattern_imgs, x, y)
            proj_raw[y, x] = uv
            if not err:
                valid_raw[y, x] = True
    print("Raw decoding done.")

    # -------------------------------
    # second pass: refine by homography
    # -------------------------------
    print("Refining projector coordinates with local homography...")

    proj_refined = proj_raw.copy()

    for y in tqdm(range(H)):
        for x in range(W):
            if not valid_raw[y, x]:
                continue

            src_pts = []
            dst_pts = []

            # gather patch
            for dy in range(-patch_half, patch_half + 1):
                for dx in range(-patch_half, patch_half + 1):
                    xx = x + dx
                    yy = y + dy
                    if xx < 0 or xx >= W or yy < 0 or yy >= H:
                        continue
                    u, v = proj_raw[yy, xx]
                    if not np.isfinite(u + v):
                        continue

                    src_pts.append([xx, yy])
                    dst_pts.append([u, v])

            if len(src_pts) < min_valid_pts:
                # cannot apply homography; keep raw decode
                print(
                    f"Pixel ({x}, {y}) is skipped: not enough valid points for homography."
                )
                continue

            src_pts = np.float32(src_pts)
            dst_pts = np.float32(dst_pts)

            Hmat, _ = cv2.findHomography(src_pts, dst_pts, method=0)

            if Hmat is None:
                print(f"Pixel ({x}, {y}) is skipped: homography computation failed.")
                continue

            pt = Hmat @ np.array([x, y, 1.0]).T
            u2, v2 = pt[0] / pt[2], pt[1] / pt[2]

            proj_refined[y, x] = (u2, v2)

    # -------------------------------
    # output
    # -------------------------------
    print("Refinement done.")
    return proj_refined, valid_raw
