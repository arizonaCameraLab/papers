import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import random
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.io as pio
import os

        
def generate_circle_tensor(size=256, radius=64):
    # Create coordinate grid
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    
    # Compute distance from center
    center = size // 2
    distance = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
    
    # Create mask for circle
    tensor = (distance <= radius).float()
    
    return tensor
def generate_quadrant_tensor(size=256):
    # Create coordinate grid
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    
    # Define the quadrants
    center = size // 2
    tensor = torch.zeros((size, size))
    
    # Set first and third quadrants to 1
    tensor[:center, :center] = 1  # First quadrant
    tensor[center:, center:] = 1  # Third quadrant
    
    return tensor

def rgb_to_gray_tensor(img_rgb: torch.Tensor) -> torch.Tensor:

    assert img_rgb.ndim in (3, 4), "Input must be 3D or 4D tensor"
    has_batch = (img_rgb.ndim == 4)

    if has_batch:
        B, C, H, W = img_rgb.shape
    else:
        C, H, W = img_rgb.shape


    assert (C == 3), "Input must have 3 channels (RGB)"


    if has_batch:
        R = img_rgb[:, 0:1, :, :]
        G = img_rgb[:, 1:2, :, :]
        B = img_rgb[:, 2:3, :, :]
    else:
        R = img_rgb[0:1]
        G = img_rgb[1:2]
        B = img_rgb[2:3]

    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray  

def apply_rect_mask(image_tensor, rect, invert=False, value_outside=0.0):



    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(1)  # [1,H,W] → [1,1,H,W]
    elif image_tensor.ndim == 2:
        raise ValueError(" [1,H,W] or [1,C,H,W]")

    B, C, H, W = image_tensor.shape
    x1, x2, y1, y2 = rect


    x1, x2 = int(np.clip(x1, 0, W)), int(np.clip(x2, 0, W))
    y1, y2 = int(np.clip(y1, 0, H)), int(np.clip(y2, 0, H))


    mask_np = np.zeros((H, W), dtype=np.float32)
    mask_np[y1:y2, x1:x2] = 1.0
    if invert:
        mask_np = 1.0 - mask_np

    mask = torch.tensor(mask_np, dtype=torch.float32, device=image_tensor.device)[None, None, :, :]
    mask = mask.expand(B, C, H, W)


    masked_img = image_tensor * mask + value_outside * (1 - mask)


    if image_tensor.ndim == 3:
        masked_img = masked_img.squeeze(1)

    return masked_img
def apply_polygon_mask(image_tensor, polygon_xy, invert=True, value_outside=0.0):



    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(1)  # [1,H,W] → [1,1,H,W]
    elif image_tensor.ndim == 2:
        raise ValueError(" [1,H,W] or [1,C,H,W]")

    B, C, H, W = image_tensor.shape

    mask_np = np.zeros((H, W), dtype=np.float32)
    polygon_np = np.array(polygon_xy, dtype=np.int32)

    cv2.fillPoly(mask_np, [polygon_np], color=1.0)

    if invert:
        mask_np = 1.0 - mask_np

    mask = torch.tensor(mask_np, dtype=torch.float32, device=image_tensor.device)[None, None, :, :]
    mask = mask.expand(B, C, H, W)

    masked_img = image_tensor * mask + value_outside * (1 -mask)

    if image_tensor.ndim == 3:
        masked_img = masked_img.squeeze(1)

    return masked_img

def draw_filled_box_with_circles(image_size=(256, 256), top_left=(15, 25), box_size=None, radius=None):
    """
    Create a 2D tensor with a filled box and two filled circles inside it.

    Parameters:
    - image_size (tuple): (H, W) size of the canvas
    - top_left (tuple): (y, x) top-left corner of the box
    - box_size (tuple): (box_height, box_width)
    - radius (int): radius of the filled circles

    Returns:
    - torch.Tensor: 2D tensor with 1s in the box and 2s in the circles
    """
    canvas = torch.zeros(image_size, dtype=torch.float32)
    H, W = image_size
    y, x = top_left
    if box_size == None :
        box_size = (int(H * 0.9), int(W * 0.9))
    box_h, box_w = box_size

    if radius == None:
        radius = int(H * 0.2)
    # Clamp box within image
    y1, y2 = max(y, 0), min(y + box_h, H)
    x1, x2 = max(x, 0), min(x + box_w, W)

    # Draw the filled box with value 1
    canvas[y1:y2, x1:x2] = 1.0

    # Create a meshgrid for coordinates
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    for _ in range(2):  # Two circles
        # Generate a random center within the box such that the entire circle fits
        cy = random.randint(y1 + radius, y2 - radius - 1)
        cx = random.randint(x1 + radius, x2 - radius - 1)

        # Create circular mask
        dist_sq = (yy - cy)**2 + (xx - cx)**2
        mask = dist_sq <= radius**2

        # Draw the circle with value 2
        canvas[mask] = 0.5

    return canvas


def crop_center(matrix, crop_height, crop_width):
    """
    Crops a 2D matrix from its center to the specified dimensions.

    Args:
        matrix (np.ndarray): The input 2D NumPy array (matrix).
        crop_height (int): The desired height of the cropped region.
        crop_width (int): The desired width of the cropped region.

    Returns:
        np.ndarray: The center-cropped matrix.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")

    matrix_height, matrix_width = matrix.shape

    if crop_height > matrix_height or crop_width > matrix_width:
        raise ValueError("Crop dimensions cannot be larger than the matrix dimensions.")

    # Calculate starting indices for the crop
    start_y = (matrix_height - crop_height) // 2
    start_x = (matrix_width - crop_width) // 2

    # Calculate ending indices for the crop
    end_y = start_y + crop_height
    end_x = start_x + crop_width

    # Perform the crop using NumPy slicing
    cropped_matrix = matrix[start_y:end_y, start_x:end_x]

    return cropped_matrix

def crop_to_size(img, size, center=None, pad_value=0):
    """
    Crop image to (out_h, out_w) with optional padding.

    Supports:
      - numpy.ndarray
      - torch.Tensor

    Shapes:
      - (H, W)
      - (H, W, C)

    Args:
        img: input image
        size: int or (out_h, out_w). If int -> (size, size)
        center: (cy, cx). If None, use image center.
        pad_value: value used for padding when crop exceeds boundary

    Returns:
        Cropped image of shape (out_h, out_w[, C])
    """
    # parse size
    if isinstance(size, int):
        out_h, out_w = size, size
    else:
        out_h, out_w = int(size[0]), int(size[1])

    h, w = img.shape[:2]

    # determine center
    if center is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = center

    # desired crop box in source coords
    top    = int(cy - out_h // 2)
    left   = int(cx - out_w // 2)
    bottom = top + out_h
    right  = left + out_w

    # intersection with source
    src_top    = max(0, top)
    src_left   = max(0, left)
    src_bottom = min(h, bottom)
    src_right  = min(w, right)

    # corresponding region in destination
    dst_top  = src_top - top
    dst_left = src_left - left
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right  = dst_left + (src_right - src_left)

    # allocate output (same type/device)
    is_torch = hasattr(img, "new_full")  # torch.Tensor has new_full
    if img.ndim == 2:
        out = img.new_full((out_h, out_w), pad_value) if is_torch else \
              __import__("numpy").full((out_h, out_w), pad_value, dtype=img.dtype)
        out[dst_top:dst_bottom, dst_left:dst_right] = img[src_top:src_bottom, src_left:src_right]
        return out

    elif img.ndim == 3:
        C = img.shape[2]
        out = img.new_full((out_h, out_w, C), pad_value) if is_torch else \
              __import__("numpy").full((out_h, out_w, C), pad_value, dtype=img.dtype)
        out[dst_top:dst_bottom, dst_left:dst_right, :] = img[src_top:src_bottom, src_left:src_right, :]
        return out

    else:
        raise ValueError(f"Unsupported shape: {img.shape}")



def pad_to_size(image, target_size, pad_dims=(0, 1), value=0):
    """
    Pad specified dimensions of a tensor or ndarray to a target size.

    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        Input image, e.g. shape (C, H, W) or (H, W)
    target_size : tuple(int, int)
        Target sizes corresponding to the pad_dims (e.g. (H, W))
    pad_dims : tuple(int, int)
        Dimensions (by index) to pad. Default (-2, -1) pads the last two dims.
    value : float
        Pad value. Default = 0.

    Returns
    -------
    padded : same type as input
    """
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise TypeError("Input must be torch.Tensor or np.ndarray")

    if len(target_size) != len(pad_dims):
        raise ValueError("target_size must match number of pad_dims")


    shape = list(image.shape)
    pad_slices = []

    for dim, tsize in zip(pad_dims, target_size):
        dim = dim if dim >= 0 else len(shape) + dim
        curr = shape[dim]
        diff = tsize - curr
        p_before = diff // 2
        p_after = diff - p_before
        pad_slices.append((dim, (p_before, p_after)))

    if isinstance(image, torch.Tensor):
        pad_list = []
        for dim in reversed(range(len(shape))):
            match = next((p for d, p in pad_slices if d == dim), None)
            if match:
                pad_list.extend(match[::-1])  # reverse (before, after)
            else:
                pad_list.extend((0, 0))
        return F.pad(image, pad_list, mode='constant', value=value)

    elif isinstance(image, np.ndarray):
        pad_config = []
        for i in range(len(shape)):
            match = next((p for d, p in pad_slices if d == i), None)
            pad_config.append(match if match else (0, 0))
        return np.pad(image, pad_config, mode='constant', constant_values=value)



def generate_test_data(n_pts, square_size=4, max_height=40,
                       pattern_source="checkboard", obj_source="sphere",
                       pattern_mode="binary", reflectance_mode="gray"):


    # -----------------------------
    def generate_spherical_cap(radius=100, max_height=100, num_points=256):
        x = np.linspace(-radius, radius, num_points)
        y = np.linspace(-radius, radius, num_points)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        mask = R <= radius
        Z = np.zeros_like(R, dtype=np.float32)
        Z[mask] = np.sqrt(radius**2 - R[mask]**2)
        offset = radius - max_height
        Z = Z - offset
        Z[Z < 0] = 0
        return X.astype(np.float32), Y.astype(np.float32), Z

    # -----------------------------
    def draw_reflectance_circles_random(image_size=(256, 256), n_circles=4, radius=None):

        H, W = image_size
        reflectance = torch.ones(3, H, W) * 0.2  
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')


        if radius is None:
            radius = int(min(H, W) * 0.12)
        else:
            radius = int(radius)

        min_dist = radius * 2.2  

        colors_all = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1)
        ]
        colors = random.sample(colors_all, n_circles)

        centers = []
        attempts = 0
        max_attempts = 500

        while len(centers) < n_circles and attempts < max_attempts:
            attempts += 1
            cy = random.randint(radius, H - radius - 1)
            cx = random.randint(radius, W - radius - 1)
            if all(np.hypot(cy - y, cx - x) > min_dist for (y, x) in centers):
                centers.append((cy, cx))

        if len(centers) < n_circles:
            print(f"⚠️ Warning: only placed {len(centers)} / {n_circles} circles (overlap constraint).")

        for (cy, cx), (r, g, b) in zip(centers, colors):
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            reflectance[0, mask] = r
            reflectance[1, mask] = g
            reflectance[2, mask] = b

        return reflectance


    # -----------------------------
    if obj_source == "slope":
        ramp = torch.linspace(0, max_height, n_pts).unsqueeze(0)
        height_map = ramp.repeat(n_pts, 1)
        X, Y = np.meshgrid(np.arange(n_pts), np.arange(n_pts))
    else:
        X, Y, height_map = generate_spherical_cap(
            radius=400, max_height=max_height, num_points=n_pts
        )
        height_map = torch.from_numpy(height_map)

    # -----------------------------
    if reflectance_mode == "gray":
        reflectance = torch.ones(n_pts, n_pts)[None,:,:]
    elif reflectance_mode == "rgb_random":
        reflectance = torch.rand(3, n_pts, n_pts) * 0.5 + 0.5
    elif reflectance_mode == "rgb_circles":
        reflectance = draw_reflectance_circles_random((n_pts, n_pts))
    else:
        raise ValueError("reflectance_mode must be 'gray', 'rgb_random', or 'rgb_circles'")

    # -----------------------------
    # Pattern
    # -----------------------------
    if pattern_mode == "binary":
        if pattern_source == "checkboard":
            ii = torch.arange(n_pts).unsqueeze(1)
            jj = torch.arange(n_pts).unsqueeze(0)
            pattern = (((ii // square_size) + (jj // square_size)) % 2 == 0).float()
        else:
            grid_dim_x = -(-n_pts // square_size)
            grid_dim_y = -(-n_pts // square_size)
            pattern = torch.zeros((n_pts, n_pts))
            for gy in range(grid_dim_y):
                for gx in range(grid_dim_x):
                    if random.random() > 0.5:
                        y0 = gy * square_size
                        x0 = gx * square_size
                        y1 = min(y0 + square_size, n_pts)
                        x1 = min(x0 + square_size, n_pts)
                        pattern[y0:y1, x0:x1] = 1.0
        pattern = pattern.unsqueeze(0)  # [1, H, W]
    else:
        grid_dim_x = -(-n_pts // square_size)
        grid_dim_y = -(-n_pts // square_size)
        pattern = torch.zeros((3, n_pts, n_pts), dtype=torch.float32)
        for gy in range(grid_dim_y):
            for gx in range(grid_dim_x):
                color = random.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
                y0 = gy * square_size
                x0 = gx * square_size
                y1 = min(y0 + square_size, n_pts)
                x1 = min(x0 + square_size, n_pts)
                pattern[0, y0:y1, x0:x1] = color[0]
                pattern[1, y0:y1, x0:x1] = color[1]
                pattern[2, y0:y1, x0:x1] = color[2]

    # -----------------------------
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title(f"Height map ({obj_source})")
    plt.imshow(height_map, aspect='equal')
    plt.colorbar(label='Height')

    plt.subplot(2, 2, 2)
    y_vals = Y[:, n_pts // 2]
    z_vals = height_map[:, n_pts // 2]
    plt.plot(y_vals, z_vals)
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.title("Cross-section")

    plt.subplot(2, 2, 3)
    if reflectance_mode.startswith("rgb"):
        plt.imshow(reflectance.permute(1, 2, 0).numpy())
    else:
        plt.imshow(reflectance.numpy().squeeze(), cmap='gray')
    plt.title(f'Reflectance ({reflectance_mode})')

    plt.subplot(2, 2, 4)
    if pattern_mode == "rgb":
        plt.imshow(pattern.permute(1, 2, 0).numpy())
    else:
        plt.imshow(pattern.squeeze().numpy(), cmap='gray')
    plt.title(f'Pattern ({pattern_mode})')

    plt.tight_layout()
    plt.show()

    return height_map, reflectance, pattern




def plot_3d_surface(height_map: torch.Tensor, n_pts: int, 
                    xy_range_mm: float = 120, 
                    title: str = '3D Surface Plot', 
                    renderer: str = 'notebook'):

    X = torch.linspace(-xy_range_mm, xy_range_mm, n_pts)
    Y = torch.linspace(-xy_range_mm, xy_range_mm, n_pts)
    X_np = X.numpy()
    Y_np = Y.numpy()
    
    Z_np = height_map.cpu().numpy()
    
    M = np.max([np.abs(X_np).max(), np.abs(Y_np).max(), np.abs(Z_np).max()])
    
    fig = go.Figure(data=[go.Surface(z=Z_np, x=X_np, y=Y_np)])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-M, M], title='X (mm)'),
            yaxis=dict(range=[-M, M], title='Y (mm)'),
            zaxis=dict(range=[-M, M], title='Height'),
            aspectmode='cube', 
        ),
    )
    
    pio.renderers.default = renderer
    fig.show()
def save_tensor_to_png(tensor: torch.Tensor, filename: str):
    """
    Save a 512x512 torch tensor to a PNG file using OpenCV.
    Assumes tensor values are in [0,1] or [0,255].
    
    Args:
        tensor (torch.Tensor): 2D (512x512) or 3D (1x512x512 / 3x512x512) tensor.
        filename (str): Output file path, must end with .png.
    """
    # Ensure tensor is on CPU and detached
    arr = tensor.detach().cpu().numpy()
    
    # If tensor has channel dimension, move to HWC
    if arr.ndim == 3:
        if arr.shape[0] == 1:  # (1, H, W) → (H, W)
            arr = arr[0]
        elif arr.shape[0] == 3:  # (3, H, W) → (H, W, 3)
            arr = np.transpose(arr, (1, 2, 0))
    
    # Normalize to 0–255 uint8 if not already
    if arr.dtype != np.uint8:
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255).astype(np.uint8)
    
    # Save using cv2 (note: cv2 expects BGR for color images)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(filename, arr)    


def crop_image(image, center_x=None, center_y=None, width=None, height=None):
    """
    Crop an image with the given center. If crop region goes outside image,
    pad with black (0) to keep requested size.

    Args:
        image: input image as numpy array (H, W, C) or (H, W).
        center_x, center_y: crop center (default: image center).
        width, height: crop size (default: 100x100, if only one -> square).

    Returns:
        Cropped image (numpy array).
    """
    h_img, w_img = image.shape[:2]

    # Default size
    if width is None and height is None:
        width = height = 100
    elif width is None:
        width = height
    elif height is None:
        height = width

    # Default center
    if center_x is None:
        center_x = w_img // 2
    if center_y is None:
        center_y = h_img // 2

    # Desired crop coordinates
    x1 = int(center_x - width  // 2)
    y1 = int(center_y - height // 2)
    x2 = x1 + width
    y2 = y1 + height

    # Compute overlap with image
    x1_img = max(0, x1)
    y1_img = max(0, y1)
    x2_img = min(w_img, x2)
    y2_img = min(h_img, y2)

    # Extract region inside image
    crop = image[y1_img:y2_img, x1_img:x2_img]

    # Create black canvas of requested size
    if image.ndim == 3:
        result = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    else:
        result = np.zeros((height, width), dtype=image.dtype)

    # Compute placement on canvas
    y_offset = max(0, -y1)
    x_offset = max(0, -x1)

    result[y_offset:y_offset+crop.shape[0], x_offset:x_offset+crop.shape[1]] = crop

    return result


def parse_intrinsics(cam_K, proj_K, cam_res, proj_res):


    cam_w, cam_h = cam_res
    proj_w, proj_h = proj_res


    cam_fx, cam_fy = cam_K[0, 0], cam_K[1, 1]
    cam_cx, cam_cy = cam_K[0, 2], cam_K[1, 2]
    cam_offset = (cam_cx - cam_w / 2, cam_cy - cam_h / 2)

    proj_fx, proj_fy = proj_K[0, 0], proj_K[1, 1]
    proj_cx, proj_cy = proj_K[0, 2], proj_K[1, 2]
    proj_offset = (proj_cx - proj_w / 2, proj_cy - proj_h / 2)

    return {
        "projector": {
            "focal": (proj_fx, proj_fy),
            "offset": proj_offset
        },
        "camera": {
            "focal": (cam_fx, cam_fy),
            "offset": cam_offset
        }

    }


def convolve_with_gaussian_psf_torch(image: torch.Tensor, fwhm: float = 1.0) -> torch.Tensor:
    """
    Convolve an image with a Gaussian PSF, where fwhm is the full-width at half-maximum (FWHM).
    
    Args:
        image (torch.Tensor): Input tensor of shape [N, C, H, W].
        fwhm (float): Gaussian FWHM in pixels. 
                      If fwhm=1, the kernel is extremely narrow (delta-like).
    
    Returns:
        torch.Tensor: Convolved image of the same shape.
    """
    # Convert FWHM to sigma
    sigma = fwhm / 2.355

    # Kernel size ~ 6*sigma, minimum 1, force odd
    ksize = max(1, int(6 * sigma + 1))
    if ksize % 2 == 0:
        ksize += 1

    # Gaussian kernel grid
    coords = torch.arange(ksize, dtype=torch.float32, device=image.device) - ksize // 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2 + 1e-12))  # add epsilon for stability
    kernel = kernel / kernel.sum()

    # Reshape for depthwise convolution
    C = image.shape[1]
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)

    # Apply convolution
    padding = ksize // 2
    out = F.conv2d(image, kernel, padding=padding, groups=C)

    return out

def COP_to_extrinsics(COP, target=None, up=None):

    if not isinstance(COP, torch.Tensor):
        COP = torch.tensor(COP, dtype=torch.float32)
    device, dtype = COP.device, COP.dtype

    if target is None:
        target = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
    elif not isinstance(target, torch.Tensor):
        target = torch.tensor(target, device=device, dtype=dtype)

    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    elif not isinstance(up, torch.Tensor):
        up = torch.tensor(up, device=device, dtype=dtype)

    # 1) forward: COP -> target
    forward = (target - COP)
    forward = forward / (forward.norm() + 1e-12)

    # 2) right = forward × up   
    right = torch.linalg.cross(forward, up)
    right = right / (right.norm() + 1e-12)

    # 3) up_vec = right × forward
    up_vec = torch.linalg.cross(right, forward)
    up_vec = up_vec / (up_vec.norm() + 1e-12)

    # 4) R_c2w
    R_c2w = torch.stack([right, up_vec, forward], dim=1)

    if torch.det(R_c2w) < 0:
        right = -right
        R_c2w = torch.stack([right, up_vec, forward], dim=1)

    # 5) world->camera
    R_w2c = R_c2w.t()
    t = -R_w2c @ COP.view(3, 1)
    return R_w2c, t

def crop_by_coords(image, simu_img_area):
    """
    Crop an image (NumPy or PyTorch) based on pixel coordinate area.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Input image. Can have any shape; singleton dimensions are squeezed first.
    simu_img_area : array-like, shape (2, 2)
        Coordinates in MATLAB style: [[x_start, x_end], [y_start, y_end]]

    Returns
    -------
    cropped : np.ndarray or torch.Tensor
        Cropped 2D image.
    """
    # squeeze first
    if isinstance(image, torch.Tensor):
        img_squeezed = np.array(image.squeeze().cpu())
    else:
        img_squeezed = np.squeeze(image)

    # unpack coordinates
    x_start, x_end = int(simu_img_area[0][0]), int(simu_img_area[0][1])
    y_start, y_end = int(simu_img_area[1][0]), int(simu_img_area[1][1])

    # crop
    if isinstance(img_squeezed, torch.Tensor):
        cropped = img_squeezed[y_start:y_end, x_start:x_end]
    else:
        cropped = img_squeezed[y_start:y_end, x_start:x_end]

    return cropped


def extrinsics_to_COP(R, t_):
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=torch.float32)
    if not isinstance(t_, torch.Tensor):
        t_ = torch.tensor(t_, dtype=torch.float32)

    COP = -R.t() @ t_
    return COP.view(-1)  


