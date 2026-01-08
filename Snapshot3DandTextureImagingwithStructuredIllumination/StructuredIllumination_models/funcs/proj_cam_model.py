import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import cv2 as cv
import random
import numpy as np
from .DIP_helper_fun import *
from scipy.ndimage import zoom

import plotly.graph_objects as go
import plotly.io as pio

class Proj_Cam_model:
    # Inner custom autograd function for soft rounding (straight-through estimator)
    class SoftRound(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return torch.round(input)
        @staticmethod
        def backward(ctx, grad_output):
            # Pass gradient through unchanged
            return grad_output

    @staticmethod
    def soft_round(x):
        return Proj_Cam_model.SoftRound.apply(x)
    
    # Inner custom autograd function for hard occlusion with STE (used也用于shadow判断)
    class HardOcclusion(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dot_prod, steepness):

            ctx.steepness = steepness
            ctx.save_for_backward(dot_prod)
            mask = (dot_prod > 0).float()
            return mask
        @staticmethod
        def backward(ctx, grad_output):

            dot_prod, = ctx.saved_tensors
            steepness = ctx.steepness
            s = torch.sigmoid(steepness * dot_prod)
            grad_dot = grad_output * steepness * s * (1 - s)
            return grad_dot, None
    class SoftOcclusion(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, steepness):
            ctx.steepness = steepness
            ctx.save_for_backward(input)
            return torch.sigmoid(steepness * input)
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            steepness = ctx.steepness
            s = torch.sigmoid(steepness * input)
            grad_input = grad_output * steepness * s * (1 - s)
            return grad_input, None
    def __init__(self, proj_f, cam_f, 
                 proj_sensor_w_px=1920, proj_sensor_h_px=1080, 
                 cam_sensor_px=3036, device='cpu',
                 proj_COP=[0.0, 0.0, 2000.0], cam_COP=[20.0, 0.0, 2000.0],
                 proj_R=None, proj_t=None, cam_R=None, cam_t=None,
                 proj_cx_px=0.0, proj_cy_px=0.0, 
                 cam_cx_px=0.0, cam_cy_px=0.0, 
                 cam_res=(512, 512)):   


        if isinstance(proj_f, (list, tuple)):
            self.proj_fx, self.proj_fy = proj_f
        else:
            self.proj_fx = self.proj_fy = proj_f

        if isinstance(cam_f, (list, tuple)):
            self.cam_fx, self.cam_fy = cam_f
        else:
            self.cam_fx = self.cam_fy = cam_f

        self.proj_sensor_w_px = proj_sensor_w_px
        self.proj_sensor_h_px = proj_sensor_h_px
        self.cam_sensor_px = cam_sensor_px
        self.device = device

        self.proj_COP = torch.as_tensor(proj_COP, dtype=torch.float32, device=device)
        self.cam_COP  = torch.as_tensor(cam_COP,  dtype=torch.float32, device=device)

        # principle point shift 
        self.proj_cx_px = -proj_cx_px
        self.proj_cy_px = -proj_cy_px
        self.cam_cx_px  = -cam_cx_px
        self.cam_cy_px  = -cam_cy_px
        if cam_res is None:
            self.cam_res_h, self.cam_res_w = None, None  
        else:
            self.cam_res_h = int(cam_res[0])
            self.cam_res_w = int(cam_res[1])

        # Helper: safely convert to tensor
        def _to_tensor(x, device, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.clone().detach().to(device=device, dtype=dtype)
            else:
                return torch.as_tensor(x, device=device, dtype=dtype)

        # Extrinsics (optional)
        self.use_proj_extrinsics = proj_R is not None and proj_t is not None
        self.use_cam_extrinsics  = cam_R is not None and cam_t is not None

        if self.use_proj_extrinsics:
            self.proj_R = _to_tensor(proj_R, device)
            self.proj_t = _to_tensor(proj_t, device).view(3,1)
            self.proj_COP = -self.proj_R.t() @ self.proj_t  # overwrite COP
            self.proj_COP = self.proj_COP.squeeze()

        if self.use_cam_extrinsics:
            self.cam_R = _to_tensor(cam_R, device)
            self.cam_t = _to_tensor(cam_t, device).view(3,1)
            self.cam_COP = -self.cam_R.t() @ self.cam_t
            self.cam_COP = self.cam_COP.squeeze()

        # To store the last inputs and outputs for visualization
        self.last_height = None
        self.last_reflectance = None
        self.last_observed_image = None
        self.last_pattern = None


    def compute_xy_range_mm(self):

        device = self.proj_COP.device
        dtype  = self.proj_COP.dtype

        proj_w = torch.as_tensor(self.proj_sensor_w_px, device=device, dtype=dtype)
        proj_h = torch.as_tensor(self.proj_sensor_h_px, device=device, dtype=dtype)

        fx = self.proj_fx
        fy = self.proj_fy
        theta_x = torch.atan((proj_w / 2) / fx)
        theta_y = torch.atan((proj_h / 2) / fy)


        # COP angles
        phi_x = torch.atan(self.proj_COP[0] / self.proj_COP[2])
        phi_y = torch.atan(self.proj_COP[1] / self.proj_COP[2])

        bb1_x = self.proj_COP[0] - self.proj_COP[2] * torch.tan(phi_x + theta_x)
        bb2_x = self.proj_COP[0] - self.proj_COP[2] * torch.tan(phi_x - theta_x)
        half_range_x = torch.max(torch.abs(bb1_x), torch.abs(bb2_x))

        bb1_y = self.proj_COP[1] - self.proj_COP[2] * torch.tan(phi_y + theta_y)
        bb2_y = self.proj_COP[1] - self.proj_COP[2] * torch.tan(phi_y - theta_y)
        half_range_y = torch.max(torch.abs(bb1_y), torch.abs(bb2_y))

        return half_range_x, half_range_y


    def projector_projection(self, points_3d, pattern, reflectance, proj_point_dir=None, shade_on=False, fov_mask_on=False):
        if self.use_proj_extrinsics:
            num_points = points_3d.shape[0] * points_3d.shape[1]
            points_flat = points_3d.reshape(num_points, 3).t()
#             points_proj_flat = self.proj_R.T @ (points_flat - self.proj_COP.view(3,1))
            points_proj_flat = self.proj_R @ (points_flat - self.proj_COP.view(3,1))  # R: world→local

            points_proj = points_proj_flat.t().reshape(points_3d.shape)

        else:
            if proj_point_dir is None:
                proj_target = torch.tensor([0.0, 0.0, 0.0], device=self.device)
                proj_point_dir = proj_target - self.proj_COP
            proj_forward_dir = proj_point_dir / torch.norm(proj_point_dir)
            proj_world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            proj_right_dir = torch.linalg.cross(proj_world_up, proj_forward_dir, dim=0)
            proj_right_dir = proj_right_dir / torch.norm(proj_right_dir)
            proj_up_dir = torch.linalg.cross(proj_forward_dir, proj_right_dir, dim=0)
            R_proj = torch.stack([proj_right_dir, proj_up_dir, proj_forward_dir], dim=0)
            points_proj = (points_3d - self.proj_COP) @ R_proj.t()

        x_proj = self.proj_fx * (points_proj[..., 0] / points_proj[..., 2])
        y_proj = self.proj_fy * (points_proj[..., 1] / points_proj[..., 2])

        projsim_h_px, projsim_w_px = pattern.shape
        cx_offset_sensor = self.proj_cx_px * (self.proj_sensor_w_px / projsim_w_px)
        cy_offset_sensor = self.proj_cy_px * (self.proj_sensor_h_px / projsim_h_px)

        # ==========================
        norm_x = (x_proj - cx_offset_sensor) / (self.proj_sensor_w_px / 2.0)
        norm_y = (y_proj - cy_offset_sensor) / (self.proj_sensor_h_px / 2.0)

        fov_mask = ((norm_x >= -1.0) & (norm_x <= 1.0) &
                    (norm_y >= -1.0) & (norm_y <= 1.0)).float()


        grid_proj = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)

        pattern_ = pattern.unsqueeze(0).unsqueeze(0)
        illumination_values = F.grid_sample(pattern_, grid_proj, mode='bilinear', align_corners=False)

        image_values = illumination_values.squeeze() * reflectance.squeeze()

        
        if fov_mask_on:
            image_values = image_values * fov_mask
        ######################################################################
        # self occlusion
        H = points_3d[..., 2]  
        res_val = H.shape[0]

        x_vals = points_3d[..., 0]
        y_vals = points_3d[..., 1]
        dx = (x_vals.max() - x_vals.min()) / (res_val - 1)
        dy = (y_vals.max() - y_vals.min()) / (res_val - 1)

        dH_dx = torch.zeros_like(H)  
        dH_dy = torch.zeros_like(H)  

        dH_dx[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / (2 * dx)
        dH_dx[:, 0]    = (H[:, 1] - H[:, 0])     / dx
        dH_dx[:, -1]   = (H[:, -1] - H[:, -2])   / dx

        dH_dy[1:-1, :] = (H[2:, :] - H[:-2, :]) / (2 * dy)
        dH_dy[0, :]    = (H[1, :] - H[0, :])    / dy
        dH_dy[-1, :]   = (H[-1, :] - H[-2, :])  / dy

        n_x, n_y, n_z = -dH_dx, -dH_dy, torch.ones_like(H)
        norm_val = torch.sqrt(n_x**2 + n_y**2 + n_z**2) + 1e-6
        normal = torch.stack([n_x / norm_val, n_y / norm_val, n_z / norm_val], dim=-1)


        L = (self.proj_COP.view(1,1,3) - points_3d)
        L = L / (torch.norm(L, dim=-1, keepdim=True) + 1e-6)

        dot_prod = torch.sum(normal * L, dim=-1) 
        occlusion_mask = Proj_Cam_model.HardOcclusion.apply(dot_prod, 100.0)

        image_values = image_values * occlusion_mask
        
        ######################################################################
        # compute_shadow_mask，
        if shade_on:
            if proj_point_dir is None:
                if self.use_proj_extrinsics:
                    proj_point_dir = (self.proj_R.t() @ torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.proj_R.dtype))
                    proj_point_dir = proj_point_dir / (torch.norm(proj_point_dir) + 1e-9)

                else:
                    proj_point_dir = torch.tensor([0.0, 0.0, -1.0], device=self.device)  # fallback
            shadow_mask = self.compute_shadow_mask(H, proj_point_dir, bias=10.0, steepness=100.0)
            image_values = image_values * shadow_mask

        return image_values


    def compute_shadow_mask(self, height, proj_point_dir, bias=1.0, steepness=100.0):
        res_val = height.shape[0]

        if not torch.is_tensor(proj_point_dir):
            proj_point_dir = torch.tensor(proj_point_dir, dtype=height.dtype, device=height.device)
        light_dir = proj_point_dir / (torch.norm(proj_point_dir) + 1e-9)
        light_xy = light_dir[:2]

        # 2) physical range
        half_range_x, half_range_y = self.compute_xy_range_mm()

        # 3) identity base_grid：grid[...,0]=x, grid[...,1]=y
        grid_lin = torch.linspace(-1, 1, res_val, device=height.device, dtype=height.dtype)
        X, Y = torch.meshgrid(grid_lin, grid_lin, indexing='xy')
        base_grid = torch.stack([X, Y], dim=-1).unsqueeze(0)  # (1,res,res,2)

        offset_norm_x =  bias * light_xy[0] / half_range_x
        offset_norm_y =  bias * light_xy[1] / half_range_y
        offset_grid   = torch.stack([-offset_norm_x, -offset_norm_y])  # [dx, dy] in normalized coords

        shifted_grid = base_grid + offset_grid.view(1, 1, 1, 2)
        height_ = height.unsqueeze(0).unsqueeze(0)  # (1,1,res,res)
        shifted_height = F.grid_sample(height_, shifted_grid, mode='bilinear', align_corners=True)
        shifted_height = shifted_height.squeeze(0).squeeze(0)

        diff = shifted_height - (height + bias)
        shadow_mask = Proj_Cam_model.HardOcclusion.apply(-diff, steepness)
        return shadow_mask


    def camera_projection(self, points_3d, image_values, cam_point_dir=None, cam_res=None):
        if self.use_cam_extrinsics:
            num_points = points_3d.shape[0] * points_3d.shape[1]
            points_flat = points_3d.reshape(num_points, 3).t()
            points_cam_flat = self.cam_R @ (points_flat - self.cam_COP.view(3,1))
            points_cam = points_cam_flat.t().reshape(points_3d.shape)
        else:
            if cam_point_dir is None:
                cam_target = torch.tensor([0.0, 0.0, 0.0], device=self.device)
                cam_point_dir = cam_target - self.cam_COP
            cam_forward_dir = cam_point_dir / torch.norm(cam_point_dir)
            cam_world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            cam_right_dir = torch.cross(cam_world_up, cam_forward_dir)
            cam_up_dir = torch.cross(cam_forward_dir, cam_right_dir)
            R_cam = torch.stack([cam_right_dir, cam_up_dir, cam_forward_dir], dim=0)
            points_cam = (points_3d - self.cam_COP) @ R_cam.t()

        x_cam = self.cam_fx * (points_cam[..., 0] / points_cam[..., 2])
        y_cam = self.cam_fy * (points_cam[..., 1] / points_cam[..., 2])

        if cam_res is not None:
            cam_res_h, cam_res_w = int(cam_res[0]), int(cam_res[1])
        else:
            cam_res_h = self.cam_res_h or points_3d.shape[0]
            cam_res_w = self.cam_res_w or points_3d.shape[1]

        base_h, base_w = self.cam_sensor_px, self.cam_sensor_px
        cx_shift = self.cam_cx_px * (cam_res_w / base_w)
        cy_shift = self.cam_cy_px * (cam_res_h / base_h) #* (4024.0 / 3036.0)

        u_cam = (x_cam / self.cam_sensor_px) * cam_res_w + cam_res_w/2 - cx_shift
        v_cam = (y_cam / self.cam_sensor_px) * cam_res_h + cam_res_h/2 - cy_shift

        valid = (u_cam >= 0) & (u_cam < cam_res_w) & (v_cam >= 0) & (v_cam < cam_res_h)

        flat_uv = torch.stack([u_cam, v_cam], dim=-1).reshape(-1, 2)
        flat_z = points_cam[..., 2].reshape(-1)
        flat_intensity = image_values.reshape(-1)
        flat_valid = valid.reshape(-1)

        valid_points = flat_uv[flat_valid]
        valid_z = flat_z[flat_valid]
        valid_intensity = flat_intensity[flat_valid]
        return valid_points, valid_z, valid_intensity, (cam_res_w, cam_res_h)



    def soft_z_buffer(self, projected_points, z_values, intensity, out_size, sigma=1.0, radius_multiplier=2, steepness=100.0):
        """
        Compute the soft Z-buffer using a local window with a differentiable (soft) rounding operation.
        
        Args:
            projected_points: Tensor of shape (N, 2) with continuous pixel coordinates (in [0, res])
            z_values: Tensor of shape (N,) containing depth values
            intensity: Tensor of shape (N,) containing intensities
            res: Resolution (scalar) for the output image (res x res)
            sigma: Standard deviation for the Gaussian kernel
            radius_multiplier: Factor to determine local window radius
            steepness: Controls the sharpness of the soft mask boundary
        
        Returns:
            final_output: Rendered image tensor of shape (res, res)
        """
        device = projected_points.device
        dtype = projected_points.dtype
        
        cam_res_w, cam_res_h = int(out_size[0]), int(out_size[1])
        window_radius = int(torch.ceil(torch.tensor(radius_multiplier * sigma, device=device, dtype=dtype)).item())
        window_size = 2 * window_radius + 1

        offsets = torch.arange(-window_radius, window_radius + 1, device=device, dtype=dtype)
        rel_x, rel_y = torch.meshgrid(offsets, offsets, indexing='xy')

        u = projected_points[:, 0]
        v = projected_points[:, 1]
        u_center = Proj_Cam_model.soft_round(u - 0.5)
        v_center = Proj_Cam_model.soft_round(v - 0.5)

        local_x = (u_center.unsqueeze(1).unsqueeze(2) + 0.5) + rel_x.unsqueeze(0)
        local_y = (v_center.unsqueeze(1).unsqueeze(2) + 0.5) + rel_y.unsqueeze(0)

        diff_x = local_x - u.unsqueeze(1).unsqueeze(2)
        diff_y = local_y - v.unsqueeze(1).unsqueeze(2)
        dist_sq = diff_x ** 2 + diff_y ** 2

        radius_sq = (radius_multiplier * sigma) ** 2
        mask = torch.sigmoid(steepness * (radius_sq - dist_sq))
        weights = torch.exp(-0.5 * dist_sq / (sigma ** 2)) * mask

        intensity_weight = (intensity / (z_values + 1e-6)).unsqueeze(1).unsqueeze(2)
        depth_weight = (1 / (z_values + 1e-6)).unsqueeze(1).unsqueeze(2)
        local_weighted_intensity = weights * intensity_weight
        local_weighted_depth = weights * depth_weight
        
        u_idx = (local_x - 0.5).long().clamp(0, cam_res_w - 1)
        v_idx = (local_y - 0.5).long().clamp(0, cam_res_h - 1)

        N = projected_points.shape[0]
        num_elems = N * window_size * window_size

        flat_intensity = local_weighted_intensity.reshape(num_elems)

        flat_depth = local_weighted_depth.reshape(num_elems)
        flat_u_idx = u_idx.reshape(num_elems)
        flat_v_idx = v_idx.reshape(num_elems)
        flat_indices = flat_v_idx * cam_res_w  + flat_u_idx

        
        final_intensity = torch.zeros(cam_res_w * cam_res_h, device=device, dtype=dtype)
        final_depth = torch.zeros(cam_res_w * cam_res_h, device=device, dtype=dtype)

        final_intensity.scatter_add_(0, flat_indices, flat_intensity)

        final_depth.scatter_add_(0, flat_indices, flat_depth)


        final_intensity = final_intensity.reshape(*final_intensity.shape[:-1], cam_res_h, cam_res_w)
        final_depth = final_depth.reshape(*final_depth.shape[:-1], cam_res_h, cam_res_w)

        final_output = final_intensity / (final_depth + 1e-6) 
        
        
        return final_output

    def render(self, height, reflectance, pattern,
           extent_mm=[-100,100,-100,100],  # [xmin, xmax, ymin, ymax] in mm
           shade_on=True, fov_mask_on=True,
           proj_point_dir=None, cam_point_dir=None, cam_res=(1024, 1024), 
           photon_count=None, show=False, print_matrix = False):

        self.last_height = height
        self.last_pattern = pattern
        self.extent_mm = extent_mm   


        height = height.to(self.device).squeeze(0)
        reflectance = reflectance.to(self.device).squeeze(0)
        pattern = pattern.to(self.device)
        
        res_h_world, res_w_world = height.shape  

        xmin, xmax, ymin, ymax = extent_mm
        X = torch.linspace(xmin, xmax, res_h_world, device=self.device)
        Y = torch.linspace(ymin, ymax, res_w_world, device=self.device)
        X, Y = torch.meshgrid(X, Y, indexing='xy')
        points_3d = torch.stack([X, Y, height], dim=-1)

        
        image_values = self.projector_projection(points_3d, pattern, reflectance,
                                                 proj_point_dir, shade_on=shade_on, fov_mask_on=fov_mask_on)


        
        valid_points, valid_z, valid_intensity, out_size = self.camera_projection(
            points_3d, image_values, cam_point_dir, cam_res=cam_res
        )

        
        observed_image = self.soft_z_buffer(valid_points, valid_z, valid_intensity,
                                            out_size, sigma=0.5, radius_multiplier=4, steepness=200.0)
        
        if photon_count is not None:
            observed_image = torch.poisson(observed_image * photon_count) / photon_count

        self.last_reflectance = reflectance
        self.last_observed_image = observed_image

        if show:
            self.visualize(print_matrix=print_matrix, margin_ratio=0.2)

        return observed_image.unsqueeze(0).unsqueeze(0)
    def get_intrinsics(self, res_val, is_proj=True):

        if is_proj:
            fx = self.proj_f * (res_val / self.proj_sensor_w_px)
            fy = self.proj_f * (res_val / self.proj_sensor_h_px)
            cx = res_val / 2.0 + self.proj_cx_px
            cy = res_val / 2.0 + self.proj_cy_px
        else:
            fx = self.cam_f * (res_val / self.cam_sensor_px)
            fy = self.cam_f * (res_val / self.cam_sensor_px)
            cx = res_val / 2.0 + self.cam_cx_px
            cy = res_val / 2.0 + self.cam_cy_px

        K = torch.tensor([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=torch.float32, device=self.device)
        return K

    def get_extrinsics(self, is_proj=True):

        if is_proj and self.use_proj_extrinsics:
            R, t = self.proj_R, self.proj_t
        elif (not is_proj) and self.use_cam_extrinsics:
            R, t = self.cam_R, self.cam_t
        else:
            COP = self.proj_COP if is_proj else self.cam_COP
            forward = -COP / torch.norm(COP)
            up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            right = torch.linalg.cross(up, forward); right /= torch.norm(right)
            up =torch.linalg.cross(forward, right); up /= torch.norm(up)
            R = torch.stack([right, up, forward], dim=0)
            t = -R @ COP.view(3,1)

        extrinsic = torch.cat([R, t], dim=1)
        return extrinsic


    def visualize(self, height_map=None, reflectance=None, observed_image=None, pattern=None,
                  show=True, print_matrix=False, margin_ratio=0.2):
        if height_map is None:
            if self.last_height is None:
                raise ValueError("No height map available. Please run render() first or provide height_map.")
            height_map = self.last_height
        if reflectance is None:
            if self.last_reflectance is None:
                raise ValueError("No reflectance available. Please run render() first or provide reflectance.")
            reflectance = self.last_reflectance
        if observed_image is None:
            if self.last_observed_image is None:
                raise ValueError("No observed image available. Please run render() first or provide observed_image.")
            observed_image = self.last_observed_image
        if pattern is None:
            if self.last_pattern is None:
                raise ValueError("No pattern available. Please run render() first or provide pattern.")
            pattern = self.last_pattern

        if print_matrix:
            res_val = height_map.shape[0]

            K_proj = np.round(self.get_intrinsics(res_val, is_proj=True).cpu().numpy(), 4)
            K_cam  = np.round(self.get_intrinsics(res_val, is_proj=False).cpu().numpy(), 4)
            E_proj = np.round(self.get_extrinsics(is_proj=True).cpu().numpy(), 4)
            E_cam  = np.round(self.get_extrinsics(is_proj=False).cpu().numpy(), 4)

            np.set_printoptions(precision=4, suppress=True)

            print("==== Projector Intrinsics (K) ====")
            print(K_proj)
            print("==== Projector Extrinsics [R|t] ====")
            print(E_proj)
            print("==== Camera Intrinsics (K) ====")
            print(K_cam)
            print("==== Camera Extrinsics [R|t] ====")
            print(E_cam)

        if show:
            x_min, x_max, y_min, y_max = self.extent_mm

            A = float(max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)))
            A = A * (1.0 + margin_ratio)  
            plot_xmin, plot_xmax = -A, A
            plot_ymin, plot_ymax = -A, A
            # ==================================

            fig = plt.figure(figsize=(12, 8))


            plt.subplot(2, 2, 1)
            plt.imshow(height_map.cpu().detach().squeeze(),
                       cmap='gray',
                       extent=[x_min, x_max, y_min, y_max],
#                        origin='lower',
                       aspect='equal')
            plt.xlim(plot_xmin, plot_xmax)
            plt.ylim(plot_ymin, plot_ymax)
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
            plt.title("Height Map (mm)")
            plt.colorbar(label="Height (mm)")
            plt.gca().set_facecolor("black")  

            # reflectance map
            plt.subplot(2, 2, 2)
            plt.imshow(reflectance.cpu().detach().squeeze(),
                       cmap='gray',
                       extent=[x_min, x_max, y_min, y_max],
#                        origin='lower',
                       aspect='equal')
            plt.xlim(plot_xmin, plot_xmax)
            plt.ylim(plot_ymin, plot_ymax)
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
            plt.title("Reflectance Map")
            plt.colorbar(label="Reflectance")
            plt.gca().set_facecolor("black")   

            # projection pattern
            plt.subplot(2, 2, 3)
            plt.imshow(pattern.cpu().detach().squeeze(),
                       cmap='gray',
                       aspect='equal')
            plt.title("Projection Pattern")
            plt.colorbar()

            # observe image
            plt.subplot(2, 2, 4)
            plt.imshow(observed_image.cpu().detach().squeeze(),
                       cmap='gray',
                       aspect='equal')
            plt.title("Observed Image")
            plt.colorbar()

            plt.tight_layout()
            plt.show()

        return


    def plot_geometry(
        self,
        grid_size: int = 200,
        axis_length: float = 200.0,
        plane_extent: float = 500.0,
        extend_ratio: float = 0.2,
        cube_view=True,
        plt_ideal: int = 0,  
    ) -> go.Figure:


        # ============ 小工具 ============

        def _add_axes(fig, origin, R_l2w, length=100, name_prefix=""):

            colors = ["red", "green", "blue"]
            axis_names = ["X", "Y", "Z"]
            for i in range(3):
                axis_dir = R_l2w[:, i]
                end = origin + axis_dir * length
                fig.add_trace(
                    go.Scatter3d(
                        x=[origin[0], end[0]],
                        y=[origin[1], end[1]],
                        z=[origin[2], end[2]],
                        mode="lines+text",
                        line=dict(color=colors[i], width=6),
                        text=[None, f"{name_prefix}{axis_names[i]}"],
                        textposition="top center",
                        name=f"{name_prefix}{axis_names[i]}-axis",
                    )
                )

        def _axis_intersection_and_far(COP, R_l2w, dir_local, extend_ratio=0.2):

            COP = COP.reshape(3)
            dir_local = np.asarray(dir_local, dtype=np.float64)
            d = R_l2w @ dir_local
            d = d / (np.linalg.norm(d) + 1e-12)

            if abs(d[2]) < 1e-12:
                return None, None, d, None

            s = -COP[2] / d[2]
            P = COP + s * d
            Far = COP + s * (1.0 + extend_ratio) * d
            return P, Far, d, s

        def _add_fov_footprint(
            fig, COP, R_w2l,
            fx, fy, W, H, cx, cy,
            color, name, linestyle="solid", opacity=0.2
        ):

            R_l2w = R_w2l.T
            COP = np.asarray(COP, dtype=np.float64).reshape(3)

            fx = float(fx)
            fy = float(fy)
            W = float(W)
            H = float(H)
            cx = float(cx)
            cy = float(cy)

            corners_c = np.array(
                [
                    [(-W / 2 + cx) / fx, (-H / 2 + cy) / fy, 1.0],
                    [( W / 2 + cx) / fx, (-H / 2 + cy) / fy, 1.0],
                    [( W / 2 + cx) / fx, ( H / 2 + cy) / fy, 1.0],
                    [(-W / 2 + cx) / fx, ( H / 2 + cy) / fy, 1.0],
                ],
                dtype=np.float64,
            )

            corners_w = []
            for p_c in corners_c:
                d_w = R_l2w @ p_c
                d_w = d_w / (np.linalg.norm(d_w) + 1e-12)
                if abs(d_w[2]) < 1e-9:
                    continue
                t = -COP[2] / d_w[2]
                P = COP + t * d_w
                corners_w.append(P)

            if len(corners_w) < 4:
                return

            corners_w = np.array(corners_w)

            fig.add_trace(
                go.Scatter3d(
                    x=np.append(corners_w[:, 0], corners_w[0, 0]),
                    y=np.append(corners_w[:, 1], corners_w[0, 1]),
                    z=np.append(corners_w[:, 2], corners_w[0, 2]),
                    mode="lines",
                    line=dict(
                        color=color,
                        width=3,
                        dash="solid" if linestyle == "solid" else "dash",
                    ),
                    name=name,
                )
            )

            fig.add_trace(
                go.Mesh3d(
                    x=corners_w[:, 0],
                    y=corners_w[:, 1],
                    z=corners_w[:, 2],
                    color=color,
                    opacity=opacity,
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    name=name + " Area",
                    showlegend=False,
                )
            )


        def _add_projector_pattern_plane(fig, COP, R_w2l, fx, fy, W, H, cx, cy):

            if self.last_pattern is None:
                return

            R_l2w = R_w2l.T
            COP = np.asarray(COP, dtype=np.float64).reshape(3)

            pattern = self.last_pattern.detach().cpu().squeeze().numpy()
            if pattern.ndim != 2:
                return

            ph, pw = pattern.shape

            fx = float(fx)
            fy = float(fy)
            W = float(W)
            H = float(H)
            cx = float(cx)
            cy = float(cy)

            u_lin = np.linspace(-W / 2 + cx, W / 2 + cx, pw)
            v_lin = np.linspace(-H / 2 + cy, H / 2 + cy, ph)
            U, V = np.meshgrid(u_lin, v_lin, indexing="xy")

            pts_local = np.stack([U / fx, V / fy, np.ones_like(U)], axis=-1)  # (ph,pw,3)
            pts_local_flat = pts_local.reshape(-1, 3).T  # (3,N)

            pts_world_flat = (R_l2w @ pts_local_flat).T + COP.reshape(1, 3)
            pts_world = pts_world_flat.reshape(ph, pw, 3)

            Xw = pts_world[..., 0]
            Yw = pts_world[..., 1]
            Zw = pts_world[..., 2]

            fig.add_trace(
                go.Surface(
                    x=Xw,
                    y=Yw,
                    z=Zw,
                    surfacecolor=pattern,
                    colorscale="Gray",
                    showscale=False,
                    opacity=0.9,
                    name="Projector Pattern on Sensor",
                )
            )

        if self.last_height is None:
            raise ValueError("None height map，run render() fitst")
        height = self.last_height.detach().cpu().squeeze()

        if hasattr(self, "extent_mm") and self.extent_mm is not None:
            xmin, xmax, ymin, ymax = self.extent_mm
        else:
            half_x, half_y = self.compute_xy_range_mm()
            half_x = float(half_x.cpu().item())
            half_y = float(half_y.cpu().item())
            xmin, xmax, ymin, ymax = -half_x, half_x, -half_y, half_y

        lin_x = np.linspace(xmin, xmax, grid_size)
        lin_y = np.linspace(ymin, ymax, grid_size)
        X, Y = np.meshgrid(lin_x, lin_y, indexing="xy")

        h_np = height.numpy()
        res = h_np.shape[0]
        if res != grid_size:
            scale = grid_size / res
            Z = zoom(h_np, zoom=(scale, scale), order=1)
        else:
            Z = h_np

        fig = go.Figure()

        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z, colorscale="Viridis", opacity=0.7, name="Object Surface"
            )
        )

        plane_lin = np.linspace(-plane_extent, plane_extent, 20)
        Xp, Yp = np.meshgrid(plane_lin, plane_lin, indexing="xy")
        Zp = np.zeros_like(Xp)
        fig.add_trace(
            go.Surface(
                x=Xp,
                y=Yp,
                z=Zp,
                colorscale=[[0, "lightgray"], [1, "lightgray"]],
                opacity=0.75,
                showscale=False,
                name="z=0 Plane",
            )
        )

        # ========== projector ==========
        proj = self.proj_COP.cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=[proj[0]],
                y=[proj[1]],
                z=[proj[2]],
                mode="markers+text",
                marker=dict(size=5, color="red"),
                text=["Projector"],
                textposition="top center",
                name="Projector",
            )
        )

        if hasattr(self, "proj_R") and self.use_proj_extrinsics:
            R_proj_w2l = self.proj_R.cpu().numpy()
            R_proj_l2w = R_proj_w2l.T
        else:
            forward = -proj / (np.linalg.norm(proj) + 1e-12)
            world_down = np.array([0.0, -1.0, 0.0])
            right = np.cross(world_down, forward)
            right = right / (np.linalg.norm(right) + 1e-12)
            down = np.cross(forward, right)
            down = down / (np.linalg.norm(down) + 1e-12)
            R_proj_l2w = np.stack([right, down, forward], axis=1)
            R_proj_w2l = R_proj_l2w.T

        _add_axes(fig, proj, R_proj_l2w, length=axis_length, name_prefix="Proj ")

        # --- ideal axis (local [0,0,1]) ---
        proj_inter_ideal, proj_far_ideal, _, _ = _axis_intersection_and_far(
            proj, R_proj_l2w, dir_local=[0.0, 0.0, 1.0], extend_ratio=extend_ratio
        )

        # --- shift axis (local [cx, cy, f]) ---
        proj_cx = float(self.proj_cx_px) 
        proj_cy = float(self.proj_cy_px)
        proj_inter_shift, proj_far_shift, _, _ = _axis_intersection_and_far(
            proj,
            R_proj_l2w,
            dir_local=[
                proj_cx / float(self.proj_fx),
                proj_cy / float(self.proj_fy),
                1.0,
            ],
            extend_ratio=extend_ratio,
        )

        print("proj_inter_ideal: ", proj_inter_ideal)
        print("proj_inter_shift: ", proj_inter_shift)

        # ===== Projector FOV: ideal + shift =====
        if (
            hasattr(self, "proj_fx") and hasattr(self, "proj_fy")
            and hasattr(self, "proj_sensor_w_px") and hasattr(self, "proj_sensor_h_px")
        ):
            R_w2l_for_proj = (
                self.proj_R.cpu().numpy()
                if (hasattr(self, "proj_R") and self.use_proj_extrinsics)
                else R_proj_w2l
            )

            if plt_ideal:
                _add_fov_footprint(
                    fig, proj, R_w2l_for_proj,
                    fx=float(self.proj_fx), fy=float(self.proj_fy),
                    W=float(self.proj_sensor_w_px), H=float(self.proj_sensor_h_px),
                    cx=0.0, cy=0.0,
                    color="orange", name="Projector FOV (ideal)",
                    linestyle="solid", opacity=0.15,
                )


            _add_fov_footprint(
                fig, proj, R_w2l_for_proj,
                fx=float(self.proj_fx), fy=float(self.proj_fy),
                W=float(self.proj_sensor_w_px), H=float(self.proj_sensor_h_px),
                cx=proj_cx, cy=proj_cy,
                color="darkorange", name="Projector FOV (shifted)",
                linestyle="dash", opacity=0.25,
            )


            _add_projector_pattern_plane(
                fig, proj, R_w2l_for_proj,
                fx=float(self.proj_fx), fy=float(self.proj_fy),
                W=float(self.proj_sensor_w_px), H=float(self.proj_sensor_h_px),
                cx=proj_cx, cy=proj_cy,
            )


        if plt_ideal:
            if proj_far_ideal is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=[proj[0], proj_far_ideal[0]],
                        y=[proj[1], proj_far_ideal[1]],
                        z=[proj[2], proj_far_ideal[2]],
                        mode="lines",
                        line=dict(color="orange", width=2),
                        name="Proj Axis (ideal)",
                    )
                )
            if proj_inter_ideal is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=[proj_inter_ideal[0]],
                        y=[proj_inter_ideal[1]],
                        z=[0],
                        mode="markers",
                        marker=dict(size=3, color="orange", symbol="circle"),
                        name="Proj Axis ∩ z=0 (ideal)",
                    )
                )

        if proj_far_shift is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[proj[0], proj_far_shift[0]],
                    y=[proj[1], proj_far_shift[1]],
                    z=[proj[2], proj_far_shift[2]],
                    mode="lines",
                    line=dict(color="darkorange", width=2, dash="dash"),
                    name="Proj Axis (shifted)",
                )
            )
        if proj_inter_shift is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[proj_inter_shift[0]],
                    y=[proj_inter_shift[1]],
                    z=[0],
                    mode="markers",
                    marker=dict(size=3, color="darkorange", symbol="circle"),
                    name="Proj Axis ∩ z=0 (shifted)",
                )
            )

        # ========== camera ==========
        cam = self.cam_COP.cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=[cam[0]],
                y=[cam[1]],
                z=[cam[2]],
                mode="markers+text",
                marker=dict(size=5, color="blue"),
                text=["Camera"],
                textposition="top center",
                name="Camera",
            )
        )

        if hasattr(self, "cam_R") and self.use_cam_extrinsics:
            R_cam_w2l = self.cam_R.cpu().numpy()
            R_cam_l2w = R_cam_w2l.T
        else:
            forward = -cam / (np.linalg.norm(cam) + 1e-12)
            world_down = np.array([0.0, -1.0, 0.0])
            right = np.cross(world_down, forward)
            right = right / (np.linalg.norm(right) + 1e-12)
            down = np.cross(forward, right)
            down = down / (np.linalg.norm(down) + 1e-12)
            R_cam_l2w = np.stack([right, down, forward], axis=1)
            R_cam_w2l = R_cam_l2w.T

        _add_axes(fig, cam, R_cam_l2w, length=axis_length, name_prefix="Cam ")

        # ideal axis
        cam_inter_ideal, cam_far_ideal, _, _ = _axis_intersection_and_far(
            cam, R_cam_l2w, dir_local=[0.0, 0.0, 1.0], extend_ratio=extend_ratio
        )

        # shift axis
        cam_cx = float(self.cam_cx_px)
        cam_cy = float(self.cam_cy_px)
        cam_inter_shift, cam_far_shift, _, _ = _axis_intersection_and_far(
            cam,
            R_cam_l2w,
            dir_local=[
                cam_cx / float(self.cam_fx),
                cam_cy / float(self.cam_fy),
                1.0,
            ],
            extend_ratio=extend_ratio,
        )
        print("cam_inter_ideal: ", cam_inter_ideal)
        print("cam_inter_shift: ", cam_inter_shift)

        # Camera FOV: ideal + shift
        if hasattr(self, "cam_fx") and hasattr(self, "cam_fy") and hasattr(self, "cam_sensor_px"):
            R_w2l_for_cam = (
                self.cam_R.cpu().numpy()
                if (hasattr(self, "cam_R") and self.use_cam_extrinsics)
                else R_cam_w2l
            )

            if plt_ideal:
                _add_fov_footprint(
                    fig, cam, R_w2l_for_cam,
                    fx=float(self.cam_fx), fy=float(self.cam_fy),
                    W=float(self.cam_sensor_px), H=float(self.cam_sensor_px),
                    cx=0.0, cy=0.0,
                    color="cyan",
                    name="Camera FOV (ideal)",
                    linestyle="solid",
                    opacity=0.15,
                )

            _add_fov_footprint(
                fig, cam, R_w2l_for_cam,
                fx=float(self.cam_fx), fy=float(self.cam_fy),
                W=float(self.cam_sensor_px), H=float(self.cam_sensor_px),
                cx=cam_cx,
                cy=cam_cy,
                color="darkcyan",
                name="Camera FOV (shifted)",
                linestyle="dash",
                opacity=0.25,
            )

        if plt_ideal:
            if cam_far_ideal is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=[cam[0], cam_far_ideal[0]],
                        y=[cam[1], cam_far_ideal[1]],
                        z=[cam[2], cam_far_ideal[2]],
                        mode="lines",
                        line=dict(color="cyan", width=2),
                        name="Cam Axis (ideal)",
                    )
                )
            if cam_inter_ideal is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=[cam_inter_ideal[0]],
                        y=[cam_inter_ideal[1]],
                        z=[0],
                        mode="markers",
                        marker=dict(size=3, color="cyan", symbol="circle"),
                        name="Cam Axis ∩ z=0 (ideal)",
                    )
                )

        if cam_far_shift is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[cam[0], cam_far_shift[0]],
                    y=[cam[1], cam_far_shift[1]],
                    z=[cam[2], cam_far_shift[2]],
                    mode="lines",
                    line=dict(color="darkcyan", width=2, dash="dash"),
                    name="Cam Axis (shifted)",
                )
            )
        if cam_inter_shift is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[cam_inter_shift[0]],
                    y=[cam_inter_shift[1]],
                    z=[0],
                    mode="markers",
                    marker=dict(size=3, color="darkcyan", symbol="circle"),
                    name="Cam Axis ∩ z=0 (shifted)",
                )
            )

        # Layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X [mm]"),
                yaxis=dict(title="Y [mm]"),
                zaxis=dict(title="Z [mm]", range=[-100, None]),
                aspectmode="cube" if cube_view else "data",
            ),
            title="Projector–Object–Camera Geometry (ideal vs shifted principal point)",
            margin=dict(l=0, r=0, b=0, t=40),
        )

#         fig.show()
        return fig





