import os
import cv2
import json
import time
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class NetParallelTrain:
    def __init__(self, base_model, n_nets, optimizers=None, schedulers=None, device="cuda"):
        self.device = device
        self.n_nets = n_nets

            # ===== MultiNet =====
        class MultiNet(nn.Module):
            def __init__(self, base_model, n_nets, device):
                super(MultiNet, self).__init__()
                self.device = device
                self.networks = nn.ModuleList([copy.deepcopy(base_model) for _ in range(n_nets)])

            def forward(self, x, msk_on=0, mask_xy=None, epoch=None, num_epochs=None, mask_warmup_ratio=0.15, mask_transition_mode="none",
):
                outputs = []
                B, _, H, W = x.shape

                # === Warm-up ===
#                 if epoch is not None and num_epochs is not None:
#                     progress_ratio = epoch / num_epochs
#                     if progress_ratio < mask_warmup_ratio:
#                         effective_mask_on = 0
#                         warm_ratio = 0.0
#                     else:
#                         effective_mask_on = msk_on
#                         warm_ratio = min((progress_ratio - mask_warmup_ratio) / (1 - mask_warmup_ratio), 1.0)
#                 else:
#                     effective_mask_on = msk_on
#                     warm_ratio = 1.0
                progress_ratio = epoch / num_epochs if (epoch is not None and num_epochs is not None) else 1.0

                warm_ratio = NetParallelTrain.compute_warm_ratio(progress_ratio, mask_warmup_ratio, mask_transition_mode)

                effective_mask_on = msk_on if warm_ratio > 1e-6 else 0

                for net in self.networks:
                    predicted_height, predicted_reflectance = net(x)

                    # === Height: Hard mask ===
                    masked_height = NetParallelTrain.apply_mask(
                        predicted_height,
                        mask_xy, H, W, effective_mask_on,
                        mode="height", soft=False  # => height >0 within ROI
                    )

                    # === Reflectance: Soft Gaussian mask (e.g., RGB) ===
                    masked_reflectance = NetParallelTrain.apply_mask(
                        predicted_reflectance,
                        mask_xy, H, W, effective_mask_on = effective_mask_on, 
                        mode="reflectance", soft=True, falloff_sigma=3
                    )

                    # === Warm-Up Linear Blending ===
                    if msk_on == 1:
                        masked_height = predicted_height * (1 - warm_ratio) + masked_height * warm_ratio
                        masked_reflectance = predicted_reflectance * (1 - warm_ratio) + masked_reflectance * warm_ratio

                    outputs.append((masked_height, masked_reflectance))

                return outputs

        self.multi_model = MultiNet(base_model, n_nets, device).to(device)

        # ===== Optimizers =====
        if optimizers is None:
            self.optimizers = [
                optim.AdamW(net.parameters(), lr=0.004) for net in self.multi_model.networks
            ]
        else:
            self.optimizers = optimizers

        # ===== Schedulers =====
        if schedulers is None:
            self.schedulers = [
                optim.lr_scheduler.CosineAnnealingLR(self.optimizers[k], T_max=800, eta_min=1e-5)
                for k in range(n_nets)
            ]
        else:
            self.schedulers = schedulers

        self.loss_lists = [[] for _ in range(n_nets)]
        # ✅ default transition mode if train() not specified
        self.mask_transition_mode = "linear"
    @staticmethod
    def compute_warm_ratio(progress_ratio, warmup_ratio, mode="linear"):
        # ✅ 1) Not yet activating the mask warm-up
        if progress_ratio < warmup_ratio:
            return 0.0

        # ✅ Normalize progress after warmup start
        progress = (progress_ratio - warmup_ratio) / max(1e-8, (1 - warmup_ratio))

        # ✅ 2) Mode-based activation after warmup starts
        if mode == "none":
            # 🚀 Immediately apply mask once warmup ends
            return 1.0

        if mode == "linear":
            # ↗ Linearly enabled mask
            return min(max(progress, 0.0), 1.0)

        if mode == "sigmoid":
            steep = 10
            return float(torch.sigmoid(torch.tensor(steep * (progress - 0.5))))

        raise ValueError(f"Invalid mask_transition_mode: {mode}")


    # ======================================================================
    # ✅ smooth
    # ======================================================================
    
    @staticmethod
    def apply_mask(pred_map, mask_xy, H, W, effective_mask_on = 1,
                   mode="height", soft=False, falloff_sigma=20):
        """
        Mask method (Height & Reflectance)
        mode: 'height' or 'reflectance'
        soft:  Gaussian soft
        falloff_sigma: soft=True; mask edge diverge
        """

        if effective_mask_on != 1:
            return pred_map

        # ===  ROI region ===
        if mask_xy is None:
            half = 75
            cx, cy = W // 2, H // 2
            x1, x2 = cx - half, cx + half
            y1, y2 = cy - half, cy + half
        else:
            x1, x2, y1, y2 = mask_xy

        device = pred_map.device
        B, C, _, _ = pred_map.shape

        # ===  ROI Box mask ===
        mask = torch.zeros_like(pred_map, device=device)
        mask[:, :, y1:y2, x1:x2] = 1.0

        # ===  not soft -> clip ===
        if not soft:
            if mode == "height":
                return mask * pred_map + (1 - mask) * torch.clamp(pred_map, max=0.0)

            elif mode == "reflectance":
                return mask * pred_map + (1 - mask) * torch.clamp(pred_map, min=1.0)

            else:
                raise ValueError("mode must be 'height' or 'reflectance'")

        # ========================
        # ✅ Soft Gaussian Mask
        # ========================
        # y = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        # x = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)

        # dist_x = torch.minimum(torch.abs(x - x1), torch.abs(x - x2))
        # dist_y = torch.minimum(torch.abs(y - y1), torch.abs(y - y2))
        # dist = torch.minimum(dist_x, dist_y)

        # soft_mask_2d = torch.exp(-(dist ** 2) / (2 * falloff_sigma ** 2))
        # soft_mask_2d[y1:y2, x1:x2] = 1.0

        # #  [B, C, H, W]
        # soft_mask = soft_mask_2d.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)

        # if mode == "height":
        #     return soft_mask * pred_map + (1 - soft_mask) * torch.clamp(pred_map, max=0.0)

        # elif mode == "reflectance":
        #     return soft_mask * pred_map + (1 - soft_mask) * torch.clamp(pred_map, min=1.0)

        # else:
        #     raise ValueError("mode must be 'height' or 'reflectance'")
        yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device),
                            indexing='ij')
        
        dx = torch.maximum(torch.maximum(x1 - xx, xx - x2), torch.zeros_like(xx))
        dy = torch.maximum(torch.maximum(y1 - yy, yy - y2), torch.zeros_like(yy))
        dist = torch.sqrt(dx ** 2 + dy ** 2)
        
        soft_mask_2d = torch.exp(-(dist ** 2) / (2 * falloff_sigma ** 2))
        soft_mask_2d[y1:y2, x1:x2] = 1.0
        
        soft_mask = soft_mask_2d.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)
        
        if mode == "height":
            return soft_mask * pred_map + (1 - soft_mask) * torch.clamp(pred_map, max=0.0)
        elif mode == "reflectance":
            return soft_mask * pred_map + (1 - soft_mask) * torch.ones_like(pred_map)



    @staticmethod
    def smoothness_loss(height_map, loss_type="tv"):
        """
        （smoothness loss）
        """
        if loss_type == "tv":
            dx = torch.abs(height_map[:, :, :, :-1] - height_map[:, :, :, 1:])
            dy = torch.abs(height_map[:, :, :-1, :] - height_map[:, :, 1:, :])
            return torch.mean(dx) + torch.mean(dy)
        elif loss_type == "laplacian":
            kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=torch.float32, device=height_map.device).view(1, 1, 3, 3)
            lap = F.conv2d(height_map, kernel, padding=1)
            return torch.mean(torch.abs(lap))
        else:
            raise ValueError("loss_type must be 'tv' or 'laplacian'")
#         return clipped
    @staticmethod
    def generate_gaussian_mask(H, W, mask_xy=None, falloff_sigma=30, device="cuda"):
        """
        ROI close to 1、edge smooth decay Gaussian mask。
        falloff_sigma: edge blur width (pixel)
        """
        y = torch.arange(H, device=device).float().unsqueeze(1).repeat(1, W)
        x = torch.arange(W, device=device).float().unsqueeze(0).repeat(H, 1)

        if mask_xy is None:
            half = 75
            cx, cy = W // 2, H // 2
            x1, x2 = cx - half, cx + half
            y1, y2 = cy - half, cy + half
        else:
            x1, x2, y1, y2 = mask_xy

        # center region
        mask = torch.zeros((H, W), device=device)
        mask[y1:y2, x1:x2] = 1.0

        # dist
        dist_x = torch.minimum(torch.abs(x - x1), torch.abs(x - x2))
        dist_y = torch.minimum(torch.abs(y - y1), torch.abs(y - y2))
        dist = torch.minimum(dist_x, dist_y)

        # Gaussian fall-off
        smooth_mask = torch.exp(-(dist ** 2) / (2 * falloff_sigma ** 2))
        smooth_mask = torch.clamp(mask + (1 - mask) * smooth_mask, 0, 1)

        return smooth_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # ======================================================================
    # ✅ save （optimizer/scheduler/loss）
    # ======================================================================
    def save_trained_networks(self, name="experiment1", save_root="checkpoints"):
        """
        save checkpoint。
        folder loke: checkpoints/experiment1/net_0_checkpoint.pth

        parameters
        ----
        name : str
            
        save_root : str
            default "checkpoints"
        """
#         import os
        os.makedirs(os.path.join(save_root, name), exist_ok=True)
        save_dir = os.path.join(save_root, name)

        for i, net in enumerate(self.multi_model.networks):
            save_path = os.path.join(save_dir, f"net_{i}_checkpoint.pth")
            checkpoint = {
                "epoch": len(self.loss_lists[i]),
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": self.optimizers[i].state_dict(),
                "scheduler_state_dict": self.schedulers[i].state_dict(),
                "loss": self.loss_lists[i],
            }
            torch.save(checkpoint, save_path)
            print(f"✅ Saved checkpoint: {save_path}")

        print(f"✅ All networks saved under: {save_dir}")

    # ======================================================================
    # ✅  load checkpoint
    # ======================================================================
    def load_trained_networks(self, name="experiment1", save_root="checkpoints"):
#         import os

        save_dir = os.path.join(save_root, name)
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"❌ Directory not found: {save_dir}")

        print(f"🔍 Loading trained networks from: {save_dir}")

        for i, net in enumerate(self.multi_model.networks):
            ckpt_path = os.path.join(save_dir, f"net_{i}_checkpoint.pth")
            if not os.path.exists(ckpt_path):
                print(f"⚠️ Warning: {ckpt_path} not found, skipping.")
                continue

            try:
                checkpoint = torch.load(
                    ckpt_path, map_location=self.device, weights_only=False
                )

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    net.load_state_dict(checkpoint["model_state_dict"])
                    if "optimizer_state_dict" in checkpoint and i < len(self.optimizers):
                        self.optimizers[i].load_state_dict(checkpoint["optimizer_state_dict"])
                    if "scheduler_state_dict" in checkpoint and i < len(self.schedulers):
                        self.schedulers[i].load_state_dict(checkpoint["scheduler_state_dict"])
                    if "loss" in checkpoint:
                        self.loss_lists[i] = checkpoint["loss"]
                    print(f"✅ Loaded checkpoint for Net{i} from {ckpt_path}")

                else:
                    # ✅  (.pth)
                    state_dict = torch.load(
                        ckpt_path, map_location=self.device, weights_only=True
                    )
                    net.load_state_dict(state_dict)
                    print(f"✅ Loaded weights-only file for Net{i} from {ckpt_path}")

            except Exception as e:
                print(f"❌ Error loading {ckpt_path}: {e}")

        print("✅ All available networks loaded successfully.")

    def save_checkpoint_epoch(self, epoch, name="experiment1", save_root="./period_checkpoints"):
        """
        ✅ Save only model weights -> file much smaller ✅
        Save checkpoint per epoch:
            checkpoints/{name}/epoch_{epoch}.pth
        """
        os.makedirs(os.path.join(save_root, name), exist_ok=True)
        save_dir = os.path.join(save_root, name)

        for i, net in enumerate(self.multi_model.networks):
            save_path = os.path.join(save_dir, f"net_{i}_epoch_{epoch}.pth")
            # ✅ Small file: only model weights
            torch.save(net.state_dict(), save_path)
        print(f"💾 Periodic Backup Saved: Epoch {epoch}")


    def rollback_training(self, checkpoint_dir, start_epoch, num_epochs_new, measured_img, proj_img_sys, pattern, extent_mm, **train_kwargs):
        """
        ✅ Rollback training from given checkpoint folder
        checkpoint_dir example:
            checkpoints/experiment1/net_0_epoch_300.pth
        Or only directory:
            checkpoints/experiment1

        start_epoch: which epoch you want to load
        num_epochs_new: how many more epochs to train
        """
        print(f"⏪ Rolling back from epoch {start_epoch}...")

        for k, net in enumerate(self.multi_model.networks):
            ckpt_file = os.path.join(checkpoint_dir, f"net_{k}_epoch_{start_epoch}.pth")
            if not os.path.exists(ckpt_file):
                raise FileNotFoundError(f"❌ Missing checkpoint: {ckpt_file}")

            net.load_state_dict(torch.load(ckpt_file, map_location=self.device))
            print(f"✅ Net{k} loaded from {ckpt_file}")

        # ✅ Reset optimizer and scheduler (no burdened state)
        self.optimizers = [
            optim.AdamW(net.parameters(), lr=0.004) for net in self.multi_model.networks
        ]
        self.schedulers = [
            optim.lr_scheduler.CosineAnnealingLR(opt, T_max=800, eta_min=1e-5)
            for opt in self.optimizers
        ]

        print(f"🔄 Resuming training from epoch {start_epoch} → +{num_epochs_new} epochs\n")

        # ✅ Call original train with offset epoch
        return self.train(
            measured_img, proj_img_sys, pattern, extent_mm,
            num_epochs=num_epochs_new,
            **train_kwargs
        )

    def train(
        self,
        measured_img,
        proj_img_sys,
        pattern,
        extent_mm,
        num_epochs=1000,
        loss_fn=None,
        print_every=100,

        # smoothness
        use_height_smooth=False,
        lambda_height_smooth=0.001,
        use_reflectance_smooth=False,
        lambda_reflectance_smooth=0.001,
        smooth_type="tv",

        # mask
        msk_on=0,
        mask_xy=None,
        mask_warmup_ratio=0.15,
        mask_transition_mode="linear",

        # visualization
        show_every=None,

        # ===== snapshot related  =====
        snapshot_every=None,          # e.g. 100
        snapshot_root=None,           # e.g. output_root/folder/seed_xx
#         snapshot_data=None,           # dict: {"height_map":..., "n_pts":...}
    ):
        """
        Full training loop with periodic snapshot saving.
        """

        if loss_fn is None:
            loss_fn = nn.MSELoss().to(self.device)

        start_time = time.perf_counter()
        B, _, H, W = measured_img.shape

        # -------------------------------
        # Helper: save snapshot
        # -------------------------------
        def save_snapshot(epoch_1based, outputs, losses, save_model=False):
            if snapshot_every is None or snapshot_root is None:
                return
            if epoch_1based % snapshot_every != 0:
                return
        
            epoch_dir = os.path.join(snapshot_root, f"epoch_{epoch_1based:04d}")
            os.makedirs(epoch_dir, exist_ok=True)
        
            # 1) optional save weights
            if save_model:
                for k, net in enumerate(self.multi_model.networks):
                    torch.save(net.state_dict(), os.path.join(epoch_dir, f"net_{k}.pth"))
        
            # 2) save loss
            loss_dict = {f"net_{k}": float(losses[k].item()) for k in range(len(losses))}
            with open(os.path.join(epoch_dir, "loss.json"), "w") as f:
                json.dump(loss_dict, f, indent=2)
        
            # 3) save predictions (USE outputs directly, no re-forward)
            for k, (pred_h, pred_r) in enumerate(outputs):
                pred_h_np = pred_h.detach().squeeze().cpu().numpy()
                pred_r_np = pred_r.detach().squeeze().cpu().numpy()
        
                np.save(os.path.join(epoch_dir, f"net_{k}_pred_h.npy"), pred_h_np)
                np.save(os.path.join(epoch_dir, f"net_{k}_pred_r.npy"), pred_r_np)
        
                # ---------- Height PNG ----------
                h = pred_h_np
                h_img = ((h - h.min()) / (h.max() - h.min() + 1e-8) * 255).astype(np.uint8)

                h_bgr = cv2.cvtColor(h_img, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(epoch_dir, f"net_{k}_height.png"), h_bgr)
        
                # ---------- Reflectance PNG ----------
                r = pred_r_np
                if r.ndim == 3:
                    r_rgb = np.transpose(r, (1, 2, 0))
                    r_img = ((r_rgb - r_rgb.min()) / (r_rgb.max() - r_rgb.min() + 1e-8) * 255).astype(np.uint8)
                    # r_img = np.flipud(r_img)
                    r_bgr = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
                else:
                    r_img = ((r - r.min()) / (r.max() - r.min() + 1e-8) * 255).astype(np.uint8)
                    # r_img = np.flipud(r_img)
                    r_bgr = cv2.cvtColor(r_img, cv2.COLOR_GRAY2BGR)
        
                cv2.imwrite(os.path.join(epoch_dir, f"net_{k}_reflectance.png"), r_bgr)


        # =========================================================
        # =================== Training Loop =======================
        # =========================================================
        for epoch in range(num_epochs):

            progress_ratio = epoch / max(1, num_epochs - 1)

            outputs = self.multi_model(
                measured_img,
                msk_on=msk_on,
                mask_xy=mask_xy,
                epoch=epoch,
                num_epochs=num_epochs,
                mask_warmup_ratio=mask_warmup_ratio,
                mask_transition_mode=mask_transition_mode,
            )

            # ---- smooth mask ----
            if msk_on == 0:
                gaussian_mask = torch.ones((1, 1, H, W), device=self.device)
            else:
                warm_ratio = NetParallelTrain.compute_warm_ratio(
                    progress_ratio, mask_warmup_ratio, mask_transition_mode
                )
                gauss = self.generate_gaussian_mask(
                    H, W, mask_xy, falloff_sigma=5, device=self.device
                )
                gaussian_mask = (1 - warm_ratio) + warm_ratio * gauss

            total_loss = 0.0
            losses = []

            for k, (pred_h, pred_r) in enumerate(outputs):

                # ---- render RGB ----
                pred_imgs = []
                for c in range(pattern.shape[0]):
                    img_c = proj_img_sys.render(
                        pred_h.squeeze(),
                        pred_r[0, c],
                        pattern[c],
                        extent_mm=extent_mm,
                        shade_on=1,
                        fov_mask_on=1,
                        cam_res=measured_img.shape[-2:],
                        photon_count=None,
                        show=0
                    )
                    pred_imgs.append(img_c.squeeze(1))
                pred_meas = torch.stack(pred_imgs, dim=1)

                recon_loss = loss_fn(pred_meas, measured_img)
                total = recon_loss

                if use_height_smooth:
                    total += lambda_height_smooth * self.smoothness_loss(
                        pred_h * gaussian_mask, smooth_type
                    )

                if use_reflectance_smooth:
                    rs = []
                    for c in range(pred_r.shape[1]):
                        rs.append(self.smoothness_loss(
                            pred_r[:, c:c+1] * gaussian_mask, smooth_type
                        ))
                    total += lambda_reflectance_smooth * torch.mean(torch.stack(rs))

                losses.append(total)
                self.loss_lists[k].append(total.item())
                total_loss += total

            # ---- backprop ----
            for opt in self.optimizers:
                opt.zero_grad()
            total_loss.backward()
            for k, opt in enumerate(self.optimizers):
                opt.step()
                self.schedulers[k].step()

            # ---- snapshot ----
            save_snapshot(epoch , outputs, losses)

            # ---- log ----
            if epoch == 0 or epoch % print_every == 0:
                msg = f"[{epoch+1}/{num_epochs}] " + " | ".join(
                    [f"net{k}: {losses[k].item():.3e}" for k in range(self.n_nets)]
                )
                print(msg)
             # ===== Visualization =====
            if show_every is not None and epoch % show_every == 0:

                best_idx = int(torch.argmin(torch.tensor([l.item() for l in losses])))

                h, r = outputs[best_idx]
                h_np = h.squeeze().detach().cpu().numpy()
                r_np = r.squeeze().detach().cpu().numpy()

                plt.figure(figsize=(10,4))
                plt.suptitle(f"Visualization @ Epoch {epoch}", fontsize=14)

                # ===== Height =====
                plt.subplot(1,2,1)
                plt.imshow(h_np, cmap='gray')
                plt.title(f"Pred Height (Net {best_idx})")  
                plt.colorbar()

                # ===== Reflectance =====
                plt.subplot(1,2,2)
                if r_np.ndim == 3:
                    plt.imshow(np.clip(np.transpose(r_np, (1,2,0)), 0,1))
                else:
                    plt.imshow(r_np, cmap='gray')
                    plt.colorbar()
                plt.title(f"Pred Reflectance (Net {best_idx})")
                plt.show()

        print(f"Training finished in {time.perf_counter() - start_time:.1f}s")
        return self.multi_model, self.loss_lists

            
    def evaluate_net(self, idx, measured_img, height_map, reflectance=None, plt_on = False, n_pts=None):
        """
        evaluate sub net：
        - gray or RGB measured_img 
        - PSNR / SSIM 
        - pred / GT / diff map
        - cross section 
        - reflectance Gray or RGB
        """
        assert 0 <= idx < self.n_nets, f"idx must be in [0, {self.n_nets-1}]"
        net = self.multi_model.networks[idx].eval()

        with torch.no_grad():
            predicted_height, predicted_reflectance = net(measured_img)  # reflectance may be RGB

        # ========== PSNR/SSIM on Height Map ==========
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        predicted_height_tensor = predicted_height.to(self.device)  # [1,1,H,W]
        gt_height_tensor = height_map[None, None, :, :].to(self.device)  # [1,1,H,W]

#         psnr_value = psnr_metric(predicted_height_tensor, gt_height_tensor).item()
#         ssim_value = ssim_metric(predicted_height_tensor, gt_height_tensor).item()
        psnr_value = 0.0
        ssim_value = 0.0
        if plt_on:
            # ========== height ==========
            fig = plt.figure(figsize=(15, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(predicted_height.squeeze().detach().cpu(), cmap='gray')
            plt.title(f'Net{idx} Recovered\nPSNR: {psnr_value:.3f} dB, SSIM: {ssim_value:.3f}')
            plt.colorbar()

            plt.subplot(1, 3, 2)
            plt.imshow(height_map.cpu().numpy(), cmap='gray')
            plt.title('GT')
            plt.colorbar()

            plt.subplot(1, 3, 3)
            plt.imshow(
                predicted_height.squeeze().detach().cpu().numpy() - height_map.cpu().numpy(),
                cmap='bwr'
            )
            plt.title('Difference (Pred - GT)')
            plt.colorbar()
            plt.show()

            # ========== cross section ==========
            if n_pts is None:
                n_pts = height_map.shape[0]

            fig = plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(predicted_height.squeeze().detach().cpu()[n_pts // 2, :], label="Recovered")
            plt.plot(height_map.cpu().numpy()[n_pts // 2, :], label="GT")
            plt.title("Height Cross Section (middle row)")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(predicted_height.squeeze().detach().cpu()[:, n_pts // 2], label="Recovered")
            plt.plot(height_map.cpu().numpy()[:, n_pts // 2], label="GT")
            plt.title("Height Cross Section (middle column)")
            plt.legend()
            plt.show()

            # ========== Reflectance（RGB ot Gray） ==========
            if reflectance is not None:
                pred_reflectance = predicted_reflectance.detach().cpu().permute(0,2,3,1).squeeze()
                gt_reflectance = reflectance.cpu().permute(1,2,0).squeeze()

                # RGB or not
                if pred_reflectance.ndim == 3 and pred_reflectance.shape[0] in [1, 3]:
                    # convert to H×W×C 
                    pred_reflectance_np = pred_reflectance.permute(1, 2, 0).numpy()
                    gt_reflectance_np = gt_reflectance.permute(1, 2, 0).numpy() if gt_reflectance.ndim == 3 else gt_reflectance.numpy()

                    fig = plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.clip(pred_reflectance_np, 0, 1))
                    plt.title(f'Pred Reflectance (RGB)\nmean={pred_reflectance.mean().item():.5f}')

                    plt.subplot(1, 2, 2)
                    plt.imshow(np.clip(gt_reflectance_np, 0, 1))
                    plt.title('GT Reflectance (RGB)')
                    plt.show()

                else:
                    # gray reflectance
                    fig = plt.figure(figsize=(12, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(pred_reflectance.squeeze().numpy(), cmap='gray')
                    plt.title(f'Pred Reflectance\nmean={pred_reflectance.mean().item():.5f}')
                    plt.colorbar()

                    plt.subplot(1, 2, 2)
                    plt.imshow(gt_reflectance.numpy(), cmap='gray')
                    plt.title('GT Reflectance')
                    plt.colorbar()
                    plt.show()

        return predicted_height.squeeze().detach().cpu(), predicted_reflectance.squeeze().detach().cpu(), psnr_value, ssim_value



    # ===============================================================
    def compare_all_nets(self, measured_img, height_map):
        """
        compare all net：
        - measured_img support RGB or gray
        -  PSNR / SSIM（height_map）
        """
        results = []
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        for idx, net in enumerate(self.multi_model.networks):
            net.eval()
            with torch.no_grad():
                predicted_height, _ = net(measured_img)

            predicted_height_tensor = predicted_height.to(self.device)  # [1,1,H,W]
            gt_height_tensor = height_map[None, None, :, :].to(self.device)  # [1,1,H,W]

#             psnr_value = psnr_metric(predicted_height_tensor, gt_height_tensor).item()
#             ssim_value = ssim_metric(predicted_height_tensor, gt_height_tensor).item()
            psnr_value = 0.0
            ssim_value = 0.0
            results.append((idx, psnr_value, ssim_value))

        results_sorted = sorted(results, key=lambda x: (x[1], x[2]), reverse=True)

        print("=== Net Performance Ranking (by PSNR, SSIM) ===")
        for idx, psnr, ssim in results_sorted:
            print(f"Net{idx}: PSNR={psnr:.3f} dB | SSIM={ssim:.3f}")

        return results_sorted

