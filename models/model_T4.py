# DOCUMENT RECONSTRUCTION MODEL

# Full detailed class containing different models for testing and experimentation
# Once final selection completed, shift to <model.py> file

""" 
OVERVIEW:
Implementing a pretrained ResNet backbone 

"""


import os
import glob
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import timm


class ResNetUnet(nn.Module):
    """
    Resnet pretrained backbone with adjusted decoder
    """

    def __init__(self, backbone_name='resnet34', pretrained=True):
        super().__init__()

        # NEW MODEL: RESNET ENCODER

        # Encoder
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)

        self.enc_channels = self.backbone.feature_info.channels()


        # Decoder 
        # e4 -> up3 (upsample from 512 -> 256) + skip e3 (256) => input to dec3 = 256 + 256 = 512
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        # e3 -> up2 (256 -> 128) + skip e2 (128) => input to dec2 = 128 + 128 = 256
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        # e2 -> up1 (128 -> 64) + skip e1 (64) => input to dec1 = 64 + 64 = 128
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Optional final conv for flow/UV prediction from e0 (64 channels)
        self.flow_predictor = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.uv_predictor = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        

        # self.up3 = nn.ConvTranspose2d(self.enc_channels[3], self.enc_channels[2], kernel_size=2, stride=2)
        # self.dec3 = self.conv_block(self.enc_channels[3]+self.enc_channels[2], self.enc_channels[2])

        # self.up2 = nn.ConvTranspose2d(self.enc_channels[2], self.enc_channels[1], kernel_size=2, stride=2)
        # self.dec2 = self.conv_block(self.enc_channels[2]+self.enc_channels[1], self.enc_channels[1])

        # self.up1 = nn.ConvTranspose2d(self.enc_channels[1], self.enc_channels[0], kernel_size=2, stride=2)
        # self.dec1 = self.conv_block(self.enc_channels[1]+self.enc_channels[0], self.enc_channels[0])

        # # Final conv to match input channels for flow prediction
        # self.flow_predictor = nn.Conv2d(self.enc_channels[0], 2, kernel_size=3, padding=1)
        # # OPTIONAL: UV PREDICTOR
        # self.uv_predictor = nn.Conv2d(self.enc_channels[0], 2, kernel_size=3, padding=1)


    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(
        self, 
        x: torch.Tensor,
        predict_uv: bool = False
    ) -> Dict[str, torch.Tensor]:

        """
        Returns dict with warped, flow, uv
        """

        B, C, H, W = x.shape
        device = x.device

        # Encoder
        features = self.backbone(x)
        # print("LENGTH OF FEATURES",len(features))
        # for i, f in enumerate(features):
        #     print(f"e{i} shape: {f.shape}")
        e0, e1, e2, e3, e4 = features

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Predict flow field
        flow = self.flow_predictor(d1)  # [B, 2, H, W]
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)

        # Constrain flow magnitude
        flow = torch.tanh(flow) * 0.3


        # Warp input
        grid = create_base_grid(B, H, W, device)  # [B, H, W, 2]
        sampling_grid = grid + flow.permute(0, 2, 3, 1)
        warped = F.grid_sample(x, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

        # uv = self.uv_predictor(d1) if predict_uv else None
        uv = None
        if predict_uv:
            uv = self.uv_predictor(d1)
            uv = F.interpolate(uv, size=(H, W), mode='bilinear', align_corners=True)

        return {
            'warped': warped,
            'flow': flow,
            'uv': uv
        }



def create_base_grid(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Helper function to create a base sampling grid for grid_sample.

    Creates a normalized coordinate grid in the range [-1, 1] as expected by grid_sample.

    Args:
        batch_size: Batch size
        height: Image height
        width: Image width
        device: Device to create tensor on

    Returns:
        Grid tensor of shape [B, H, W, 2] with normalized coordinates

    Usage example:
        # Create base grid
        grid = create_base_grid(batch_size, H, W, device)

        # Predict flow field [B, 2, H, W]
        flow = model.predict_flow(features)

        # Add flow to grid (need to permute flow to [B, H, W, 2])
        sampling_grid = grid + flow.permute(0, 2, 3, 1)

        # Warp image
        warped = F.grid_sample(input, sampling_grid, align_corners=True)
    """
    # Create coordinate grids
    y_coords = torch.linspace(-1, 1, height, device=device)
    x_coords = torch.linspace(-1, 1, width, device=device)

    # Create meshgrid
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Stack to create [H, W, 2] grid
    grid = torch.stack([xx, yy], dim=-1)

    # Expand to batch size [B, H, W, 2]
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return grid




# ============================================================================
# LOSS FUNCTIONS FOR DOCUMENT RECONSTRUCTION
# ============================================================================
#
# This section provides loss functions specifically designed for document
# reconstruction. The key insight is that backgrounds should be ignored during
# training, allowing the network to focus on the document surface.
#
# Available loss functions:
# 1. MaskedL1Loss - L1 loss with optional document masking
# 2. MaskedMSELoss - MSE loss with optional document masking
# 3. UVReconstructionLoss - Combined loss for UV-based reconstruction
#
# Usage:
#   criterion = MaskedL1Loss(use_mask=True)
#   loss = criterion(prediction, ground_truth, mask)
#
# For UV-based models that predict flow fields:
#   criterion = UVReconstructionLoss(
#       reconstruction_weight=1.0,
#       uv_weight=0.5,
#       smoothness_weight=0.01,
#       use_mask=True
#   )
# ============================================================================

class MaskedL1Loss(nn.Module):
    """
    L1 Loss with optional masking to focus on document regions.

    If a border mask is provided, this loss will only compute the error
    on pixels where the document exists, ignoring the background.

    This is useful because:
    - The background is not part of the reconstruction task
    - Focusing on document pixels improves convergence
    - Prevents the model from "cheating" by predicting background

    Args:
        use_mask: Whether to apply masking (requires 'border' in batch)
    """

    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.use_mask = use_mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute masked L1 loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            mask: Optional mask [B, 1, H, W] where 1=document, 0=background

        Returns:
            Scalar loss value
        """
        # Compute element-wise L1 distance
        l1_loss = torch.abs(pred - target)

        if self.use_mask and mask is not None:
            # Apply mask (broadcast across channels)
            l1_loss = l1_loss * mask

            # Average only over masked pixels
            # This prevents background from contributing to loss
            num_pixels = mask.sum() * pred.shape[1]  # Total masked pixels across channels
            if num_pixels > 0:
                return l1_loss.sum() / num_pixels
            else:
                return l1_loss.mean()  # Fallback if mask is empty
        else:
            # Standard L1 loss
            return l1_loss.mean()


class MaskedMSELoss(nn.Module):
    """
    MSE Loss with optional masking to focus on document regions.

    Similar to MaskedL1Loss but uses squared error (L2).

    Args:
        use_mask: Whether to apply masking (requires 'border' in batch)
    """

    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.use_mask = use_mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute masked MSE loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            mask: Optional mask [B, 1, H, W] where 1=document, 0=background

        Returns:
            Scalar loss value
        """
        # Compute element-wise squared error
        mse_loss = (pred - target) ** 2

        if self.use_mask and mask is not None:
            # Apply mask
            mse_loss = mse_loss * mask

            # Average only over masked pixels
            num_pixels = mask.sum() * pred.shape[1]
            if num_pixels > 0:
                return mse_loss.sum() / num_pixels
            else:
                return mse_loss.mean()
        else:
            # Standard MSE loss
            return mse_loss.mean()


class SSIMLoss(nn.Module):
    """
    SSIM (Structural Similarity) Loss - RECOMMENDED for geometric reconstruction.

    SSIM measures structural similarity rather than pixel-wise differences, making it
    ideal for this task where lighting effects differ between input and ground truth.

    Benefits:
    - Focuses on structure (edges, patterns) not pixel intensities
    - Robust to lighting/shading differences
    - More perceptually aligned than L1/L2

    Args:
        data_range: Expected range of input values (1.0 for normalized images)
        channel: Number of channels (3 for RGB)

    Requires: pip install pytorch-msssim
    """

    def __init__(self, data_range: float = 1.0, channel: int = 3):
        super().__init__()
        try:
            from pytorch_msssim import ssim
            self.ssim_func = ssim
        except ImportError:
            raise ImportError(
                "pytorch-msssim not installed. Install with: pip install pytorch-msssim"
            )
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]

        Returns:
            Scalar loss value (lower is better, range 0-1)
        """
        # SSIM returns value in range [0, 1] where 1 is perfect
        # Convert to loss by: loss = 1 - SSIM
        ssim_val = self.ssim_func(pred, target, data_range=self.data_range)
        return 1 - ssim_val


class UVReconstructionLoss(nn.Module):
    """
    Combined loss for UV-based document reconstruction.

    This loss combines:
    1. Reconstruction loss (L1, MSE, or SSIM) on the final image
    2. Optional UV map supervision (if your model predicts UV explicitly)
    3. Optional flow smoothness regularization

    Args:
        reconstruction_weight: Weight for image reconstruction loss
        uv_weight: Weight for UV map loss (set to 0 if not predicting UV)
        smoothness_weight: Weight for flow smoothness regularization
        use_mask: Whether to use masking
        loss_type: Type of loss ('l1', 'mse', or 'ssim')
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        uv_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        use_mask: bool = False,
        loss_type: str = 'ssim'  # 'l1', 'mse', or 'ssim' (recommended!)
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.uv_weight = uv_weight
        self.smoothness_weight = smoothness_weight

        # Choose base loss function
        if loss_type == 'ssim':
            self.recon_loss = SSIMLoss()
            self.use_ssim = True
        elif loss_type == 'l1':
            self.recon_loss = MaskedL1Loss(use_mask=use_mask)
            self.use_ssim = False
        else:  # mse
            self.recon_loss = MaskedMSELoss(use_mask=use_mask)
            self.use_ssim = False

    def forward(
        self,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        pred_uv: Optional[torch.Tensor] = None,
        target_uv: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred_image: Predicted reconstructed image [B, 3, H, W]
            target_image: Ground truth image [B, 3, H, W]
            pred_uv: Predicted UV map [B, 2, H, W] (optional)
            target_uv: Ground truth UV map [B, 2, H, W] (optional)
            flow: Predicted flow field [B, 2, H, W] (optional, for smoothness)
            mask: Document mask [B, 1, H, W] (optional)

        Returns:
            Dictionary with 'total' loss and individual loss components
        """
        losses = {}

        device = pred_image.device
        target_image = target_image.to(device)
        if pred_uv is not None and target_uv is not None:
            target_uv = target_uv.to(pred_uv.device)
        if flow is not None:
            flow = flow.to(device)
        if mask is not None:
            mask = mask.to(device)

        # Slice target UV channels if they don't match pred_uv
        if pred_uv is not None and target_uv is not None:
            if target_uv.shape[1] != pred_uv.shape[1]:
                target_uv = target_uv[:, :pred_uv.shape[1], :, :]

            
        # 1. Reconstruction loss
        # Apply SSIM with a mask
        if self.use_ssim:
            if mask is not None:
                pred = pred_image * mask
                tgt = target_image * mask
            else:
                pred = pred_image
                tgt = target_image

            losses['reconstruction'] = self.recon_loss(pred, tgt)
        else:
            losses['reconstruction'] = self.recon_loss(pred_image, target_image, mask)
        
        total_loss = self.reconstruction_weight * losses['reconstruction']

        # 2. UV supervision loss (if applicable)
        if self.uv_weight > 0 and pred_uv is not None and target_uv is not None:

            uv_diff = torch.abs(pred_uv - target_uv)
            if mask is not None:
                uv_diff = uv_diff * mask 
            losses['uv'] = uv_diff.mean()
            total_loss += self.uv_weight * losses['uv']

        # 3. Flow smoothness regularization (Total Variation)
        if self.smoothness_weight > 0 and flow is not None:
            # Compute gradients
            dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
            dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
            losses['smoothness'] = (dx.abs().mean() + dy.abs().mean())
            total_loss += self.smoothness_weight * losses['smoothness']

        losses['total'] = total_loss
        return losses
