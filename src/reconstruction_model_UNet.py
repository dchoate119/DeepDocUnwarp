# DOCUMENT RECONSTRUCTION MODEL

# Full detailed class containing different models for testing and experimentation
# Once final selection completed, shift to <model.py> file

""" 
OVERVIEW:

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


class DocumentReconstructionModel(nn.Module):
    """
    Starter model for document dewarping (geometric correction).

    IMPORTANT: The goal is GEOMETRIC RECONSTRUCTION, not photometric matching!
    - The rendered images have lighting/shading effects
    - Your model should focus on learning the geometric transformation (UV/flow field)
    - Don't worry about exact pixel intensities - focus on structure

    TODO: Implement your own architecture here.
    This is a simple U-Net-style baseline to get started.

    Suggestions for improvement:
    - Use a pretrained encoder from HuggingFace (e.g., ResNet, EfficientNet)
    - Add attention mechanisms
    - Use depth/UV information if available
    - Experiment with different loss functions (SSIM is recommended!)
    - Add skip connections
    - Try different decoder architectures

    IMPORTANT HINT: Consider using torch.nn.functional.grid_sample for differentiable warping!

    One powerful approach for document reconstruction is to:
    1. Predict a deformation/flow field (mapping from distorted space to flat space)
    2. Use grid_sample to warp the input image according to this field
    3. This allows the network to learn geometric transformations explicitly

    Example usage of grid_sample:
        # Predict a flow field [B, 2, H, W] representing (x, y) offsets
        flow = self.flow_predictor(features)

        # Create base grid and add flow to get sampling coordinates
        grid = create_base_grid(B, H, W) + flow

        # Sample from input image using the predicted grid
        warped = torch.nn.functional.grid_sample(
            input_image,
            grid.permute(0, 2, 3, 1),  # [B, H, W, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # TODO: Replace this simple architecture with your own design
        # Consider using HuggingFace transformers or timm models as backbone

        # NEW MODEL: U-Net with skip connections, flow predictor, differentiable warping

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels*2)
        self.enc3 = self.conv_block(base_channels*2, base_channels*4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(base_channels*4, base_channels*8)

        # Decoder 
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = self.conv_block(base_channels*8, base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = self.conv_block(base_channels*4, base_channels*2)

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = self.conv_block(base_channels*2, base_channels)

        # Flow predictor: predicts (x, y) offsets for each pixel
        self.flow_predictor = nn.Conv2d(base_channels, 2, kernel_size=3, padding=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Predict flow field
        flow = self.flow_predictor(d1)  # [B, 2, H, W]

        # Warp input
        grid = create_base_grid(B, H, W, device)  # [B, H, W, 2]
        sampling_grid = grid + flow.permute(0, 2, 3, 1)
        warped = F.grid_sample(x, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

        return warped, flow



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

        # 1. Reconstruction loss
        if self.use_ssim:
            # SSIM doesn't use mask (operates on full image structure)
            losses['reconstruction'] = self.recon_loss(pred_image, target_image)
        else:
            losses['reconstruction'] = self.recon_loss(pred_image, target_image, mask)
        total_loss = self.reconstruction_weight * losses['reconstruction']

        # 2. UV supervision loss (if applicable)
        if self.uv_weight > 0 and pred_uv is not None and target_uv is not None:
            if self.use_ssim:
                losses['uv'] = self.recon_loss(pred_uv, target_uv)
            else:
                losses['uv'] = self.recon_loss(pred_uv, target_uv, mask)
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

