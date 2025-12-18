# Starter code provided by Roy Shilkrot

# Computer Vision Final Project 
# Daniel Choate: Tufts University


"""
Document Reconstruction Dataset Loader
=======================================

This script provides a PyTorch dataset loader for the synthetic document dataset.
Focus on implementing the reconstruction model and training loop.

"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


class DocumentDataset(Dataset):
    """
    PyTorch Dataset for document reconstruction.

    Loads RGB images of warped documents and their corresponding ground truth
    flat textures. Optionally loads depth maps, UV maps, and border masks.

    Args:
        data_dir: Root directory containing the dataset
        use_depth: Whether to load depth maps
        use_uv: Whether to load UV maps
        use_border: Whether to load border masks
        transform: Optional transform to apply to images
        img_size: Tuple of (height, width) to resize images to
    """

    def __init__(
        self,
        data_dir: str,
        use_depth: bool = False,
        use_uv: bool = False,
        use_border: bool = False,
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = (512, 512)
    ):
        self.data_dir = Path(data_dir)
        self.use_depth = use_depth
        self.use_uv = use_uv
        self.use_border = use_border
        self.transform = transform
        self.img_size = img_size

        # Find all RGB images
        self.rgb_dir = self.data_dir / 'rgb'
        self.gt_dir = self.data_dir / 'ground_truth'
        self.depth_dir = self.data_dir / 'depth'
        self.uv_dir = self.data_dir / 'uv'
        self.border_dir = self.data_dir / 'border'

        if not self.rgb_dir.exists():
            raise ValueError(f"RGB directory not found: {self.rgb_dir}")
        if not self.gt_dir.exists():
            raise ValueError(f"Ground truth directory not found: {self.gt_dir}")

        # Get list of all samples (based on RGB images)
        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.rgb_dir}")

        print(f"Found {len(self.samples)} samples in {self.data_dir}")

        # Define default transforms if none provided
        if self.transform is None:
            self.transform = self._get_default_transform()

    def _find_samples(self) -> List[str]:
        """Find all valid samples (those with both RGB and ground truth)."""
        samples = []

        # Find all RGB images
        rgb_files = sorted(glob.glob(str(self.rgb_dir / "*.jpg")))

        for rgb_path in rgb_files:
            # Extract base filename (without extension)
            base_name = Path(rgb_path).stem

            # Check if ground truth exists
            gt_path = self.gt_dir / f"{base_name}.png"
            if gt_path.exists():
                samples.append(base_name)

        return samples

    def _get_default_transform(self):
        """Get default image transforms."""
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_rgb(self, base_name: str) -> Image.Image:
        """Load RGB image."""
        rgb_path = self.rgb_dir / f"{base_name}.jpg"
        return Image.open(rgb_path).convert('RGB')

    def _load_ground_truth(self, base_name: str) -> Image.Image:
        """Load ground truth image."""
        gt_path = self.gt_dir / f"{base_name}.png"
        return Image.open(gt_path).convert('RGB')

    def _load_depth(self, base_name: str) -> Optional[np.ndarray]:
        """Load depth map (EXR format)."""
        if not self.use_depth:
            return None

        depth_path = self.depth_dir / f"{base_name}.exr"
        if not depth_path.exists():
            return None

        try:
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(str(depth_path))
            dw = exr_file.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

            depth_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
            depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])

            return depth
        except ImportError:
            print("Warning: OpenEXR not installed. Install with: pip install OpenEXR")
            return None

    def _load_uv(self, base_name: str) -> Optional[Image.Image]:
        """Load UV map."""
        if not self.use_uv:
            return None

        uv_path = self.uv_dir / f"{base_name}.png"
        if not uv_path.exists():
            return None

        return Image.open(uv_path).convert('RGB')

    def _load_border(self, base_name: str) -> Optional[Image.Image]:
        """Load border mask."""
        if not self.use_border:
            return None

        border_path = self.border_dir / f"{base_name}.png"
        if not border_path.exists():
            return None

        return Image.open(border_path).convert('L')  # Grayscale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - 'rgb': Input warped document image [3, H, W]
                - 'ground_truth': Target flat texture [3, H, W]
                - 'depth': Depth map [1, H, W] (if use_depth=True)
                - 'uv': UV map [3, H, W] (if use_uv=True)
                - 'border': Border mask [1, H, W] (if use_border=True)
                - 'filename': Base filename (string)
        """
        base_name = self.samples[idx]

        # Load RGB and ground truth (required)
        rgb = self._load_rgb(base_name)
        ground_truth = self._load_ground_truth(base_name)

        # Apply transforms
        rgb_tensor = self.transform(rgb)
        gt_tensor = self.transform(ground_truth)

        # Build output dictionary
        sample = {
            'rgb': rgb_tensor,
            'ground_truth': gt_tensor,
            'filename': base_name
        }

        # Load optional modalities
        if self.use_depth:
            depth = self._load_depth(base_name)
            if depth is not None:
                # Normalize depth and convert to tensor
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth = torch.from_numpy(depth).unsqueeze(0).float()
                # Resize to match img_size
                depth = transforms.Resize(self.img_size)(depth)
                sample['depth'] = depth

        if self.use_uv:
            uv = self._load_uv(base_name)
            if uv is not None:
                uv_tensor = self.transform(uv)
                sample['uv'] = uv_tensor

        if self.use_border:
            border = self._load_border(base_name)
            if border is not None:
                border_tensor = transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.ToTensor()
                ])(border)
                sample['border'] = border_tensor

        return sample


def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    train_split: float = 0.8,
    use_depth: bool = False,
    use_uv: bool = False,
    use_border: bool = False,
    img_size: Tuple[int, int] = (512, 512),
    num_workers: int = 4, # CHANGED FROM 4 to 8
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Root directory containing the dataset
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training (0.0 to 1.0)
        use_depth: Whether to load depth maps
        use_uv: Whether to load UV maps
        use_border: Whether to load border masks
        img_size: Tuple of (height, width) to resize images to
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle training data
        random_seed: Random seed for reproducible splits

    Returns:
        (train_loader, val_loader): Tuple of DataLoader objects
    """
    # Create dataset
    dataset = DocumentDataset(
        data_dir=data_dir,
        use_depth=use_depth,
        use_uv=use_uv,
        use_border=use_border,
        img_size=img_size
    )

    # Split into train and validation
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    # Use random_split with generator for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def visualize_batch(batch: Dict[str, torch.Tensor], num_samples: int = 4):
    """
    Visualize a batch of samples.

    Args:
        batch: Dictionary containing batch data
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt

    num_samples = min(num_samples, batch['rgb'].shape[0])

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Denormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        rgb = batch['rgb'][i] * std + mean
        gt = batch['ground_truth'][i] * std + mean

        # Convert to numpy and transpose to HWC
        rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
        gt_np = gt.permute(1, 2, 0).cpu().numpy()

        # Clip values to [0, 1]
        rgb_np = np.clip(rgb_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)

        # Plot
        axes[i, 0].imshow(rgb_np)
        axes[i, 0].set_title(f"Input RGB - {batch['filename'][i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_np)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_batch_pred(batch: Dict[str, torch.Tensor], num_samples: int = 4):
    """
    Visualize a batch of input RGB, ground truth, and predicted output.

    Args:
        batch: Dictionary with keys 'rgb', 'ground_truth', 'predicted'
        num_samples: Number of samples to visualize
    """

    import matplotlib.pyplot as plt


    num_samples = min(num_samples, batch['rgb'].shape[0])

    # Always 3 columns: RGB, GT, Pred
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    # Ensure axes is 2D array
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)  # shape (1,3)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    for i in range(num_samples):
        rgb = batch['rgb'][i] * std + mean
        gt = batch['ground_truth'][i] * std + mean
        pred = batch['predicted'][i] * std + mean


        rgb_np = np.clip(rgb.permute(1,2,0).cpu().numpy(), 0, 1)
        gt_np = np.clip(gt.permute(1,2,0).cpu().numpy(), 0, 1)
        pred_np = np.clip(pred.permute(1,2,0).cpu().numpy(), 0, 1)

        axes[i,0].imshow(rgb_np)
        axes[i,0].set_title("Input RGB")
        axes[i,0].axis('off')

        axes[i,1].imshow(gt_np)
        axes[i,1].set_title("Ground Truth")
        axes[i,1].axis('off')

        axes[i,2].imshow(pred_np)
        axes[i,2].set_title("Predicted")
        axes[i,2].axis('off')


    plt.tight_layout()
    plt.show()


def visualize_uv_flow(
    uv_pred: torch.Tensor,
    uv_gt: Optional[torch.Tensor] = None,
    num_samples: int = 4,
    title_pred: str = "Predicted UV Flow",
    title_gt: str = "Ground Truth UV"
):
    """
    Visualize predicted UV maps (and optionally ground truth UV) as optical-flow-style color fields.

    Args:
        uv_pred: [B,2,H,W] predicted UV tensor
        uv_gt: [B,2,H,W] optional ground truth UV tensor
        num_samples: number of samples to visualize
        title_pred: title for predicted UV
        title_gt: title for GT UV
    """

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    num_samples = min(num_samples, uv_pred.shape[0])
    num_cols = 2 if uv_gt is not None else 1
    fig, axes = plt.subplots(1, num_samples * num_cols, figsize=(5 * num_samples * num_cols, 5))
    if num_samples * num_cols == 1:
        axes = [axes]
    elif num_samples == 1 and uv_gt is not None:
        axes = [axes[0], axes[1]]

    def uv_to_rgb(uv: torch.Tensor):
        u = uv[0]
        v = uv[1]
        mag = torch.sqrt(u ** 2 + v ** 2)
        mag = mag / (mag.max() + 1e-8)
        ang = torch.atan2(v, u)
        hsv = torch.zeros(3, *u.shape, device=uv.device)
        hsv[0] = (ang + torch.pi) / (2 * torch.pi)
        hsv[1] = 1.0
        hsv[2] = mag
        hsv_np = hsv.permute(1, 2, 0).cpu().numpy()
        return mcolors.hsv_to_rgb(hsv_np)

    for i in range(num_samples):
        axes[i * num_cols].imshow(uv_to_rgb(uv_pred[i]))
        axes[i * num_cols].set_title(f"{title_pred} #{i}")
        axes[i * num_cols].axis("off")

        if uv_gt is not None:
            axes[i * num_cols + 1].imshow(uv_to_rgb(uv_gt[i]))
            axes[i * num_cols + 1].set_title(f"{title_gt} #{i}")
            axes[i * num_cols + 1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_flow_and_warp(model, batch, device, num_samples: int = 4):
    """
    Visualize input, GT, warped output, and flow magnitude for a batch.
    """
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        rgb = batch['rgb'].to(device)
        outputs = model(rgb, predict_uv=False)
        warped = outputs['warped']          # [B, 3, H, W]
        flow = outputs['flow']              # [B, 2, H, W]

    num_samples = min(num_samples, rgb.shape[0])

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    fig, axes = plt.subplots(num_samples, 4, figsize=(18, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, 4)

    for i in range(num_samples):
        # Denormalize
        rgb_i = rgb[i] * std + mean
        gt_i = batch['ground_truth'][i].to(device) * std + mean
        warped_i = warped[i] * std + mean

        rgb_np    = torch.clamp(rgb_i, 0, 1).permute(1, 2, 0).cpu().numpy()
        gt_np     = torch.clamp(gt_i, 0, 1).permute(1, 2, 0).cpu().numpy()
        warped_np = torch.clamp(warped_i, 0, 1).permute(1, 2, 0).cpu().numpy()

        # Flow magnitude
        flow_i = flow[i]                     # [2, H, W]
        mag = torch.norm(flow_i, dim=0)      # [H, W] in roughly [0, 0.3*sqrt(2)]
        mag_np = mag.cpu().numpy()

        # Plots
        axes[i, 0].imshow(rgb_np)
        axes[i, 0].set_title(f"Input RGB - {batch['filename'][i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_np)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(warped_np)
        axes[i, 2].set_title("Warped (Predicted)")
        axes[i, 2].axis('off')

        im = axes[i, 3].imshow(mag_np, cmap='viridis')
        axes[i, 3].set_title("Flow magnitude")
        axes[i, 3].axis('off')
        fig.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


