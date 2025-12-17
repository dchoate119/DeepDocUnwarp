"""
Document Reconstruction Dataset Loader
=======================================

This script provides a PyTorch dataset loader for the synthetic document dataset.
Focus on implementing the reconstruction model and training loop.

IMPORTANT: FOCUS ON GEOMETRIC RECONSTRUCTION, NOT PHOTOMETRIC MATCHING!
-----------------------------------------------------------------------
The rendered images include realistic lighting effects (shadows, shading, specular
highlights). Even with perfect geometric dewarping, the lighting will NOT match
the flat ground truth. This is expected and acceptable!

Your goal: Learn the geometric transformation (UV mapping / flow field)
NOT your goal: Match pixel intensities exactly (lighting is different!)

Recommended approach:
- Use SSIMLoss (focuses on structure, robust to lighting)
- Predict flow fields with grid_sample (explicit geometric reasoning)
- Evaluate with SSIM (structure) not just PSNR (pixels)
- Use border masks to focus on document geometry

Dataset Structure:
    renders/synthetic_data_pitch_sweep/
        ├── rgb/           # Input images (warped documents with backgrounds)
        ├── ground_truth/  # Target images (flat paper - DIFFERENT LIGHTING!)
        ├── depth/         # Depth maps (optional, for advanced methods)
        ├── uv/           # UV maps (optional, for advanced methods)
        └── border/       # Border masks (optional, for advanced methods)

Usage Example:
    from dataset_loader import DocumentDataset, get_dataloaders

    # Quick start - get train and val dataloaders
    train_loader, val_loader = get_dataloaders(
        data_dir='renders/synthetic_data_pitch_sweep',
        batch_size=8,
        train_split=0.8
    )

    # Or create custom dataset
    dataset = DocumentDataset(
        data_dir='renders/synthetic_data_pitch_sweep',
        use_depth=True,
        use_uv=False,
        transform=None
    )
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




# ********************** TRAINING AND MAIN LOOP ****************************

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    Train for one epoch.

    TODO: Modify this to add:
    - Additional metrics (PSNR, SSIM)
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training
    - Logging to tensorboard/wandb
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        rgb = batch['rgb'].to(device)
        ground_truth = batch['ground_truth'].to(device)

        # Optional: Load mask if using masked loss
        mask = batch.get('border', None)
        if mask is not None:
            mask = mask.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(rgb)

        # Compute loss (handles both standard and masked losses)
        if isinstance(criterion, (MaskedL1Loss, MaskedMSELoss)):
            loss = criterion(output, ground_truth, mask)
        elif isinstance(criterion, SSIMLoss):
            loss = criterion(output, ground_truth)
        elif isinstance(criterion, UVReconstructionLoss):
            # For advanced UV-based losses, extract additional outputs if available
            # This assumes your model returns (image, uv, flow) - adapt as needed
            losses = criterion(pred_image=output, target_image=ground_truth, mask=mask)
            loss = losses['total']
        else:
            # Standard loss (MSE, L1, etc.)
            loss = criterion(output, ground_truth)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate the model.

    TODO: Modify this to add more metrics here (PSNR, SSIM, etc.)
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch['rgb'].to(device)
            ground_truth = batch['ground_truth'].to(device)

            # Optional: Load mask if using masked loss
            mask = batch.get('border', None)
            if mask is not None:
                mask = mask.to(device)

            output = model(rgb)

            # Compute loss (handles both standard and masked losses)
            if isinstance(criterion, (MaskedL1Loss, MaskedMSELoss)):
                loss = criterion(output, ground_truth, mask)
            elif isinstance(criterion, SSIMLoss):
                loss = criterion(output, ground_truth)
            elif isinstance(criterion, UVReconstructionLoss):
                losses = criterion(pred_image=output, target_image=ground_truth, mask=mask)
                loss = losses['total']
            else:
                loss = criterion(output, ground_truth)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# def main():
#     """
#     Main training loop - STARTER CODE

#     TODO: Modify this to customize for your experiments:
#     1. Implement a better model architecture
#     2. Try different loss functions
#     3. Add learning rate scheduling
#     4. Implement early stopping
#     5. Add visualization and logging
#     6. Experiment with data augmentation
#     7. Use pretrained models from HuggingFace
#     """

#     # Configuration
#     DATA_DIR = 'renders/synthetic_data_pitch_sweep'
#     BATCH_SIZE = 8
#     NUM_EPOCHS = 50
#     LEARNING_RATE = 1e-4
#     IMG_SIZE = (512, 512)

#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Create dataloaders
#     # IMPORTANT: Set use_border=True to enable masked losses!
#     train_loader, val_loader = get_dataloaders(
#         data_dir=DATA_DIR,
#         batch_size=BATCH_SIZE,
#         img_size=IMG_SIZE,
#         use_depth=False,  # TODO: Set to True if you want to use depth information
#         use_uv=False,     # TODO: Set to True if you want to use UV maps
#         use_border=False  # TODO: Set to True if you want to use border masks for better training
#     )

#     # Visualize a batch (optional)
#     sample_batch = next(iter(train_loader))
#     print(f"Batch RGB shape: {sample_batch['rgb'].shape}")
#     print(f"Batch GT shape: {sample_batch['ground_truth'].shape}")
#     if 'border' in sample_batch:
#         print(f"Batch Border mask shape: {sample_batch['border'].shape}")
#     # visualize_batch(sample_batch)  # Uncomment to visualize

#     # Create model
#     model = DocumentReconstructionModel().to(device)
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

#     # TODO: Try different loss functions
#     # Option 1: Simple losses (baseline, not recommended)
#     criterion = nn.MSELoss()  # Simple L2 loss - sensitive to lighting!
#     # criterion = nn.L1Loss()  # Try L1 loss - also sensitive to lighting

#     # Option 2: SSIM Loss (RECOMMENDED - focuses on structure, not lighting!)
#     # Uncomment this line (requires: pip install pytorch-msssim)
#     # criterion = SSIMLoss()

#     # Option 3: Masked losses (focuses on document pixels)
#     # Uncomment these lines and set use_border=True above
#     # criterion = MaskedL1Loss(use_mask=True)
#     # criterion = MaskedMSELoss(use_mask=True)

#     # Option 4: Combined loss with UV supervision (ADVANCED)
#     # Uncomment and set use_uv=True, use_border=True above
#     # criterion = UVReconstructionLoss(
#     #     reconstruction_weight=1.0,
#     #     uv_weight=0.5,
#     #     smoothness_weight=0.01,
#     #     use_mask=True,
#     #     loss_type='ssim'  # Use SSIM for geometric reconstruction!
#     # )

#     # TODO: Try different optimizers
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

#     # Training loop
#     best_val_loss = float('inf')

#     for epoch in range(NUM_EPOCHS):
#         print(f"\n{'='*50}")
#         print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
#         print(f"{'='*50}")

#         # Train
#         train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
#         print(f"Train Loss: {train_loss:.4f}")

#         # Validate
#         val_loss = validate(model, val_loader, criterion, device)
#         print(f"Val Loss: {val_loss:.4f}")

#         # Save best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), 'best_model.pth')
#             print(f"Saved best model with val loss: {val_loss:.4f}")

#     print("\nTraining complete!")
#     print(f"Best validation loss: {best_val_loss:.4f}")


# if __name__ == '__main__':
#     # Example: Just load and visualize data
#     print("Document Reconstruction Dataset Loader")
#     print("="*50)

#     # Quick test
#     try:
#         train_loader, val_loader = get_dataloaders(
#             data_dir='renders/synthetic_data_pitch_sweep',
#             batch_size=4,
#             img_size=(512, 512)
#         )

#         print("\nDataset loaded successfully!")

#         # Visualize a sample batch
#         print("\nVisualizing a sample batch...")
#         sample_batch = next(iter(train_loader))
#         print(f"Batch shape - RGB: {sample_batch['rgb'].shape}, Ground Truth: {sample_batch['ground_truth'].shape}")
#         visualize_batch(sample_batch, num_samples=min(4, sample_batch['rgb'].shape[0]))

#         print("\nTo start training, uncomment the main() function call below")
#         # main()  # Uncomment this to start training

#     except Exception as e:
#         print(f"\nError loading dataset: {e}")
#         print("Please check that the data directory exists and contains the required files.")
