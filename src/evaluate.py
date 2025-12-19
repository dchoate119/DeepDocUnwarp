# Evaluation of trained model and results
# Daniel Choate: Tufts University 




import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from dataset_loader import get_dataloaders, visualize_batch, visualize_batch_pred, visualize_uv_flow
from model import ResNetUnet, MaskedL1Loss, MaskedMSELoss, SSIMLoss, UVReconstructionLoss
from training_val import validate


import torch
from typing import Dict, List, Tuple, Optional, Callable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np




# Configuration
DATA_DIR = '/home/daniel-choate/Datasets/DocUnwarp/renders/synthetic_data_pitch_sweep'
BATCH_SIZE = 8
IMG_SIZE = (512, 512) # Or 256, 256 depending on results
# WEIGHTS_PATH = 'best_model_256.pth'
WEIGHTS_PATH = 'best_model_512.pth'
NUM_VISUALS = 5  # number of validation samples to visualize

# LOAD DATA
_, val_loader = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    use_depth=False,
    use_uv=True,      # we want GT UV for evaluation & viz
    use_border=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Load Model & Weights
model = ResNetUnet(
    backbone_name='resnet34',
    pretrained=True
).to(device)

state = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Loaded weights from {WEIGHTS_PATH}")



# Compute validation loss
criterion_full = UVReconstructionLoss(
    reconstruction_weight=1.0,
    uv_weight=0.0,
    smoothness_weight=0.01,
    use_mask=True,
    loss_type='ssim'
)

val_loss = validate(model, val_loader, criterion_full, device)
print(f"Validation loss (UVReconstructionLoss total): {val_loss:.4f}")



# Compute SSIM metric explicitly
ssim_loss_module = SSIMLoss()  # returns 1 - SSIM
total_ssim = 0.0
num_batches = 0

with torch.no_grad():
    for batch in val_loader:
        rgb = batch['rgb'].to(device)
        gt = batch['ground_truth'].to(device)

        outputs = model(rgb, predict_uv=True)
        warped = outputs['warped']

        # SSIMLoss returns 1 - SSIM, so:
        ssim_loss = ssim_loss_module(warped, gt)
        batch_ssim = 1.0 - ssim_loss.item()

        total_ssim += batch_ssim
        num_batches += 1

mean_ssim = total_ssim / max(1, num_batches)
print(f"Final mean SSIM on validation set: {mean_ssim:.4f}")


# Qualitative visualization
model.eval()
visualized = 0

with torch.no_grad():
    for batch in val_loader:
        rgb = batch['rgb'].to(device)
        gt = batch['ground_truth'].to(device)
        uv_gt = batch['uv'].to(device)

        outputs = model(rgb, predict_uv=True)
        warped = outputs['warped']
        uv_pred = outputs['uv']

        # # Optional: print UV ranges once for debugging
        # if visualized == 0:
        #     print("Pred UV range: ", uv_pred.min().item(), "→", uv_pred.max().item())
        #     print("GT   UV range: ", uv_gt.min().item(), "→", uv_gt.max().item())
        #     print("UV shapes: pred =", uv_pred.shape, ", gt =", uv_gt.shape)

        for i in range(rgb.size(0)):
            if visualized >= NUM_VISUALS:
                break

            # 3a) RGB / GT / Predicted reconstruction
            visualize_batch_pred({
                'rgb': rgb[i].unsqueeze(0).cpu(),
                'ground_truth': gt[i].unsqueeze(0).cpu(),
                'predicted': warped[i].unsqueeze(0).cpu(),
            }, num_samples=1)

            # 3b) Predicted vs Ground Truth UV flow
            visualize_uv_flow(
                uv_pred[i].unsqueeze(0).cpu(),   # [1,2,H,W]
                uv_gt[i].unsqueeze(0).cpu(),     # [1,2,H,W]
                num_samples=1,
                title_pred="Predicted UV Flow",
                title_gt="Ground Truth UV Flow"
            )

            visualized += 1

        if visualized >= NUM_VISUALS:
            break