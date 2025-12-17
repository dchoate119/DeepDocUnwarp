# Training loop and validation functions
# Daniel Choate 
# Starter code provided by Professor Roy Shilkrot



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

from dataset_loader import get_dataloaders, visualize_batch
# from reconstruction_model import DocumentReconstructionModel
from reconstruction_model import ResNetUnet
from reconstruction_model import MaskedL1Loss, MaskedMSELoss, SSIMLoss, UVReconstructionLoss


# Training loop 

def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    epoch: int
) -> float:
    """ 
    Train for one epoch

    *** CHECK TODO LIST ***
    """
    model.train()
    total_loss = 0.0

    # Loop through each batch 
    # t_batch_start = time.time()
    for batch_idx, batch in enumerate(dataloader):

        # t0 = time.time() # TIMING CHECK 0
        # print(f"DataLoader prep time: {(t0 - t_batch_start)*1000:.1f}ms")
        
        # Move data to device 
        rgb = batch['rgb'].to(device)
        ground_truth = batch['ground_truth'].to(device)

        # Load mask if using masked loss
        mask = batch.get('border', None)
        if mask is not None:
            mask = mask.to(device)

        # torch.cuda.synchronize()
        # t1 = time.time() # TIMING CHECK 1

        # Forward pass
        optimizer.zero_grad()
        # output = model(rgb) # IF NOT USING FLOW PREDICTOR
        outputs = model(rgb, predict_uv=True)
        warped = outputs['warped']
        flow = outputs['flow']
        uv = outputs.get('uv', None)
        # output, flow = model(rgb)
        # print(f"Finished batch {batch_idx}")

        # Compute loss
        if isinstance(criterion, (MaskedL1Loss, MaskedMSELoss)):
            loss = criterion(output, ground_truth, mask)
        elif isinstance(criterion, SSIMLoss):
            loss = criterion(output, ground_truth)
        elif isinstance(criterion, UVReconstructionLoss):
            # Extract additional outputs if avail for UV-based
            losses = criterion(
                pred_image=warped,
                target_image=ground_truth,
                pred_uv=uv,
                target_uv=batch.get('uv', None),  # optional UV supervision
                flow=flow,
                mask=mask
            )
            loss=losses['total']
        else:
            # Standard (MSE, L1)
            # print(f"Standard loss: {criterion}")
            loss = criterion(output, ground_truth)

        # torch.cuda.synchronize()
        # t2 = time.time() # TIMING CHECK 2

        # Backward pass 
        loss.backward()
        # UPDATE MODEL
        optimizer.step()

        # torch.cuda.synchronize()
        # t3 = time.time()
        
        total_loss += loss.item()

        # if batch_idx % 20 == 0:
        #     print(
        #         f"Batch {batch_idx}: "
        #         f"h2d={(t1-t0)*1000:.1f}ms | "
        #         f"fwd+loss={(t2-t1)*1000:.1f}ms | "
        #         f"bwd+step={(t3-t2)*1000:.1f}"
        #     )

        # # Print progess 
        # if batch_idx % 10 == 0:
        #     print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
        # t_batch_start = time.time()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# Validation function 

def validate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> float:
    """
    Validate the model

    TODO: MODIFY to add more metrics (PSNR, SSIM, etc)
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch['rgb'].to(device)
            ground_truth = batch['ground_truth'].to(device)

            # Optional: MASKED LOSS ***
            mask = batch.get('border', None)
            if mask is not None:
                mask = mask.to(device)

            # output = model(rgb)
            # output, flow = model(rgb)
            outputs = model(rgb, predict_uv=True)
            warped = outputs['warped']
            flow = outputs['flow']
            uv = outputs.get('uv', None)

            # Compute loss (standard or masked 
            if isinstance(criterion, (MaskedL1Loss, MaskedMSELoss)):
                loss = criterion(output, ground_truth, mask)
            elif isinstance(criterion, SSIMLoss):
                loss = criterion(output, ground_truth)
            elif isinstance(criterion, UVReconstructionLoss):
                losses = criterion(
                    pred_image=warped,
                    target_image=ground_truth,
                    pred_uv=uv,
                    target_uv=batch.get('uv', None),
                    flow=flow,
                    mask=mask
                )
                loss = losses['total']
            else:
                loss = criterion(output, ground_truth)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
