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
from reconstruction_model import *
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
        C_gt = batch['coords'].to(device)            # 3D ground truth
        B_gt = batch['backward_map'].to(device)     # backward map ground truth
        D_gt = batch['ground_truth'].to(device)     # unwarped image
        mask = batch.get('border', None)
        if mask is not None:
            mask = mask.to(device)

        optimizer.zero_grad()

        # DewarpNet forward
        D_hat, C_hat, B_hat = model(rgb)

        # Compute multi-component loss
        losses = compute_loss(
            C_hat=C_hat,
            C=C_gt,
            B_hat=B_hat,
            B=B_gt,
            D_hat=D_hat,
            D=D_gt,
            mask=mask,
            alpha=alpha,
            beta=beta,
            lambda_grad=lambda_grad,
            gamma=gamma,
            delta=delta
        )

        loss = losses['total']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss, losses  # Return last batch losses for logging


# Validation function 

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    alpha=1.0,
    beta=1.0,
    lambda_grad=0.1,
    gamma=1.0,
    delta=1.0,
) -> float:
    """Validate DewarpNet."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch['rgb'].to(device)
            C_gt = batch['coords'].to(device)
            B_gt = batch['backward_map'].to(device)
            D_gt = batch['ground_truth'].to(device)
            mask = batch.get('border', None)
            if mask is not None:
                mask = mask.to(device)

            D_hat, C_hat, B_hat = model(rgb)

            losses = compute_loss(
                C_hat=C_hat,
                C=C_gt,
                B_hat=B_hat,
                B=B_gt,
                D_hat=D_hat,
                D=D_gt,
                mask=mask,
                alpha=alpha,
                beta=beta,
                lambda_grad=lambda_grad,
                gamma=gamma,
                delta=delta
            )
            total_loss += losses['total'].item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss