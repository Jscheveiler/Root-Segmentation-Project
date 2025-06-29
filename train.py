"""
This file contains the training script for the networks.
NOTE: It does not include a validation function, because the data is relatively scarce (380 images), so there's no dataset splitting.
"""
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import RootSegmentationDataset
from utils.losses import bce_dice_loss
from utils.metrics import dice_coefficient
from utils.helpers import save_checkpoint, print_model_summary
from models.unet import UNet

def train(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Dataset and dataloader
    dataset = RootSegmentationDataset(root_dir=args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Initializing model and printing its number of trainable parameters
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)
    print_model_summary(model)

    # Setting up Loss and Optimizer
    criterion = bce_dice_loss  # Uses BCE_Dice_Loss with its default parameter (0.7*bce + 0.3*dice)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # Training loop
    best_dice = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0

        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()

            # Calculate dice coefficient
            with torch.no_grad():
                dice = dice_coefficient(outputs, masks)
                epoch_dice += dice

            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}',
                            'Dice': f'{dice:.4f}'})

        # Calculate averages
        avg_loss = epoch_loss / len(dataloader)
        avg_dice = epoch_dice / len(dataloader)

        # Update learning rate
        scheduler.step(avg_loss)

        # Print epoch results
        elapsed_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Time elapsed: {elapsed_time / 60:.1f}min")
        print("-" * 50)

        # Save best model
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, best_ckpt_path)
            print(f"New best model saved! Dice: {best_dice:.4f}")

        # Save periodic checkpoints
        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_epoch{epoch + 1}.pt")
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, ckpt_path)

    print(f"\nTraining completed! Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Root Segmentation Model")
    parser.add_argument("--data_dir", type=str, default="data/1_mesocosm",
                        help="Path to dataset root")
    parser.add_argument("--ckpt_dir", type=str, default="logs/model_checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loading workers")
    parser.add_argument("--save_freq", type=int, default=100,
                        help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Print configuration
    print("Training Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-" * 50)

    train(args)