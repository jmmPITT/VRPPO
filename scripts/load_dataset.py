#!/usr/bin/env python3
"""
Script to load and explore the Food101 dataset for RL training.
This script demonstrates how to use the dataset and extract patches.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.food101_dataset import Food101H5Dataset, Food101PatchDataset, create_dataloaders

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load and explore the Food101 dataset')
    
    parser.add_argument('--data_path', type=str, 
                        default='/home/jonathan/claude_projects/ImageClassifierVRM/efficient_dataset',
                        help='Path to Food101 dataset directory with H5 files')
    
    parser.add_argument('--output_dir', type=str, 
                        default='/home/jonathan/claude_projects/VisualReasoningPPO/data',
                        help='Directory to save exploration results')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data loading')
    
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    parser.add_argument('--patch_size', type=int, default=80,
                        help='Size of image patches')
    
    parser.add_argument('--num_patches_h', type=int, default=3,
                        help='Number of patches in height dimension')
    
    parser.add_argument('--num_patches_w', type=int, default=4,
                        help='Number of patches in width dimension')
    
    return parser.parse_args()

def visualize_samples(dataset, output_dir, num_samples=5):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset: Food101H5Dataset instance
        output_dir: Directory to save the visualizations
        num_samples: Number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    indices = np.random.randint(0, len(dataset), num_samples)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Move to numpy for plotting
        image = image.numpy().transpose(1, 2, 0)
        
        # Plot the image
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples.png'))
    plt.close()
    print(f"Saved sample visualization to {os.path.join(output_dir, 'samples.png')}")

def visualize_patches(dataset, output_dir, idx=0):
    """
    Visualize patches from a single image.
    
    Args:
        dataset: Food101PatchDataset instance
        output_dir: Directory to save the visualizations
        idx: Index of the image to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sequential patches
    patches, label = dataset.get_sequential_patches(idx)
    
    # Calculate grid size for visualization
    num_patches_h = dataset.num_patches_h
    num_patches_w = dataset.num_patches_w
    
    fig, axes = plt.subplots(num_patches_h, num_patches_w, figsize=(3*num_patches_w, 3*num_patches_h))
    
    patch_idx = 0
    for h in range(num_patches_h):
        for w in range(num_patches_w):
            # Get the patch
            patch = patches[patch_idx]
            
            # Move to numpy for plotting
            patch = patch.numpy().transpose(1, 2, 0)
            
            # Plot the patch
            if num_patches_h > 1 and num_patches_w > 1:
                axes[h, w].imshow(patch)
                axes[h, w].set_title(f"Patch ({h},{w})")
                axes[h, w].axis('off')
            else:
                axes[patch_idx].imshow(patch)
                axes[patch_idx].set_title(f"Patch {patch_idx}")
                axes[patch_idx].axis('off')
            
            patch_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'patches_img{idx}.png'))
    plt.close()
    print(f"Saved patch visualization to {os.path.join(output_dir, f'patches_img{idx}.png')}")

def explore_dataset(args):
    """
    Load and explore the Food101 dataset.
    
    Args:
        args: Command-line arguments
    """
    print(f"Loading Food101 dataset from {args.data_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    train_file = os.path.join(args.data_path, 'train.h5')
    test_file = os.path.join(args.data_path, 'test.h5')
    
    train_dataset = Food101H5Dataset(data_file=train_file)
    patch_dataset = Food101PatchDataset(
        data_file=train_file,
        patch_size=args.patch_size,
        num_patches_h=args.num_patches_h,
        num_patches_w=args.num_patches_w
    )
    
    # Visualize samples
    print("Visualizing random samples...")
    visualize_samples(train_dataset, args.output_dir, args.num_samples)
    
    # Visualize patches for a few examples
    print("Visualizing patches...")
    for i in range(min(args.num_samples, 3)):
        visualize_patches(patch_dataset, args.output_dir, i)
    
    # Test dataloader
    print("Testing dataloader...")
    train_loader, val_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get first batch as a test
    batch = next(iter(train_loader))
    images, labels = batch
    
    print(f"Loaded batch with shape: {images.shape}, labels shape: {labels.shape}")
    
    # Get dataset stats
    print("\nDataset Stats:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Total patches: {len(patch_dataset)}")
    print(f"  Image shape: {images.shape[1:]}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of patches per image: {args.num_patches_h * args.num_patches_w}")
    
    # Write stats to a file
    with open(os.path.join(args.output_dir, 'dataset_stats.txt'), 'w') as f:
        f.write("Food101 Dataset Statistics\n")
        f.write("=========================\n\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Total patches: {len(patch_dataset)}\n")
        f.write(f"Image shape: {images.shape[1:]}\n")
        f.write(f"Patch size: {args.patch_size}x{args.patch_size}\n")
        f.write(f"Patch grid: {args.num_patches_h}x{args.num_patches_w}\n")
        f.write(f"Patches per image: {args.num_patches_h * args.num_patches_w}\n")
    
    print(f"Dataset stats saved to {os.path.join(args.output_dir, 'dataset_stats.txt')}")
    
    # Close datasets
    train_dataset.close()
    patch_dataset.close()

if __name__ == '__main__':
    args = parse_args()
    explore_dataset(args)