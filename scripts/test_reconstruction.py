#!/usr/bin/env python3
"""
Test script for reconstructing images from the VQEnvironment states.
This script tests the environment by running random actions and visualizing
the reconstructed images at each step, including class labels.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py

# Add parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.vqvae import VQVAEWrapper
from utils.food101_dataset import Food101LatentDataset
from environments import VQEnvironment

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test reconstruction from VQ Environment states')
    
    parser.add_argument('--data_path', type=str, 
                        default='/home/jonathan/claude_projects/ImageClassifierVRM/efficient_dataset',
                        help='Path to Food101 dataset directory with H5 files')
    
    parser.add_argument('--vqvae_model', type=str, 
                        default='/home/jonathan/claude_projects/Food101_VQGAN/checkpoints/best_model_stage1.pt',
                        help='Path to trained VQ-VAE model checkpoint')
    
    parser.add_argument('--output_dir', type=str, 
                        default='/home/jonathan/claude_projects/VisualReasoningPPO/data/reconstructions',
                        help='Directory to save results')
    
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Hidden dimensions for the VQ-VAE model')
    
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent dimension for the VQ-VAE model')
    
    parser.add_argument('--num_embeddings', type=int, default=256,
                        help='Number of embeddings in the VQ-VAE codebook')
    
    parser.add_argument('--patch_size', type=int, default=80,
                        help='Size of image patches')
    
    parser.add_argument('--num_patches_h', type=int, default=3,
                        help='Number of patches in height dimension')
    
    parser.add_argument('--num_patches_w', type=int, default=4,
                        help='Number of patches in width dimension')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes to run')
    
    parser.add_argument('--sequence_length', type=int, default=3,
                        help='Number of steps in each episode')
    
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for random sampling (higher = more diverse)')
    
    return parser.parse_args()

def load_class_names(data_path):
    """
    Load class names from the dataset.
    
    Args:
        data_path: Path to the dataset directory
        
    Returns:
        class_names: List of class names
    """
    train_file = os.path.join(data_path, 'train.h5')
    
    with h5py.File(train_file, 'r') as f:
        if 'class_names' in f:
            class_names = f['class_names'][:]
            class_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in class_names]
        else:
            # Try to load class names from a separate file
            try:
                class_names = np.load(os.path.join(data_path, 'classes.npy'))
                class_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in class_names]
            except:
                print("Warning: Could not load class names")
                class_names = [str(i) for i in range(101)]  # Default to numeric labels
    
    return class_names

def reconstruct_state(state, encoder, num_patches, grid_height, grid_width, latent_dim):
    """
    Reconstruct an image from a flattened state.
    
    Args:
        state: Flattened state tensor of shape [total_positions, embedding_dim]
        encoder: VQVAEWrapper for decoding
        num_patches: Number of patches in the image
        grid_height: Height of the latent grid
        grid_width: Width of the latent grid
        latent_dim: Dimension of the latent vectors
        
    Returns:
        reconstructed_image: Reconstructed image as a tensor
    """
    # Reshape the state back to [num_patches, latent_dim, grid_height, grid_width]
    reshaped_state = state.reshape(
        num_patches, grid_height, grid_width, latent_dim
    ).permute(0, 3, 1, 2)
    
    # Decode with the VQ-VAE decoder
    with torch.no_grad():
        reconstructed_image = encoder.decode_image(reshaped_state)
    
    return reconstructed_image

def visualize_reconstructions(episode_states, encoder, class_names, label, output_path, episode_idx=0):
    """
    Visualize the reconstructions at each step with class label.
    
    Args:
        episode_states: List of states from an episode
        encoder: VQVAEWrapper for decoding
        class_names: List of class names
        label: Class label of the image
        output_path: Path to save the visualization
        episode_idx: Index of the episode
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Get encoder parameters
    num_patches = encoder.num_patches_h * encoder.num_patches_w
    grid_height = encoder.grid_height
    grid_width = encoder.grid_width
    latent_dim = encoder.latent_dim
    
    # Create a figure with the reconstructions
    num_images = len(episode_states)
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    
    # If there's only one step, axes is not a list
    if num_images == 1:
        axes = [axes]
    
    # Get the class name (if available)
    class_name = class_names[label] if label < len(class_names) else f"Unknown ({label})"
    
    # Plot each step's reconstruction
    for i, state in enumerate(episode_states):
        # Reconstruct the image
        reconstructed = reconstruct_state(
            state, encoder, num_patches, grid_height, grid_width, latent_dim
        )
        
        # Convert to numpy for plotting
        recon_np = reconstructed.permute(1, 2, 0).cpu().numpy()
        recon_np = np.clip(recon_np, 0, 1)
        
        # Plot the reconstruction
        axes[i].imshow(recon_np)
        
        # Add step and class label for the first step
        if i == 0:
            title = f"Step {i}: {class_name}"
        else:
            title = f"Step {i}"
            
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'recon_episode_{episode_idx}.png'))
    plt.close()
    print(f"Saved reconstruction visualization to {os.path.join(output_path, f'recon_episode_{episode_idx}.png')}")

def test_reconstruction(args):
    """
    Test reconstructing images from VQEnvironment states.
    
    Args:
        args: Command line arguments
    """
    print(f"Testing reconstruction from VQ Environment states")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the VQ-VAE wrapper
    print(f"Initializing VQ-VAE wrapper with model from {args.vqvae_model}")
    encoder = VQVAEWrapper(
        model_path=args.vqvae_model,
        device=args.device,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings
    )
    
    # Create latent dataset for the environment
    print(f"Loading latent dataset from {args.data_path}")
    train_file = os.path.join(args.data_path, 'train.h5')
    latent_dataset = Food101LatentDataset(
        data_file=train_file,
        encoder=encoder,
        patch_size=args.patch_size,
        num_patches_h=args.num_patches_h,
        num_patches_w=args.num_patches_w
    )
    
    # Load class names
    print("Loading class names...")
    class_names = load_class_names(args.data_path)
    print(f"Loaded {len(class_names)} class names")
    
    # Create environment
    print(f"Creating environment with sequence length {args.sequence_length}")
    env = VQEnvironment(
        dataset=latent_dataset,
        encoder=encoder,
        sequence_length=args.sequence_length,
        device=device
    )
    
    # Print codebook stats
    print(f"Codebook size: {env.codebook_size}")
    print(f"Codebook shape: {env.codebook.shape}")
    
    # Run episodes with random actions
    for episode in range(args.num_episodes):
        print(f"\nRunning episode {episode+1}/{args.num_episodes}")
        
        # Reset the environment and get the starting state
        state = env.reset()
        
        # Get the label for the current state
        label = env.current_label
        print(f"  Selected image with label: {label} ({class_names[label] if label < len(class_names) else 'Unknown'})")
        
        done = False
        step = 0
        
        # Store states for visualization
        episode_states = [state.detach()]
        
        # Run until done
        while not done:
            # Generate random action probabilities
            # Use a higher temperature for more diversity
            random_logits = torch.randn(env.get_action_shape(), device=device) * args.temperature
            
            # Take a step in the environment
            next_state, reward, done, info = env.step(random_logits)
            
            # Print some info about the sampled indices
            indices = info['sampled_indices']
            print(f"  Step {step+1}/{args.sequence_length}:")
            print(f"    Unique codebook indices used: {len(np.unique(indices))}")
            print(f"    Min index: {indices.min()}, Max index: {indices.max()}")
            
            # Store the state for visualization
            episode_states.append(next_state.detach())
            
            # Update state for next iteration
            state = next_state
            step += 1
        
        # Visualize the reconstructions
        visualize_reconstructions(
            episode_states,
            encoder, 
            class_names,
            label,
            args.output_dir,
            episode
        )
    
    # Close the environment and dataset
    env.close()
    latent_dataset.close()
    
    print("\nReconstruction test complete!")

if __name__ == '__main__':
    args = parse_args()
    test_reconstruction(args)