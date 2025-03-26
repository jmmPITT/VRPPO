#!/usr/bin/env python3
"""
Test script for the VQEnvironment class.
This script tests the environment by using random actions.
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

from utils.vqvae import VQVAEWrapper
from utils.food101_dataset import Food101LatentDataset
from environments import VQEnvironment

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the VQ Environment')
    
    parser.add_argument('--data_path', type=str, 
                        default='/home/jonathan/claude_projects/ImageClassifierVRM/efficient_dataset',
                        help='Path to Food101 dataset directory with H5 files')
    
    parser.add_argument('--vqvae_model', type=str, 
                        default='/home/jonathan/claude_projects/Food101_VQGAN/checkpoints/best_model_stage1.pt',
                        help='Path to trained VQ-VAE model checkpoint')
    
    parser.add_argument('--output_dir', type=str, 
                        default='/home/jonathan/claude_projects/VisualReasoningPPO/data',
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
    
    return parser.parse_args()

def visualize_episode(episode_states, encoder, output_path, episode_idx=0):
    """
    Visualize the states in an episode.
    
    Args:
        episode_states: List of states from an episode
        encoder: VQVAEWrapper for decoding
        output_path: Path to save the visualization
        episode_idx: Index of the episode
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Create a figure with a row for each step in the episode
    fig, axes = plt.subplots(
        len(episode_states), 1, 
        figsize=(12, 6 * len(episode_states))
    )
    
    # If there's only one step, axes is not a list
    if len(episode_states) == 1:
        axes = [axes]
    
    for i, state in enumerate(episode_states):
        # Reshape the state back to patch format
        # From: [total_positions, embedding_dim]
        # To: [num_patches, latent_dim, grid_height, grid_width]
        num_patches = encoder.num_patches_h * encoder.num_patches_w
        grid_height, grid_width = encoder.grid_height, encoder.grid_width
        latent_dim = encoder.latent_dim
        
        reshaped_state = state.reshape(
            num_patches, grid_height, grid_width, latent_dim
        ).permute(0, 3, 1, 2)
        
        # Decode the state to get an image
        with torch.no_grad():
            image = encoder.decode_image(reshaped_state)
            image = image.cpu()
        
        # Convert to numpy for plotting
        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1)  # Ensure values are in [0, 1]
        
        # Plot the image
        axes[i].imshow(image_np)
        axes[i].set_title(f"Step {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'episode_{episode_idx}.png'))
    plt.close()
    print(f"Saved episode visualization to {os.path.join(output_path, f'episode_{episode_idx}.png')}")

def test_environment(args):
    """
    Test the VQEnvironment by running random actions.
    
    Args:
        args: Command line arguments
    """
    print(f"Testing VQ Environment with random actions")
    
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
    
    # Create dataset
    print(f"Loading dataset from {args.data_path}")
    train_file = os.path.join(args.data_path, 'train.h5')
    dataset = Food101LatentDataset(
        data_file=train_file,
        encoder=encoder,
        patch_size=args.patch_size,
        num_patches_h=args.num_patches_h,
        num_patches_w=args.num_patches_w
    )
    
    # Create environment
    print(f"Creating environment with sequence length {args.sequence_length}")
    env = VQEnvironment(
        dataset=dataset,
        encoder=encoder,
        sequence_length=args.sequence_length,
        device=device
    )
    
    # Get state and action shapes
    state_shape = env.get_state_shape()
    action_shape = env.get_action_shape()
    print(f"State shape: {state_shape}")
    print(f"Action shape: {action_shape}")
    
    # Run episodes with random actions
    for episode in range(args.num_episodes):
        print(f"\nRunning episode {episode+1}/{args.num_episodes}")
        
        # Reset the environment
        state = env.reset()
        done = False
        step = 0
        
        # Store states for visualization
        episode_states = [state.detach()]
        
        # Run until done
        while not done:
            # Generate random action probabilities
            # In practice, these would come from the policy network
            random_logits = torch.randn(action_shape, device=device)
            
            # Take a step in the environment
            next_state, reward, done, info = env.step(random_logits)
            
            # Store the state for visualization
            episode_states.append(next_state.detach())
            
            # Print step information
            print(f"  Step {step+1}/{args.sequence_length}:")
            print(f"    Label: {info['label']}")
            print(f"    State shape: {next_state.shape}")
            print(f"    Reward: {reward}")
            print(f"    Done: {done}")
            
            # Update state for next iteration
            state = next_state
            step += 1
        
        # Visualize the episode
        visualize_episode(
            episode_states, 
            encoder, 
            os.path.join(args.output_dir, 'episodes'),
            episode
        )
    
    # Close the environment and dataset
    env.close()
    dataset.close()
    
    print("\nEnvironment test complete!")

if __name__ == '__main__':
    args = parse_args()
    test_environment(args)