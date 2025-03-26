#!/usr/bin/env python3
"""
Script to load and test the VQ-VAE model for the RL agent.
This script demonstrates how to use the VQ-VAE and access its codebook.
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
from utils.food101_dataset import Food101H5Dataset, Food101PatchDataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load and test the VQ-VAE model')
    
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
    
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to test')
    
    return parser.parse_args()

def visualize_reconstructions(encoder, patches, labels, output_dir, idx=0):
    """
    Visualize original patches and their reconstructions.
    
    Args:
        encoder: VQVAEWrapper instance
        patches: List of patches to encode/decode
        labels: Label of the image
        output_dir: Directory to save visualizations
        idx: Index for the filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the number of patches
    num_patches = len(patches)
    if num_patches > 16:  # Limit visualization to max 16 patches
        patches = patches[:16]
        num_patches = 16
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_patches * 2)))  # *2 for original + reconstruction
    grid_cols = grid_size
    grid_rows = int(np.ceil((num_patches * 2) / grid_cols))
    
    # Create the figure
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3*grid_cols, 3*grid_rows))
    axes = axes.flatten()
    
    # Process each patch
    for i, patch in enumerate(patches):
        # Add batch dimension for the model
        patch_batch = patch.unsqueeze(0).to(encoder.device)
        
        # Encode and decode
        with torch.no_grad():
            latent, _ = encoder.encode_patch(patch_batch)
            reconstruction = encoder.decode_patch(latent)
            
            # Move to CPU and remove batch dimension
            reconstruction = reconstruction.squeeze(0).cpu()
        
        # Plot original patch
        axes[i*2].imshow(patch.numpy().transpose(1, 2, 0))
        axes[i*2].set_title(f"Original {i}")
        axes[i*2].axis('off')
        
        # Plot reconstruction
        axes[i*2+1].imshow(reconstruction.numpy().transpose(1, 2, 0))
        axes[i*2+1].set_title(f"Recon {i}")
        axes[i*2+1].axis('off')
    
    # Hide empty subplot
    for i in range(num_patches * 2, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'reconstructions_img{idx}.png'))
    plt.close()
    print(f"Saved reconstruction visualization to {os.path.join(output_dir, f'reconstructions_img{idx}.png')}")

def visualize_codebook(encoder, output_dir):
    """
    Visualize the VQ-VAE codebook embeddings using a PCA projection.
    
    Args:
        encoder: VQVAEWrapper instance
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the codebook
    codebook = encoder.get_codebook().cpu().numpy()
    print(f"Codebook shape: {codebook.shape}")
    
    # Get codebook usage if available
    try:
        usage = encoder.get_codebook_usage().cpu().numpy()
        usage_normalized = (usage - usage.min()) / (usage.max() - usage.min() + 1e-8)
    except:
        usage_normalized = np.ones(encoder.num_embeddings)
        print("Codebook usage not available, using uniform usage")
    
    # Use PCA to visualize the embeddings (using scikit-learn internally)
    try:
        from sklearn.decomposition import PCA
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(codebook)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=usage_normalized,  # Color by usage
            cmap='viridis', 
            alpha=0.7, 
            s=50
        )
        plt.colorbar(scatter, label='Normalized Usage')
        plt.title('VQ-VAE Codebook Embeddings (PCA projection)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'codebook_pca.png'))
        plt.close()
        print(f"Saved codebook PCA visualization to {os.path.join(output_dir, 'codebook_pca.png')}")
        
    except ImportError:
        print("scikit-learn not available, skipping PCA visualization")
    
    # Create a histogram of codebook usage
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(usage_normalized)), usage_normalized)
    plt.title('VQ-VAE Codebook Usage')
    plt.xlabel('Codebook Index')
    plt.ylabel('Normalized Usage')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'codebook_usage.png'))
    plt.close()
    print(f"Saved codebook usage histogram to {os.path.join(output_dir, 'codebook_usage.png')}")
    
    # Save codebook statistics
    with open(os.path.join(output_dir, 'codebook_stats.txt'), 'w') as f:
        f.write("VQ-VAE Codebook Statistics\n")
        f.write("=========================\n\n")
        f.write(f"Codebook size: {encoder.num_embeddings}\n")
        f.write(f"Embedding dimension: {encoder.latent_dim}\n")
        f.write(f"Latent grid shape: {encoder.grid_height}x{encoder.grid_width}\n")
        
        # Calculate codebook diversity
        norms = np.linalg.norm(codebook, axis=1)
        min_norm = norms.min()
        max_norm = norms.max()
        mean_norm = norms.mean()
        
        # Calculate cosine similarities
        similarity_matrix = codebook @ codebook.T
        norms_matrix = norms.reshape(-1, 1) @ norms.reshape(1, -1)
        cosine_matrix = similarity_matrix / (norms_matrix + 1e-8)
        np.fill_diagonal(cosine_matrix, 0)  # Remove self-similarity
        max_similarity = cosine_matrix.max()
        mean_similarity = cosine_matrix.mean()
        
        f.write(f"\nCodebook Statistics:\n")
        f.write(f"  Min embedding norm: {min_norm:.6f}\n")
        f.write(f"  Max embedding norm: {max_norm:.6f}\n")
        f.write(f"  Mean embedding norm: {mean_norm:.6f}\n")
        f.write(f"  Max cosine similarity: {max_similarity:.6f}\n")
        f.write(f"  Mean cosine similarity: {mean_similarity:.6f}\n")
    
    print(f"Saved codebook statistics to {os.path.join(output_dir, 'codebook_stats.txt')}")

def test_vqvae(args):
    """
    Load and test the VQ-VAE model.
    
    Args:
        args: Command-line arguments
    """
    print(f"Loading VQ-VAE model from {args.vqvae_model}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the VQ-VAE wrapper
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    encoder = VQVAEWrapper(
        model_path=args.vqvae_model,
        device=args.device,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings
    )
    
    # Load dataset for testing
    train_file = os.path.join(args.data_path, 'train.h5')
    patch_dataset = Food101PatchDataset(
        data_file=train_file,
        patch_size=args.patch_size,
        num_patches_h=args.num_patches_h,
        num_patches_w=args.num_patches_w
    )
    
    # Test reconstructions
    print("Testing reconstructions...")
    for i in range(min(args.num_samples, 3)):
        patches, label = patch_dataset.get_sequential_patches(i)
        visualize_reconstructions(encoder, patches, label, args.output_dir, i)
    
    # Visualize and analyze the codebook
    print("Analyzing codebook...")
    visualize_codebook(encoder, args.output_dir)
    
    # Test latent dimensions and shapes
    print("\nTesting latent dimensions:")
    
    # Get a sample patch
    sample_patch, _ = patch_dataset[0]
    sample_patch = sample_patch.unsqueeze(0).to(device)  # Add batch dimension
    
    # Encode the patch
    with torch.no_grad():
        latent, indices = encoder.encode_patch(sample_patch)
    
    print(f"Input patch shape: {sample_patch.shape}")
    print(f"Encoded latent shape: {latent.shape}")
    print(f"Indices shape: {indices.shape}")
    
    # Print statistics to a file
    with open(os.path.join(args.output_dir, 'vqvae_stats.txt'), 'w') as f:
        f.write("VQ-VAE Model Statistics\n")
        f.write("======================\n\n")
        f.write(f"Model path: {args.vqvae_model}\n")
        f.write(f"Hidden dimensions: {args.hidden_dims}\n")
        f.write(f"Latent dimension: {args.latent_dim}\n")
        f.write(f"Codebook size: {args.num_embeddings}\n")
        f.write(f"\nInput/Output Shapes:\n")
        f.write(f"  Input patch shape: {sample_patch.shape}\n")
        f.write(f"  Encoded latent shape: {latent.shape}\n")
        f.write(f"  Codebook indices shape: {indices.shape}\n")
        f.write(f"\nLatent Grid Dimensions:\n")
        f.write(f"  Grid height: {encoder.grid_height}\n")
        f.write(f"  Grid width: {encoder.grid_width}\n")
        f.write(f"  Flattened patch embedding dimension: {encoder.get_patch_embedding_dim()}\n")
    
    print(f"VQ-VAE stats saved to {os.path.join(args.output_dir, 'vqvae_stats.txt')}")
    
    # Close dataset
    patch_dataset.close()

if __name__ == '__main__':
    args = parse_args()
    test_vqvae(args)