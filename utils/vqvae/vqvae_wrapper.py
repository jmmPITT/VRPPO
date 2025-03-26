import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add path to the Food101_VQGAN project to import models
food101_vqgan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'Food101_VQGAN')
sys.path.append(food101_vqgan_path)

from models.vqgan_optimized import OptimizedVQGAN

class VQVAEWrapper:
    """
    Wrapper for the VQ-VAE model from Food101_VQGAN to handle encoding and decoding.
    This wrapper simplifies the interface for encoding and decoding patches and
    provides access to the codebook for the RL agent.
    """
    def __init__(self, model_path, device='cuda', 
                 hidden_dims=[32, 64, 128, 256], 
                 latent_dim=16, 
                 num_embeddings=256):
        """
        Initialize the VQ-VAE wrapper.
        
        Args:
            model_path: Path to the VQ-VAE model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
            hidden_dims: Hidden dimensions of the encoder/decoder
            latent_dim: Dimension of the latent space
            num_embeddings: Number of embeddings in the codebook
        """
        # Ensure device is properly set
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"VQVAEWrapper using device: {self.device}")
        
        # Initialize the VQ-VAE model
        self.model = OptimizedVQGAN(
            in_channels=3,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,  # Must be the same as latent_dim
        )
        
        # Load the model weights
        try:
            # Load model weights with the correct device mapping
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move model to device after loading weights
            self.model = self.model.to(self.device)
            
            print(f"Model loaded successfully from {model_path}")
            self.model.eval()  # Set model to evaluation mode
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        # Get the codebook size and latent dimensions
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        
        # Calculate patch grid size from a test forward pass
        with torch.no_grad():
            test_input = torch.zeros(1, 3, 80, 80).to(self.device)
            test_latent = self.model.encoder(test_input)
            self.grid_height, self.grid_width = test_latent.shape[2], test_latent.shape[3]
            print(f"Encoder output shape for a single patch: {test_latent.shape}")
            print(f"Grid dimensions: {self.grid_height}x{self.grid_width}")
            
            # Store number of patches per dimension for the environment
            self.num_patches_h = 3  # Default values for Food101
            self.num_patches_w = 4
            
    def encode_patch(self, patch):
        """
        Encode a single patch.
        
        Args:
            patch: Tensor of shape [1, 3, H, W] representing a single patch
            
        Returns:
            latent: Encoded latent representation [1, latent_dim, grid_height, grid_width]
            indices: Codebook indices [1, grid_height, grid_width]
        """
        with torch.no_grad():
            # Ensure patch is on the correct device
            patch = patch.to(self.device)
            
            # Encode the patch
            _, quantized, _, indices = self.model.encode(patch)
            return quantized, indices
            
    def decode_patch(self, latent):
        """
        Decode a single patch latent.
        
        Args:
            latent: Tensor of shape [1, latent_dim, grid_height, grid_width]
            
        Returns:
            patch: Decoded image patch [1, 3, H, W]
        """
        with torch.no_grad():
            decoded = self.model.decode(latent)
            return decoded
    
    def encode_image(self, image_patches):
        """
        Encode an entire image represented as a tensor of patches.
        
        Args:
            image_patches: Tensor of shape [n_patches, 3, H, W] representing image patches
            
        Returns:
            latents: Encoded latent representations [n_patches, latent_dim, grid_height, grid_width]
            indices: Codebook indices [n_patches, grid_height, grid_width]
        """
        n_patches = image_patches.shape[0]
        latents = []
        indices_list = []
        
        # Ensure patches are on the correct device
        image_patches = image_patches.to(self.device)
        
        with torch.no_grad():
            for i in range(n_patches):
                patch = image_patches[i:i+1]
                latent, indices = self.encode_patch(patch)
                latents.append(latent)
                indices_list.append(indices)
                
        # Concatenate results
        latents_tensor = torch.cat(latents, dim=0)
        indices_tensor = torch.cat(indices_list, dim=0)
        
        return latents_tensor, indices_tensor
    
    def decode_image(self, latents, num_patches_h=3, num_patches_w=4, patch_size=80):
        """
        Decode an entire image from latent representations.
        
        Args:
            latents: Tensor of shape [n_patches, latent_dim, grid_height, grid_width]
            num_patches_h: Number of patches in height dimension
            num_patches_w: Number of patches in width dimension
            patch_size: Size of each patch (assuming square patches)
            
        Returns:
            image: Decoded image [3, H, W]
        """
        n_patches = latents.shape[0]
        assert n_patches == num_patches_h * num_patches_w, "Number of patches doesn't match the grid"
        
        # Create empty image
        image_height = num_patches_h * patch_size
        image_width = num_patches_w * patch_size
        image = torch.zeros(3, image_height, image_width).to(self.device)
        
        with torch.no_grad():
            patch_idx = 0
            for h in range(num_patches_h):
                for w in range(num_patches_w):
                    # Decode the patch
                    latent = latents[patch_idx:patch_idx+1]
                    decoded_patch = self.decode_patch(latent).squeeze(0)
                    
                    # Place in the right position in the image
                    h_start = h * patch_size
                    w_start = w * patch_size
                    image[:, h_start:h_start+patch_size, w_start:w_start+patch_size] = decoded_patch
                    
                    patch_idx += 1
        
        return image
    
    def get_patch_embedding_dim(self):
        """Get the dimension of a patch embedding when flattened."""
        return self.latent_dim * self.grid_height * self.grid_width
    
    def get_latent_grid_shape(self):
        """Get the shape of the latent grid for a single patch."""
        return (self.latent_dim, self.grid_height, self.grid_width)
    
    def get_codebook(self):
        """Get the codebook embeddings for the RL agent."""
        return self.model.vector_quantizer.embeddings.clone().detach()
    
    def get_codebook_usage(self):
        """Get the current usage of the codebook embeddings."""
        return self.model.vector_quantizer.get_codebook_usage().clone().detach()