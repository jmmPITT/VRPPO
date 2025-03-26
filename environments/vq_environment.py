import os
import numpy as np
import torch
import torch.nn.functional as F
import h5py

class VQEnvironment:
    """
    Environment for RL agent to interact with VQ-VAE latent spaces.
    
    This environment loads the entire dataset (images and labels) from an HDF5 file
    into memory (and closes the file immediately) so that each call to reset samples
    the image and label from the in-memory dataset.
    """
    def __init__(self, encoder, patch_size=80, num_patches_h=3, num_patches_w=4,
                 sequence_length=3, device='cuda'):
        
        """
        Initialize the environment.
        
        Args:
            data_file: Path to the HDF5 file containing images and labels.
            encoder: VQVAEWrapper instance with access to the codebook and encoder.
            patch_size: Size of each patch to extract.
            num_patches_h: Number of patches along the height.
            num_patches_w: Number of patches along the width.
            sequence_length: Number of steps in each episode.
            device: Device to run on (cuda or cpu).
        """
        self.device = device
        self.encoder = encoder
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.sequence_length = sequence_length
        data_file = '/home/jonathan/claude_projects/ImageClassifierVRM/efficient_dataset/train.h5'
        
        # Load the dataset into memory and immediately close the file.
        with h5py.File(data_file, 'r') as f:
            # Determine half the number of images directly from the dataset
            half = f['images'].shape[0] // 2
            # Load only the first half of images and labels
            self.images = np.array(f['images'][:half])
            self.labels = np.array(f['labels'][:half])
        self.n_images = self.images.shape[0]
        print(f"Loaded {self.n_images} images from {data_file} into memory.")

        
        # Get latent dimensions and codebook from the encoder.
        latent_dim, grid_height, grid_width = self.encoder.get_latent_grid_shape()
        self.latent_dim = latent_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.codebook = self.encoder.get_codebook().to(device)
        self.codebook_size = self.codebook.shape[0]
        self.embedding_dim = self.codebook.shape[1]
        
        # Calculate number of patches and total embedding positions.
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.total_positions = self.num_patches * grid_height * grid_width
        
        # Initialize current state.
        self.current_state = None
        self.current_label = None
        self.current_step = 0
        
        print("VQEnvironment initialized with:")
        print(f"  {self.n_images} images")
        print(f"  Codebook size: {self.codebook_size}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Latent shape: ({latent_dim}, {grid_height}, {grid_width})")
        print(f"  Patches per image: {self.num_patches}")
        print(f"  Total embedding positions: {self.total_positions}")
        print(f"  Sequence length: {self.sequence_length}")
    
    def extract_patches(self, image):
        """
        Given an image (assumed shape [C, H, W]), extract patches in sequential order.
        Returns a list of patches (each a torch tensor).
        """
        patches = []
        C, H, W = image.shape
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                y_start = i * self.patch_size
                x_start = j * self.patch_size
                y_end = min(y_start + self.patch_size, H)
                x_end = min(x_start + self.patch_size, W)
                patch = image[:, y_start:y_end, x_start:x_end]
                # Pad if necessary.
                if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
                    padded = np.zeros((C, self.patch_size, self.patch_size), dtype=patch.dtype)
                    padded[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded
                patch = patch.astype(np.float32) / 255.0
                patch = torch.from_numpy(patch).float()
                patches.append(patch)
        return patches
    
    def get_sequential_latents(self, img_idx):
        """
        For a given image index, extract patches, encode them, and return the list of latents.
        """
        image = self.images[img_idx]  # [C, H, W]
        label = self.labels[img_idx]
        patches = self.extract_patches(image)
        latents = []
        for patch in patches:
            patch = patch.unsqueeze(0).to(self.device)  # Add batch dimension.
            with torch.no_grad():
                latent, _ = self.encoder.encode_patch(patch)
            latents.append(latent.squeeze(0).cpu())
        return latents, label
    
    def reset(self):
        """
        Reset the environment: sample a random image, extract and encode its patches,
        then prepare the initial state.
        """
        img_idx = np.random.randint(0, self.n_images)
        patches, label = self.get_sequential_latents(img_idx)
        patch_tensor = torch.stack(patches).to(self.device)  # [num_patches, latent_dim, grid_height, grid_width]
        # Reshape to [total_positions, latent_dim]
        self.current_state = patch_tensor.permute(0, 2, 3, 1).reshape(-1, self.latent_dim)
        self.current_label = label
        self.current_step = 0
        return self.current_state.clone()
    
    def step(self, sampled_indices):
        """
        Update the state using sampled codebook indices.
        
        Args:
            sampled_indices: Tensor of shape [total_positions] (indices into the codebook)
        
        Returns:
            next_state, reward, done, info
        """
        new_embeddings = self.codebook[sampled_indices]  # [total_positions, embedding_dim]
        self.current_state = new_embeddings
        self.current_step += 1
        done = self.current_step >= self.sequence_length
        reward = 0.0  # Placeholder reward.
        info = {
            'label': self.current_label,
            'step': self.current_step,
            'sampled_indices': sampled_indices.cpu().numpy()
        }
        return self.current_state.clone(), reward, done, info
    
    def render(self, mode='human'):
        """
        Render the current state by decoding it back to image space.
        """
        if self.current_state is None:
            return None
        reshaped_state = self.current_state.reshape(self.num_patches, self.grid_height, self.grid_width, self.latent_dim)
        reshaped_state = reshaped_state.permute(0, 3, 1, 2)
        with torch.no_grad():
            image = self.encoder.decode_image(reshaped_state)
        if mode == 'rgb_array':
            return image.cpu().numpy()
        return None
    
    def get_state_shape(self):
        return (self.total_positions, self.embedding_dim)
    
    def get_action_shape(self):
        return (self.total_positions, self.codebook_size)
    
    def close(self):
        pass  # No additional cleanup needed.

# Example usage:
# env = VQEnvironment(data_file='path/to/train.h5', encoder=my_encoder)
# state = env.reset()
