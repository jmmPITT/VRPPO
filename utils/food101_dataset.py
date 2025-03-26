import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Food101H5Dataset(Dataset):
    """
    Dataset for loading Food101 data from H5 files.
    This dataset loads images by index without breaking them into patches.
    """
    def __init__(self, data_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to the H5 data file
            transform: Transformations to apply to the images
        """
        self.data_file = data_file
        self.transform = transform
        
        # Open the HDF5 file but don't load everything into memory
        self.h5_file = h5py.File(data_file, 'r')
        
        # Get dataset shape
        self.images = self.h5_file['images']
        self.labels = self.h5_file['labels']
        
        # Store total number of images
        self.n_images = self.images.shape[0]
        
        print(f"Dataset loaded from {data_file} with {self.n_images} images")
        
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            image: Image tensor [C, H, W]
            label: Class label
        """
        # Get the image and label
        image = self.images[idx]  # [C, H, W]
        label = self.labels[idx]
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def close(self):
        """Close the HDF5 file when done."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
    
    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset is deleted."""
        self.close()


class Food101PatchDataset(Dataset):
    """
    Dataset that extracts patches from Food101 images for RL training.
    Each patch will be used as an environment state for the RL agent.
    """
    def __init__(self, data_file, patch_size=80, num_patches_h=3, num_patches_w=4):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to the H5 data file with images
            patch_size: Size of each image patch
            num_patches_h: Number of patches in height dimension
            num_patches_w: Number of patches in width dimension
        """
        self.data_file = data_file
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        # Open the HDF5 file
        self.h5_file = h5py.File(data_file, 'r')
        
        # Get dataset info
        self.images = self.h5_file['images']
        self.labels = self.h5_file['labels']
        self.n_images = self.images.shape[0]
        
        # Total number of patches
        self.total_patches = self.n_images * num_patches_h * num_patches_w
        
        print(f"Patch dataset initialized with {self.n_images} images, {self.total_patches} total patches")
    
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        """
        Get a patch from the dataset.
        
        Args:
            idx: Index of the patch to retrieve
            
        Returns:
            patch: Image patch tensor [C, patch_size, patch_size]
            label: Class label of the parent image
        """
        # Convert flat index to image index and patch position
        img_idx = idx // (self.num_patches_h * self.num_patches_w)
        patch_idx = idx % (self.num_patches_h * self.num_patches_w)
        
        # Convert patch index to grid position
        patch_h = patch_idx // self.num_patches_w
        patch_w = patch_idx % self.num_patches_w
        
        # Get the image and label
        image = self.images[img_idx]  # [C, H, W]
        label = self.labels[img_idx]
        
        # Calculate patch coordinates
        y_start = patch_h * self.patch_size
        x_start = patch_w * self.patch_size
        y_end = min(y_start + self.patch_size, image.shape[1])
        x_end = min(x_start + self.patch_size, image.shape[2])
        
        # Extract patch
        patch = image[:, y_start:y_end, x_start:x_end]
        
        # Handle patches at the edges that might be smaller
        if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
            padded_patch = np.zeros((image.shape[0], self.patch_size, self.patch_size), dtype=patch.dtype)
            padded_patch[:, :patch.shape[1], :patch.shape[2]] = patch
            patch = padded_patch
        
        # Convert to float and normalize to [0, 1]
        patch = patch.astype(np.float32) / 255.0
        
        # Convert to torch tensor
        patch = torch.from_numpy(patch).float()
        
        return patch, label
    
    def get_sequential_patches(self, img_idx):
        """
        Get all patches for a specific image in sequential order.
        Useful for creating RL environments that navigate through an image.
        
        Args:
            img_idx: Index of the image
            
        Returns:
            patches: List of image patches [num_patches_h * num_patches_w, C, patch_size, patch_size]
            label: Class label of the image
        """
        # Get the image and label
        image = self.images[img_idx]  # [C, H, W]
        label = self.labels[img_idx]
        
        patches = []
        
        # Extract all patches
        for h in range(self.num_patches_h):
            for w in range(self.num_patches_w):
                # Calculate patch coordinates
                y_start = h * self.patch_size
                x_start = w * self.patch_size
                y_end = min(y_start + self.patch_size, image.shape[1])
                x_end = min(x_start + self.patch_size, image.shape[2])
                
                # Extract patch
                patch = image[:, y_start:y_end, x_start:x_end]
                
                # Handle patches at the edges that might be smaller
                if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
                    padded_patch = np.zeros((image.shape[0], self.patch_size, self.patch_size), dtype=patch.dtype)
                    padded_patch[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded_patch
                
                # Convert to float and normalize to [0, 1]
                patch = patch.astype(np.float32) / 255.0
                
                # Convert to torch tensor
                patch = torch.from_numpy(patch).float()
                
                patches.append(patch)
        
        return patches, label
    
    def close(self):
        """Close the HDF5 file when done."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
    
    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset is deleted."""
        self.close()


class Food101LatentDataset(Dataset):
    """
    Dataset that provides encoded patches for the RL agent.
    This dataset encodes the patches on-the-fly using the VQ-VAE model.
    """
    def __init__(self, data_file, encoder, patch_size=80, num_patches_h=3, num_patches_w=4):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to the H5 data file with images
            encoder: VQVAEWrapper instance for encoding patches
            patch_size: Size of each image patch
            num_patches_h: Number of patches in height dimension
            num_patches_w: Number of patches in width dimension
        """
        self.data_file = data_file
        self.encoder = encoder
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        # Open the HDF5 file
        self.h5_file = h5py.File(data_file, 'r')
        
        # Get dataset info
        self.images = self.h5_file['images']
        self.labels = self.h5_file['labels']
        self.n_images = self.images.shape[0]
        
        # Total number of patches
        self.total_patches = self.n_images * num_patches_h * num_patches_w
        
        # Get latent shape from encoder
        self.latent_shape = encoder.get_latent_grid_shape()
        print(f"Latent patch shape: {self.latent_shape}")
        
        print(f"Latent dataset initialized with {self.n_images} images, {self.total_patches} total patches")
    
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        """
        Get an encoded patch from the dataset.
        
        Args:
            idx: Index of the patch to retrieve
            
        Returns:
            latent: Encoded patch tensor [latent_dim, grid_height, grid_width]
            label: Class label of the parent image
        """
        # Convert flat index to image index and patch position
        img_idx = idx // (self.num_patches_h * self.num_patches_w)
        patch_idx = idx % (self.num_patches_h * self.num_patches_w)
        
        # Convert patch index to grid position
        patch_h = patch_idx // self.num_patches_w
        patch_w = patch_idx % self.num_patches_w
        
        # Get the image and label
        image = self.images[img_idx]  # [C, H, W]
        label = self.labels[img_idx]
        
        # Calculate patch coordinates
        y_start = patch_h * self.patch_size
        x_start = patch_w * self.patch_size
        y_end = min(y_start + self.patch_size, image.shape[1])
        x_end = min(x_start + self.patch_size, image.shape[2])
        
        # Extract patch
        patch = image[:, y_start:y_end, x_start:x_end]
        
        # Handle patches at the edges that might be smaller
        if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
            padded_patch = np.zeros((image.shape[0], self.patch_size, self.patch_size), dtype=patch.dtype)
            padded_patch[:, :patch.shape[1], :patch.shape[2]] = patch
            patch = padded_patch
        
        # Convert to float and normalize to [0, 1]
        patch = patch.astype(np.float32) / 255.0
        
        # Convert to torch tensor and add batch dimension
        patch = torch.from_numpy(patch).float().unsqueeze(0)
        
        # Encode the patch
        with torch.no_grad():
            latent, _ = self.encoder.encode_patch(patch)
            
            # Remove batch dimension
            latent = latent.squeeze(0).cpu()
        
        return latent, label
    
    def get_sequential_latents(self, img_idx):
        """
        Get encoded latents for all patches of a specific image in sequential order.
        
        Args:
            img_idx: Index of the image
            
        Returns:
            latents: List of encoded patches [num_patches_h * num_patches_w, latent_dim, grid_height, grid_width]
            label: Class label of the image
        """
        # Get the image and label
        image = self.images[img_idx]  # [C, H, W]
        label = self.labels[img_idx]
        
        patches = []
        
        # Extract all patches
        for h in range(self.num_patches_h):
            for w in range(self.num_patches_w):
                # Calculate patch coordinates
                y_start = h * self.patch_size
                x_start = w * self.patch_size
                y_end = min(y_start + self.patch_size, image.shape[1])
                x_end = min(x_start + self.patch_size, image.shape[2])
                
                # Extract patch
                patch = image[:, y_start:y_end, x_start:x_end]
                
                # Handle patches at the edges that might be smaller
                if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
                    padded_patch = np.zeros((image.shape[0], self.patch_size, self.patch_size), dtype=patch.dtype)
                    padded_patch[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded_patch
                
                # Convert to float and normalize to [0, 1]
                patch = patch.astype(np.float32) / 255.0
                
                # Convert to torch tensor and add batch dimension
                patch = torch.from_numpy(patch).float().unsqueeze(0)
                
                patches.append(patch)
        
        # Encode all patches
        latents = []
        with torch.no_grad():
            for patch in patches:
                latent, _ = self.encoder.encode_patch(patch)
                latents.append(latent.squeeze(0).cpu())
        
        return latents, label
    
    def close(self):
        """Close the HDF5 file when done."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
    
    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset is deleted."""
        self.close()


def create_dataloaders(data_path, batch_size=32, num_workers=0):
    """
    Create standard data loaders for the Food101 dataset.
    
    Args:
        data_path: Path to the directory containing the H5 data files
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    train_file = os.path.join(data_path, 'train.h5')
    test_file = os.path.join(data_path, 'test.h5')
    
    # Create datasets
    train_dataset = Food101H5Dataset(data_file=train_file)
    test_dataset = Food101H5Dataset(data_file=test_file)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader