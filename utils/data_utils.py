import numpy as np
import torch

def generate_dummy_data(patch_size=1200, seq_len=3, state_dim=16, action_dim=256):
    """
    Generate dummy data for testing the PPO agent
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence (fixed at 3 for this task)
        state_dim: Dimension of the state vector
        action_dim: Dimension of the action space
        
    Returns:
        states: (batch_size, seq_len, state_dim) array of states
        actions: (batch_size, seq_len) array of actions
        log_probs: (batch_size, seq_len) array of log probabilities
        rewards: (batch_size, seq_len) array of rewards
        next_states: (batch_size, seq_len, state_dim) array of next states
        dones: (batch_size, seq_len) array of done flags
    """
    # Generate random states and next states
    states = np.random.randn(patch_size, seq_len, state_dim).astype(np.float32)
    next_states = np.random.randn(patch_size, seq_len, state_dim).astype(np.float32)
    
    # Generate random actions (integers between 0 and action_dim-1)
    actions = np.random.randint(0, action_dim, size=(patch_size, seq_len))
    
    # Generate random log probabilities
    log_probs = np.random.randn(patch_size, seq_len).astype(np.float32)
    
    # Generate random rewards
    rewards = np.random.randn(patch_size, seq_len).astype(np.float32)
    
    # Generate random done flags (mostly False, with some True)
    dones = np.random.random((patch_size, seq_len)) < 0.1
    
    return states, actions, log_probs, rewards, next_states, dones