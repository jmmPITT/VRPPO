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
# from models.networks import ActorNetwork, CriticNetwork
from models.ppo_agent import PPOAgent
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
    
    parser.add_argument('--num_episodes', type=int, default=5000000,
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
    # print(f"Loading dataset from {args.data_path}")
    # train_file = os.path.join(args.data_path, 'train.h5')
    # dataset = Food101LatentDataset(
    #     data_file=train_file,
    #     encoder=encoder,
    #     patch_size=args.patch_size,
    #     num_patches_h=args.num_patches_h,
    #     num_patches_w=args.num_patches_w
    # )
    
    # Create environment
    print(f"Creating environment with sequence length {args.sequence_length}")
    env = VQEnvironment(
        encoder=encoder,
        sequence_length=args.sequence_length,
        device=device
    )

    agent = PPOAgent(
        input_dim=16,
        dff=512,
        hidden_dim=256,
        action_dim=256,
        lr=3e-5,
        gamma=0.99999999,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.001,
        device=device
    )

    # agent.load("./ppo_model.pt")
    
    # Get state and action shapes
    state_shape = env.get_state_shape()
    action_shape = env.get_action_shape()
    print(f"State shape: {state_shape}")
    print(f"Action shape: {action_shape}")

    # Initialize arrays to collect data across episodes
    # all_states = []
    # all_actions = []
    # all_log_probs = []
    # all_rewards = []
    # all_next_states = []
    # all_dones = []

    # Training loop
    actor_losses = []
    critic_losses = []
    entropies = []
    predicted_losses = []
    rewards_list = []

        # Run episodes with random actions
    for episode in range(args.num_episodes):
        # print(f"\nRunning episode {episode+1}/{args.num_episodes}")
        
        # Reset the environment
        state = env.reset()
        done = False
        step = 0
        
        # Store states for visualization
        episode_states = [state.detach()]
        
        # Initialize episode data collection
        episode_states_list = []
        episode_actions_list = []
        episode_log_probs_list = []
        episode_rewards_list = []
        episode_next_states_list = []
        episode_dones_list = []

        with torch.no_grad():
            C1 = torch.zeros(12*10*10+1, 512).to(device)
            M1 = torch.zeros(12*10*10+1, 512).to(device)
            H1 = torch.zeros(12*10*10+1, 512).to(device)
            N1 = torch.zeros(12*10*10+1, 512).to(device)
            C2 = torch.zeros(12*10*10+1, 512).to(device)
            M2 = torch.zeros(12*10*10+1, 512).to(device)
            H2 = torch.zeros(12*10*10+1, 512).to(device)
            N2 = torch.zeros(12*10*10+1, 512).to(device)
        
        # Run until done
        while not done:
            # Store current state
            episode_states_list.append(state)
            
            # Get action from agent
            # print(f"State shape: {state.shape}")
            with torch.no_grad():

                _, C1, M1, H1, N1, C2, M2, H2, N2 = agent.process_with_vit(state, C1, M1, H1, N1, C2, M2, H2, N2)
                # cls = H2[0, :]
                s = H2[1:, :]
            # print(f"State shape after processing: {state.shape}")
            actions, log_probs = agent.get_action(s)
            # print(f"actions shape in main_two: {actions.shape}")
            # Store action and log_probs
            episode_actions_list.append(actions)
            episode_log_probs_list.append(log_probs)
            
            # Take a step in the environment
            next_state, reward, done, info = env.step(actions)
            label = info['label']

            if done:
                with torch.no_grad():
                    c = agent.classifier.classify(H2)
                    # print('c', c)
                    # print('label', label)
                    predicted_label = torch.argmax(c)
                    # print('predicted_label', predicted_label)
                    reward = 10.0 if predicted_label == label else -10.0
            
                    rewards_list.append(reward)
            # Store remaining data
            episode_next_states_list.append(next_state)
            episode_rewards_list.append(reward)
            episode_dones_list.append(1.0 if done else 0.0)
            
            # Store the state for visualization
            episode_states.append(next_state.detach())
            
            
            # Update state for next iteration
            state = next_state
            # print(f"state shape in main_two: {state.shape}")
            step += 1
        
        # Convert episode lists to tensors with proper shapes
        # Each list element is a tensor of shape [1200, 16]
        states_tensor = torch.stack(episode_states_list, dim=1)
        next_states_tensor = torch.stack(episode_next_states_list, dim=1)
        
        # For actions, log_probs, rewards, and dones, reshape to [1200, 3]
        actions_tensor = torch.stack(episode_actions_list, dim=1)
        log_probs_tensor = torch.stack(episode_log_probs_list, dim=1)
        rewards_tensor = torch.tensor(episode_rewards_list).repeat(1, 1)
        dones_tensor = torch.tensor(episode_dones_list).repeat(1, 1)
        
        # Append to overall collection
        # all_states.append(states_tensor)
        # all_actions.append(actions_tensor)
        # all_log_probs.append(log_probs_tensor)
        # all_rewards.append(rewards_tensor)
        # all_next_states.append(next_states_tensor)
        # all_dones.append(dones_tensor)

        # Concatenate all episodes into the final arrays
        states = states_tensor
        actions = actions_tensor
        log_probs = log_probs_tensor
        rewards = rewards_tensor
        next_states = next_states_tensor
        dones = dones_tensor



        metrics = agent.learn(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            label=label,
            epochs=1,
            patch_size=1200  # Changed from patch_size to batch_size
        )

        actor_losses.append(metrics['actor_loss'])
        critic_losses.append(metrics['critic_loss'])
        entropies.append(metrics['entropy'])
        predicted_losses.append(metrics['predicted_loss'])

        # Print progress
        if episode % 500 == 0:
            print(f"Iteration { episode+1  }/{args.num_episodes}")
            print(f"  Actor Loss: {sum(actor_losses[-500:])/500:.4f}")
            print(f"  Critic Loss: {sum(critic_losses[-500:])/500:.4f}")
            print(f"  Entropy: {sum(entropies[-500:])/500:.4f}")
            print(f"  Predicted Loss: {sum(predicted_losses[-500:])/500:.4f}")
            print(f"  Rewards: {sum(rewards_list[-500:])/500:.4f}")

            # Save the trained agent
            agent.save("./ppo_model.pt")
            print("Model saved to ./ppo_model.pt")

            # Initialize episode data collection
            episode_states_list = []
            episode_actions_list = []
            episode_log_probs_list = []
            episode_rewards_list = []
            episode_next_states_list = []
            episode_dones_list = []
            rewards_list = []
            # # Plot training curves
            # plt.figure(figsize=(12, 4))
            
            # plt.subplot(1, 4, 1)
            # plt.plot(actor_losses)
            # plt.title('Actor Loss')
            # plt.xlabel('Iterations')
            
            # plt.subplot(1, 4, 2)
            # plt.plot(critic_losses)
            # plt.title('Critic Loss')
            # plt.xlabel('Iterations')
            
            # plt.subplot(1, 4, 3)
            # plt.plot(entropies)
            # plt.title('Entropy')
            # plt.xlabel('Iterations')
            
            # plt.subplot(1, 4, 4)
            # plt.plot(predicted_losses)
            # plt.title('Predicted Loss')
            # plt.xlabel('Iterations')
            
            # plt.tight_layout()
            # plt.savefig("./training_curves.png")
            # print("Training curves saved to ./training_curves.png")
        
    # Close the environment and dataset
    # env.close()
    # dataset.close()
    
    # print("\nEnvironment test complete!")

if __name__ == '__main__':
    args = parse_args()
    test_environment(args)