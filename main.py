import torch
import numpy as np
from models.ppo_agent import PPOAgent
from utils.data_utils import generate_dummy_data
import matplotlib.pyplot as plt

def main():
    # Configuration
    state_dim = 16
    hidden_dim = 256
    action_dim = 256
    patch_size = 1200
    seq_len = 3
    epochs = 5
    training_iterations = 100
    
    # Initialize the PPO agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    agent = PPOAgent(
        input_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        device=device
    )
    
    # Training loop
    actor_losses = []
    critic_losses = []
    entropies = []
    
    for iteration in range(training_iterations):
        # Generate dummy batch of data
        states, actions, log_probs, rewards, next_states, dones = generate_dummy_data(
            patch_size=patch_size,
            seq_len=seq_len,
            state_dim=state_dim,
            action_dim=action_dim
        )

        print('states', states.shape)
        print('actions', actions.shape)
        print('log_probs', log_probs.shape)
        print('rewards', rewards.shape)
        print('next_states', next_states.shape)
        print('dones', dones.shape)
        # Train the agent
        metrics = agent.learn(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            epochs=epochs,
            patch_size=1200  # Mini-batch size for PPO updates
        )
        
        # Store metrics
        actor_losses.append(metrics['actor_loss'])
        critic_losses.append(metrics['critic_loss'])
        entropies.append(metrics['entropy'])
        
        # Print progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{training_iterations}")
            print(f"  Actor Loss: {metrics['actor_loss']:.4f}")
            print(f"  Critic Loss: {metrics['critic_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
    
    # Save the trained agent
    agent.save("./ppo_model.pt")
    print("Model saved to ./ppo_model.pt")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(actor_losses)
    plt.title('Actor Loss')
    plt.xlabel('Iterations')
    
    plt.subplot(1, 3, 2)
    plt.plot(critic_losses)
    plt.title('Critic Loss')
    plt.xlabel('Iterations')
    
    plt.subplot(1, 3, 3)
    plt.plot(entropies)
    plt.title('Entropy')
    plt.xlabel('Iterations')
    
    plt.tight_layout()
    plt.savefig("./training_curves.png")
    print("Training curves saved to ./training_curves.png")

if __name__ == "__main__":
    main()