import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.networks import ActorNetwork, CriticNetwork, ViTClassifier

class PPOAgent:
    def __init__(self, 
                 input_dim=16, 
                 dff=512,
                 hidden_dim=256, 
                 action_dim=256,
                 lr=3e-4,
                 gamma=0.99999999,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.dff = dff
        self.total_grid_points = 12*10*10
        
        self.actor = ActorNetwork(input_dim=dff, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
        self.critic = CriticNetwork(input_dim=dff, hidden_dim=hidden_dim).to(device)

        self.classifier = ViTClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=2, num_layers=1,
                 mlp_dim=128, num_patches=12, grid_height=10, grid_width=10, 
                 num_classes=101, dropout=0.1, dff=dff).to(device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

    def process_with_vit(self, state, C1, M1, H1, N1, C2, M2, H2, N2):
        """
        Process a state through the ViT encoder to get embeddings.
        
        Args:
            state: Input state with shape matching the ViT's expected input
                [batch_size, num_patches, latent_dim, grid_height, grid_width]
        
        Returns:
            embeddings: Transformer embeddings including CLS token
                    [batch_size, num_patches*grid_height*grid_width+1, hidden_dim]
        """

        embeddings, C1, M1, H1, N1, C2, M2, H2, N2 = self.classifier.encode(state, C1, M1, H1, N1, C2, M2, H2, N2)
        
        return embeddings, C1, M1, H1, N1, C2, M2, H2, N2

    def classify_state(self, state=None, embeddings=None, return_logits=False):
        """
        Classify a state using the ViT classifier.
        
        Args:
            state: Optional - Input state to classify
                [batch_size, num_patches, latent_dim, grid_height, grid_width]
                If provided, will be encoded first
            embeddings: Optional - Pre-computed embeddings from process_with_vit
                    If provided, will be used directly for classification
            return_logits: Whether to return raw logits or class probabilities
        
        Returns:
            predictions: Either class probabilities [batch_size, num_classes]
                        or class indices [batch_size] depending on return_logits
        """
        if embeddings is None and state is not None:
            # Process the state through the ViT encoder first
            embeddings, C1, M1, H1, N1, C2, M2, H2, N2 = self.process_with_vit(state, C1, M1, H1, N1, C2, M2, H2, N2)
        
        if embeddings is None:
            raise ValueError("Either state or embeddings must be provided")
        
        # Classify using the embeddings
        with torch.no_grad():
            logits = self.classifier.classify(embeddings)
        
        if return_logits:
            return logits
        else:
            # Return probabilities and predicted class
            probs = F.softmax(logits, dim=-1)
            pred_classes = torch.argmax(probs, dim=-1)
            return probs, pred_classes

        
    def get_action(self, state):
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(self.device)
            action, log_prob = self.actor.get_action(state)
        return action, log_prob
    
    def compute_gae(self, rewards, values, dones, next_value, gamma=0.99, tau=0.95):
        values = values + [next_value]
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * tau * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def learn(self, states, actions, old_log_probs, rewards, next_states, dones, epochs=1, patch_size=None, label=None):
        """
        PPO update that loops over the 3 timesteps to accumulate losses,
        then performs a single backward pass and optimizer step.
        """
        # Expect states to be of shape: [patch_size, seq_len, state_dim] where patch_size == self.total_grid_points
        patch_size, seq_len, state_dim = states.shape
        assert seq_len == 3, "Expected sequence length of 3 for each episode"
        
        # Convert inputs to tensors on the proper device
        states_tensor = states.to(self.device)
        actions_tensor = actions.to(self.device)
        old_log_probs_tensor = old_log_probs.to(self.device)
        rewards_tensor = rewards.to(self.device)
        next_states_tensor = next_states.to(self.device)
        dones_tensor = dones.to(self.device)
        
        # Process states through the ViT encoder for each timestep.
        # Note: processed_states holds grid-level features, processed_states_cls holds the CLS token per timestep.
        processed_states = torch.zeros((self.total_grid_points, seq_len, self.dff), device=self.device)
        processed_next_states = torch.zeros((self.total_grid_points, seq_len, self.dff), device=self.device)
        processed_states_cls = torch.zeros((seq_len, self.dff), device=self.device)
        processed_next_states_cls = torch.zeros((seq_len, self.dff), device=self.device)

        # Initialize ViT memory tokens (or any required buffers)
        C1 = torch.zeros(self.total_grid_points+1, self.dff).to(self.device)
        M1 = torch.zeros(self.total_grid_points+1, self.dff).to(self.device)
        H1 = torch.zeros(self.total_grid_points+1, self.dff).to(self.device)
        N1 = torch.zeros(self.total_grid_points+1, self.dff).to(self.device)
        C2 = torch.zeros(self.total_grid_points+1, self.dff).to(self.device)
        M2 = torch.zeros(self.total_grid_points+1, self.dff).to(self.device)
        H2 = torch.zeros(self.total_grid_points+1, self.dff).to(self.device)
        N2 = torch.zeros(self.total_grid_points+1, self.dff).to(self.device)
        
        for i in range(seq_len):
            current_states = states_tensor[:, i, :]
            current_next_states = next_states_tensor[:, i, :]
            
            # Process current states (gradients enabled so that the ViT is trainable)
            _, C1, M1, H1, N1, C2, M2, H2, N2 = self.process_with_vit(current_states, C1, M1, H1, N1, C2, M2, H2, N2)
            processed_states_cls[i, :] = H2[0, :]
            processed_states[:, i, :] = H2[1:, :]
            
            # Process next states (no gradients)
            with torch.no_grad():
                _, _, _, _, _, _, _, H2_next, _ = self.process_with_vit(current_next_states, C1, M1, H1, N1, C2, M2, H2, N2)
                processed_next_states_cls[i, :] = H2_next[0, :]
                processed_next_states[:, i, :] = H2_next[1:, :]

        # Use the CLS token of the last timestep for classification loss
        c = self.classifier.classify(processed_states_cls[-1, :])
        if c.dim() == 1:
            c = c.unsqueeze(0)
        # Create a batch of size 1 for the label (remains unchanged)
        label = torch.tensor([label], device=self.device, dtype=torch.long)
        # print('c', c)
        predicted_loss = F.cross_entropy(c, label)
        
        # # Get value estimates from the critic (reshape to [1, seq_len])
        values = self.critic(processed_states_cls.reshape(-1, self.dff)).view(1, seq_len)
        next_values = self.critic(processed_next_states_cls.reshape(-1, self.dff)).view(1, seq_len)
        
        # Compute advantages using GAE over the timesteps
        advantages = torch.zeros_like(rewards_tensor, device=self.device)
        last_gae = torch.zeros(1, device=self.device)
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_val = next_values[:, t].detach() * (1 - dones_tensor[:, t])
            else:
                next_val = values[:, t+1].detach()
            delta = rewards_tensor[:, t] + self.gamma * next_val * (1 - dones_tensor[:, t]) - values[:, t]
            last_gae = delta + self.gamma * 0.95 * (1 - dones_tensor[:, t]) * last_gae
            advantages[:, t] = last_gae
        returns = advantages + values
        
        # Normalize advantages (important for stable PPO training)
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # print('norm_advantages', norm_advantages)
        # Initialize cumulative losses for the sequence
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        
        # Loop through each timestep, compute and accumulate losses.
        for t in range(seq_len):
            # Evaluate the actor on the grid-level features for timestep t.
            new_log_probs_t, entropy_t = self.actor.evaluate_actions(processed_states[:, t, :], actions_tensor[:, t])
            ratio_t = torch.exp(new_log_probs_t - old_log_probs_tensor[:, t])
            # print('ratio_t', ratio_t.shape)
            # print('new_log_probs_t', new_log_probs_t)
            # print('old_log_probs_tensor[:, t]', old_log_probs_tensor[:, t])
            adv_t = norm_advantages[0, t]  # batch size is 1, so select the first element
            # print('adv_t', adv_t.shape)
            surr1_t = ratio_t * adv_t
            surr2_t = torch.clamp(ratio_t, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_t
            # print('surr1_t', surr1_t.shape)
            # print('surr2_t', surr2_t.shape)
            actor_loss_t = -torch.min(surr1_t, surr2_t).mean()
            total_actor_loss += actor_loss_t
            

            # Critic loss is computed on the CLS token features for timestep t.
            value_t = self.critic(processed_states_cls[t, :].unsqueeze(0))
            critic_loss_t = F.mse_loss(value_t.view(-1), returns[:, t].view(-1))
            total_critic_loss += critic_loss_t
            
            # Accumulate the entropy loss.
            total_entropy_loss += -entropy_t.mean()
        
        # Optionally average the losses over timesteps.
        total_actor_loss = total_actor_loss / seq_len
        total_critic_loss = total_critic_loss / seq_len
        total_entropy_loss = total_entropy_loss / seq_len
        
        # Combine losses using the same scaling factors as before.
        total_loss = 1000000.0*total_actor_loss + self.value_coef * total_critic_loss + self.entropy_coef * total_entropy_loss + 10.1 * predicted_loss
        # total_loss = 1.1 * predicted_loss
        
        # Zero gradients, backpropagate, and perform the optimizer steps for all networks.
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        self.optimizer_classifier.zero_grad()
        total_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()
        self.optimizer_classifier.step()
        
        return {
            'actor_loss': 1000000.0*total_actor_loss.item(),
            'critic_loss': total_critic_loss.item(),
            'entropy': total_entropy_loss.item(),
            'predicted_loss': predicted_loss.item()
        }
    
        # return {
        #         'predicted_loss': predicted_loss.item()
        #     }


        
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'optimizer_classifier': self.optimizer_classifier.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        self.optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier'])