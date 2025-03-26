import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torch as T

class CustomLSTMCell(nn.Module):
    """
    Custom LSTM Cell implementation with numerically stable operations.
    Uses a modified version of the LSTM architecture for improved stability and performance.
    """
    def __init__(self, patch_size=12*25, d_model=64, dff=256):
        super(CustomLSTMCell, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff  # Hidden dimension size
        self.input_size = self.d_model

        # Linear transformations for hidden state
        self.WI = nn.Linear(self.dff, self.dff)  # Input gate
        self.WF = nn.Linear(self.dff, self.dff)  # Forget gate
        self.WO = nn.Linear(self.dff, self.dff)  # Output gate
        self.WZ = nn.Linear(self.dff, self.dff)  # Cell state update

        # Linear transformations for input
        self.RI = nn.Linear(self.input_size, self.dff)  # Input gate
        self.RF = nn.Linear(self.input_size, self.dff)  # Forget gate
        self.RO = nn.Linear(self.input_size, self.dff)  # Output gate 
        self.RZ = nn.Linear(self.input_size, self.dff)  # Cell state update

    def forward(self, Zi, Ci, Mi, Hi, Ni):
        """
        Forward pass through the LSTM cell.
        
        Args:
            Zi: Input data
            Ci: Cell state
            Mi: Maximum value tracker for numerical stability
            Hi: Hidden state
            Ni: Normalization factor
            
        Returns:
            C_t: Updated cell state
            M_t: Updated maximum value tracker
            H_t: Updated hidden state
            N_t: Updated normalization factor
        """

        # Reshape inputs to expected dimensions
        Zi = Zi.view(-1, self.patch_size, self.input_size)
        Ci = Ci.view(-1, self.patch_size, self.dff)
        Hi = Hi.view(-1, self.patch_size, self.dff)
        Ni = Ni.view(-1, self.patch_size, self.dff)

        # Store previous states
        C_prev = Ci
        M_prev = Mi
        H_prev = Hi
        N_prev = Ni

        # Gate computations
        I_tilde = self.WI(H_prev) + self.RI(Zi)  # Input gate pre-activation
        F_tilde = self.WF(H_prev) + self.RF(Zi)  # Forget gate pre-activation
        O_tilde = self.WO(H_prev) + self.RO(Zi)  # Output gate pre-activation
        Z_tilde = self.WZ(H_prev) + self.RZ(Zi)  # Cell update pre-activation

        # Numerically stable computation to prevent exploding/vanishing gradients
        M_t = torch.max(F_tilde + M_prev, I_tilde)
        I_t = torch.exp(I_tilde - M_t)  # Input gate activation
        F_t = torch.exp(F_tilde + M_prev - M_t)  # Forget gate activation

        # Final gate and state computations
        O_t = F.sigmoid(O_tilde)  # Output gate activation
        N_t = F_t*N_prev + I_t  # Update normalization factor
        Z_t = F.tanh(Z_tilde)  # Cell state update activation
        C_t = (C_prev * F_t + Z_t * I_t)  # Update cell state
        H_t = O_t * (C_t / N_t)  # Update hidden state with normalization

        return C_t, M_t, H_t, N_t

class TransformerBlock(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """
    def __init__(self, input_dims=64, patch_length=12*25, dff=256, dropout=0.01, num_heads=2):
        super(TransformerBlock, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate
        
        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layersC.dropout)
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model*4)
        self.linear2 = nn.Linear(self.d_model*4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, H):
        """
        Forward pass through the transformer block.
        
        Args:
            state: Input state tensor
            H1: First hidden state from LSTM
            H2: Second hidden state from LSTM
            
        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        state = state.view(-1, self.patch_length, self.input_dims)
        
        # Concatenate hidden states for context
        # print(H1.shape, H2.shape)
        # H = torch.cat((H1, H2), dim=2).view(-1, self.patch_length, self.dff*2)
        H = H.view(-1, self.patch_length, self.dff)
        batch_size = state.shape[0]
        # print("H.shape", H.shape)
        # Attention mechanism
        # Construct query, key, and value with both input state and hidden state context
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
                self.W_C1q(H)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(H)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(H)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, _ = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)
        
        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm))
        Z4 = self.linear2(Z3)
        Z5 = Z1 + self.dropout2(Z4)

        return Z5

    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            
        Returns:
            attention_output: Output after applying attention to values
            attention_weights: The attention weights
        """
        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute attention output
        attention_output = torch.matmul(attention_weights, v)
        
        return attention_output, attention_weights
    
    

class ActorNetwork(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, action_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs
    
    def get_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def evaluate_actions(self, states, actions):
        action_probs = self.forward(states)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_log_probs, dist_entropy


class CriticNetwork(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value
    

class ViTClassifier(nn.Module):
    """
    Alternative classifier using a Vision Transformer architecture.
    This version uses a proper ViT with attention mechanism.
    """
    def __init__(self, input_dim, hidden_dim=64, num_heads=2, num_layers=2,
                 mlp_dim=128, num_patches=12, grid_height=19, grid_width=19, 
                 num_classes=101, dropout=0.1, dff=512):
        """
        Initialize the Vision Transformer classifier.
        
        Args:
            input_dim: Dimension of each latent vector
            hidden_dim: Hidden dimension of the transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_dim: Dimension of the MLP in the transformer
            num_patches: Number of patches per image (e.g., 12 for a 3x4 grid)
            grid_height: Height of the grid for each patch latent
            grid_width: Width of the grid for each patch latent
            num_classes: Number of output classes (101 for Food101)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_classes = num_classes
        self.dff = dff
        # Total number of grid points across all patches
        self.total_grid_points = num_patches * grid_height * grid_width
        # print(f"self.total_grid_points: {self.total_grid_points}")
        # Input projection to hidden dimension
        # self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(self.total_grid_points + 1, input_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        
        # Transformer encoder
        self.transformer1 = TransformerBlock(
            input_dims=input_dim,
            patch_length=self.total_grid_points+1,
            dff=self.dff,
            dropout=dropout,
            num_heads=num_heads
        )

        self.lstm1 = CustomLSTMCell(
            patch_size=self.total_grid_points+1,
            d_model=input_dim,
            dff = self.dff
        )

        # Transformer encoder
        self.transformer2 = TransformerBlock(
            input_dims=input_dim,
            patch_length=self.total_grid_points+1,
            dff=self.dff,
            dropout=dropout,
            num_heads=num_heads
        )

        self.lstm2 = CustomLSTMCell(
            patch_size=self.total_grid_points+1,
            d_model=input_dim,
            dff = self.dff
        )
        
        # Output projection
        self.fc1_classifier = nn.Linear(self.dff, self.dff)
        self.fc2_classifier = nn.Linear(self.dff, self.dff)
        self.fc3_classifier = nn.Linear(self.dff, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights for the model."""
        # Initialize position embedding
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Initialize class token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x, C1, M1, H1, N1, C2, M2, H2, N2):
        """
        Encode input through the Vision Transformer and output embeddings.
        
        Args:
            x: Tensor of shape [batch_size, num_patches, latent_dim, grid_height, grid_width]
            
        Returns:
            embeddings: Transformer output embeddings 
                       [batch_size, num_patches*grid_height*grid_width+1, hidden_dim]
            The first embedding (index 0) is the CLS token.
        """
        x = x.view(-1,  self.total_grid_points, self.input_dim)
        batch_size = x.shape[0]
        # print(f"x shape: {x.shape}")
        # Reshape to properly handle dimensions
        # x = x.permute(0, 1, 3, 4, 2)  # [batch_size, num_patches, grid_height, grid_width, latent_dim]
        x = x.reshape(batch_size, self.total_grid_points, self.input_dim)  # [batch_size, total_grid_points, latent_dim]
        # print(f"x shape: {x.shape}")
        # Project to hidden dimension
        # x = self.input_projection(x)  # [batch_size, total_grid_points, hidden_dim]
        
        # Add class token
        # if x.shape[0] == 1200:
        cls_tokens = self.cls_token.expand(1, 1,-1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        # print(f"x shape: {x.shape}")
        # print(f"self.pos_embedding shape: {self.pos_embedding.shape}")
        x = x.view(-1, self.input_dim)
        x = x + self.pos_embedding
        
        # Apply transformer
        Z1 = self.transformer1(x, H1)  # [batch_size, total_grid_points+1, hidden_dim]
        C1, M1, H1, N1 = self.lstm1(Z1, C1, M1, H1, N1)
        
        Z2 = self.transformer2(Z1, H1)  # [batch_size, total_grid_points+1, hidden_dim]
        C2, M2, H2, N2 = self.lstm2(Z2, C2, M2, H2, N2)
        
        return Z2.view(-1, self.input_dim), C1.view(-1, self.dff), M1.view(-1, self.dff), H1.view(-1, self.dff), N1.view(-1, self.dff), C2.view(-1, self.dff), M2.view(-1, self.dff), H2.view(-1, self.dff), N2.view(-1, self.dff)
    
    def classify(self, h_cls):
        """
        Classify from transformer embeddings using the CLS token.
        
        Args:
            embeddings: Transformer output with CLS token at index 0
                       [batch_size, total_grid_points+1, hidden_dim]
            
        print(f"Dataset loaded from {data_file} with {self.n_images} images")
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Use the class token for classification
        cls_token = h_cls
        
        # Classification
        cls_token = F.elu(self.fc1_classifier(cls_token))
        cls_token = F.elu(self.fc2_classifier(cls_token))
        logits = self.fc3_classifier(cls_token)
        
        return logits
    
    def forward(self, x, C1, M1, H1, N1, C2, M2, H2, N2):
        """
        Forward pass through the Vision Transformer.
        
        Args:
            x: Tensor of shape [batch_size, num_patches, latent_dim, grid_height, grid_width]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Get embeddings from encoder
        embeddings, C1, M1, H1, N1, C2, M2, H2, N2 = self.encode(x, C1, M1, H1, N1, C2, M2, H2, N2)
        
        # Get classification from CLS token
        logits = self.classify(H2)
        
        return logits