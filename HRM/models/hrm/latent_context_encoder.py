"""
Latent Context Encoder - VAE-based task context encoding

🧠 SMART VERSION: Daha güçlü representation, better diversity
Manifesto ruhunda: "Context should capture the ESSENCE of reasoning patterns"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentContextEncoder(nn.Module):
    """
    VAE-based encoder that creates a latent context representation
    from H and L module states.
    
    🧠 MANIFESTO PRINCIPLE: "Different tasks need different reasoning strategies"
    → Encoder should learn to distinguish task characteristics
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        z_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        
        # 🧠 SMART: Multi-scale pooling to capture both local and global patterns
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Encoder: (H, L states) -> latent space
        # Input: concatenated z_H and z_L features
        encoder_layers = []
        
        # 🧠 Include both mean-pooled and attention-pooled features
        input_dim = 4 * hidden_size  # H_mean, H_attn, L_mean, L_attn
        current_dim = input_dim
        
        # Progressive dimensionality reduction with residual connections
        dims = [input_dim]
        for i in range(num_layers):
            next_dim = max(z_dim * 4, current_dim // 2)
            dims.append(next_dim)
            
            encoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = next_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE heads with better initialization
        self.fc_mu = nn.Linear(current_dim, z_dim)
        self.fc_logvar = nn.Linear(current_dim, z_dim)
        
        # 🧠 SMART INIT: Start with reasonable variance (not too small)
        nn.init.constant_(self.fc_logvar.bias, -1.0)  # log(0.37) ≈ -1
        nn.init.normal_(self.fc_logvar.weight, std=0.02)
        
        # Initialize mu with small weights but not zero
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.1)
        nn.init.normal_(self.fc_mu.bias, std=0.01)
    
    def attention_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention-based pooling over sequence dimension
        🧠 Captures important positions in the sequence
        """
        # x: [batch, seq_len, hidden]
        attn_weights = self.attention_pool(x)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (x * attn_weights).sum(dim=1)  # [batch, hidden]
        return pooled
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        
        🧠 SMART: Better clamping for stability
        """
        # Clamp logvar to reasonable range
        logvar = torch.clamp(logvar, min=-5, max=2)
        
        std = torch.exp(0.5 * logvar)
        
        # Clamp std to prevent extreme values
        std = torch.clamp(std, min=1e-3, max=5.0)
        
        if self.training:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Use mean during inference
        
        # Safety check
        z = torch.nan_to_num(z, nan=0.0)
        
        return z
    
    def forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode H and L states into latent context
        
        🧠 MANIFESTO: "Context should be rich and diverse"
        
        Args:
            z_H: High-level state [batch, seq_len, hidden_size]
            z_L: Low-level state [batch, seq_len, hidden_size]
        
        Returns:
            z_context: Latent context [batch, z_dim]
            mu: Mean of latent distribution [batch, z_dim]
            logvar: Log variance of latent distribution [batch, z_dim]
        """
        batch_size = z_H.size(0)
        
        # NaN check at input
        if torch.isnan(z_H).any() or torch.isnan(z_L).any():
            z_H = torch.nan_to_num(z_H, nan=0.0)
            z_L = torch.nan_to_num(z_L, nan=0.0)
        
        # 🧠 SMART: Multi-scale feature extraction
        # 1. Mean pooling (global context)
        h_mean = z_H.mean(dim=1)  # [batch, hidden]
        l_mean = z_L.mean(dim=1)  # [batch, hidden]
        
        # 2. Attention pooling (important positions)
        h_attn = self.attention_pooling(z_H)  # [batch, hidden]
        l_attn = self.attention_pooling(z_L)  # [batch, hidden]
        
        # 🧠 Don't normalize - let the network learn the scale
        # This preserves information about reasoning intensity
        
        # Concatenate all features
        combined = torch.cat([h_mean, h_attn, l_mean, l_attn], dim=-1)  # [batch, 4*hidden]
        
        # Encode
        encoded = self.encoder(combined)  # [batch, encoder_dim]
        
        # NaN check after encoder
        if torch.isnan(encoded).any():
            encoded = torch.nan_to_num(encoded, nan=0.0)
        
        # VAE parameters
        mu = self.fc_mu(encoded)  # [batch, z_dim]
        logvar = self.fc_logvar(encoded)  # [batch, z_dim]
        
        # Clamp mu to prevent extreme values
        mu = torch.clamp(mu, min=-5, max=5)
        
        # Reparameterize
        z_context = self.reparameterize(mu, logvar)
        
        # Final safety check
        z_context = torch.nan_to_num(z_context, nan=0.0)
        mu = torch.nan_to_num(mu, nan=0.0)
        logvar = torch.nan_to_num(logvar, nan=-1.0)
        
        return z_context, mu, logvar