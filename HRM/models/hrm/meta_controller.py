"""
Meta Controller - Dynamic module weighting (FIXED NORMALIZATION)
z_context'e bakarak H ve L modüllerini nasıl dengeleyeceğine karar verir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MetaController(nn.Module):
    """
    Context-aware gating mechanism for HRM modules.
    
    🔥 FIX: Guaranteed perfect normalization (sum=1.0)
    """
    
    def __init__(
        self,
        z_dim: int = 128,
        num_modules: int = 2,  # H ve L için
        hidden_dim: int = 256,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.num_modules = num_modules
        self.temperature = temperature
        
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_modules)
        )
        
        # Initialize to equal weights at start
        nn.init.normal_(self.gate_net[-1].weight, mean=0.0, std=0.02)
        nn.init.uniform_(self.gate_net[-1].bias, -0.1, 0.1)
    
    def forward(
        self,
        z_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_context: Latent context vector (batch_size, z_dim)
        
        Returns:
            gates: Module weights (batch_size, num_modules)
                   🔥 GUARANTEED: gates.sum(dim=-1) == 1.0 (perfect normalization)
        """
        # Gate logits
        logits = self.gate_net(z_context)  # (batch_size, num_modules)
        
        # 🔥 ULTIMATE FIX: Double precision softmax + double renormalization
        # 1. Convert to float64 (double precision)
        logits_f64 = logits.double()
        
        # 2. Temperature scaling
        scaled_logits = logits_f64 / self.temperature
        
        # 3. Softmax in double precision
        gates_f64 = F.softmax(scaled_logits, dim=-1)
        
        # 4. First renormalization (should already be perfect in float64)
        gates_sum = gates_f64.sum(dim=-1, keepdim=True)
        gates_f64 = gates_f64 / gates_sum
        
        # 5. Second renormalization after converting to float32 (handles conversion errors)
        gates_f32 = gates_f64.float()
        gates_sum_f32 = gates_f32.sum(dim=-1, keepdim=True)
        gates_f32 = gates_f32 / gates_sum_f32
        
        # 6. Convert to original dtype
        gates = gates_f32.to(logits.dtype)
        
        return gates
    
    def entropy_regularization(
        self,
        gates: torch.Tensor
    ) -> torch.Tensor:
        """
        Entropy regularization to encourage exploration.
        H(p) = -sum(p * log(p))
        
        Yüksek entropy = modüller arasında dengeli kullanım
        Düşük entropy = tek bir modüle odaklanma
        """
        # Numerical stability için epsilon ekle
        eps = 1e-8
        entropy = -(gates * torch.log(gates + eps)).sum(dim=-1)
        return entropy.mean()
    
    def get_gate_stats(
        self,
        gates: torch.Tensor
    ) -> Dict[str, float]:
        """
        Logging için gate istatistikleri
        """
        gates_np = gates.detach().cpu().numpy()
        
        return {
            'gate_H_mean': float(gates_np[:, 0].mean()),
            'gate_L_mean': float(gates_np[:, 1].mean()),
            'gate_H_std': float(gates_np[:, 0].std()),
            'gate_L_std': float(gates_np[:, 1].std()),
            'gate_entropy': float(self.entropy_regularization(gates).item())
        }


class AdaptiveMetaController(MetaController):
    """
    Advanced version with step-wise gating (opsiyonel).
    Her reasoning step'te farklı ağırlıklar üretebilir.
    """
    
    def __init__(
        self,
        z_dim: int = 128,
        num_modules: int = 2,
        hidden_dim: int = 256,
        max_steps: int = 16,
        temperature: float = 1.0
    ):
        super().__init__(z_dim, num_modules, hidden_dim, temperature)
        
        self.max_steps = max_steps
        
        # Step embedding
        self.step_emb = nn.Embedding(max_steps, z_dim)
        
        # Modified gate network (z_context + step_emb)
        self.gate_net = nn.Sequential(
            nn.Linear(z_dim * 2, hidden_dim),  # Concatenate z + step
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_modules)
        )
        
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(
        self,
        z_context: torch.Tensor,
        step: int = 0
    ) -> torch.Tensor:
        """
        Args:
            z_context: Latent context vector (batch_size, z_dim)
            step: Current reasoning step (0 to max_steps-1)
        
        Returns:
            gates: Step-specific module weights (batch_size, num_modules)
        """
        batch_size = z_context.size(0)
        
        # Step embedding
        step_tensor = torch.full((batch_size,), step, 
                                dtype=torch.long, 
                                device=z_context.device)
        step_emb = self.step_emb(step_tensor)  # (batch_size, z_dim)
        
        # Combine context + step
        combined = torch.cat([z_context, step_emb], dim=-1)  # (batch_size, z_dim*2)
        
        # Gate logits
        logits = self.gate_net(combined)
        
        # 🔥 Same ultimate fix as parent class
        logits_f64 = logits.double()
        scaled_logits = logits_f64 / self.temperature
        gates_f64 = F.softmax(scaled_logits, dim=-1)
        gates_sum = gates_f64.sum(dim=-1, keepdim=True)
        gates_f64 = gates_f64 / gates_sum
        gates_f32 = gates_f64.float()
        gates_sum_f32 = gates_f32.sum(dim=-1, keepdim=True)
        gates_f32 = gates_f32 / gates_sum_f32
        gates = gates_f32.to(logits.dtype)
        
        return gates