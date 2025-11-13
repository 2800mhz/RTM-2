"""
Meta-Learning Loss Functions

🧠 MANIFESTO: "The model should learn WHEN to use which reasoning mode"

These losses ensure:
1. VAE learns meaningful task representations (diversity)
2. Controller learns to adapt gates to tasks (differentiation)
3. System maintains stability (regularization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MetaLearningLoss(nn.Module):
    """
    Combined loss for meta-learning components
    
    🧠 PHILOSOPHY:
    - KL loss: Encourages diverse, well-distributed latent contexts
    - Gate entropy: Prevents collapse to always 50/50
    - Gate commitment: Encourages confident decisions when appropriate
    """
    
    def __init__(
        self,
        kl_weight: float = 0.001,  # 🧠 Small KL to not overpower task loss
        entropy_weight: float = 0.01,  # 🧠 Encourage exploration
        diversity_weight: float = 0.01,  # 🧠 Prevent context collapse
    ):
        super().__init__()
        
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between learned distribution and standard normal
        
        🧠 MANIFESTO: "Context should be diverse but not chaotic"
        
        KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()
    
    def gate_entropy(self, gates: torch.Tensor) -> torch.Tensor:
        """
        Entropy of gate distribution
        
        🧠 MANIFESTO: "Don't always be 50/50, learn to commit"
        
        We want NEGATIVE entropy loss to encourage LOW entropy (confident gates)
        But not too low - we still want the ability to be balanced when needed
        
        Target entropy ≈ 0.5 * log(2) = 0.35 (slightly biased, not uniform)
        """
        # Entropy: -sum(p * log(p))
        entropy = -(gates * torch.log(gates + 1e-8)).sum(dim=-1)
        
        # 🧠 We want entropy around 0.35 (slightly confident)
        # Penalize being too uniform (entropy=0.693) or too extreme (entropy=0)
        target_entropy = 0.35
        entropy_loss = (entropy - target_entropy).pow(2).mean()
        
        return entropy_loss
    
    def diversity_loss(self, z_context: torch.Tensor) -> torch.Tensor:
        """
        Encourage diversity in latent contexts across batch
        
        🧠 MANIFESTO: "Different examples should produce different contexts"
        
        Maximize variance of contexts within batch
        """
        # Compute variance along batch dimension
        mean = z_context.mean(dim=0, keepdim=True)
        variance = ((z_context - mean) ** 2).mean()
        
        # We want HIGH variance, so minimize negative variance
        return 1.0 / (variance + 1e-6)  
    def forward(
        self,
        meta_info: Dict[str, torch.Tensor],
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all meta-learning losses
        
        Args:
            meta_info: Dict with keys 'mu', 'logvar', 'gates', 'z_context'
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            losses: Dict with individual losses and total
        """
        mu = meta_info['mu']
        logvar = meta_info['logvar']
        gates = meta_info['gates']
        z_context = meta_info['z_context']
        
        # Individual losses
        kl_loss = self.kl_divergence(mu, logvar)
        entropy_loss = self.gate_entropy(gates)
        div_loss = self.diversity_loss(z_context)
        
        # Total meta loss
        total_meta_loss = (
            self.kl_weight * kl_loss +
            self.entropy_weight * entropy_loss +
            self.diversity_weight * div_loss
        )
        
        return {
            'meta/kl_loss': kl_loss,
            'meta/entropy_loss': entropy_loss,
            'meta/diversity_loss': div_loss,
            'meta/total_loss': total_meta_loss,
            
            # Metrics (for monitoring)
            'meta/gate_H_mean': gates[:, 0].mean(),
            'meta/gate_L_mean': gates[:, 1].mean(),
            'meta/gate_std': gates.std(),
            'meta/z_context_norm': z_context.norm(dim=-1).mean(),
            'meta/mu_norm': mu.norm(dim=-1).mean(),
            'meta/logvar_mean': logvar.mean(),
        }


def compute_meta_loss(
    meta_info: Dict[str, torch.Tensor],
    kl_weight: float = 0.001,
    entropy_weight: float = 0.01,
    diversity_weight: float = 0.01
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Convenience function to compute meta losses
    
    🧠 USAGE in training loop:
    ```python
    carry, logits, meta_info = model.inner(carry, batch)
    
    # Task loss (cross-entropy, etc)
    task_loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
    
    # Meta loss
    meta_loss, metrics = compute_meta_loss(meta_info)
    
    # Total loss
    total_loss = task_loss + meta_loss
    total_loss.backward()
    ```
    
    Returns:
        loss: Scalar tensor (total meta loss)
        metrics: Dict of individual losses and metrics
    """
    loss_fn = MetaLearningLoss(
        kl_weight=kl_weight,
        entropy_weight=entropy_weight,
        diversity_weight=diversity_weight
    )
    
    losses = loss_fn(meta_info)
    total_loss = losses['meta/total_loss']
    
    return total_loss, losses


# 🧠 ADAPTIVE WEIGHTING: Adjust loss weights during training
class AdaptiveMetaLoss(nn.Module):
    """
    Automatically adjust meta loss weights based on training progress
    
    🧠 MANIFESTO: "Be gentle early, be demanding later"
    
    Early training: Focus on task loss, gentle meta regularization
    Later training: Increase meta loss to encourage specialization
    """
    
    def __init__(
        self,
        kl_weight_start: float = 0.0001,
        kl_weight_end: float = 0.001,
        entropy_weight_start: float = 0.001,
        entropy_weight_end: float = 0.01,
        warmup_steps: int = 10000
    ):
        super().__init__()
        
        self.kl_weight_start = kl_weight_start
        self.kl_weight_end = kl_weight_end
        self.entropy_weight_start = entropy_weight_start
        self.entropy_weight_end = entropy_weight_end
        self.warmup_steps = warmup_steps
        
        self.register_buffer('step', torch.tensor(0))
    
    def get_weights(self):
        """Get current loss weights based on training step"""
        progress = min(1.0, self.step.item() / self.warmup_steps)
        
        kl_weight = self.kl_weight_start + progress * (self.kl_weight_end - self.kl_weight_start)
        entropy_weight = self.entropy_weight_start + progress * (self.entropy_weight_end - self.entropy_weight_start)
        
        return kl_weight, entropy_weight
    
    def forward(self, meta_info: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute losses with adaptive weights"""
        kl_w, entropy_w = self.get_weights()
        
        loss_fn = MetaLearningLoss(
            kl_weight=kl_w,
            entropy_weight=entropy_w,
            diversity_weight=0.01
        )
        
        losses = loss_fn(meta_info)
        
        # Increment step
        self.step += 1
        
        # Add weight info to metrics
        losses['meta/kl_weight'] = kl_w
        losses['meta/entropy_weight'] = entropy_w
        
        return losses