"""
Meta-Learning Loss Head for HRM-Free-Meta
"""
print("Meta-Learning Loss Head for HRM-Free-Meta")

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from models.losses import ACTLossHead, stablemax_cross_entropy


class MetaLearningLossHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        loss_type: str = "stablemax_cross_entropy",
        alpha_kl: float = 0.001,
        beta_entropy: float = 0.01,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.alpha_kl = alpha_kl
        self.beta_entropy = beta_entropy
        self.base_loss_head = ACTLossHead(model, loss_type=loss_type, **kwargs)
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_loss.mean()
    
    def compute_gate_entropy(self, gates: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        entropy = -(gates * torch.log(gates + eps)).sum(dim=-1)
        return -entropy.mean()
    
    def forward(self, carry: Any, batch: Dict[str, torch.Tensor], return_keys: list = []):
        carry, _, metrics, preds, all_finish = self.model(carry, batch, return_keys)
        
        if 'labels' in batch and 'logits' in preds:
            logits = preds['logits']
            labels = batch['labels']
            if self.loss_type == "stablemax_cross_entropy":
                loss_task = stablemax_cross_entropy(logits, labels)
            else:
                loss_task = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction='mean'
                )
            if loss_task.numel() > 1:
                loss_task = loss_task.mean()
        else:
            loss_task = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        meta_info = self.model.get_meta_info(carry)
        mu = meta_info.get('mu')
        logvar = meta_info.get('logvar')
        gates = meta_info.get('gates')
        
        loss_kl = torch.tensor(0.0, device=loss_task.device)
        loss_entropy = torch.tensor(0.0, device=loss_task.device)
        
        if mu is not None and logvar is not None:
            loss_kl = self.compute_kl_loss(mu, logvar)
        if gates is not None:
            loss_entropy = self.compute_gate_entropy(gates)
        
        total_loss = loss_task + self.alpha_kl * loss_kl + self.beta_entropy * loss_entropy
        if total_loss.numel() > 1:
            total_loss = total_loss.mean()
        
        metrics.update({
            'meta/loss_task': loss_task.detach(),
            'meta/loss_kl': loss_kl.detach(),
            'meta/loss_entropy': loss_entropy.detach(),
            'meta/total_loss': total_loss.detach(),
        })
        
        if gates is not None:
            metrics.update({
                'meta/gate_H_mean': gates[:, 0].mean().detach(),
                'meta/gate_L_mean': gates[:, 1].mean().detach(),
                'meta/gate_std': gates.std().detach(),
            })
        if mu is not None:
            metrics.update({
                'meta/z_mu_norm': mu.norm(dim=-1).mean().detach(),
                'meta/z_std_mean': torch.exp(0.5 * logvar).mean().detach(),
            })
        
        return carry, total_loss, metrics, preds, all_finish
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> Any:
        return self.model.initial_carry(batch)


class AdaptiveMetaLossHead(MetaLearningLossHead):
    def __init__(self, model: nn.Module, loss_type: str = "stablemax_cross_entropy",
                 alpha_kl_init: float = 0.001, beta_entropy_init: float = 0.01,
                 adaptive_rate: float = 0.99, target_kl: float = 1.0,
                 target_entropy: float = 0.5, **kwargs):
        super().__init__(model, loss_type, alpha_kl_init, beta_entropy_init, **kwargs)
        self.adaptive_rate = adaptive_rate
        self.target_kl = target_kl
        self.target_entropy = target_entropy
        self.log_alpha_kl = nn.Parameter(torch.tensor(alpha_kl_init).log())
        self.log_beta_entropy = nn.Parameter(torch.tensor(beta_entropy_init).log())
        self.register_buffer('kl_ema', torch.tensor(target_kl))
        self.register_buffer('entropy_ema', torch.tensor(target_entropy))
    
    @property
    def alpha_kl(self) -> float:
        return self.log_alpha_kl.exp().item()
    
    @property
    def beta_entropy(self) -> float:
        return self.log_beta_entropy.exp().item()
    
    def update_adaptive_weights(self, current_kl: float, current_entropy: float):
        with torch.no_grad():
            self.kl_ema = self.adaptive_rate * self.kl_ema + (1 - self.adaptive_rate) * current_kl
            self.entropy_ema = self.adaptive_rate * self.entropy_ema + (1 - self.adaptive_rate) * current_entropy
            if self.kl_ema > self.target_kl * 1.5:
                self.log_alpha_kl.data -= 0.01
            elif self.kl_ema < self.target_kl * 0.5:
                self.log_alpha_kl.data += 0.01
            if self.entropy_ema < self.target_entropy * 0.5:
                self.log_beta_entropy.data += 0.01
            elif self.entropy_ema > self.target_entropy * 1.5:
                self.log_beta_entropy.data -= 0.01
            self.log_alpha_kl.data.clamp_(min=-10, max=0)
            self.log_beta_entropy.data.clamp_(min=-10, max=0)
    
    def forward(self, carry: Any, batch: Dict[str, torch.Tensor], return_keys: list = []):
        carry, total_loss, metrics, preds, all_finish = super().forward(carry, batch, return_keys)
        if self.training:
            current_kl = metrics.get('meta/loss_kl', 0.0).item()
            current_entropy = -metrics.get('meta/loss_entropy', 0.0).item()
            self.update_adaptive_weights(current_kl, current_entropy)
        metrics.update({
            'meta/alpha_kl': self.alpha_kl,
            'meta/beta_entropy': self.beta_entropy,
            'meta/kl_ema': self.kl_ema.item(),
            'meta/entropy_ema': self.entropy_ema.item(),
        })
        return carry, total_loss, metrics, preds, all_finish
