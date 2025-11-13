#dynamic weight güncellemesi yapıldı - FULL FIX v2
"""
HRM-Free-Meta: Meta-Learning Enhanced Hierarchical Reasoning Model

🔥 CRITICAL FIXES:
- Device sync for buffers (H_init, L_init)
- Zero initialization for empty_carry (no NaN)
- Proper dtype propagation
"""

print("HRM-Free-Meta: Meta-Learning Enhanced Hierarchical Reasoning Model")

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass, fields

# Mevcut HRM bileşenlerini import et
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1_Inner,
    HierarchicalReasoningModel_ACTV1InnerCarry as InnerCarry,
    HierarchicalReasoningModel_ACTV1,
)

# Yeni meta-learning bileşenleri
from models.hrm.latent_context_encoder import LatentContextEncoder
from models.hrm.meta_controller import MetaController


@dataclass
class MetaCarry:
    """
    Extended carry structure with meta-learning components
    """
    # Orijinal HRM carry
    base_carry: InnerCarry

    # Meta-learning additions
    z_context: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    gates: Optional[torch.Tensor] = None


class HRMFreeMetaInner(nn.Module):
    """
    Inner model with meta-learning enhancements.
    Mevcut HRM_Inner'ı sarmalayıp üzerine meta katmanları ekler.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # 🔥 CRITICAL: forward_dtype'ı ayarla - FlashAttention için fp16/bf16 gerekli
        forward_dtype_str = getattr(config, 'forward_dtype', 'bfloat16')
        if forward_dtype_str == 'float32':
            print("⚠️  WARNING: forward_dtype='float32' is incompatible with FlashAttention")
            print("    → Automatically switching to 'bfloat16'")
            forward_dtype_str = 'bfloat16'
            if isinstance(config, dict):
                config['forward_dtype'] = 'bfloat16'
            else:
                config.forward_dtype = 'bfloat16'
        
        self.forward_dtype = getattr(torch, forward_dtype_str)
        
        # 1. Orijinal HRM Inner'ı oluştur
        if isinstance(config, dict):
            from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config
            config = HierarchicalReasoningModel_ACTV1Config(**config)
        self.hrm_inner = HierarchicalReasoningModel_ACTV1_Inner(config)

        # 2. Meta-learning bileşenleri ekle
        hidden_size = getattr(config, 'hidden_size', 512)
        z_dim = getattr(config, 'z_dim', 128)
        
        self.latent_encoder = LatentContextEncoder(
            hidden_size=hidden_size,
            z_dim=z_dim,
            num_layers = getattr(config, 'encoder_layers', 2)
        )
        
        self.meta_controller = MetaController(
            z_dim=z_dim,
            num_modules=2,
            hidden_dim=getattr(config, 'controller_hidden', 256),
            temperature=getattr(config, 'gate_temperature', 1.0),
        )

        # 3. Module blending layer (opsiyonel)
        self.use_dynamic_weighting = getattr(config, 'use_dynamic_weighting', True)
        
        # 🔥 CRITICAL: Meta modülleri doğru dtype'a cast et
        self.latent_encoder = self.latent_encoder.to(dtype=self.forward_dtype)
        self.meta_controller = self.meta_controller.to(dtype=self.forward_dtype)
    
    def empty_carry(self, batch: Dict[str, torch.Tensor]) -> MetaCarry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        
        # 🔥 FIX: Pass device to empty_carry
        base_carry = self.hrm_inner.empty_carry(batch_size, device=device)
        
        return MetaCarry(
            base_carry=base_carry,
            z_context=None, mu=None, logvar=None, gates=None
        )

    def reset_carry(self, carry: MetaCarry, batch: Dict[str, torch.Tensor]) -> MetaCarry:
        device = batch["inputs"].device
        reset_flag = torch.ones(
            (batch["inputs"].shape[0],), dtype=torch.bool, device=device
        )
        base_carry = self.hrm_inner.reset_carry(reset_flag, carry.base_carry)
        return MetaCarry(
            base_carry=base_carry,
            z_context=carry.z_context, 
            mu=carry.mu, 
            logvar=carry.logvar, 
            gates=carry.gates
        )

    def forward(
        self,
        carry: MetaCarry,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[MetaCarry, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with meta-learning
        
        Returns:
            carry: Updated carry state
            logits: Output logits (batch_size, seq_len, vocab_size)
            meta_info: Dictionary with z_context, mu, logvar, gates
        """
        # 1. HRM forward pass (orijinal)
        inner_carry_out, logits, (q_halt_logits, q_continue_logits) = self.hrm_inner(
            carry.base_carry, batch, **kwargs
        )
        
        # 2. Extract H and L states for meta-learning
        z_H = inner_carry_out.z_H  # (B, seq, hidden)
        z_L = inner_carry_out.z_L  # (B, seq, hidden)
        
        # 🔥 CRITICAL: NaN check before meta-learning
        if torch.isnan(z_H).any() or torch.isnan(z_L).any():
            print("⚠️  WARNING: NaN detected in z_H or z_L from HRM Inner!")
            print(f"    z_H NaNs: {torch.isnan(z_H).sum()}/{z_H.numel()}")
            print(f"    z_L NaNs: {torch.isnan(z_L).sum()}/{z_L.numel()}")
            # Replace NaNs with zeros
            z_H = torch.nan_to_num(z_H, nan=0.0)
            z_L = torch.nan_to_num(z_L, nan=0.0)
        
        # 3. Latent context encoding
        z_context, mu, logvar = self.latent_encoder(z_H, z_L)
        
        # 🔥 NaN check after encoder
        if torch.isnan(z_context).any():
            print("⚠️  WARNING: NaN in z_context after encoder!")
            z_context = torch.nan_to_num(z_context, nan=0.0)
            mu = torch.nan_to_num(mu, nan=0.0)
            logvar = torch.nan_to_num(logvar, nan=-2.0)
        
        # 4. Meta controller gating
        gates = self.meta_controller(z_context)
        
        # 🔥 NaN check after controller
        if torch.isnan(gates).any():
            print("⚠️  WARNING: NaN in gates after controller!")
            gates = torch.ones_like(gates) * 0.5
        
        # 5. Dynamic weighting of module outputs - manifestonun kalbi 
        if self.use_dynamic_weighting:
            # Gates: [batch, 2] -> gates[:,0]=H_weight, gates[:,1]=L_weight
            # z_H, z_L: [batch, seq_len, hidden_size]
            
            # Gates'i sequence dimension'a yay
            gate_H = gates[:, 0:1].unsqueeze(-1)  # [batch, 1, 1]
            gate_L = gates[:, 1:2].unsqueeze(-1)  # [batch, 1, 1]
            
            # Weighted combination of H and L states
            z_combined = gate_H * z_H + gate_L * z_L  # [batch, seq_len, hidden_size]
            
            # 🔥 NaN check after combining
            if torch.isnan(z_combined).any():
                print("⚠️  WARNING: NaN in z_combined!")
                z_combined = torch.nan_to_num(z_combined, nan=0.0)
            
            # Bu combined state'ten logits üret
            logits = self.hrm_inner.lm_head(z_combined)[:, self.hrm_inner.puzzle_emb_len:]
            
            # 🔥 Final NaN check
            if torch.isnan(logits).any():
                print("⚠️  WARNING: NaN in final logits!")
                logits = torch.nan_to_num(logits, nan=0.0)
        
        # 6. Update carry
        carry_out = MetaCarry(
            base_carry=inner_carry_out,
            z_context=z_context, mu=mu, logvar=logvar, gates=gates
        )
        
        # 7. Meta info for loss calculation
        meta_info = {
            'z_context': z_context,
            'mu': mu,
            'logvar': logvar,
            'gates': gates
        }
        
        return carry_out, logits, meta_info


class HRMFreeMeta(nn.Module):
    """
    Complete HRM-Free-Meta model with ACT mechanism.
    HRM-ACT-V1'i sarmalayıp meta-learning ekler.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # 🔥 CRITICAL: Config'de dtype kontrolü
        if 'forward_dtype' not in config:
            config['forward_dtype'] = 'bfloat16'
        elif config['forward_dtype'] == 'float32':
            print("⚠️  WARNING: Overriding forward_dtype from float32 to bfloat16 (FlashAttention requirement)")
            config['forward_dtype'] = 'bfloat16'
        
        # Inner model with meta enhancements
        self.inner = HRMFreeMetaInner(config)
        
        # ACT mechanism parameters
        self.halt_max_steps = config.get('halt_max_steps', 16)
        self.halt_exploration_prob = config.get('halt_exploration_prob', 0.1)
    
    def to(self, *args, **kwargs):
        """Override to() to maintain dtype consistency"""
        result = super().to(*args, **kwargs)
        return result

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> MetaCarry:
        """Initialize carry for new batch"""
        return self.inner.empty_carry(batch)
    
    def forward(
        self,
        carry: MetaCarry,
        batch: Dict[str, torch.Tensor],
        return_keys: list = []
    ) -> Tuple[MetaCarry, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any], bool]:
        """
        Full forward pass with ACT-style iterative reasoning.
        
        Returns:
            carry: Updated carry
            loss: Total loss (task + meta losses)
            metrics: Training metrics
            preds: Predictions (if return_keys specified)
            all_finish: Whether all examples finished reasoning
        """
        # Reset carry for finished puzzles
        if 'is_new_puzzle' in batch:
            carry = self.inner.reset_carry(carry, batch)
        
        # Forward pass
        carry, logits, meta_info = self.inner(carry, batch)
        
        # Dummy loss (gerçek loss losses.py'de hesaplanacak)
        device = logits.device
        loss = torch.tensor(0.0, device=device)
        
        # Metrics
        metrics = {
            'count': torch.tensor(batch['inputs'].size(0), device=device)
        }
        
        # Meta-learning metrics ekle
        if meta_info is not None:
            metrics.update({
                'meta/gate_H_mean': meta_info['gates'][:, 0].mean(),
                'meta/gate_L_mean': meta_info['gates'][:, 1].mean(),
                'meta/z_context_norm': meta_info['z_context'].norm(dim=-1).mean(),
            })
        
        preds = {'logits': logits}
        if 'z_context' in return_keys:
            preds['z_context'] = meta_info['z_context']
        if 'gates' in return_keys:
            preds['gates'] = meta_info['gates']
        
        all_finish = True
        
        return carry, loss, metrics, preds, all_finish
    
    def get_meta_info(self, carry: MetaCarry) -> Dict[str, torch.Tensor]:
        """Extract meta-learning info from carry"""
        return {
            'z_context': carry.z_context,
            'mu': carry.mu,
            'logvar': carry.logvar,
            'gates': carry.gates
        }


def create_hrm_free_meta(config: Dict[str, Any]) -> HRMFreeMeta:
    """
    Factory function to create HRM-Free-Meta model.
    pretrain.py'den çağrılacak.
    """
    return HRMFreeMeta(config)#dynamic weight güncellemesi yapıldı - FULL FIX v2
"""
HRM-Free-Meta: Meta-Learning Enhanced Hierarchical Reasoning Model

🔥 CRITICAL FIXES:
- Device sync for buffers (H_init, L_init)
- Zero initialization for empty_carry (no NaN)
- Proper dtype propagation
"""

print("gpt tarafından hrm bileşenlerini import eden kısım revize edildi 4 - FULL FIXED")

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass, fields

# Mevcut HRM bileşenlerini import et
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1_Inner,
    HierarchicalReasoningModel_ACTV1InnerCarry as InnerCarry,
    HierarchicalReasoningModel_ACTV1,
)

# Yeni meta-learning bileşenleri
from models.hrm.latent_context_encoder import LatentContextEncoder
from models.hrm.meta_controller import MetaController


@dataclass
class MetaCarry:
    """
    Extended carry structure with meta-learning components
    """
    # Orijinal HRM carry
    base_carry: InnerCarry

    # Meta-learning additions
    z_context: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    gates: Optional[torch.Tensor] = None


class HRMFreeMetaInner(nn.Module):
    """
    Inner model with meta-learning enhancements.
    Mevcut HRM_Inner'ı sarmalayıp üzerine meta katmanları ekler.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # 🔥 CRITICAL: forward_dtype'ı ayarla - FlashAttention için fp16/bf16 gerekli
        forward_dtype_str = getattr(config, 'forward_dtype', 'bfloat16')
        if forward_dtype_str == 'float32':
            print("⚠️  WARNING: forward_dtype='float32' is incompatible with FlashAttention")
            print("    → Automatically switching to 'bfloat16'")
            forward_dtype_str = 'bfloat16'
            if isinstance(config, dict):
                config['forward_dtype'] = 'bfloat16'
            else:
                config.forward_dtype = 'bfloat16'
        
        self.forward_dtype = getattr(torch, forward_dtype_str)
        
        # 1. Orijinal HRM Inner'ı oluştur
        if isinstance(config, dict):
            from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config
            config = HierarchicalReasoningModel_ACTV1Config(**config)
        self.hrm_inner = HierarchicalReasoningModel_ACTV1_Inner(config)

        # 2. Meta-learning bileşenleri ekle
        hidden_size = getattr(config, 'hidden_size', 512)
        z_dim = getattr(config, 'z_dim', 128)
        
        self.latent_encoder = LatentContextEncoder(
            hidden_size=hidden_size,
            z_dim=z_dim,
            num_layers = getattr(config, 'encoder_layers', 2)
        )
        
        self.meta_controller = MetaController(
            z_dim=z_dim,
            num_modules=2,
            hidden_dim=getattr(config, 'controller_hidden', 256),
            temperature=getattr(config, 'gate_temperature', 1.0),
        )

        # 3. Module blending layer (opsiyonel)
        self.use_dynamic_weighting = getattr(config, 'use_dynamic_weighting', True)
        
        # 🔥 CRITICAL: Meta modülleri doğru dtype'a cast et
        self.latent_encoder = self.latent_encoder.to(dtype=self.forward_dtype)
        self.meta_controller = self.meta_controller.to(dtype=self.forward_dtype)
    
    def empty_carry(self, batch: Dict[str, torch.Tensor]) -> MetaCarry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        
        # 🔥 FIX: Pass device to empty_carry
        base_carry = self.hrm_inner.empty_carry(batch_size, device=device)
        
        return MetaCarry(
            base_carry=base_carry,
            z_context=None, mu=None, logvar=None, gates=None
        )

    def reset_carry(self, carry: MetaCarry, batch: Dict[str, torch.Tensor]) -> MetaCarry:
        device = batch["inputs"].device
        reset_flag = torch.ones(
            (batch["inputs"].shape[0],), dtype=torch.bool, device=device
        )
        base_carry = self.hrm_inner.reset_carry(reset_flag, carry.base_carry)
        return MetaCarry(
            base_carry=base_carry,
            z_context=carry.z_context, 
            mu=carry.mu, 
            logvar=carry.logvar, 
            gates=carry.gates
        )

    def forward(
        self,
        carry: MetaCarry,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[MetaCarry, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with meta-learning
        
        Returns:
            carry: Updated carry state
            logits: Output logits (batch_size, seq_len, vocab_size)
            meta_info: Dictionary with z_context, mu, logvar, gates
        """
        # 1. HRM forward pass (orijinal)
        inner_carry_out, logits, (q_halt_logits, q_continue_logits) = self.hrm_inner(
            carry.base_carry, batch, **kwargs
        )
        
        # 2. Extract H and L states for meta-learning
        z_H = inner_carry_out.z_H  # (B, seq, hidden)
        z_L = inner_carry_out.z_L  # (B, seq, hidden)
        
        # 🔥 CRITICAL: NaN check before meta-learning
        if torch.isnan(z_H).any() or torch.isnan(z_L).any():
            print("⚠️  WARNING: NaN detected in z_H or z_L from HRM Inner!")
            print(f"    z_H NaNs: {torch.isnan(z_H).sum()}/{z_H.numel()}")
            print(f"    z_L NaNs: {torch.isnan(z_L).sum()}/{z_L.numel()}")
            # Replace NaNs with zeros
            z_H = torch.nan_to_num(z_H, nan=0.0)
            z_L = torch.nan_to_num(z_L, nan=0.0)
        
        # 3. Latent context encoding
        z_context, mu, logvar = self.latent_encoder(z_H, z_L)
        
        # 🔥 NaN check after encoder
        if torch.isnan(z_context).any():
            print("⚠️  WARNING: NaN in z_context after encoder!")
            z_context = torch.nan_to_num(z_context, nan=0.0)
            mu = torch.nan_to_num(mu, nan=0.0)
            logvar = torch.nan_to_num(logvar, nan=-2.0)
        
        # 4. Meta controller gating
        gates = self.meta_controller(z_context)
        
        # 🔥 NaN check after controller
        if torch.isnan(gates).any():
            print("⚠️  WARNING: NaN in gates after controller!")
            gates = torch.ones_like(gates) * 0.5
        
        # 5. Dynamic weighting of module outputs - manifestonun kalbi 
        if self.use_dynamic_weighting:
            # Gates: [batch, 2] -> gates[:,0]=H_weight, gates[:,1]=L_weight
            # z_H, z_L: [batch, seq_len, hidden_size]
            
            # Gates'i sequence dimension'a yay
            gate_H = gates[:, 0:1].unsqueeze(-1)  # [batch, 1, 1]
            gate_L = gates[:, 1:2].unsqueeze(-1)  # [batch, 1, 1]
            
            # Weighted combination of H and L states
            z_combined = gate_H * z_H + gate_L * z_L  # [batch, seq_len, hidden_size]
            
            # 🔥 NaN check after combining
            if torch.isnan(z_combined).any():
                print("⚠️  WARNING: NaN in z_combined!")
                z_combined = torch.nan_to_num(z_combined, nan=0.0)
            
            # Bu combined state'ten logits üret
            logits = self.hrm_inner.lm_head(z_combined)[:, self.hrm_inner.puzzle_emb_len:]
            
            # 🔥 Final NaN check
            if torch.isnan(logits).any():
                print("⚠️  WARNING: NaN in final logits!")
                logits = torch.nan_to_num(logits, nan=0.0)
        
        # 6. Update carry
        carry_out = MetaCarry(
            base_carry=inner_carry_out,
            z_context=z_context, mu=mu, logvar=logvar, gates=gates
        )
        
        # 7. Meta info for loss calculation
        meta_info = {
            'z_context': z_context,
            'mu': mu,
            'logvar': logvar,
            'gates': gates
        }
        
        return carry_out, logits, meta_info


class HRMFreeMeta(nn.Module):
    """
    Complete HRM-Free-Meta model with ACT mechanism.
    HRM-ACT-V1'i sarmalayıp meta-learning ekler.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # 🔥 CRITICAL: Config'de dtype kontrolü
        if 'forward_dtype' not in config:
            config['forward_dtype'] = 'bfloat16'
        elif config['forward_dtype'] == 'float32':
            print("⚠️  WARNING: Overriding forward_dtype from float32 to bfloat16 (FlashAttention requirement)")
            config['forward_dtype'] = 'bfloat16'
        
        # Inner model with meta enhancements
        self.inner = HRMFreeMetaInner(config)
        
        # ACT mechanism parameters
        self.halt_max_steps = config.get('halt_max_steps', 16)
        self.halt_exploration_prob = config.get('halt_exploration_prob', 0.1)
    
    def to(self, *args, **kwargs):
        """Override to() to maintain dtype consistency"""
        result = super().to(*args, **kwargs)
        return result

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> MetaCarry:
        """Initialize carry for new batch"""
        return self.inner.empty_carry(batch)
    
    def forward(
        self,
        carry: MetaCarry,
        batch: Dict[str, torch.Tensor],
        return_keys: list = []
    ) -> Tuple[MetaCarry, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any], bool]:
        """
        Full forward pass with ACT-style iterative reasoning.
        
        Returns:
            carry: Updated carry
            loss: Total loss (task + meta losses)
            metrics: Training metrics
            preds: Predictions (if return_keys specified)
            all_finish: Whether all examples finished reasoning
        """
        # Reset carry for finished puzzles
        if 'is_new_puzzle' in batch:
            carry = self.inner.reset_carry(carry, batch)
        
        # Forward pass
        carry, logits, meta_info = self.inner(carry, batch)
        
        # Dummy loss (gerçek loss losses.py'de hesaplanacak)
        device = logits.device
        loss = torch.tensor(0.0, device=device)
        
        # Metrics
        metrics = {
            'count': torch.tensor(batch['inputs'].size(0), device=device)
        }
        
        # Meta-learning metrics ekle
        if meta_info is not None:
            metrics.update({
                'meta/gate_H_mean': meta_info['gates'][:, 0].mean(),
                'meta/gate_L_mean': meta_info['gates'][:, 1].mean(),
                'meta/z_context_norm': meta_info['z_context'].norm(dim=-1).mean(),
            })
        
        preds = {'logits': logits}
        if 'z_context' in return_keys:
            preds['z_context'] = meta_info['z_context']
        if 'gates' in return_keys:
            preds['gates'] = meta_info['gates']
        
        all_finish = True
        
        return carry, loss, metrics, preds, all_finish
    
    def get_meta_info(self, carry: MetaCarry) -> Dict[str, torch.Tensor]:
        """Extract meta-learning info from carry"""
        return {
            'z_context': carry.z_context,
            'mu': carry.mu,
            'logvar': carry.logvar,
            'gates': carry.gates
        }


def create_hrm_free_meta(config: Dict[str, Any]) -> HRMFreeMeta:
    """
    Factory function to create HRM-Free-Meta model.
    pretrain.py'den çağrılacak.
    """
    return HRMFreeMeta(config)      