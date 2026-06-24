"""
HRM-Free-Meta model.

This wraps the base HRM implementation with:
- a latent task-context encoder,
- a small controller that predicts H/L mixing gates,
- a carry object that keeps the meta state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1_Inner,
    HierarchicalReasoningModel_ACTV1InnerCarry as InnerCarry,
    HierarchicalReasoningModel_ACTV1Config,
)
from models.hrm.latent_context_encoder import LatentContextEncoder
from models.hrm.meta_controller import MetaController


@dataclass
class MetaCarry:
    base_carry: InnerCarry
    z_context: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    gates: Optional[torch.Tensor] = None


class HRMFreeMetaInner(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        cfg_dict = dict(config)
        forward_dtype_name = cfg_dict.get("forward_dtype", "bfloat16")
        if forward_dtype_name == "float32":
            forward_dtype_name = "bfloat16"
            cfg_dict["forward_dtype"] = forward_dtype_name

        self.forward_dtype = getattr(torch, forward_dtype_name)
        self.config = HierarchicalReasoningModel_ACTV1Config(**cfg_dict)
        self.hrm_inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

        hidden_size = self.config.hidden_size
        z_dim = cfg_dict.get("z_dim", 128)

        self.latent_encoder = LatentContextEncoder(
            hidden_size=hidden_size,
            z_dim=z_dim,
            num_layers=cfg_dict.get("encoder_layers", 2),
        ).to(dtype=self.forward_dtype)

        self.meta_controller = MetaController(
            z_dim=z_dim,
            num_modules=2,
            hidden_dim=cfg_dict.get("controller_hidden", 256),
            temperature=cfg_dict.get("gate_temperature", 1.0),
        ).to(dtype=self.forward_dtype)

        self.use_dynamic_weighting = cfg_dict.get("use_dynamic_weighting", True)

    def empty_carry(self, batch: Dict[str, torch.Tensor]) -> MetaCarry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        base_carry = self.hrm_inner.empty_carry(batch_size, device=device)
        return MetaCarry(base_carry=base_carry)

    def reset_carry(self, carry: MetaCarry, batch: Dict[str, torch.Tensor]) -> MetaCarry:
        device = batch["inputs"].device
        reset_flag = torch.ones((batch["inputs"].shape[0],), dtype=torch.bool, device=device)
        base_carry = self.hrm_inner.reset_carry(reset_flag, carry.base_carry)
        return MetaCarry(
            base_carry=base_carry,
            z_context=carry.z_context,
            mu=carry.mu,
            logvar=carry.logvar,
            gates=carry.gates,
        )

    def forward(
        self,
        carry: MetaCarry,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Tuple[MetaCarry, torch.Tensor, Dict[str, torch.Tensor]]:
        inner_carry_out, logits, _ = self.hrm_inner(carry.base_carry, batch, **kwargs)

        z_H = torch.nan_to_num(inner_carry_out.z_H, nan=0.0)
        z_L = torch.nan_to_num(inner_carry_out.z_L, nan=0.0)

        z_context, mu, logvar = self.latent_encoder(z_H, z_L)
        z_context = torch.nan_to_num(z_context, nan=0.0)
        mu = torch.nan_to_num(mu, nan=0.0)
        logvar = torch.nan_to_num(logvar, nan=-2.0)

        gates = self.meta_controller(z_context)
        if torch.isnan(gates).any():
            gates = torch.full_like(gates, 0.5)

        if self.use_dynamic_weighting:
            gate_H = gates[:, 0:1].unsqueeze(-1)
            gate_L = gates[:, 1:2].unsqueeze(-1)
            z_combined = torch.nan_to_num(gate_H * z_H + gate_L * z_L, nan=0.0)
            logits = self.hrm_inner.lm_head(z_combined)[:, self.hrm_inner.puzzle_emb_len :]
            logits = torch.nan_to_num(logits, nan=0.0)

        carry_out = MetaCarry(
            base_carry=inner_carry_out,
            z_context=z_context,
            mu=mu,
            logvar=logvar,
            gates=gates,
        )
        meta_info = {
            "z_context": z_context,
            "mu": mu,
            "logvar": logvar,
            "gates": gates,
        }
        return carry_out, logits, meta_info


class HRMFreeMeta(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        cfg_dict = dict(config)
        if cfg_dict.get("forward_dtype") == "float32":
            cfg_dict["forward_dtype"] = "bfloat16"

        self.config = cfg_dict
        self.inner = HRMFreeMetaInner(cfg_dict)
        self.halt_max_steps = cfg_dict.get("halt_max_steps", 16)
        self.halt_exploration_prob = cfg_dict.get("halt_exploration_prob", 0.1)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> MetaCarry:
        return self.inner.empty_carry(batch)

    def forward(
        self,
        carry: MetaCarry,
        batch: Dict[str, torch.Tensor],
        return_keys: list[str] | None = None,
    ) -> Tuple[MetaCarry, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any], bool]:
        return_keys = return_keys or []

        if "is_new_puzzle" in batch:
            carry = self.inner.reset_carry(carry, batch)

        carry, logits, meta_info = self.inner(carry, batch)
        device = logits.device

        metrics: Dict[str, torch.Tensor] = {
            "count": torch.tensor(batch["inputs"].size(0), device=device),
            "meta/gate_H_mean": meta_info["gates"][:, 0].mean(),
            "meta/gate_L_mean": meta_info["gates"][:, 1].mean(),
            "meta/z_context_norm": meta_info["z_context"].norm(dim=-1).mean(),
        }

        preds: Dict[str, Any] = {"logits": logits}
        if "z_context" in return_keys:
            preds["z_context"] = meta_info["z_context"]
        if "gates" in return_keys:
            preds["gates"] = meta_info["gates"]

        loss = torch.tensor(0.0, device=device)
        all_finish = True
        return carry, loss, metrics, preds, all_finish

    def get_meta_info(self, carry: MetaCarry) -> Dict[str, torch.Tensor]:
        return {
            "z_context": carry.z_context,
            "mu": carry.mu,
            "logvar": carry.logvar,
            "gates": carry.gates,
        }


def create_hrm_free_meta(config: Dict[str, Any]) -> HRMFreeMeta:
    return HRMFreeMeta(config)
