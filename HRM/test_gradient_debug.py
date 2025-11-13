#!/usr/bin/env python3
"""
Gradient Flow Debug - Controller'a neden gradient gitmiyor?
"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 GRADIENT FLOW DEBUG\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.hrm.hrm_free_meta import HRMFreeMeta

config = {
    'batch_size': 8, 'seq_len': 81, 'vocab_size': 10,
    'num_puzzle_identifiers': 100, 'hidden_size': 256,
    'num_heads': 4, 'expansion': 4, 'H_layers': 2, 'L_layers': 2,
    'H_cycles': 1, 'L_cycles': 1, 'halt_max_steps': 4,
    'z_dim': 128, 'encoder_layers': 2, 'controller_hidden': 256,
    'gate_temperature': 1.0, 'use_dynamic_weighting': True,
    'causal': False, 'pos_encodings': 'rope',
    'puzzle_emb_ndim': 256, 'halt_exploration_prob': 0.05,
    'forward_dtype': 'bfloat16',
}

model = HRMFreeMeta(config).to(device)
model.train()

batch = {
    'inputs': torch.randint(0, 10, (8, 81), device=device),
    'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
}

print("Step 1: Check if controller outputs require grad")
carry = model.initial_carry(batch)
carry_out, logits, meta_info = model.inner(carry, batch)

print(f"  gates.requires_grad: {meta_info['gates'].requires_grad}")
print(f"  z_context.requires_grad: {meta_info['z_context'].requires_grad}")
print(f"  logits.requires_grad: {logits.requires_grad}\n")

print("Step 2: Backward from gates directly")
model.zero_grad()
carry = model.initial_carry(batch)
carry_out, logits, meta_info = model.inner(carry, batch)

# Direct backward from gates
loss = meta_info['gates'].sum()
print(f"  Loss from gates: {loss.item()}")
loss.backward()

controller_params = list(model.inner.meta_controller.parameters())
has_grad = [p.grad is not None and p.grad.abs().sum() > 0 for p in controller_params]
print(f"  Controller params with grad: {sum(has_grad)}/{len(controller_params)}")

if sum(has_grad) > 0:
    print("  ✅ Controller CAN receive gradients!")
else:
    print("  ❌ Controller CANNOT receive gradients - something is detached")

print("\nStep 3: Check if gates are detached somewhere")
carry = model.initial_carry(batch)

# Manual trace
with torch.enable_grad():
    # Inner HRM forward
    inner_carry_out, logits_hrm, _ = model.inner.hrm_inner(carry.base_carry, batch)
    
    print(f"  After HRM: z_H.requires_grad = {inner_carry_out.z_H.requires_grad}")
    print(f"  After HRM: z_L.requires_grad = {inner_carry_out.z_L.requires_grad}")
    
    # Encoder
    z_context, mu, logvar = model.inner.latent_encoder(
        inner_carry_out.z_H, inner_carry_out.z_L
    )
    
    print(f"  After Encoder: z_context.requires_grad = {z_context.requires_grad}")
    
    # Controller
    gates = model.inner.meta_controller(z_context)
    
    print(f"  After Controller: gates.requires_grad = {gates.requires_grad}")
    
    # Dynamic weighting
    if model.inner.use_dynamic_weighting:
        gate_H = gates[:, 0:1].unsqueeze(-1)
        gate_L = gates[:, 1:2].unsqueeze(-1)
        z_combined = gate_H * inner_carry_out.z_H + gate_L * inner_carry_out.z_L
        
        print(f"  After Weighting: z_combined.requires_grad = {z_combined.requires_grad}")
        
        logits_final = model.inner.hrm_inner.lm_head(z_combined)[:, model.inner.hrm_inner.puzzle_emb_len:]
        
        print(f"  After LM Head: logits.requires_grad = {logits_final.requires_grad}")

print("\nStep 4: Test backward through full pipeline")
model.zero_grad()
carry = model.initial_carry(batch)
carry_out, logits, meta_info = model.inner(carry, batch)

# Loss that uses gates in computation graph
if model.inner.use_dynamic_weighting:
    # Logits should depend on gates
    loss = logits.sum() + meta_info['gates'].sum()
else:
    loss = logits.sum()

print(f"  Total loss: {loss.item()}")
loss.backward()

encoder_grads = sum(1 for p in model.inner.latent_encoder.parameters() 
                    if p.grad is not None and p.grad.abs().sum() > 0)
controller_grads = sum(1 for p in model.inner.meta_controller.parameters() 
                       if p.grad is not None and p.grad.abs().sum() > 0)

print(f"  Encoder params with grad: {encoder_grads}")
print(f"  Controller params with grad: {controller_grads}")

if controller_grads == 0:
    print("\n❌ PROBLEM FOUND: Controller'a gradient gitmiyor!")
    print("   → Muhtemelen hrm_free_meta.py'de z_H veya z_L detach ediliyor")
    print("   → Line 108-120 civarına bakın")
else:
    print("\n✅ Controller receives gradients when using dynamic weighting!")