#!/usr/bin/env python3
"""
Smart Model Test - Does it actually LEARN?

🧠 MANIFESTO TEST: "Intelligence is adaptation, not memorization"

This test simulates training by:
1. Forward pass
2. Compute losses  
3. Backward pass
4. Check if gradients flow and model adapts
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("🧠 SMART MODEL TEST: Intelligence Check")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

from models.hrm.hrm_free_meta import HRMFreeMeta
from models.hrm.meta_losses import compute_meta_loss

# Config
config = {
    'batch_size': 8,
    'seq_len': 81,
    'vocab_size': 10,
    'num_puzzle_identifiers': 100,
    'hidden_size': 256,
    'num_heads': 4,
    'expansion': 4,
    'H_layers': 2,
    'L_layers': 2,
    'H_cycles': 1,
    'L_cycles': 1,
    'halt_max_steps': 4,
    'z_dim': 128,  # 🧠 Increased from 64
    'encoder_layers': 2,
    'controller_hidden': 256,
    'gate_temperature': 1.0,
    'use_dynamic_weighting': True,
    'causal': False,
    'pos_encodings': 'rope',
    'puzzle_emb_ndim': 256,
    'halt_exploration_prob': 0.05,
    'forward_dtype': 'bfloat16',
}

print("🏗️  Creating model...")
model = HRMFreeMeta(config).to(device)

print(f"✅ Model created (dtype: {next(model.parameters()).dtype})")
print(f"   Encoder params: {sum(p.numel() for p in model.inner.latent_encoder.parameters()):,}")
print(f"   Controller params: {sum(p.numel() for p in model.inner.meta_controller.parameters()):,}")

# ============================================================================
# TEST 1: Initial State Check
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Initial State Check")
print("=" * 80)

model.eval()

# Create two DIFFERENT batches
batch1 = {
    'inputs': torch.randint(0, 10, (8, 81), device=device),
    'puzzle_identifiers': torch.randint(0, 50, (8,), device=device),  # First half
}

batch2 = {
    'inputs': torch.randint(0, 10, (8, 81), device=device),
    'puzzle_identifiers': torch.randint(50, 100, (8,), device=device),  # Second half
}

with torch.no_grad():
    carry1 = model.initial_carry(batch1)
    _, _, meta1 = model.inner(carry1, batch1)
    
    carry2 = model.initial_carry(batch2)
    _, _, meta2 = model.inner(carry2, batch2)

gates1 = meta1['gates'].float()
gates2 = meta2['gates'].float()
z1 = meta1['z_context'].float()
z2 = meta2['z_context'].float()

print(f"\n📊 Initial Statistics:")
print(f"   Batch 1 - H: {gates1[:, 0].mean():.4f}, L: {gates1[:, 1].mean():.4f}")
print(f"   Batch 2 - H: {gates2[:, 0].mean():.4f}, L: {gates2[:, 1].mean():.4f}")
print(f"   Gate difference: {(gates1 - gates2).abs().mean():.6f}")
print(f"   Context norm 1: {z1.norm(dim=-1).mean():.4f}")
print(f"   Context norm 2: {z2.norm(dim=-1).mean():.4f}")
print(f"   Context difference: {(z1 - z2).norm(dim=-1).mean():.4f}")

initial_gate_diff = (gates1 - gates2).abs().mean().item()
initial_context_diff = (z1 - z2).norm(dim=-1).mean().item()

if initial_gate_diff < 0.001:
    print("   ⚠️  Gates are identical - controller not sensitive to inputs")
else:
    print(f"   ✅ Gates show variation")

# ============================================================================
# TEST 2: Gradient Flow & Loss Computation
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Gradient Flow & Loss Computation")
print("=" * 80)

model.train()

batch = {
    'inputs': torch.randint(0, 10, (8, 81), device=device),
    'labels': torch.randint(0, 10, (8, 81), device=device),
    'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
}

# Zero gradients
model.zero_grad()

# Forward pass
carry = model.initial_carry(batch)
carry_out, logits, meta_info = model.inner(carry, batch)

# Compute losses
task_loss = F.cross_entropy(
    logits.reshape(-1, config['vocab_size']),
    batch['labels'].reshape(-1)
)

meta_loss, meta_metrics = compute_meta_loss(
    meta_info,
    kl_weight=0.001,
    entropy_weight=0.01,
    diversity_weight=0.01
)

total_loss = task_loss + meta_loss

print(f"\n📊 Loss Values:")
print(f"   Task loss:     {task_loss.item():.4f}")
print(f"   Meta loss:     {meta_loss.item():.6f}")
print(f"   Total loss:    {total_loss.item():.4f}")
print(f"\n   KL loss:       {meta_metrics['meta/kl_loss'].item():.6f}")
print(f"   Entropy loss:  {meta_metrics['meta/entropy_loss'].item():.6f}")
print(f"   Diversity:     {meta_metrics['meta/diversity_loss'].item():.6f}")

# Backward
total_loss.backward()

# Check gradients
encoder_grads = [p.grad.abs().mean().item() for p in model.inner.latent_encoder.parameters() 
                 if p.grad is not None]
controller_grads = [p.grad.abs().mean().item() for p in model.inner.meta_controller.parameters() 
                    if p.grad is not None]

print(f"\n📊 Gradient Statistics:")
print(f"   Encoder gradients:   mean={sum(encoder_grads)/len(encoder_grads):.6f}, max={max(encoder_grads):.6f}")
print(f"   Controller gradients: mean={sum(controller_grads)/len(controller_grads):.6f}, max={max(controller_grads):.6f}")

test2_pass = len(encoder_grads) > 0 and len(controller_grads) > 0 and max(controller_grads) > 1e-6

if test2_pass:
    print("   ✅ PASS: Gradients flowing through meta components")
else:
    print("   ❌ FAIL: Weak or no gradients")

# ============================================================================
# TEST 3: Simulated Training (10 steps)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Simulated Training (10 mini-steps)")
print("=" * 80)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Save initial state
with torch.no_grad():
    carry_init = model.initial_carry(batch)
    _, _, meta_init = model.inner(carry_init, batch)
    gates_before = meta_init['gates'].clone()

print("\n📊 Training...")
for step in range(10):
    optimizer.zero_grad()
    
    # New random batch each step
    batch = {
        'inputs': torch.randint(0, 10, (8, 81), device=device),
        'labels': torch.randint(0, 10, (8, 81), device=device),
        'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
    }
    
    carry = model.initial_carry(batch)
    carry_out, logits, meta_info = model.inner(carry, batch)
    
    task_loss = F.cross_entropy(logits.reshape(-1, config['vocab_size']), batch['labels'].reshape(-1))
    meta_loss, _ = compute_meta_loss(meta_info, kl_weight=0.001, entropy_weight=0.01)
    
    total_loss = task_loss + meta_loss
    total_loss.backward()
    optimizer.step()
    
    if step % 3 == 0:
        print(f"   Step {step}: loss={total_loss.item():.4f}, gates=[{meta_info['gates'][0,0]:.3f}, {meta_info['gates'][0,1]:.3f}]")

# Check if gates changed after training
model.eval()
with torch.no_grad():
    carry_after = model.initial_carry(batch)
    _, _, meta_after = model.inner(carry_after, batch)
    gates_after = meta_after['gates']

gate_change = (gates_after.float() - gates_before.float()).abs().mean().item()

print(f"\n📊 Training Effect:")
print(f"   Gate change: {gate_change:.6f}")
print(f"   Before: H={gates_before[0,0]:.4f}, L={gates_before[0,1]:.4f}")
print(f"   After:  H={gates_after[0,0]:.4f}, L={gates_after[0,1]:.4f}")

test3_pass = gate_change > 0.001

if test3_pass:
    print("   ✅ PASS: Model is learning (gates changed)")
else:
    print("   ⚠️  WEAK: Gates barely changed, may need more training")

# ============================================================================
# TEST 4: Diversity Check
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Context Diversity")
print("=" * 80)

contexts = []
gates_list = []

for i in range(20):
    batch = {
        'inputs': torch.randint(0, 10, (8, 81), device=device),
        'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
    }
    
    with torch.no_grad():
        carry = model.initial_carry(batch)
        _, _, meta = model.inner(carry, batch)
    
    contexts.append(meta['z_context'].cpu().float())
    gates_list.append(meta['gates'].cpu().float())

contexts = torch.cat(contexts, dim=0)  # [160, z_dim]
gates_all = torch.cat(gates_list, dim=0)  # [160, 2]

z_std = contexts.std(dim=0).mean().item()
gate_std = gates_all.std(dim=0).mean().item()

print(f"\n📊 Diversity Metrics (20 batches):")
print(f"   Context std:  {z_std:.4f}")
print(f"   Gate std:     {gate_std:.4f}")
print(f"   H-gate: mean={gates_all[:,0].mean():.4f}, std={gates_all[:,0].std():.4f}")
print(f"   L-gate: mean={gates_all[:,1].mean():.4f}, std={gates_all[:,1].std():.4f}")

test4_pass = z_std > 0.1 and gate_std > 0.01

if test4_pass:
    print("   ✅ PASS: Good diversity in contexts and gates")
else:
    print("   ⚠️  WEAK: Low diversity, needs more training or tuning")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📋 INTELLIGENCE CHECK SUMMARY")
print("=" * 80)

tests = [
    ("Gradient Flow", test2_pass),
    ("Learning Ability", test3_pass),
    ("Diversity", test4_pass),
]

print("\nTest Results:")
for name, passed in tests:
    status = "✅ PASS" if passed else "⚠️  WEAK"
    print(f"   {status}: {name}")

total_passed = sum(1 for _, p in tests if p)

print(f"\n{'='*80}")
if total_passed == len(tests):
    print("🎉 MODEL IS INTELLIGENT! Ready for real training.")
    print("   → All systems functional")
    print("   → Meta-learning components active")
    print("   → Dynamic weighting operational")
elif total_passed >= 2:
    print("✅ MODEL IS FUNCTIONAL! Good foundation.")
    print("   → Core systems work")
    print("   → Needs more training to show full intelligence")
    print("   → Consider increasing learning rate or training steps")
else:
    print("⚠️  MODEL NEEDS TUNING")
    print("   → Check hyperparameters")
    print("   → May need architecture adjustments")

print("=" * 80)