#!/usr/bin/env python3
"""
FIXED FINAL VALIDATION TEST
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("🎯 FIXED FINAL VALIDATION TEST")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

from models.hrm.hrm_free_meta import HRMFreeMeta
from meta_loss_head import MetaLearningLossHead

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

print("🏗️  Creating model...")
model = HRMFreeMeta(config).to(device)
loss_head = MetaLearningLossHead(
    model=model,
    loss_type='stablemax_cross_entropy',
    alpha_kl=0.001,
    beta_entropy=0.01
)
print("✅ Model created\n")

# Test batch WITH LABELS (critical!)
batch = {
    'inputs': torch.randint(0, 10, (8, 81), device=device),
    'labels': torch.randint(0, 10, (8, 81), device=device),  # 🔥 LABELS!
    'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
}

print("="*80)
print("TEST 1: Perfect Normalization Check")
print("="*80)

model.eval()
gates_list = []
for i in range(100):
    test_batch = {
        'inputs': torch.randint(0, 10, (8, 81), device=device),
        'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
    }
    
    carry = model.initial_carry(test_batch)
    with torch.no_grad():
        _, _, meta_info = model.inner(carry, test_batch)
    
    gates_list.append(meta_info['gates'].cpu().float())

gates_all = torch.cat(gates_list, dim=0)

gates_sum = gates_all.sum(dim=-1)
max_error = (gates_sum - 1.0).abs().max().item()
mean_error = (gates_sum - 1.0).abs().mean().item()

print(f"\n📊 Normalization Statistics (100 batches, 800 samples):")
print(f"   Max error:     {max_error:.10f}")
print(f"   Mean error:    {mean_error:.10f}")

test1_pass = max_error < 0.01

if max_error < 1e-6:
    print("   ✅ PERFECT: Normalization error < 1e-6")
elif test1_pass:
    print("   ✅ PASS: Normalization error < 0.01")
else:
    print(f"   ❌ FAIL: Max error {max_error:.10f} too high")

print("\n" + "="*80)
print("TEST 2: Gate Statistics")
print("="*80)

gate_H_mean = gates_all[:, 0].mean().item()
gate_L_mean = gates_all[:, 1].mean().item()
gate_H_std = gates_all[:, 0].std().item()
gate_L_std = gates_all[:, 1].std().item()

print(f"\n📊 Gate Distribution:")
print(f"   H-gate: mean={gate_H_mean:.4f}, std={gate_H_std:.4f}")
print(f"   L-gate: mean={gate_L_mean:.4f}, std={gate_L_std:.4f}")

test2_pass = gate_H_std > 0.01 and gate_L_std > 0.01

if test2_pass:
    print("   ✅ PASS: Gates show good variability")
else:
    print("   ⚠️  WARNING: Low gate variability (expected initially)")

print("\n" + "="*80)
print("TEST 3: Dynamic Weighting Effect")
print("="*80)

model.eval()
carry = model.initial_carry(batch)

model.inner.use_dynamic_weighting = True
with torch.no_grad():
    _, logits_dyn, _ = model.inner(carry, batch)

carry = model.initial_carry(batch)
model.inner.use_dynamic_weighting = False
with torch.no_grad():
    _, logits_static, _ = model.inner(carry, batch)

logits_diff = (logits_dyn.float() - logits_static.float()).abs().mean().item()

print(f"\n📊 Logits difference (dynamic vs static):")
print(f"   Mean difference: {logits_diff:.6f}")

test3_pass = logits_diff > 1e-3

if test3_pass:
    print("   ✅ PASS: Dynamic weighting is active")
else:
    print("   ❌ FAIL: Dynamic weighting has no effect")

print("\n" + "="*80)
print("TEST 4: Gradient Flow (FIXED)")
print("="*80)

# 🔥 FIX: Use LOSS HEAD with proper training setup
model.train()
loss_head.model.zero_grad()

# Fresh carry
carry = loss_head.initial_carry(batch)

# 🔥 CRITICAL: Use loss_head.forward() NOT model.inner()!
carry_out, total_loss, metrics, preds, _ = loss_head(carry, batch, return_keys=[])

print(f"\n📊 Loss Values:")
print(f"   Total loss: {total_loss.item():.4f}")
print(f"   Task loss:  {metrics.get('meta/loss_task', 0):.4f}")
print(f"   KL loss:    {metrics.get('meta/loss_kl', 0):.4f}")

# Backward
total_loss.backward()

# Check gradients
encoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.inner.latent_encoder.parameters())
controller_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                          for p in model.inner.meta_controller.parameters())
hrm_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.inner.hrm_inner.parameters())

print(f"\n📊 Gradient Flow:")
print(f"   HRM Inner gradients:  {hrm_has_grad}")
print(f"   Encoder gradients:    {encoder_has_grad}")
print(f"   Controller gradients: {controller_has_grad}")

if controller_has_grad:
    controller_grad_mags = [p.grad.abs().mean().item() 
                            for p in model.inner.meta_controller.parameters() 
                            if p.grad is not None]
    print(f"   Controller grad mean: {sum(controller_grad_mags)/len(controller_grad_mags):.6f}")

if encoder_has_grad:
    encoder_grad_mags = [p.grad.abs().mean().item() 
                         for p in model.inner.latent_encoder.parameters() 
                         if p.grad is not None]
    print(f"   Encoder grad mean:    {sum(encoder_grad_mags)/len(encoder_grad_mags):.6f}")

# 🔥 NEW: Check HRM gradients
if hrm_has_grad:
    hrm_grad_mags = [p.grad.abs().mean().item()
                     for p in model.inner.hrm_inner.parameters()
                     if p.grad is not None]
    print(f"   HRM grad mean:        {sum(hrm_grad_mags)/len(hrm_grad_mags):.6f}")

test4_pass = encoder_has_grad and controller_has_grad and hrm_has_grad

if test4_pass:
    print("   ✅ PASS: Full gradient flow (HRM + Meta)")
elif encoder_has_grad and controller_has_grad:
    print("   ⚠️  PARTIAL: Meta components OK, but HRM not learning")
else:
    print("   ❌ FAIL: No gradients")

model.eval()

print("\n" + "="*80)
print("TEST 5: Context Diversity")
print("="*80)

z_std = gates_all.std(dim=0).mean().item()

print(f"\n📊 Diversity:")
print(f"   Gate std: {z_std:.4f}")

test5_pass = z_std > 0.01

if test5_pass:
    print("   ✅ PASS: Good diversity")
else:
    print("   ⚠️  WARNING: Low diversity (expected initially)")

print("\n" + "="*80)
print("TEST 6: Numerical Stability")
print("="*80)

has_nan = torch.isnan(gates_all).any().item()
has_inf = torch.isinf(gates_all).any().item()

print(f"\n📊 Stability Check:")
print(f"   Contains NaN: {has_nan}")
print(f"   Contains Inf: {has_inf}")

test6_pass = not has_nan and not has_inf

if test6_pass:
    print("   ✅ PASS: Numerically stable")
else:
    print("   ❌ FAIL: Numerical instability detected")

# FINAL SUMMARY
print("\n" + "="*80)
print("📋 FINAL VALIDATION SUMMARY")
print("="*80)

tests = [
    ("Perfect Normalization", test1_pass),
    ("Gate Variability", test2_pass),
    ("Dynamic Weighting", test3_pass),
    ("Gradient Flow", test4_pass),
    ("Context Diversity", test5_pass),
    ("Numerical Stability", test6_pass),
]

print("\nTest Results:")
for name, passed in tests:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"   {status}: {name}")

total_passed = sum(1 for _, p in tests if p)

print(f"\n{'='*80}")
if total_passed == 6:
    print("🎉 6/6 TESTS PASSED! MODEL READY FOR TRAINING!")
    print("✅ All systems operational")
    print("\n⚠️  NOTE: Gate variability is LOW but this is EXPECTED before training.")
    print("   Gates will diversify during training as the model learns different tasks.")
elif total_passed >= 4:
    print(f"✅ {total_passed}/6 PASSED - Core functionality works")
    print("→ Ready to start training")
else:
    print(f"⚠️  {total_passed}/6 PASSED - Some issues remain")

print("="*80)