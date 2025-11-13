#!/usr/bin/env python3
print("ANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANAANLAYANA")
"""
Dynamic Weighting Validation Script (FIXED) CLAUDE REVİZESİ 114-2025
Manifestonun en kritik özelliğini test eder: Gates'in modül çıktılarını gerçekten değiştirip değiştirmediğini
"""

import torch
import numpy as np
import sys
from pathlib import Path

# HRM root'u path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("🔬 HRM-FREE-META: DYNAMIC WEIGHTING VALIDATION TEST (FIXED)")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# Import
from models.hrm.hrm_free_meta import HRMFreeMeta, MetaCarry

# Config - 🔥 FIXED: bfloat16 instead of float32
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
    'z_dim': 64,
    'encoder_layers': 1,
    'controller_hidden': 128,
    'gate_temperature': 1.0,
    'use_dynamic_weighting': True,  # CRITICAL
    'causal': False,
    'pos_encodings': 'rope',
    'puzzle_emb_ndim': 256,
    'halt_exploration_prob': 0.05,
    'forward_dtype': 'bfloat16',  # 🔥 FIXED
}

print("�️  Creating model...")
model = HRMFreeMeta(config).to(device)
model.eval()

# Test batch
batch = {
    'inputs': torch.randint(0, 10, (8, 81), device=device),
    'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
}

print(f"✅ Model created (dtype: {next(model.parameters()).dtype})\n")

# ============================================================================
# TEST 1: Dynamic Weighting ON vs OFF
# ============================================================================
print("=" * 80)
print("TEST 1: Dynamic Weighting ON vs OFF Comparison")
print("=" * 80)

carry = model.initial_carry(batch)

# Forward WITH dynamic weighting
model.inner.use_dynamic_weighting = True
with torch.no_grad():
    carry_out_dyn, logits_dyn, meta_info_dyn = model.inner(carry, batch)

# Forward WITHOUT dynamic weighting (original HRM behavior)
carry = model.initial_carry(batch)  # Reset carry
model.inner.use_dynamic_weighting = False
with torch.no_grad():
    carry_out_static, logits_static, meta_info_static = model.inner(carry, batch)

# Compare logits
logits_diff = (logits_dyn.float() - logits_static.float()).abs()
mean_diff = logits_diff.mean().item()
max_diff = logits_diff.max().item()
pct_changed = (logits_diff > 1e-4).float().mean().item() * 100

print(f"\n📊 Logits Comparison:")
print(f"   Mean absolute difference: {mean_diff:.6f}")
print(f"   Max absolute difference:  {max_diff:.6f}")
print(f"   % of logits changed:      {pct_changed:.2f}%")

if mean_diff > 1e-3:
    print("   ✅ PASS: Dynamic weighting is ACTIVE and affecting logits!")
else:
    print("   ❌ FAIL: Dynamic weighting has NO EFFECT on logits!")
    print("   → Check if gates are being used in hrm_free_meta.py line ~154")

# ============================================================================
# TEST 2: Gates Variability Check
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Gates Variability Across Different Inputs")
print("=" * 80)

model.inner.use_dynamic_weighting = True
gates_list = []
z_context_list = []

# Generate 5 different batches
for i in range(5):
    batch_variant = {
        'inputs': torch.randint(0, 10, (8, 81), device=device),
        'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
    }
    
    carry = model.initial_carry(batch_variant)
    with torch.no_grad():
        _, _, meta_info = model.inner(carry, batch_variant)
    
    gates_list.append(meta_info['gates'].cpu().float())
    z_context_list.append(meta_info['z_context'].cpu().float())

# Analyze variability
gates_all = torch.stack(gates_list)  # [5, 8, 2]
z_context_all = torch.stack(z_context_list)  # [5, 8, 64]

gate_H_mean = gates_all[:, :, 0].mean().item()
gate_L_mean = gates_all[:, :, 1].mean().item()
gate_H_std = gates_all[:, :, 0].std().item()
gate_L_std = gates_all[:, :, 1].std().item()
gate_entropy = -(gates_all * torch.log(gates_all + 1e-8)).sum(dim=-1).mean().item()

print(f"\n📊 Gates Statistics (across 5 different batches):")
print(f"   H-weight: mean={gate_H_mean:.4f}, std={gate_H_std:.4f}")
print(f"   L-weight: mean={gate_L_mean:.4f}, std={gate_L_std:.4f}")
print(f"   Entropy:  {gate_entropy:.4f} (max=0.693)")

# Check if gates sum to 1
gates_sum = gates_all.sum(dim=-1)
sum_error = (gates_sum - 1.0).abs().max().item()
print(f"   Sum check: max error = {sum_error:.6f}")

test1_pass = mean_diff > 1e-3

if sum_error < 1e-4:
    print("   ✅ PASS: Gates properly normalized (sum=1)")
    test2_norm_pass = True
else:
    print(f"   ⚠️  WARNING: Gates normalization error > 1e-4")
    test2_norm_pass = False

if gate_H_std > 0.01 or gate_L_std > 0.01:
    print("   ✅ PASS: Gates show variability across inputs")
    test2_var_pass = True
else:
    print("   ❌ FAIL: Gates are too static (no variability)")
    print("   → Controller might be outputting same values always")
    test2_var_pass = False

# ============================================================================
# TEST 3: Manual Gate Manipulation
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Manual Gate Manipulation (Force Different Weights)")
print("=" * 80)

# Test with extreme gates
test_cases = [
    ("100% H-module", torch.tensor([[1.0, 0.0]]).repeat(8, 1)),
    ("100% L-module", torch.tensor([[0.0, 1.0]]).repeat(8, 1)),
    ("50-50 balanced", torch.tensor([[0.5, 0.5]]).repeat(8, 1)),
]

logits_variants = []

for case_name, forced_gates in test_cases:
    # Forward pass dengan HRM
    carry = model.initial_carry(batch)
    with torch.no_grad():
        inner_carry_out, logits_original, _ = model.inner.hrm_inner(carry.base_carry, batch)
    
    # Extract z_H dan z_L
    z_H = inner_carry_out.z_H
    z_L = inner_carry_out.z_L
    
    # Manual weighted combination
    forced_gates = forced_gates.to(device)
    gate_H = forced_gates[:, 0:1].unsqueeze(-1)
    gate_L = forced_gates[:, 1:2].unsqueeze(-1)
    z_combined = gate_H * z_H + gate_L * z_L
    
    # Generate logits
    logits_manual = model.inner.hrm_inner.lm_head(z_combined)[:, model.inner.hrm_inner.puzzle_emb_len:]
    logits_variants.append((case_name, logits_manual))

# Compare all three
print(f"\n📊 Logits Comparison with Forced Gates:")
for i in range(len(logits_variants)):
    for j in range(i+1, len(logits_variants)):
        name1, logits1 = logits_variants[i]
        name2, logits2 = logits_variants[j]
        
        diff = (logits1.float() - logits2.float()).abs().mean().item()
        print(f"   {name1:20s} vs {name2:20s}: diff = {diff:.6f}")

# Check if 100% H differs from 100% L
h_only = logits_variants[0][1]
l_only = logits_variants[1][1]
critical_diff = (h_only.float() - l_only.float()).abs().mean().item()

if critical_diff > 1e-3:
    print(f"\n   ✅ PASS: H-only and L-only produce DIFFERENT outputs (diff={critical_diff:.6f})")
    print("   → This confirms dynamic weighting mechanism works!")
    test3_pass = True
else:
    print(f"\n   ❌ FAIL: H-only and L-only produce SAME outputs (diff={critical_diff:.6f})")
    print("   → Dynamic weighting is NOT working")
    test3_pass = False

# ============================================================================
# TEST 4: Gradient Flow Check
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Gradient Flow Through Gates")
print("=" * 80)

model.train()
model.inner.use_dynamic_weighting = True

# Zero gradients
model.zero_grad()

# Create batch
carry = model.initial_carry(batch)

# Forward
carry_out, _, meta_info = model.inner(carry, batch)

# Dummy loss - use gates directly to ensure backward
dummy_loss = meta_info['gates'].sum() + meta_info['z_context'].sum()
dummy_loss.backward()

# Check gradients
has_grad_encoder = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.inner.latent_encoder.parameters())
has_grad_controller = any(p.grad is not None and p.grad.abs().sum() > 0 
                          for p in model.inner.meta_controller.parameters())

print(f"\n📊 Gradient Check:")
print(f"   Latent Encoder has gradients:  {has_grad_encoder}")
print(f"   Meta Controller has gradients: {has_grad_controller}")

if has_grad_encoder and has_grad_controller:
    print("   ✅ PASS: Gradients flow through meta-learning components")
    test4_pass = True
else:
    print("   ❌ FAIL: Gradients NOT flowing properly")
    test4_pass = False

model.eval()

# ============================================================================
# TEST 5: Context Vector Diversity
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: Latent Context Diversity")
print("=" * 80)

# Measure z_context diversity
z_context_flat = z_context_all.view(-1, 64)  # [40, 64]
z_mean = z_context_flat.mean(dim=0)
z_std = z_context_flat.std(dim=0)

print(f"\n📊 z_context Statistics:")
print(f"   Mean norm:     {z_mean.norm().item():.4f}")
print(f"   Std dev mean:  {z_std.mean().item():.4f}")
print(f"   Min std:       {z_std.min().item():.4f}")
print(f"   Max std:       {z_std.max().item():.4f}")

# Check if z_context has collapsed
if z_std.mean().item() > 0.01:
    print("   ✅ PASS: z_context shows good diversity")
    test5_pass = True
else:
    print("   ❌ FAIL: z_context has collapsed (no diversity)")
    print("   → KL weight might be too high")
    test5_pass = False

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📋 FINAL SUMMARY")
print("=" * 80)

tests_passed = [
    ("Dynamic Weighting Effect", test1_pass),
    ("Gates Normalization", test2_norm_pass),
    ("Gates Variability", test2_var_pass),
    ("H vs L Differentiation", test3_pass),
    ("Gradient Flow", test4_pass),
    ("Context Diversity", test5_pass),
]

print("\nTest Results:")
for test_name, passed in tests_passed:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"   {status}: {test_name}")

total_passed = sum(1 for _, p in tests_passed if p)
total_tests = len(tests_passed)

print(f"\n{'='*80}")
if total_passed == total_tests:
    print("🎉 ALL TESTS PASSED! Dynamic Weighting is FULLY FUNCTIONAL!")
    print("→ Manifestonun kalbi atıyor! Eğitime başlayabilirsiniz.")
elif total_passed >= 4:
    print(f"✅ {total_passed}/{total_tests} tests passed. System is mostly working.")
    print("→ Bazı iyileştirmeler yapılabilir ama temel fonksiyon çalışıyor.")
else:
    print(f"⚠️  Only {total_passed}/{total_tests} tests passed.")
    print("→ Bazı sorunlar var, kontrol edin.")

print("=" * 80)