#!/usr/bin/env python3
"""
Deep Debug - HRM Inner'daki NaN'ın kaynağını bulalım
"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    'use_dynamic_weighting': True,
    'causal': False,
    'pos_encodings': 'rope',
    'puzzle_emb_ndim': 256,
    'halt_exploration_prob': 0.05,
    'forward_dtype': 'bfloat16',
}

print("="*80)
print("🔍 DEEP DEBUG: Finding NaN source in HRM Inner")
print("="*80)

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config, HierarchicalReasoningModel_ACTV1_Inner

# Create HRM Inner directly
hrm_config = HierarchicalReasoningModel_ACTV1Config(**config)
hrm_inner = HierarchicalReasoningModel_ACTV1_Inner(hrm_config).to(device)
hrm_inner.eval()

print(f"✅ HRM Inner created")
print(f"   forward_dtype: {hrm_inner.forward_dtype}")
print(f"   Model dtype: {next(hrm_inner.parameters()).dtype}")

# Check if dtype matches
param_dtype = next(hrm_inner.parameters()).dtype
if param_dtype != hrm_inner.forward_dtype:
    print(f"⚠️  WARNING: Dtype mismatch!")
    print(f"   Parameters: {param_dtype}")
    print(f"   Forward dtype: {hrm_inner.forward_dtype}")

batch = {
    'inputs': torch.randint(0, 10, (8, 81), device=device),
    'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
}

print("\n" + "="*80)
print("Step 1: Check Initial Carry")
print("="*80)

carry = hrm_inner.empty_carry(8)
print(f"z_H: shape={carry.z_H.shape}, dtype={carry.z_H.dtype}, has_nan={torch.isnan(carry.z_H).any()}")
print(f"z_L: shape={carry.z_L.shape}, dtype={carry.z_L.dtype}, has_nan={torch.isnan(carry.z_L).any()}")

# Reset carry
reset_flag = torch.ones(8, dtype=torch.bool, device=device)
carry = hrm_inner.reset_carry(reset_flag, carry)
print(f"\nAfter reset:")
print(f"z_H: has_nan={torch.isnan(carry.z_H).any()}, mean={carry.z_H.mean():.4f}, std={carry.z_H.std():.4f}")
print(f"z_L: has_nan={torch.isnan(carry.z_L).any()}, mean={carry.z_L.mean():.4f}, std={carry.z_L.std():.4f}")

print("\n" + "="*80)
print("Step 2: Check Input Embeddings")
print("="*80)

input_embeddings = hrm_inner._input_embeddings(batch['inputs'], batch['puzzle_identifiers'])
print(f"input_embeddings: shape={input_embeddings.shape}, dtype={input_embeddings.dtype}")
print(f"  has_nan={torch.isnan(input_embeddings).any()}")
print(f"  mean={input_embeddings.mean():.4f}, std={input_embeddings.std():.4f}")
print(f"  min={input_embeddings.min():.4f}, max={input_embeddings.max():.4f}")

print("\n" + "="*80)
print("Step 3: RoPE Check")
print("="*80)

if hasattr(hrm_inner, 'rotary_emb'):
    cos, sin = hrm_inner.rotary_emb()
    print(f"cos: shape={cos.shape}, dtype={cos.dtype}, has_nan={torch.isnan(cos).any()}")
    print(f"sin: shape={sin.shape}, dtype={sin.dtype}, has_nan={torch.isnan(sin).any()}")

print("\n" + "="*80)
print("Step 4: Manual Forward Pass with Hooks")
print("="*80)

# Hook to catch NaNs
nan_locations = []

def make_nan_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                nan_count = torch.isnan(output).sum().item()
                total = output.numel()
                nan_locations.append({
                    'name': name,
                    'shape': output.shape,
                    'dtype': output.dtype,
                    'nan_count': nan_count,
                    'total': total,
                    'pct': 100 * nan_count / total
                })
                print(f"❌ NaN at {name}: {nan_count}/{total} ({100*nan_count/total:.2f}%)")
        return output
    return hook

# Register hooks on all layers
for name, module in hrm_inner.named_modules():
    if len(list(module.children())) == 0:  # Only leaf modules
        module.register_forward_hook(make_nan_hook(name))

print("\nRunning forward pass...")
with torch.no_grad():
    try:
        carry_out, logits, (q_halt, q_cont) = hrm_inner(carry, batch)
        
        print("\n" + "="*80)
        print("✅ Forward pass completed")
        print("="*80)
        
        print(f"\nOutputs:")
        print(f"  z_H: has_nan={torch.isnan(carry_out.z_H).any()}")
        print(f"  z_L: has_nan={torch.isnan(carry_out.z_L).any()}")
        print(f"  logits: has_nan={torch.isnan(logits).any()}")
        
        if torch.isnan(carry_out.z_H).any():
            print(f"\n🔍 z_H NaN Analysis:")
            nan_mask = torch.isnan(carry_out.z_H)
            print(f"  Total NaNs: {nan_mask.sum()}/{carry_out.z_H.numel()}")
            print(f"  First NaN position: {torch.where(nan_mask)[0][0] if nan_mask.any() else 'N/A'}")
        
        if torch.isnan(carry_out.z_L).any():
            print(f"\n🔍 z_L NaN Analysis:")
            nan_mask = torch.isnan(carry_out.z_L)
            print(f"  Total NaNs: {nan_mask.sum()}/{carry_out.z_L.numel()}")
            print(f"  First NaN position: {torch.where(nan_mask)[0][0] if nan_mask.any() else 'N/A'}")
        
    except Exception as e:
        print(f"\n💥 Exception during forward: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("📋 NaN Timeline")
print("="*80)

if nan_locations:
    print("\nNaN appeared in these layers:")
    for i, loc in enumerate(nan_locations, 1):
        print(f"{i}. {loc['name']}")
        print(f"   Shape: {loc['shape']}, Dtype: {loc['dtype']}")
        print(f"   NaNs: {loc['nan_count']}/{loc['total']} ({loc['pct']:.2f}%)")
    
    print(f"\n⚠️  FIRST NaN SOURCE: {nan_locations[0]['name']}")
else:
    print("✅ No NaNs detected in intermediate layers!")
    print("   Problem might be in carry initialization or reset")

print("\n" + "="*80)
print("Step 5: Check H_init and L_init buffers")
print("="*80)

print(f"H_init: dtype={hrm_inner.H_init.dtype}, has_nan={torch.isnan(hrm_inner.H_init).any()}")
print(f"  mean={hrm_inner.H_init.mean():.4f}, std={hrm_inner.H_init.std():.4f}")
print(f"L_init: dtype={hrm_inner.L_init.dtype}, has_nan={torch.isnan(hrm_inner.L_init).any()}")
print(f"  mean={hrm_inner.L_init.mean():.4f}, std={hrm_inner.L_init.std():.4f}")

print("\n" + "="*80)
print("🎯 DIAGNOSIS COMPLETE")
print("="*80)