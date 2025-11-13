#!/usr/bin/env python3
"""
NaN Debug Script - Nerede patlıyor bulalım
"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.hrm.hrm_free_meta import HRMFreeMeta

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
    'forward_dtype': 'float32',  # 🔥 BU ÖNEMLİ
}

print("Creating model...")
model = HRMFreeMeta(config).to(device)
model.eval()

batch = {
    'inputs': torch.randint(0, 10, (8, 81), device=device),
    'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
}

print("Running forward pass with debug hooks...\n")

carry = model.initial_carry(batch)

# Hook her katmanda NaN kontrolü
def check_nan_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"❌ NaN detected at: {name}")
                print(f"   Output shape: {output.shape}")
                print(f"   Output dtype: {output.dtype}")
                print(f"   Sample values: {output.flatten()[:10]}")
            else:
                print(f"✅ {name}: OK (dtype={output.dtype})")
        elif isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor) and torch.isnan(o).any():
                    print(f"❌ NaN detected at: {name}[{i}]")
    return hook

# Register hooks
model.inner.hrm_inner.register_forward_hook(check_nan_hook("HRM Inner"))
model.inner.latent_encoder.register_forward_hook(check_nan_hook("Latent Encoder"))
model.inner.meta_controller.register_forward_hook(check_nan_hook("Meta Controller"))

with torch.no_grad():
    try:
        carry_out, logits, meta_info = model.inner(carry, batch)
        
        print("\n" + "="*60)
        print("Final outputs:")
        print(f"  logits: shape={logits.shape}, has_nan={torch.isnan(logits).any()}")
        print(f"  gates: shape={meta_info['gates'].shape}, has_nan={torch.isnan(meta_info['gates']).any()}")
        print(f"  z_context: shape={meta_info['z_context'].shape}, has_nan={torch.isnan(meta_info['z_context']).any()}")
        
        if torch.isnan(meta_info['gates']).any():
            print("\n🔍 Gates değerleri:")
            print(meta_info['gates'])
            
        if torch.isnan(meta_info['z_context']).any():
            print("\n🔍 z_context değerleri:")
            print(meta_info['z_context'])
            print(f"\nmu: {meta_info['mu']}")
            print(f"logvar: {meta_info['logvar']}")
            
    except Exception as e:
        print(f"\n💥 Exception: {e}")
        import traceback
        traceback.print_exc()