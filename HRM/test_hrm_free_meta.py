#!/usr/bin/env python3
"""
HRM-Free-Meta quick smoke test.
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
# HRM root directory'yi Python path'e ekle
hrm_root = Path(__file__).parent
sys.path.insert(0, str(hrm_root))

print("=" * 60)
print("HRM-Free-Meta Quick Test")
print("=" * 60)

# Test 1: Import kontrolü
print("\n[Test 1/6] Import kontrolü...")
try:
    # Yeni HRM-Free-Meta yapısına göre importlar düzeltildi
    from models.hrm.latent_context_encoder import LatentContextEncoder
    from models.hrm.meta_controller import MetaController
    from models.hrm.hrm_free_meta import HRMFreeMeta, MetaCarry
    from meta_loss_head import MetaLearningLossHead

    print("✅ Tüm modüller başarıyla import edildi")
except Exception as e:
    print(f"❌ Import hatası: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Test 2: Latent Encoder
print("\n[Test 2/6] Latent Context Encoder testi...")
try:
    from models.hrm.meta_losses import MetaLearningLoss
    
    encoder = LatentContextEncoder(hidden_size=512, z_dim=128)
    z_H = torch.randn(4, 81, 512)
    z_L = torch.randn(4, 81, 512)

    z_context, mu, logvar = encoder(z_H, z_L)
    assert z_context.shape == (4, 128)
    assert mu.shape == (4, 128)
    assert logvar.shape == (4, 128)
    meta_loss_fn = MetaLearningLoss(kl_weight=0.001)
    kl_loss = meta_loss_fn.kl_divergence(mu, logvar)
    assert kl_loss.numel() == 1

    print(f"✅ Encoder çalışıyor (z_context: {z_context.shape}, KL: {kl_loss.item():.4f})")
except Exception as e:
    print(f"❌ Encoder hatası: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Test 3: Meta Controller
print("\n[Test 3/6] Meta Controller testi...")
try:
    controller = MetaController(z_dim=128, num_modules=2)
    z_context = torch.randn(4, 128)
    gates = controller(z_context)
    assert gates.shape == (4, 2)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(4), atol=1e-5)
    entropy = controller.entropy_regularization(gates)
    stats = controller.get_gate_stats(gates)
    print(f"✅ Controller çalışıyor (gates H={stats['gate_H_mean']:.3f}, L={stats['gate_L_mean']:.3f})")
except Exception as e:
    print(f"❌ Controller hatası: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Test 4: HRM-Free-Meta Model
print("\n[Test 4/6] HRM-Free-Meta model testi...")
try:
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

    model = HRMFreeMeta(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model oluşturuldu (Toplam parametre: {total_params:,})")

    batch = {
        'inputs': torch.randint(0, 10, (8, 81), device=device),
        'puzzle_identifiers': torch.randint(0, 100, (8,), device=device),
    }
    carry = model.initial_carry(batch)
    assert isinstance(carry, MetaCarry)

    model.eval()
    with torch.no_grad():
        carry_out, loss, metrics, preds, all_finish = model(carry, batch, return_keys=['z_context', 'gates'])

    assert 'logits' in preds
    assert 'z_context' in preds
    assert 'gates' in preds
    print("✅ Model forward pass başarılı")
except Exception as e:
    print(f"❌ Model hatası: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Test 5: Meta Loss Head
print("\n[Test 5/6] Meta Loss Head testi...")
try:
    loss_head = MetaLearningLossHead(model=model, loss_type='stablemax_cross_entropy', alpha_kl=0.001, beta_entropy=0.01)
    batch_with_labels = {**batch, 'labels': torch.randint(0, 10, (8, 81), device=device)}
    carry = loss_head.initial_carry(batch_with_labels)
    model.train()
    carry_out, total_loss, metrics, preds, all_finish = loss_head(carry, batch_with_labels, return_keys=[])

    print(f"✅ Loss head çalışıyor (Total Loss: {metrics['meta/total_loss'].item():.4f})")

except Exception as e:
    print(f"❌ Loss head hatası: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Test 6: Backward pass
print("\n[Test 6/6] Backward pass (gradient) testi...")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("✅ Backward pass ve optimizer step başarılı")
except Exception as e:
    print(f"❌ Backward pass hatası: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 TÜM TESTLER BAŞARILI!")
print("=" * 60)
