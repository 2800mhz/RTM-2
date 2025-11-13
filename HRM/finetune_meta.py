"""
BAŞTAN EĞİTİM YAPACAĞIZ 
HRM-Free-Meta Fine-tuning Script
Mevcut HRM checkpoint'inden meta-learning katmanlarını ekleyip sadece onları eğitir.

"""

import os
import torch
from pathlib import Path

# Import edilecek modüller
from models.hrm.hrm_free_meta import HRMFreeMeta
from models.hrm.meta_loss_head import MetaLearningLossHead
from adam_atan2_pytorch import AdamAtan2

def freeze_hrm_weights(model):
    """HRM'nin orijinal parametrelerini dondur"""
    for name, param in model.named_parameters():
        if 'hrm_inner' in name:
            param.requires_grad = False
            print(f"❄️  Frozen: {name}")
    
    # Sadece meta katmanlarını train et
    meta_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            meta_params.append(param)
            print(f"🔥 Trainable: {name}")
    
    return meta_params


def load_pretrained_hrm(checkpoint_path, meta_config):
    """
    Pretrained HRM'yi yükleyip HRM-Free-Meta'ya dönüştür
    """
    print(f"\n📦 Loading pretrained HRM from: {checkpoint_path}")
    
    # 1. HRM-Free-Meta modelini oluştur
    model = HRMFreeMeta(meta_config)
    
    # 2. Mevcut checkpoint'i yükle
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 3. Sadece HRM kısmını yükle (meta katmanları yeni, onları skip et)
    model_state = model.state_dict()
    pretrained_state = checkpoint
    
    # HRM inner parametrelerini transfer et
    loaded_keys = []
    skipped_keys = []
    
    for key in pretrained_state.keys():
        # Meta-learning katmanlarını atla
        if any(x in key for x in ['latent_encoder', 'meta_controller']):
            skipped_keys.append(key)
            continue
        
        # HRM parametrelerini yükle
        new_key = f'inner.hrm_inner.{key}' if not key.startswith('inner.') else key
        
        if new_key in model_state:
            model_state[new_key] = pretrained_state[key]
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)
    
    model.load_state_dict(model_state, strict=False)
    
    print(f"✅ Loaded {len(loaded_keys)} HRM parameters")
    print(f"⚠️  Skipped {len(skipped_keys)} keys (meta layers)")
    
    return model


def finetune_meta_layers(
    checkpoint_path: str,
    data_path: str,
    epochs: int = 5000,
    lr: float = 1e-4,
    output_dir: str = "checkpoints/hrm_free_meta_finetuned" #fine tune planı iptal baştan eğiteceğiz
):
    """
    Fine-tune meta layers on top of pretrained HRM
    """
    
    # Config
    config = {
        'batch_size': 384,
        'seq_len': 81,
        'vocab_size': 10,
        'num_puzzle_identifiers': 1000,
        'hidden_size': 512,
        'num_heads': 8,
        'expansion': 4,
        'H_layers': 4,
        'L_layers': 4,
        'H_cycles': 2,
        'L_cycles': 2,
        'halt_max_steps': 16,
        'z_dim': 128,
        'encoder_layers': 2,
        'controller_hidden': 256,
        'gate_temperature': 1.0,
        'use_dynamic_weighting': True,
        'causal': False,
        'pos_encodings': 'rope',
        'puzzle_emb_ndim': 512,
    }
    
    # Load model
    model = load_pretrained_hrm(checkpoint_path, config)
    model = model.cuda()
    
    # Freeze HRM, only train meta layers
    meta_params = freeze_hrm_weights(model)
    
    print(f"\n🎯 Training {len(meta_params)} meta parameters")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in meta_params):,}")
    
    # Loss head
    loss_head = MetaLearningLossHead(
        model=model,
        alpha_kl=0.001,
        beta_entropy=0.01
    )
    
    # Optimizer (sadece meta params)
    optimizer = AdamAtan2(meta_params, lr=lr, weight_decay=0.1)
    
    # Data loader
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    from torch.utils.data import DataLoader
    
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            dataset_path=data_path,
            seed=42,
            rank=0,
            num_replicas=1
        ),
        split='train'
    )
    
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1)
    
    # Training loop
    print("\n🚀 Starting fine-tuning...")
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (set_name, batch, global_batch_size) in enumerate(dataloader):
            # Move to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Initialize carry
            carry = loss_head.initial_carry(batch)
            
            # Forward
            carry, loss, metrics, preds, _ = loss_head(carry, batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                print(f"  KL: {metrics['meta/loss_kl'].item():.4f} | "
                      f"Gate H: {metrics['meta/gate_H_mean'].item():.3f} | "
                      f"Gate L: {metrics['meta/gate_L_mean'].item():.3f}")
        
        # Save checkpoint
        if epoch % 500 == 0:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/epoch_{epoch}.pt")
            print(f"💾 Saved checkpoint: {output_dir}/epoch_{epoch}.pt")
    
    print("\n✅ Fine-tuning complete!")


if __name__ == "__main__":
    # Kullanım
    finetune_meta_layers(
        checkpoint_path="checkpoints/hrm_sudoku/step_50000",  # Mevcut HRM checkpoint
        data_path="data/sudoku-extreme-1k-aug-1000",
        epochs=5000,
        lr=1e-4
    )