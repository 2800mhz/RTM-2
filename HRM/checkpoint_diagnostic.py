#!/usr/bin/env python3
"""
🔍 Checkpoint Diagnostic Tool
Analyze checkpoints and identify training issues
"""

import torch
import os
import sys
from pathlib import Path
import yaml

def load_and_analyze_checkpoint(checkpoint_path: str):
    """Load checkpoint and print detailed analysis"""
    
    print("\n" + "="*80)
    print("🔍 CHECKPOINT DIAGNOSTIC")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"📂 Loading: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("\n" + "="*80)
        print("📊 CHECKPOINT CONTENTS")
        print("="*80)
        
        # Basic info
        print("\n🔑 Keys in checkpoint:")
        for key in checkpoint.keys():
            if key == 'config':
                print(f"  ✓ {key}: {type(checkpoint[key])}")
            elif key.endswith('_state_dict'):
                print(f"  ✓ {key}: {len(checkpoint[key])} parameters")
            else:
                print(f"  ✓ {key}: {checkpoint[key]}")
        
        # Training state
        print("\n📈 Training State:")
        print(f"  Step: {checkpoint.get('step', 'N/A')}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Metric: {checkpoint.get('best_metric', 'N/A')}")
        
        # Model state
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"\n🏗️  Model State:")
            print(f"  Total parameters: {len(model_state)}")
            
            # Show first few parameter shapes
            print(f"\n  Sample parameters:")
            for i, (name, param) in enumerate(list(model_state.items())[:5]):
                print(f"    {name}: {param.shape}")
            print(f"    ... ({len(model_state) - 5} more)")
            
            # Check for NaN/Inf
            has_nan = False
            has_inf = False
            for name, param in model_state.items():
                if torch.isnan(param).any():
                    print(f"    ⚠️  NaN found in: {name}")
                    has_nan = True
                if torch.isinf(param).any():
                    print(f"    ⚠️  Inf found in: {name}")
                    has_inf = True
            
            if not has_nan and not has_inf:
                print(f"    ✅ No NaN/Inf detected")
        
        # Optimizer state
        if 'optimizer_state_dicts' in checkpoint:
            print(f"\n⚙️  Optimizers: {len(checkpoint['optimizer_state_dicts'])}")
            for i, opt_state in enumerate(checkpoint['optimizer_state_dicts']):
                print(f"  Optimizer {i}:")
                if 'state' in opt_state:
                    print(f"    State entries: {len(opt_state['state'])}")
                if 'param_groups' in opt_state:
                    for j, pg in enumerate(opt_state['param_groups']):
                        print(f"    Param group {j}:")
                        print(f"      lr: {pg.get('lr', 'N/A')}")
                        print(f"      weight_decay: {pg.get('weight_decay', 'N/A')}")
        
        # Meta optimizer
        if 'meta_optimizer_state_dict' in checkpoint:
            print(f"\n🎯 Meta Optimizer: Present")
        
        # Config
        if 'config' in checkpoint:
            print(f"\n⚙️  Config:")
            config = checkpoint['config']
            print(f"  Mode: {config.get('mode', 'N/A')}")
            print(f"  Epochs: {config.get('epochs', 'N/A')}")
            print(f"  Batch size: {config.get('global_batch_size', 'N/A')}")
            print(f"  LR: {config.get('lr', 'N/A')}")
            
            if 'arch' in config:
                print(f"  Architecture: {config['arch'].get('name', 'N/A')}")
                print(f"  Loss: {config['arch'].get('loss', {}).get('name', 'N/A')}")
        
        print("\n" + "="*80)
        print("✅ DIAGNOSTIC COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()


def check_checkpoint_directory(checkpoint_dir: str):
    """Analyze all checkpoints in directory"""
    
    print("\n" + "="*80)
    print("📁 CHECKPOINT DIRECTORY ANALYSIS")
    print("="*80)
    print(f"Directory: {checkpoint_dir}\n")
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Directory not found: {checkpoint_dir}")
        return
    
    # Find all checkpoints
    checkpoint_files = list(Path(checkpoint_dir).glob("checkpoint_*.pt"))
    
    if not checkpoint_files:
        print("⚠️  No checkpoints found")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints:\n")
    
    # Sort by step number
    checkpoints_info = []
    for cp_file in checkpoint_files:
        try:
            cp = torch.load(cp_file, map_location='cpu')
            step = cp.get('step', 0)
            best_metric = cp.get('best_metric', 0.0)
            checkpoints_info.append({
                'file': cp_file.name,
                'step': step,
                'epoch': cp.get('epoch', 0),
                'best_metric': best_metric,
                'size_mb': cp_file.stat().st_size / (1024*1024)
            })
        except:
            print(f"  ⚠️  Failed to load: {cp_file.name}")
    
    # Sort by step
    checkpoints_info.sort(key=lambda x: x['step'])
    
    # Print table
    print(f"{'Checkpoint':<30} {'Step':<8} {'Epoch':<8} {'Best Metric':<12} {'Size (MB)':<10}")
    print("-" * 80)
    for info in checkpoints_info:
        print(f"{info['file']:<30} {info['step']:<8} {info['epoch']:<8} "
              f"{info['best_metric']:<12.4f} {info['size_mb']:<10.2f}")
    
    # Check for config
    config_file = Path(checkpoint_dir) / "config.yaml"
    if config_file.exists():
        print(f"\n✅ Config file found: config.yaml")
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"   Mode: {config.get('mode', 'N/A')}")
            print(f"   Data: {config.get('data_path', 'N/A')}")
        except:
            print(f"   ⚠️  Failed to read config")
    else:
        print(f"\n⚠️  No config.yaml found")
    
    # Check for W&B run ID
    wandb_file = Path(checkpoint_dir) / "wandb_run_id.txt"
    if wandb_file.exists():
        with open(wandb_file, 'r') as f:
            run_id = f.read().strip()
        print(f"✅ W&B Run ID: {run_id}")
    
    print("\n" + "="*80)


def suggest_fixes(checkpoint_path: str):
    """Suggest fixes based on checkpoint analysis"""
    
    print("\n" + "="*80)
    print("💡 SUGGESTED FIXES")
    print("="*80)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        issues = []
        
        # Check best metric
        if checkpoint.get('best_metric', 0.0) == 0.0:
            issues.append({
                'issue': 'Best metric is 0.0',
                'likely_cause': 'Model not learning or accuracy metric missing',
                'fix': [
                    'Check if loss_head returns accuracy metric',
                    'Verify data is correctly labeled',
                    'Try lower learning rate',
                    'Check if model forward pass works correctly'
                ]
            })
        
        # Check config
        if 'config' in checkpoint:
            config = checkpoint['config']
            
            # Check if using pretrain mode with meta loss
            if config.get('mode') == 'pretrain':
                loss_name = config.get('arch', {}).get('loss', {}).get('name', '')
                if 'meta' in loss_name.lower():
                    issues.append({
                        'issue': 'Using meta loss in pretrain mode',
                        'likely_cause': 'Meta loss may not be optimized for standard pretraining',
                        'fix': [
                            'Consider using standard cross_entropy loss for pretraining',
                            'Or switch to mode=meta for meta-learning',
                            'Check loss_head implementation'
                        ]
                    })
        
        # Check for NaN/Inf
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            for name, param in model_state.items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    issues.append({
                        'issue': f'NaN/Inf in model parameters: {name}',
                        'likely_cause': 'Training instability or gradient explosion',
                        'fix': [
                            'Reduce learning rate',
                            'Increase gradient clipping',
                            'Check for division by zero',
                            'Use mixed precision training carefully'
                        ]
                    })
                    break
        
        if not issues:
            print("\n✅ No obvious issues detected!")
            print("\nGeneral suggestions:")
            print("  • Monitor training loss - should be decreasing")
            print("  • Check evaluation metrics - should improve over time")
            print("  • Verify data preprocessing is correct")
            print("  • Try visualizing predictions on validation set")
        else:
            for i, issue_info in enumerate(issues, 1):
                print(f"\n{i}. {issue_info['issue']}")
                print(f"   Likely cause: {issue_info['likely_cause']}")
                print(f"   Suggested fixes:")
                for fix in issue_info['fix']:
                    print(f"     • {fix}")
        
    except Exception as e:
        print(f"\n❌ Error analyzing checkpoint: {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint Diagnostic Tool")
    parser.add_argument("checkpoint", type=str, 
                       help="Path to checkpoint file or directory")
    parser.add_argument("--directory", action="store_true",
                       help="Analyze entire directory instead of single file")
    parser.add_argument("--suggest", action="store_true",
                       help="Suggest fixes based on analysis")
    
    args = parser.parse_args()
    
    if args.directory:
        check_checkpoint_directory(args.checkpoint)
    else:
        load_and_analyze_checkpoint(args.checkpoint)
    
    if args.suggest:
        suggest_fixes(args.checkpoint)