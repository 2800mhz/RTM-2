#!/usr/bin/env python3
"""
🎯 UNIFIED TRAINING PIPELINE FOR HRM-FREE-META
Complete training system with meta-learning, standard pretraining, and distributed support
NOW WITH RESUME TRAINING SUPPORT!

Usage:
    # Standard pretraining
    python unified_training.py mode=pretrain
    
    # Meta-learning (MAML-style)
    python unified_training.py mode=meta
    
    # Resume training
    python unified_training.py --resume_from checkpoints/my_run/checkpoint_latest.pt
    python unified_training.py --resume_auto --checkpoint_path checkpoints/my_run
    
    # Distributed training
    torchrun --nproc_per_node=4 unified_training.py mode=pretrain
"""

import os
import sys
import math
import yaml
import shutil
import glob
import coolname
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Sequence, List, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

import tqdm
import wandb
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf

# Import model components
from models.hrm.hrm_free_meta import HRMFreeMeta
from meta_loss_head import MetaLearningLossHead
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from utils.functions import load_model_class, get_model_source_path

try:
    from adam_atan2_pytorch import AdamAtan2
except ImportError:
    print("⚠️  AdamAtan2 not found, using standard AdamW")
    AdamAtan2 = None


# ============================================================================
# CONFIGURATION
# ============================================================================

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class MetaConfig(pydantic.BaseModel):
    """Meta-learning specific config"""
    enabled: bool = False
    n_support: int = 5
    n_query: int = 10
    inner_steps: int = 5
    inner_lr: float = 1e-3
    task_batch_size: int = 4
    meta_batches_per_epoch: int = 100
    kl_weight_start: float = 0.0001
    kl_weight_end: float = 0.001
    kl_warmup_steps: int = 5000


class TrainingConfig(pydantic.BaseModel):
    # Mode
    mode: str = "pretrain"  # "pretrain" or "meta"
    
    # Architecture
    arch: ArchConfig
    
    # Data
    data_path: str
    
    # Hyperparameters
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Puzzle embedding
    puzzle_emb_lr: float = 1e-4
    puzzle_emb_weight_decay: float = 0.0
    
    # Meta-learning
    meta: MetaConfig = MetaConfig()
    
    # Logging & checkpointing
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []
    log_interval: int = 10
    
    # W&B
    wandb_enabled: bool = True
    
    # Misc
    seed: int = 0
    grad_clip: float = 1.0
    
    # Resume training fields (NEW!)
    resume_from: Optional[str] = None  # Path to checkpoint file or directory
    resume_auto: bool = False          # Auto-resume from latest checkpoint
    reset_optimizer: bool = False      # Reset optimizer state when resuming
    reset_lr_schedule: bool = False    # Reset learning rate schedule
    resume_epoch_offset: int = 0       # Add offset to epoch count (for extending training)


# ============================================================================
# CHECKPOINT LOADING FUNCTIONS (NEW!)
# ============================================================================

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # First try checkpoint_latest.pt
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    if latest_path.exists():
        return str(latest_path)
    
    # Otherwise find highest step number
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    
    if not checkpoint_files:
        return None
    
    # Extract step numbers and find max
    steps = []
    for f in checkpoint_files:
        try:
            step = int(f.stem.split('_')[-1])
            steps.append((step, f))
        except:
            continue
    
    if not steps:
        return None
    
    steps.sort(reverse=True)
    return str(steps[0][1])


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizers: Sequence[torch.optim.Optimizer],
    config: TrainingConfig,
    rank: int
) -> Tuple[int, int, float, Optional[dict]]:
    """
    Load checkpoint and restore training state
    
    Returns:
        (step, epoch, best_metric, meta_optimizer_state)
    """
    if rank != 0:
        # Wait for rank 0 to load
        if dist.is_initialized():
            dist.barrier()
        return 0, 0, 0.0, None
    
    # Find checkpoint file
    checkpoint_file = checkpoint_path
    
    if os.path.isdir(checkpoint_path):
        checkpoint_file = find_latest_checkpoint(checkpoint_path)
        if checkpoint_file is None:
            print(f"⚠️  No checkpoint found in {checkpoint_path}")
            return 0, 0, 0.0, None
    
    if not os.path.exists(checkpoint_file):
        print(f"⚠️  Checkpoint not found: {checkpoint_file}")
        return 0, 0, 0.0, None
    
    print(f"\n{'='*80}")
    print(f"📥 RESUMING FROM CHECKPOINT")
    print(f"{'='*80}")
    print(f"File: {checkpoint_file}")
    
    try:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model state")
        
        # Load optimizer states (unless reset requested)
        if not config.reset_optimizer and 'optimizer_state_dicts' in checkpoint:
            for opt, opt_state in zip(optimizers, checkpoint['optimizer_state_dicts']):
                opt.load_state_dict(opt_state)
            print(f"✅ Loaded optimizer states")
        else:
            print(f"⚠️  Optimizer states reset (training from scratch LR schedule)")
        
        # Load meta optimizer if exists
        meta_optimizer_state = None
        if 'meta_optimizer_state_dict' in checkpoint and not config.reset_optimizer:
            meta_optimizer_state = checkpoint['meta_optimizer_state_dict']
            print(f"✅ Loaded meta-optimizer state")
        
        # Get training progress
        step = checkpoint.get('step', 0)
        epoch = checkpoint.get('epoch', 0)
        best_metric = checkpoint.get('best_metric', 0.0)
        
        # Apply epoch offset if requested (for extending training)
        if config.resume_epoch_offset > 0:
            print(f"📈 Applying epoch offset: +{config.resume_epoch_offset}")
        
        print(f"\nResumed from:")
        print(f"  Step: {step}")
        print(f"  Epoch: {epoch}")
        print(f"  Best metric: {best_metric:.4f}")
        print(f"{'='*80}\n")
        
        # Broadcast to all ranks
        if dist.is_initialized():
            dist.barrier()
        
        return step, epoch, best_metric, meta_optimizer_state
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0.0, None


# ============================================================================
# TASK BATCH FOR META-LEARNING
# ============================================================================

@dataclass
class TaskBatch:
    """Single task's support and query sets"""
    support_inputs: torch.Tensor
    support_labels: torch.Tensor
    query_inputs: torch.Tensor
    query_labels: torch.Tensor
    puzzle_ids: torch.Tensor
    task_id: int


def pad_to_batch_size(batch: Dict[str, torch.Tensor], target_size: int) -> Tuple[Dict[str, torch.Tensor], int]:
    """Pad batch to target size for consistent processing"""
    current_size = batch['inputs'].shape[0]
    actual_size = current_size
    
    if current_size == target_size:
        return batch, actual_size
    
    if current_size > target_size:
        return {k: v[:target_size] for k, v in batch.items()}, target_size
    
    # Pad
    pad_size = target_size - current_size
    padded_batch = {}
    
    for key, value in batch.items():
        if key in ['inputs', 'labels']:
            pad_shape = (pad_size,) + value.shape[1:]
            padding = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            padded_batch[key] = torch.cat([value, padding], dim=0)
        elif key == 'puzzle_identifiers':
            padding = value[-1:].expand(pad_size, *value.shape[1:]) if value.shape[0] > 0 else \
                     torch.zeros((pad_size,) + value.shape[1:], dtype=value.dtype, device=value.device)
            padded_batch[key] = torch.cat([value, padding], dim=0)
        else:
            padded_batch[key] = value
    
    return padded_batch, actual_size


def create_task_batch_from_puzzle_batch(
    batch: Dict[str, torch.Tensor],
    n_support: int = 5,
    n_query: int = 10
) -> TaskBatch:
    """Split puzzle batch into support/query for meta-learning"""
    total = batch['inputs'].size(0)
    assert total >= n_support + n_query, f"Batch too small: {total} < {n_support + n_query}"
    
    indices = torch.randperm(total)
    support_idx = indices[:n_support]
    query_idx = indices[n_support:n_support+n_query]
    
    return TaskBatch(
        support_inputs=batch['inputs'][support_idx],
        support_labels=batch['labels'][support_idx],
        query_inputs=batch['inputs'][query_idx],
        query_labels=batch['labels'][query_idx],
        puzzle_ids=batch['puzzle_identifiers'][:1],
        task_id=batch['puzzle_identifiers'][0].item(),
    )


# ============================================================================
# TRAINING STATE
# ============================================================================

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_base_lrs: Sequence[float]
    carry: Any
    
    step: int
    total_steps: int
    epoch: int
    
    # Meta-learning specific
    meta_optimizer: Optional[torch.optim.Optimizer] = None
    best_metric: float = 0.0


# ============================================================================
# DATA LOADING
# ============================================================================

def create_dataloader(config: TrainingConfig, split: str, rank: int, world_size: int, **kwargs):
    """Create dataloader for training or evaluation"""
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    return dataloader, dataset.metadata


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(config: TrainingConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    """Create model with loss head"""
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False
    )
    
    # Instantiate model
    model_cls = load_model_class(config.arch.name)
    
    # Load loss head class - handle special case for meta_loss_head
    if config.arch.loss.name == "meta_loss_head@MetaLearningLossHead":
        from meta_loss_head import MetaLearningLossHead
        loss_head_cls = MetaLearningLossHead
    else:
        loss_head_cls = load_model_class(config.arch.loss.name)
    
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
        
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)
        
        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)
    
    # Create optimizers
    optimizers = []
    optimizer_lrs = []
    
    # Puzzle embedding optimizer
    if hasattr(model, 'model') and hasattr(model.model, 'puzzle_emb'):
        optimizers.append(
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),
                lr=config.puzzle_emb_lr,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        )
        optimizer_lrs.append(config.puzzle_emb_lr)
    
    # Main optimizer
    if AdamAtan2 is not None:
        main_optimizer = AdamAtan2(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    else:
        main_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    
    optimizers.append(main_optimizer)
    optimizer_lrs.append(config.lr)
    
    return model, optimizers, optimizer_lrs


# ============================================================================
# LEARNING RATE SCHEDULING
# ============================================================================

def cosine_schedule_with_warmup(
    current_step: int,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.1
):
    """Cosine learning rate schedule with warmup"""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def compute_lr(base_lr: float, config: TrainingConfig, train_state: TrainState):
    """Compute current learning rate"""
    return cosine_schedule_with_warmup(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


# ============================================================================
# STANDARD PRETRAINING
# ============================================================================

def train_batch_standard(
    config: TrainingConfig,
    train_state: TrainState,
    batch: Dict[str, torch.Tensor],
    global_batch_size: int,
    rank: int,
    world_size: int
) -> Optional[Dict[str, float]]:
    """Standard training step"""
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return None
    
    # Move to device
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Initialize carry
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)
    
    # Adaptive meta loss weighting (if using meta loss head)
    warmup_steps = 1000
    use_meta_loss = train_state.step > warmup_steps
    
    if use_meta_loss and hasattr(train_state.model, 'alpha_kl'):
        progress = min(1.0, (train_state.step - warmup_steps) / config.meta.kl_warmup_steps)
        kl_weight = config.meta.kl_weight_start + progress * (config.meta.kl_weight_end - config.meta.kl_weight_start)
        train_state.model.alpha_kl = kl_weight
    
    # Forward pass
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry,
        batch=batch,
        return_keys=[]
    )
    
    # Backward pass
    ((1 / global_batch_size) * loss).backward()
    
    # Gradient synchronization
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), config.grad_clip)
    
    # Optimizer step
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_base_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()
    
    # Metrics processing (only on rank 0)
    if rank == 0 and len(metrics) > 0:
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        
        if world_size > 1:
            dist.reduce(metric_values, dst=0)
        
        metric_values_np = metric_values.detach().cpu().numpy()

        reduced_metrics = {k: metric_values_np[i] for i, k in enumerate(metric_keys)}
        
        count = max(reduced_metrics.get("count", 1), 1)
        
        # Normalize metrics properly
        normalized_metrics = {}
        for k, v in reduced_metrics.items():
            if k == "count":
                continue
            # Losses should be normalized by global_batch_size
            # Other metrics (accuracy, etc.) by count
            if "loss" in k.lower():
                normalized_metrics[f"train/{k}"] = v / global_batch_size
            else:
                normalized_metrics[f"train/{k}"] = v / count
        
        # Add learning rate
        if lr_this_step is not None:
            normalized_metrics["train/lr"] = lr_this_step
        
        # Add KL weight if using meta loss
        if use_meta_loss and hasattr(train_state.model, 'alpha_kl'):
            normalized_metrics["train/kl_weight"] = train_state.model.alpha_kl
        
        # Print summary every log_interval steps
        if train_state.step % config.log_interval == 0:
            loss_str = f"{normalized_metrics.get('train/loss', normalized_metrics.get('train/meta/total_loss', 0)):.4f}"
            acc_str = f"{normalized_metrics.get('train/accuracy', 0)*100:.2f}%"
            print(f"[Step {train_state.step}] Loss: {loss_str} | Acc: {acc_str} | LR: {lr_this_step:.2e}")
        
        return normalized_metrics
    
    return None


# ============================================================================
# META-LEARNING (MAML-STYLE)
# ============================================================================

def inner_loop_adapt(
    model: nn.Module,
    loss_head: nn.Module,
    task: TaskBatch,
    inner_lr: float,
    inner_steps: int,
    fixed_batch_size: int
) -> Dict[str, List[float]]:
    """Inner loop adaptation on support set"""
    model.train()
    
    inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
    
    metrics = {'inner_loss': [], 'inner_accuracy': []}
    
    # Prepare support batch
    support_batch = {
        'inputs': task.support_inputs,
        'labels': task.support_labels,
        'puzzle_identifiers': task.puzzle_ids,
    }
    
    support_size = task.support_inputs.shape[0]
    if task.puzzle_ids.shape[0] == 1 and support_size > 1:
        support_batch['puzzle_identifiers'] = task.puzzle_ids.expand(support_size)
    
    support_batch, actual_support_size = pad_to_batch_size(support_batch, fixed_batch_size)
    
    # Inner loop training
    for step in range(inner_steps):
        inner_optimizer.zero_grad()
        
        carry = loss_head.initial_carry(support_batch)
        carry, loss, step_metrics, preds, _ = loss_head(carry, support_batch, return_keys=[])
        
        if actual_support_size < fixed_batch_size:
            loss = loss * (actual_support_size / fixed_batch_size)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        inner_optimizer.step()
        
        metrics['inner_loss'].append(loss.item())
        
        with torch.no_grad():
            pred_labels = preds['logits'][:actual_support_size].argmax(dim=-1)
            acc = (pred_labels == task.support_labels).float().mean().item()
            metrics['inner_accuracy'].append(acc)
    
    return metrics


def outer_loop_meta_update(
    config: TrainingConfig,
    train_state: TrainState,
    task_batch: List[TaskBatch],
    fixed_batch_size: int,
    rank: int,
    world_size: int
) -> Dict[str, float]:
    """MAML outer loop meta-update"""
    train_state.model.train()
    
    # Save original parameters
    original_params = {name: param.data.clone() 
                      for name, param in train_state.model.parameters()}
    
    all_metrics = {
        'query_loss': [],
        'query_accuracy': [],
        'inner_loss': [],
    }
    
    # Process each task
    for task_idx, task in enumerate(task_batch):
        # Zero gradients only on first task
        if task_idx == 0:
            train_state.meta_optimizer.zero_grad(set_to_none=True)
        
        # Restore original parameters
        with torch.no_grad():
            for name, param in train_state.model.parameters():
                param.data.copy_(original_params[name])
        
        # Inner loop adaptation
        inner_metrics = inner_loop_adapt(
            train_state.model,
            train_state.model,
            task,
            config.meta.inner_lr,
            config.meta.inner_steps,
            fixed_batch_size
        )
        
        # Query evaluation
        query_batch = {
            'inputs': task.query_inputs,
            'labels': task.query_labels,
            'puzzle_identifiers': task.puzzle_ids,
        }
        
        query_size = task.query_inputs.shape[0]
        if task.puzzle_ids.shape[0] == 1 and query_size > 1:
            query_batch['puzzle_identifiers'] = task.puzzle_ids.expand(query_size)
        
        query_batch, actual_query_size = pad_to_batch_size(query_batch, fixed_batch_size)
        
        carry = train_state.model.initial_carry(query_batch)
        carry, query_loss, query_metrics, preds, _ = train_state.model(
            carry, query_batch, return_keys=[]
        )
        
        if actual_query_size < fixed_batch_size:
            query_loss = query_loss * (actual_query_size / fixed_batch_size)
        
        # Backward for this task
        task_loss = query_loss / len(task_batch)
        task_loss.backward()
        
        # Track metrics
        all_metrics['query_loss'].append(query_loss.detach().item())
        all_metrics['inner_loss'].append(inner_metrics['inner_loss'][-1])
        
        with torch.no_grad():
            pred_labels = preds['logits'][:actual_query_size].argmax(dim=-1)
            acc = (pred_labels == task.query_labels).float().mean().item()
            all_metrics['query_accuracy'].append(acc)
    
    # Restore original parameters
    with torch.no_grad():
        for name, param in train_state.model.parameters():
            param.data.copy_(original_params[name])
    
    # Gradient synchronization
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
    
    # Meta-update
    torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), config.grad_clip)
    train_state.meta_optimizer.step()
    
    train_state.step += 1
    
    return {
        'meta_loss': sum(all_metrics['query_loss']) / len(all_metrics['query_loss']),
        'query_accuracy': sum(all_metrics['query_accuracy']) / len(all_metrics['query_accuracy']),
        'inner_loss': sum(all_metrics['inner_loss']) / len(all_metrics['inner_loss']),
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(
    config: TrainingConfig,
    train_state: TrainState,
    eval_loader: DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int
) -> Optional[Dict]:
    """Evaluate model on test set with safety limits"""
    train_state.model.eval()
    
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        all_preds = {}
        
        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
        batch_count = 0
        max_batches = 1000
        
        for set_name, batch, global_batch_size in eval_loader:
            batch_count += 1
            
            if batch_count > max_batches:
                if rank == 0:
                    print(f"\n⚠️  Reached max eval batches ({max_batches}), stopping early")
                break
            
            if rank == 0 and batch_count % 50 == 0:
                print(f"  Evaluating batch {batch_count}...", end='\r')
            
            batch = {k: v.cuda() for k, v in batch.items()}
            
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)
            
            max_forward_steps = 100
            step_count = 0
            
            while step_count < max_forward_steps:
                carry, _, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=config.eval_save_outputs
                )
                step_count += 1
                
                if all_finish:
                    break
                
                if step_count % 20 == 0 and rank == 0:
                    print(f"  Batch {batch_count}, step {step_count}/{max_forward_steps}...", end='\r')
            
            if rank == 0 and step_count >= max_forward_steps:
                print(f"\n⚠️  Warning: Batch {batch_count} reached max steps ({max_forward_steps})")
            
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())
            
            del carry, preds, batch, all_finish
            
            set_id = set_ids[set_name]
            
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), 
                                           dtype=torch.float32, device="cuda")
            
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size
        
        if rank == 0:
            print(f"\n✅ Evaluated {batch_count} batches")
        
        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, 
                      os.path.join(config.checkpoint_path, f"step_{train_state.step}_preds.{rank}"))
        
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id] 
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }
                
                # Normalize by count
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}
                
                return reduced_metrics
    
    return None


# ============================================================================
# CHECKPOINTING (UPDATED WITH AUTO-CLEANUP!)
# ============================================================================

def save_checkpoint(config: TrainingConfig, train_state: TrainState, rank: int, 
                   keep_last_n: int = 3):
    """
    Save model checkpoint with automatic cleanup of old checkpoints
    
    Args:
        keep_last_n: Number of recent checkpoints to keep (default: 3)
                     Set to -1 to keep all checkpoints
    """
    if rank != 0 or config.checkpoint_path is None:
        return
    
    os.makedirs(config.checkpoint_path, exist_ok=True)
    
    checkpoint = {
        'step': train_state.step,
        'epoch': train_state.epoch,
        'model_state_dict': train_state.model.state_dict(),
        'optimizer_state_dicts': [opt.state_dict() for opt in train_state.optimizers],
        'best_metric': train_state.best_metric,
        'config': config.model_dump(),  # Save config for reference
    }
    
    if train_state.meta_optimizer is not None:
        checkpoint['meta_optimizer_state_dict'] = train_state.meta_optimizer.state_dict()
    
    # Save step checkpoint
    step_path = os.path.join(config.checkpoint_path, f"checkpoint_step_{train_state.step}.pt")
    torch.save(checkpoint, step_path)
    
    # Save latest checkpoint
    latest_path = os.path.join(config.checkpoint_path, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)
    
    print(f"💾 Checkpoint saved: {step_path}")
    
    # Cleanup old checkpoints (keep last N)
    if keep_last_n > 0:
        checkpoint_files = sorted(
            glob.glob(os.path.join(config.checkpoint_path, "checkpoint_step_*.pt")),
            key=lambda x: int(x.split('_')[-1].replace('.pt', ''))
        )
        
        # Keep the latest N checkpoints
        if len(checkpoint_files) > keep_last_n:
            for old_checkpoint in checkpoint_files[:-keep_last_n]:
                try:
                    os.remove(old_checkpoint)
                    print(f"🗑️  Removed old checkpoint: {os.path.basename(old_checkpoint)}")
                except:
                    pass


def save_code_and_config(config: TrainingConfig, rank: int):
    """Save training code and config"""
    if rank != 0 or config.checkpoint_path is None:
        return
    
    os.makedirs(config.checkpoint_path, exist_ok=True)
    
    # Save config
    config_path = os.path.join(config.checkpoint_path, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config.model_dump(), f)
    
    # Copy model files
    code_files = []
    
    # Handle main model
    model_path = get_model_source_path(config.arch.name)
    if model_path:
        code_files.append(model_path)
    
    # Handle loss head - special case for meta_loss_head
    if config.arch.loss.name == "meta_loss_head@MetaLearningLossHead":
        loss_path = "meta_loss_head.py"
        if os.path.exists(loss_path):
            code_files.append(loss_path)
    else:
        loss_path = get_model_source_path(config.arch.loss.name)
        if loss_path:
            code_files.append(loss_path)
    
    # Copy files
    for code_file in code_files:
        if code_file is not None and os.path.exists(code_file):
            dst = os.path.join(config.checkpoint_path, os.path.basename(code_file))
            shutil.copy(code_file, dst)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def init_train_state(
    config: TrainingConfig, 
    train_metadata: PuzzleDatasetMetadata, 
    world_size: int,
    rank: int
) -> TrainState:
    """Initialize training state with optional checkpoint loading (UPDATED!)"""
    
    # Estimate total steps
    total_steps = int(
        config.epochs * train_metadata.total_groups * 
        train_metadata.mean_puzzle_examples / config.global_batch_size
    )
    
    # Create model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size)
    
    # Create meta-optimizer if needed
    meta_optimizer = None
    if config.mode == "meta":
        meta_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    
    # Initialize state
    train_state = TrainState(
        step=0,
        total_steps=total_steps,
        epoch=0,
        model=model,
        optimizers=optimizers,
        optimizer_base_lrs=optimizer_lrs,
        carry=None,
        meta_optimizer=meta_optimizer,
        best_metric=0.0
    )
    
    # Resume from checkpoint if requested (NEW!)
    resume_path = None
    
    if config.resume_from:
        resume_path = config.resume_from
    elif config.resume_auto and config.checkpoint_path:
        # Auto-resume from checkpoint_path
        resume_path = config.checkpoint_path
    
    if resume_path:
        step, epoch, best_metric, meta_opt_state = load_checkpoint(
            resume_path,
            model,
            optimizers,
            config,
            rank
        )
        
        # Update train state
        train_state.step = step
        train_state.epoch = epoch
        train_state.best_metric = best_metric
        
        # Restore meta optimizer if available
        if meta_opt_state and train_state.meta_optimizer:
            train_state.meta_optimizer.load_state_dict(meta_opt_state)
        
        # Adjust total steps if extending training
        if config.resume_epoch_offset > 0:
            additional_steps = int(
                config.resume_epoch_offset * train_metadata.total_groups * 
                train_metadata.mean_puzzle_examples / config.global_batch_size
            )
            train_state.total_steps += additional_steps
            
            if rank == 0:
                print(f"📈 Extended training: +{additional_steps} steps (total: {train_state.total_steps})")
    
    return train_state


def train_epoch_standard(
    config: TrainingConfig,
    train_state: TrainState,
    train_loader: DataLoader,
    progress_bar: Optional[tqdm.tqdm],
    rank: int,
    world_size: int
):
    """Standard training epoch"""
    train_state.model.train()
    
    for set_name, batch, global_batch_size in train_loader:
        metrics = train_batch_standard(
            config, train_state, batch, global_batch_size, rank, world_size
        )
        
        if rank == 0 and metrics is not None:
            if config.wandb_enabled and wandb.run is not None:
                wandb.log(metrics, step=train_state.step)
            
            if progress_bar is not None:
                progress_bar.update(train_state.step - progress_bar.n)
                progress_bar.set_postfix({
                    'loss': f"{metrics.get('train/loss', 0):.4f}",
                    'lr': f"{metrics.get('train/lr', 0):.2e}"
                })


def train_epoch_meta(
    config: TrainingConfig,
    train_state: TrainState,
    train_loader: DataLoader,
    rank: int,
    world_size: int
):
    """Meta-learning training epoch"""
    train_state.model.train()
    
    fixed_batch_size = config.global_batch_size // world_size
    epoch_metrics = []
    
    # Collect batches and create tasks
    batches_collected = 0
    task_batch = []
    
    for set_name, batch, global_batch_size in train_loader:
        try:
            task = create_task_batch_from_puzzle_batch(
                batch,
                n_support=config.meta.n_support,
                n_query=config.meta.n_query
            )
            task_batch.append(task)
            batches_collected += 1
            
            # When we have enough tasks, do meta-update
            if len(task_batch) >= config.meta.task_batch_size:
                metrics = outer_loop_meta_update(
                    config, train_state, task_batch, 
                    fixed_batch_size, rank, world_size
                )
                epoch_metrics.append(metrics)
                task_batch = []
                
                if rank == 0 and config.wandb_enabled and wandb.run is not None:
                    wandb.log({
                        'meta_loss': metrics['meta_loss'],
                        'query_accuracy': metrics['query_accuracy'],
                        'inner_loss': metrics['inner_loss'],
                    }, step=train_state.step)
                
                if rank == 0 and train_state.step % config.log_interval == 0:
                    print(f"\n[Step {train_state.step}] Meta Loss: {metrics['meta_loss']:.4f} | "
                          f"Query Acc: {metrics['query_accuracy']*100:.2f}%")
            
            if batches_collected >= config.meta.meta_batches_per_epoch:
                break
                
        except Exception as e:
            print(f"⚠️  Error creating task from batch: {e}")
            continue
    
    # Process remaining tasks
    if len(task_batch) > 0:
        metrics = outer_loop_meta_update(
            config, train_state, task_batch, 
            fixed_batch_size, rank, world_size
        )
        epoch_metrics.append(metrics)
    
    # Epoch summary
    if rank == 0 and len(epoch_metrics) > 0:
        avg_meta_loss = sum(m['meta_loss'] for m in epoch_metrics) / len(epoch_metrics)
        avg_query_acc = sum(m['query_accuracy'] for m in epoch_metrics) / len(epoch_metrics)
        
        print(f"\n{'='*60}")
        print(f"EPOCH {train_state.epoch} SUMMARY")
        print(f"{'='*60}")
        print(f"  Avg Meta Loss: {avg_meta_loss:.4f}")
        print(f"  Avg Query Accuracy: {avg_query_acc*100:.2f}%")
        
        return avg_query_acc
    
    return 0.0


def main_training_loop(
    config: TrainingConfig,
    train_state: TrainState,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int
):
    """Main training loop"""
    
    # Progress bar (rank 0 only)
    progress_bar = None
    if rank == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, desc="Training")
        # Update progress bar to current step if resuming
        if train_state.step > 0:
            progress_bar.update(train_state.step)
    
    # Training epochs
    eval_interval = config.eval_interval if config.eval_interval else config.epochs
    total_iters = config.epochs // eval_interval
    
    for iter_id in range(total_iters):
        train_state.epoch = iter_id * eval_interval
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"EPOCH {train_state.epoch}/{config.epochs}")
            print(f"{'='*60}")
        
        # Training
        if config.mode == "pretrain":
            train_epoch_standard(
                config, train_state, train_loader, 
                progress_bar, rank, world_size
            )
        else:  # meta
            train_epoch_meta(
                config, train_state, train_loader,
                rank, world_size
            )
        
        # Evaluation
        train_state.model.eval()
        eval_metrics = evaluate(
            config, train_state, eval_loader, 
            eval_metadata, rank, world_size
        )
        
        if rank == 0 and eval_metrics is not None:
            print(f"\n{'='*60}")
            print("EVALUATION RESULTS")
            print(f"{'='*60}")
            
            for set_name, metrics in eval_metrics.items():
                print(f"\n{set_name}:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")
            
            if config.wandb_enabled and wandb.run is not None:
                wandb.log({
                    f"eval/{set_name}/{k}": v 
                    for set_name, metrics in eval_metrics.items()
                    for k, v in metrics.items()
                }, step=train_state.step)
            
            # Track best metric
            # Try to find accuracy in any set, fallback to other metrics
            main_metric = 0.0
            for set_metrics in eval_metrics.values():
                if 'accuracy' in set_metrics:
                    main_metric = max(main_metric, set_metrics['accuracy'])
                elif 'loss' in set_metrics:
                    # Use negative loss as metric if accuracy not available
                    main_metric = max(main_metric, -set_metrics['loss'])
            
            if main_metric > train_state.best_metric:
                train_state.best_metric = main_metric
                if rank == 0:
                    print(f"\n🎯 New best metric: {main_metric:.4f}")
        
        # Checkpointing
        if rank == 0 and (config.checkpoint_every_eval or iter_id == total_iters - 1):
            save_checkpoint(config, train_state, rank)
    
    if progress_bar is not None:
        progress_bar.close()


# ============================================================================
# HYDRA LAUNCHER
# ============================================================================

def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> TrainingConfig:
    """Load and synchronize config across ranks"""
    objects = [None]
    
    if rank == 0:
        config = TrainingConfig(**hydra_config)
        
        # Auto-naming
        if config.project_name is None:
            dataset_name = os.path.basename(config.data_path).capitalize()
            config.project_name = f"{dataset_name} HRM-Free-Meta"
        
        if config.run_name is None:
            mode_str = config.mode.capitalize()
            model_str = config.arch.name.split('@')[-1]
            config.run_name = f"{mode_str}-{model_str}-{coolname.generate_slug(2)}"
        
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(
                "checkpoints",
                config.project_name,
                config.run_name
            )
        
        objects = [config]
    
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    
    return objects[0]


@hydra.main(config_path="config", config_name="unified_training", version_base=None)
def launch(hydra_config: DictConfig):
    """Main entry point with Hydra config"""
    
    # Distributed setup
    RANK = 0
    WORLD_SIZE = 1
    
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    # Load config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    
    # Print config
    if RANK == 0:
        print("\n" + "="*80)
        print("🚀 HRM-FREE-META UNIFIED TRAINING")
        print("="*80)
        print(f"Mode: {config.mode.upper()}")
        print(f"Project: {config.project_name}")
        print(f"Run: {config.run_name}")
        print(f"Device: cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"World Size: {WORLD_SIZE}")
        
        # Show resume info if applicable
        if config.resume_from or config.resume_auto:
            print(f"\n📥 RESUME MODE ENABLED")
            if config.resume_from:
                print(f"   Resume from: {config.resume_from}")
            if config.resume_auto:
                print(f"   Auto-resume: True")
            if config.reset_optimizer:
                print(f"   Reset optimizer: True")
            if config.resume_epoch_offset > 0:
                print(f"   Extend epochs: +{config.resume_epoch_offset}")
        
        print("="*80 + "\n")
    
    # Seed
    torch.manual_seed(config.seed + RANK)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + RANK)
    
    # Create dataloaders
    train_epochs_per_iter = config.eval_interval if config.eval_interval else config.epochs
    assert config.epochs % train_epochs_per_iter == 0, \
        "eval_interval must divide epochs evenly"
    
    if RANK == 0:
        print("📦 Loading datasets...")
    
    train_loader, train_metadata = create_dataloader(
        config, "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE
    )
    
    eval_loader, eval_metadata = create_dataloader(
        config, "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE
    )
    
    if RANK == 0:
        print(f"✅ Train: {train_metadata.total_groups} groups")
        print(f"✅ Eval: {len(eval_metadata.sets)} sets\n")
    
    # Initialize training state (NOW WITH RESUME SUPPORT!)
    if RANK == 0:
        print("🏗️  Creating model...")
    
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE, rank=RANK)
    
    if RANK == 0:
        num_params = sum(p.numel() for p in train_state.model.parameters())
        print(f"✅ Model created: {num_params/1e6:.2f}M parameters")
        print(f"✅ Total training steps: {train_state.total_steps:,}")
        if train_state.step > 0:
            print(f"✅ Resumed from step: {train_state.step}")
        print()
    
    # Initialize W&B
    if RANK == 0 and config.wandb_enabled:
        # Resume W&B run if resuming training
        wandb_id = None
        if config.resume_from or config.resume_auto:
            # Try to load W&B run ID from checkpoint directory
            wandb_id_file = os.path.join(config.checkpoint_path, "wandb_run_id.txt")
            if os.path.exists(wandb_id_file):
                with open(wandb_id_file, 'r') as f:
                    wandb_id = f.read().strip()
                print(f"📊 Resuming W&B run: {wandb_id}")
        
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
            id=wandb_id,
            resume="allow" if wandb_id else None
        )
        
        # Save W&B run ID for future resumes
        if wandb.run and config.checkpoint_path:
            os.makedirs(config.checkpoint_path, exist_ok=True)
            with open(os.path.join(config.checkpoint_path, "wandb_run_id.txt"), 'w') as f:
                f.write(wandb.run.id)
        
        wandb.log({
            "num_params": sum(p.numel() for p in train_state.model.parameters())
        }, step=train_state.step)
    
    # Save code and config
    if RANK == 0:
        save_code_and_config(config, RANK)
    
    # Main training loop
    try:
        main_training_loop(
            config, train_state, train_loader, eval_loader,
            eval_metadata, RANK, WORLD_SIZE
        )
        
        if RANK == 0:
            print("\n" + "="*80)
            print("✅ TRAINING COMPLETE!")
            print("="*80)
            print(f"Best metric: {train_state.best_metric:.4f}")
            print(f"Checkpoint: {config.checkpoint_path}")
            print("="*80 + "\n")
    
    except KeyboardInterrupt:
        if RANK == 0:
            print("\n\n⚠️  Training interrupted by user")
            save_checkpoint(config, train_state, RANK)
    
    except Exception as e:
        if RANK == 0:
            print(f"\n\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
        
        if RANK == 0 and config.wandb_enabled:
            wandb.finish()


# ============================================================================
# CLI ENTRY POINT (WITHOUT HYDRA) - UPDATED WITH RESUME ARGS!
# ============================================================================

def launch_simple(
    mode: str = "pretrain",
    data_path: str = "data/sudoku-extreme-1k-aug-1000",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    **kwargs
):
    """
    Simple entry point without Hydra
    
    New resume arguments:
        resume_from: Path to checkpoint file or directory
        resume_auto: Auto-resume from checkpoint_path
        reset_optimizer: Reset optimizer when resuming
        extend_epochs: Add more epochs to existing training
    
    Usage:
        python unified_training.py --mode pretrain --epochs 10
        python unified_training.py --resume_from checkpoints/my_run/checkpoint_latest.pt
    """
    
    # Create minimal config
    config_dict = {
        'mode': mode,
        'data_path': data_path,
        'global_batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'lr_warmup_steps': 1000,
        'lr_min_ratio': 0.1,
        'weight_decay': 0.1,
        'beta1': 0.9,
        'beta2': 0.999,
        'puzzle_emb_lr': 1e-4,
        'puzzle_emb_weight_decay': 0.0,
        
        # Resume options (NEW!)
        'resume_from': kwargs.get('resume_from', None),
        'resume_auto': kwargs.get('resume_auto', False),
        'reset_optimizer': kwargs.get('reset_optimizer', False),
        'reset_lr_schedule': kwargs.get('reset_lr_schedule', False),
        'resume_epoch_offset': kwargs.get('extend_epochs', 0),
        
        # Eval interval - None means eval only at end
        'eval_interval': kwargs.get('eval_interval', None),
        'checkpoint_every_eval': kwargs.get('checkpoint_every_eval', True),
        'eval_save_outputs': [],
        
        'arch': {
            'name': 'hrm.hrm_free_meta@HRMFreeMeta',
            'loss': {
                'name': 'meta_loss_head@MetaLearningLossHead',
                'loss_type': 'stablemax_cross_entropy',
                'alpha_kl': 0.001,
                'beta_entropy': 0.01,
            },
            # HRM core parameters
            'halt_exploration_prob': 0.1,
            'halt_max_steps': 16,
            'H_cycles': 2,
            'L_cycles': 2,
            'H_layers': 4,
            'L_layers': 4,
            'hidden_size': 512,
            'num_heads': 8,
            'expansion': 4,
            'puzzle_emb_ndim': 512,
            'pos_encodings': 'rope',
            'rope_theta': 10000.0,
            'rms_norm_eps': 1e-5,
            # Meta-learning parameters
            'z_dim': 128,
            'encoder_layers': 2,
            'controller_hidden': 256,
            'gate_temperature': 1.0,
            'use_dynamic_weighting': True,
            'forward_dtype': 'bfloat16',
        },
        'meta': {
            'enabled': mode == 'meta',
            'n_support': 5,
            'n_query': 10,
            'inner_steps': 5,
            'inner_lr': 1e-3,
            'task_batch_size': 4,
            'meta_batches_per_epoch': 100,
            'kl_weight_start': 0.0001,
            'kl_weight_end': 0.001,
            'kl_warmup_steps': 5000,
        },
        'seed': 0,
        'grad_clip': 1.0,
        'log_interval': 10,
        'wandb_enabled': kwargs.get('wandb_enabled', True),
        
        # Optional overrides from kwargs
        'project_name': kwargs.get('project_name', None),
        'run_name': kwargs.get('run_name', None),
        'checkpoint_path': kwargs.get('checkpoint_path', None),
    }
    
    config_dict.update(kwargs)
    
    # Convert to DictConfig
    omega_config = OmegaConf.create(config_dict)
    
    # Launch
    launch(omega_config)


if __name__ == "__main__":
    import argparse
    import sys
    
    # Check if using Hydra syntax (key=value) or CLI syntax (--key value)
    if len(sys.argv) > 1 and '=' in sys.argv[1]:
        # Hydra mode
        print("🔧 Using Hydra config mode")
        launch()
    else:
        # CLI mode
        print("🔧 Using CLI argument mode")
        
        parser = argparse.ArgumentParser(description="HRM-Free-Meta Unified Training")
        
        # Basic training args
        parser.add_argument("--mode", type=str, default="pretrain", 
                           choices=["pretrain", "meta"],
                           help="Training mode")
        parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug-1000",
                           help="Path to dataset")
        parser.add_argument("--epochs", type=int, default=10,
                           help="Number of epochs")
        parser.add_argument("--batch_size", type=int, default=32,
                           help="Global batch size")
        parser.add_argument("--lr", type=float, default=1e-4,
                           help="Learning rate")
        parser.add_argument("--eval_interval", type=int, default=None,
                           help="Evaluation interval (epochs). None = eval only at end")
        
        # Resume arguments (NEW!)
        resume_group = parser.add_argument_group('Resume Training')
        resume_group.add_argument("--resume_from", type=str, default=None,
                                 help="Path to checkpoint file or directory to resume from")
        resume_group.add_argument("--resume_auto", action="store_true",
                                 help="Auto-resume from latest checkpoint in checkpoint_path")
        resume_group.add_argument("--reset_optimizer", action="store_true",
                                 help="Reset optimizer state when resuming (keeps model weights)")
        resume_group.add_argument("--extend_epochs", type=int, default=0,
                                 help="Add more epochs to extend training beyond original target")
        resume_group.add_argument("--checkpoint_path", type=str, default=None,
                                 help="Directory to save/load checkpoints")
        
        # Other args
        parser.add_argument("--no_eval", action="store_true",
                           help="Skip evaluation completely (fast training)")
        parser.add_argument("--wandb", action="store_true", default=True,
                           help="Enable W&B logging")
        parser.add_argument("--no-wandb", action="store_false", dest="wandb",
                           help="Disable W&B logging")
        
        args = parser.parse_args()
        
        # Handle eval
        eval_interval = None if not args.no_eval else args.eval_interval
        if args.no_eval:
            eval_interval = 999999
        
        # Launch with CLI args
        launch_simple(
            mode=args.mode,
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_interval=eval_interval,
            wandb_enabled=args.wandb,
            # Resume args (NEW!)
            resume_from=args.resume_from,
            resume_auto=args.resume_auto,
            reset_optimizer=args.reset_optimizer,
            extend_epochs=args.extend_epochs,
            checkpoint_path=args.checkpoint_path,
        )