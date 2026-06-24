#!/usr/bin/env python3
"""
Evaluate a checkpoint produced by unified_training.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from unified_training import (
    TrainingConfig,
    create_dataloader,
    evaluate as run_evaluation,
    init_train_state,
)


def _setup_distributed() -> tuple[int, int]:
    rank = 0
    world_size = 1
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return rank, world_size


def _load_training_config(checkpoint_dir: Path) -> TrainingConfig:
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file next to checkpoint: {config_path}")

    config_data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(config_data, dict)
    return TrainingConfig(**config_data)


@hydra.main(version_base=None, config_path=None)
def launch(cfg: DictConfig) -> None:
    checkpoint = Path(cfg.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    rank, world_size = _setup_distributed()
    checkpoint_dir = checkpoint.parent
    train_cfg = _load_training_config(checkpoint_dir)

    if "save_outputs" in cfg and cfg.save_outputs is not None:
        train_cfg.eval_save_outputs = list(cfg.save_outputs)
    if "data_path" in cfg and cfg.data_path:
        train_cfg.data_path = cfg.data_path
    train_cfg.checkpoint_path = str(checkpoint_dir)
    train_cfg.wandb_enabled = False

    train_loader, train_metadata = create_dataloader(
        train_cfg,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=train_cfg.global_batch_size,
        rank=rank,
        world_size=world_size,
    )
    eval_loader, eval_metadata = create_dataloader(
        train_cfg,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=train_cfg.global_batch_size,
        rank=rank,
        world_size=world_size,
    )

    train_state = init_train_state(train_cfg, train_metadata, world_size=world_size, rank=rank)
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    train_state.model.load_state_dict(state_dict, strict=False)
    train_state.step = int(payload.get("step", 0)) if isinstance(payload, dict) else 0

    metrics = run_evaluation(
        train_cfg,
        train_state,
        eval_loader,
        eval_metadata,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0 and metrics is not None:
        print("\nEvaluation results")
        print("=" * 80)
        for set_name, set_metrics in metrics.items():
            print(f"\n[{set_name}]")
            for key, value in set_metrics.items():
                print(f"{key}: {value:.6f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    launch()
