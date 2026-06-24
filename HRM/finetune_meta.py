#!/usr/bin/env python3
"""
Fine-tune only the meta layers starting from an existing checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch

from meta_loss_head import MetaLearningLossHead
from models.hrm.hrm_free_meta import HRMFreeMeta
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

try:
    from adam_atan2_pytorch import AdamAtan2
except ImportError:  # pragma: no cover - optional dependency
    AdamAtan2 = None


def freeze_base_hrm(model: HRMFreeMeta) -> list[torch.nn.Parameter]:
    for name, param in model.named_parameters():
        if "hrm_inner" in name:
            param.requires_grad = False

    return [param for param in model.parameters() if param.requires_grad]


def load_pretrained_hrm(checkpoint_path: str, config: dict) -> HRMFreeMeta:
    model = HRMFreeMeta(config)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model


def create_train_loader(data_path: str, batch_size: int):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            dataset_path=data_path,
            seed=42,
            global_batch_size=batch_size,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="train",
    )
    return torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=1)


def finetune_meta_layers(
    checkpoint_path: str,
    data_path: str,
    epochs: int = 1,
    lr: float = 1e-4,
    output_dir: str = "checkpoints/hrm_free_meta_finetuned",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "batch_size": 32,
        "seq_len": 81,
        "vocab_size": 10,
        "num_puzzle_identifiers": 1000,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4,
        "H_layers": 4,
        "L_layers": 4,
        "H_cycles": 2,
        "L_cycles": 2,
        "halt_max_steps": 16,
        "z_dim": 128,
        "encoder_layers": 2,
        "controller_hidden": 256,
        "gate_temperature": 1.0,
        "use_dynamic_weighting": True,
        "causal": False,
        "pos_encodings": "rope",
        "puzzle_emb_ndim": 512,
        "forward_dtype": "bfloat16",
    }

    model = load_pretrained_hrm(checkpoint_path, config).to(device)
    trainable_params = freeze_base_hrm(model)
    loss_head = MetaLearningLossHead(model=model, alpha_kl=0.001, beta_entropy=0.01).to(device)

    optimizer_cls = AdamAtan2 if AdamAtan2 is not None else torch.optim.AdamW
    optimizer = optimizer_cls(trainable_params, lr=lr, weight_decay=0.1)

    dataloader = create_train_loader(data_path, batch_size=config["batch_size"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(epochs):
        for batch_index, (_set_name, batch, _global_batch_size) in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = loss_head.initial_carry(batch)

            optimizer.zero_grad()
            _, loss, metrics, _, _ = loss_head(carry, batch, return_keys=[])
            loss.backward()
            optimizer.step()

            if batch_index % 50 == 0:
                print(
                    f"epoch={epoch} batch={batch_index} "
                    f"loss={loss.item():.4f} "
                    f"kl={metrics['meta/loss_kl'].item():.4f}"
                )

        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch},
            output_path / f"checkpoint_epoch_{epoch}.pt",
        )


if __name__ == "__main__":
    checkpoint = os.environ.get("HRM_CHECKPOINT")
    data_path = os.environ.get("HRM_DATA_PATH")
    if not checkpoint or not data_path:
        raise SystemExit("Set HRM_CHECKPOINT and HRM_DATA_PATH to run fine-tuning.")

    finetune_meta_layers(checkpoint_path=checkpoint, data_path=data_path)
