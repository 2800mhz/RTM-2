# HRM / HRM-Free-Meta

This directory contains the active experimental code for RTM-2.

At its core, this project starts from HRM (Hierarchical Reasoning Model) and
extends it with a more adaptive branch we call **HRM-Free-Meta**.

The local research direction is to move beyond a model that only "computes
harder", toward one that can also infer **how it should think** about a task.

That idea shows up here as:

- latent task-context encoding,
- dynamic weighting between hierarchical reasoning modules,
- meta-learning style losses and diagnostics,
- and a more coherent training / evaluation / resume workflow.

## What this repo is trying to do

A rough intuition:

- standard reasoning systems often apply one fixed computation pattern,
- but different tasks may benefit from different mixtures of abstract planning
  and detailed stepwise processing,
- so we want the model to infer a task-dependent reasoning mode rather than use
  the same internal balance every time.

This repo is an attempt in that direction.

## Current repository status

This is still a research workspace.

It is now in a much cleaner state than before, but you should still think of it
as:

- experimental,
- partially validated,
- and optimized for iteration rather than publication-grade packaging.

The maintained workflow is centered on:

- `unified_training.py`
- `evaluate.py`
- `test.py`

Older debug scripts are still present because they were useful during
stabilization work.

## Repository layout

Key files:

- [unified_training.py](C:/Users/emreg/Desktop/RTM-2/HRM/unified_training.py) — main training loop, checkpointing, resume flow
- [evaluate.py](C:/Users/emreg/Desktop/RTM-2/HRM/evaluate.py) — evaluate checkpoints produced by `unified_training.py`
- [test.py](C:/Users/emreg/Desktop/RTM-2/HRM/test.py) — curated smoke-test runner
- [meta_loss_head.py](C:/Users/emreg/Desktop/RTM-2/HRM/meta_loss_head.py) — meta-aware loss head
- [models/hrm/hrm_free_meta.py](C:/Users/emreg/Desktop/RTM-2/HRM/models/hrm/hrm_free_meta.py) — main HRM-Free-Meta model wrapper
- [models/hrm/latent_context_encoder.py](C:/Users/emreg/Desktop/RTM-2/HRM/models/hrm/latent_context_encoder.py) — latent task-context encoder
- [models/hrm/meta_controller.py](C:/Users/emreg/Desktop/RTM-2/HRM/models/hrm/meta_controller.py) — module-weight controller
- [config/unified_training.yaml](C:/Users/emreg/Desktop/RTM-2/HRM/config/unified_training.yaml) — default config

Supporting directories:

- `dataset/` — dataset builders and raw inputs
- `models/` — base model code and HRM variants
- `utils/` — loading / utility helpers
- `wandb/` — local W&B run artifacts already present in the repo snapshot

## Environment setup

Create a Python environment with PyTorch first, then install the rest:

```bash
pip install -r requirements.txt
```

If you want experiment tracking:

```bash
wandb login
```

Notes:

- Some runs were originally developed in a Linux / WSL environment.
- Current files are cleaned for the local workspace, but you still need a valid
  local runtime with PyTorch available.
- CUDA is strongly recommended for meaningful experiments.

## Dataset layout

`unified_training.py` expects `data_path` to point to the **dataset root**, not
to the `train` subfolder.

Expected structure:

```text
data/sudoku-extreme-1k-aug-1000/
  train/
    dataset.json
    *.npy
  test/
    dataset.json
    *.npy
```

Included raw asset:

- `dataset/raw-data/ARC-AGI/arc_data.tar.gz`

Dataset builders:

```bash
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
python dataset/build_maze_dataset.py
python dataset/build_arc_dataset.py
```

## Training

### Default run

```bash
python unified_training.py mode=pretrain
```

### HRM-Free-Meta example

```bash
python unified_training.py mode=pretrain \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  arch.name=hrm.hrm_free_meta@HRMFreeMeta \
  arch.loss.name=meta_loss_head@MetaLearningLossHead \
  global_batch_size=32 \
  epochs=10 \
  lr=1e-4 \
  wandb_enabled=false
```

### Original HRM-style loss head example

```bash
python unified_training.py mode=pretrain \
  arch.name=hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1 \
  arch.loss.name=losses@ACTLossHead
```

### Distributed training

```bash
torchrun --nproc_per_node=4 unified_training.py mode=pretrain
```

## Resume training

Resume from a concrete checkpoint:

```bash
python unified_training.py --resume_from checkpoints/my_run/checkpoint_latest.pt
```

Auto-resume from a run directory:

```bash
python unified_training.py --resume_auto --checkpoint_path checkpoints/my_run
```

Reset optimizer state while keeping weights:

```bash
python unified_training.py \
  --resume_from checkpoints/my_run/checkpoint_latest.pt \
  --reset_optimizer
```

More examples:

- [resume_quickstart.md](C:/Users/emreg/Desktop/RTM-2/HRM/resume_quickstart.md)

## Evaluate a checkpoint

Basic evaluation:

```bash
python evaluate.py checkpoint=checkpoints/my_run/checkpoint_latest.pt
```

Save extra tensors during evaluation:

```bash
python evaluate.py checkpoint=checkpoints/my_run/checkpoint_latest.pt save_outputs=[logits,gates,z_context]
```

Override dataset path if needed:

```bash
python evaluate.py checkpoint=checkpoints/my_run/checkpoint_latest.pt data_path=data/sudoku-extreme-1k-aug-1000
```

## Smoke tests

Run the curated smoke test runner:

```bash
python test.py
```

Run the single direct smoke test:

```bash
python test_hrm_free_meta.py
```

## What changed in this cleaned-up version

Compared with the earlier messy workspace state, the current repo now has:

- a single active training entrypoint,
- a checkpoint evaluation path aligned with the same training pipeline,
- duplicate model definitions removed from `hrm_free_meta.py`,
- config defaults aligned with the active codepath,
- resume docs that match actual behavior,
- and simpler top-level documentation.

## Known limitations

To keep expectations honest:

- not every older debug script has been fully modernized,
- some exploratory tests are still rough,
- runtime behavior has only been partially verified from this workspace because
  full execution depends on local PyTorch / CUDA availability,
- this is still best viewed as an experimental reasoning research branch.

## Project intent in one line

> Not just solving tasks — learning how to approach them.
