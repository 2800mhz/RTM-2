# HRM / HRM-Free-Meta

This repository now uses a single main training pipeline:

- `python unified_training.py ...`

The original upstream HRM code is still present, but the maintained local
workflow in this workspace is centered on:

- `unified_training.py` for training and resume
- `evaluate.py` for checkpoint evaluation
- `test.py` for a quick smoke test pass

## Install

Create an environment with PyTorch and then install:

```bash
pip install -r requirements.txt
```

If you want W&B logging:

```bash
wandb login
```

## Dataset layout

`unified_training.py` expects `data_path` to point to the dataset root, not the
`train` subfolder. A valid dataset looks like:

```text
data/sudoku-extreme-1k-aug-1000/
  train/
  test/
```

Each split must contain `dataset.json` and the generated `.npy` files.

Raw ARC data currently included in the repo lives under:

- `dataset/raw-data/ARC-AGI/`

Use the dataset builder scripts to generate processed datasets:

```bash
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
python dataset/build_maze_dataset.py
python dataset/build_arc_dataset.py
```

## Training

Default config:

```bash
python unified_training.py mode=pretrain
```

Example HRM-Free-Meta pretraining run:

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

Example original HRM run:

```bash
python unified_training.py mode=pretrain \
  arch.name=hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1 \
  arch.loss.name=losses@ACTLossHead
```

Example distributed run:

```bash
torchrun --nproc_per_node=4 unified_training.py mode=pretrain
```

## Resume training

Resume from an explicit checkpoint:

```bash
python unified_training.py --resume_from checkpoints/my_run/checkpoint_latest.pt
```

Auto-resume from a run directory:

```bash
python unified_training.py --resume_auto --checkpoint_path checkpoints/my_run
```

See also [resume_quickstart.md](C:/Users/emreg/Desktop/RTM-2/HRM/resume_quickstart.md).

## Evaluate a checkpoint

```bash
python evaluate.py checkpoint=checkpoints/my_run/checkpoint_latest.pt
```

Optional overrides:

```bash
python evaluate.py checkpoint=checkpoints/my_run/checkpoint_latest.pt save_outputs=[logits,gates]
```

## Smoke tests

Quick validation:

```bash
python test.py
```

Direct single smoke script:

```bash
python test_hrm_free_meta.py
```

## Notes

- This workspace contains experimental and debugging scripts from earlier
  iterations. The files above are the maintained entrypoints.
- Checkpoints saved by `unified_training.py` store `config.yaml` beside the
  checkpoint for later resume and evaluation.
