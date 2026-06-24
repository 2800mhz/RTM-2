# Resume quickstart

The active resume flow in this repo is implemented in `unified_training.py`.

## Resume from a concrete checkpoint

```bash
python unified_training.py --resume_from checkpoints/my_run/checkpoint_latest.pt
```

## Auto-resume from a checkpoint directory

```bash
python unified_training.py --resume_auto --checkpoint_path checkpoints/my_run
```

## Resume but reset optimizer state

Useful when weights are good but the optimizer state is unstable:

```bash
python unified_training.py \
  --resume_from checkpoints/my_run/checkpoint_latest.pt \
  --reset_optimizer
```

## Extend a finished run

```bash
python unified_training.py \
  --resume_from checkpoints/my_run/checkpoint_latest.pt \
  --extend_epochs 10
```

## What gets restored

- model weights
- optimizer state, unless `--reset_optimizer` is used
- best metric
- training step / epoch counters
- W&B run id when present beside the checkpoint

## Files expected in a run directory

Typical directory contents:

```text
checkpoints/my_run/
  checkpoint_latest.pt
  checkpoint_step_XXXX.pt
  config.yaml
  wandb_run_id.txt
```
