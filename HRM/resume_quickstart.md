# Resume quickstart

This repo’s active resume flow lives in `unified_training.py`.

The checkpoint format, config saving, and W&B resume handling are all designed
around that script.

## When to use resume

Resume is useful when you want to:

- continue an interrupted run,
- extend a finished run for more epochs,
- keep learned weights but restart optimizer dynamics,
- recover evaluation or prediction export from an existing checkpoint.

## Resume from a specific checkpoint

```bash
python unified_training.py --resume_from checkpoints/my_run/checkpoint_latest.pt
```

You can also point to a step checkpoint directly:

```bash
python unified_training.py --resume_from checkpoints/my_run/checkpoint_step_5000.pt
```

## Auto-resume from a run directory

```bash
python unified_training.py --resume_auto --checkpoint_path checkpoints/my_run
```

This checks the directory for:

- `checkpoint_latest.pt`, or
- the highest available `checkpoint_step_*.pt`

## Resume but reset optimizer state

This is useful when the weights are worth keeping but the optimizer state may be
hurting stability:

```bash
python unified_training.py \
  --resume_from checkpoints/my_run/checkpoint_latest.pt \
  --reset_optimizer
```

## Extend training

If the original run ended cleanly but you want to keep going:

```bash
python unified_training.py \
  --resume_from checkpoints/my_run/checkpoint_latest.pt \
  --extend_epochs 10
```

## What gets restored

By default, resume restores:

- model weights,
- optimizer state,
- best metric,
- step / epoch progress,
- meta-optimizer state when applicable.

If a `wandb_run_id.txt` file exists beside the checkpoint, the code also tries
to reconnect the run for W&B logging continuity.

## What gets saved in a run directory

Typical checkpoint directory:

```text
checkpoints/my_run/
  checkpoint_latest.pt
  checkpoint_step_XXXX.pt
  config.yaml
  wandb_run_id.txt
```

## Practical advice

- Use `--reset_optimizer` if a resumed run becomes unstable immediately.
- Keep `config.yaml` beside checkpoints; evaluation relies on it.
- If you move checkpoints to a new machine, confirm the dataset path still makes
  sense in the restored config.
- If you only want evaluation, you can skip resume entirely and call
  `evaluate.py` on the saved checkpoint.
