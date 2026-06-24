# RTM-2

RTM-2 is an experimental reasoning project built on top of HRM (Hierarchical
Reasoning Model), with a local research branch focused on making the model more
adaptive, more context-aware, and a little closer to what we informally kept
calling a "sixth-sense" style of reasoning.

The goal is not just to make a model solve structured tasks, but to push it
toward learning how it should approach different tasks:

- when to lean on high-level planning,
- when to rely on fast low-level computation,
- and how to infer the right reasoning style from the task context itself.

In practice, this repo currently explores that idea through a variant called
HRM-Free-Meta, which adds:

- latent task-context modeling,
- dynamic weighting between hierarchical modules,
- meta-learning inspired regularization,
- and a cleaned local training / evaluation / resume workflow.

## Current status

This repository is an experimental research workspace, not a polished library.

What is already in decent shape:

- a coherent local training entrypoint,
- checkpoint save / resume support,
- checkpoint evaluation,
- basic smoke tests,
- cleaner documentation than before.

What is still true:

- some files remain exploratory or debug-oriented,
- full runtime verification depends on a local PyTorch + CUDA environment,
- this is still a research repo, not a finished product.

## Main entrypoints

The active code lives under [HRM](C:/Users/emreg/Desktop/RTM-2/HRM).

The most important entrypoints are:

- [HRM/unified_training.py](C:/Users/emreg/Desktop/RTM-2/HRM/unified_training.py) — main training and resume pipeline
- [HRM/evaluate.py](C:/Users/emreg/Desktop/RTM-2/HRM/evaluate.py) — evaluate saved checkpoints
- [HRM/test.py](C:/Users/emreg/Desktop/RTM-2/HRM/test.py) — quick smoke-test runner
- [HRM/README.md](C:/Users/emreg/Desktop/RTM-2/HRM/README.md) — detailed project usage

## Project direction

The bigger research idea behind RTM-2 is simple:

> Not just solving tasks — learning how to approach them.

That means this repo is less about scaling parameter count blindly, and more
about experimenting with:

- hierarchical reasoning,
- adaptive computation,
- task-aware control,
- context-sensitive routing,
- and meta-level signals that influence reasoning behavior.

## Quick start

See the main project guide here:

- [HRM/README.md](C:/Users/emreg/Desktop/RTM-2/HRM/README.md)

If you mainly want resume examples:

- [HRM/resume_quickstart.md](C:/Users/emreg/Desktop/RTM-2/HRM/resume_quickstart.md)

## Suggested GitHub description

Experimental HRM fork exploring more intuitive task-aware reasoning with
meta-learning, latent context modeling, dynamic module weighting, and a cleaned
training/evaluation workflow.

## Suggested GitHub topics

- `pytorch`
- `deep-learning`
- `machine-learning`
- `meta-learning`
- `reasoning`
- `hierarchical-reasoning`
- `adaptive-reasoning`
- `context-aware`
- `research`
- `experimental`
- `arc-agi`
