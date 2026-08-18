# HiPAD-PNN

HiPAD-PNN connects HiP-AD perception/planning with a physics-aware secondary
planner. The release path uses HiP-AD route intent, temporal actor alignment
v3, a ControlNet that predicts 3 seconds of acceleration and front-wheel
steering, bicycle-model rollout, and the horizon-v2.3 receding-horizon CARLA
controller.

This repository contains source code only. It intentionally excludes model
weights, Bench2Drive data, converted training tensors, experiment outputs, and
raw evaluation results.

## Repository layout

```text
projects/                         HiP-AD model and dataset plugin
pnn/Main/                         PNN model and training implementation
pnn/nnc/                          bicycle dynamics and controller primitives
hipad_pnn_adapter.py              HiP-AD to PNN interface
pnn_temporal_alignment.py         shared actor temporal alignment v3
tools/pnn_pipeline/               data preparation and training recipes
bench2drive/                      closed-loop integration
scripts/                          stable user-facing commands
tests/                            data-free smoke tests
```

## Quick Start

The tested stack is Python 3.8.20, PyTorch 1.13.0 + CUDA 11.7,
MMCV 1.7.1, and MMDetection 2.28.2. A CUDA toolkit compatible with the local
PyTorch build is required to compile deformable aggregation.

```bash
git clone https://github.com/YOUR_ACCOUNT/HiPAD-PNN.git
cd HiPAD-PNN

conda env create -f environment.yml
conda activate hipad-pnn

python -m pip install -e ./projects/mmdet3d_plugin/ops
bash scripts/smoke_test.sh
```

The smoke test requires no dataset or checkpoint. A successful run verifies
the coordinate bridge, temporal alignment, ControlNet forward pass, and output
shapes.

### Prepare external artifacts

```bash
mkdir -p data/pnn checkpoints outputs evaluation
ln -s /absolute/path/to/Bench2Drive data/bench2drive

cp /path/to/hipad_stage2.pth checkpoints/hipad_stage2.pth
cp /path/to/pnn_control.pth checkpoints/pnn_control.pth
cp /path/to/pnn_stats.pt checkpoints/pnn_stats.pt
```

These locations are ignored by Git. Published downloads should include a
SHA256 checksum; see [data preparation](docs/DATA_PREPARATION.md).

### Open-loop evaluation

```bash
HIPAD_CKPT=checkpoints/hipad_stage2.pth \
PNN_CONTROL_CKPT=checkpoints/pnn_control.pth \
PNN_STATS_PATH=checkpoints/pnn_stats.pt \
GPUS=1 bash scripts/eval_openloop.sh
```

### Train the current physical ControlNet recipe

After completing the four data-preparation commands in
[Data preparation](docs/DATA_PREPARATION.md), the default recipe reads:

```text
data/pnn/time_aligned_v3/train_old.pt
data/pnn/static_v3/train_new_with_hipad_plan.pt
data/pnn/static_v31/solid_lane_supervision.pt
```

```bash
PNN_GPUS=0,1,2,3 bash scripts/train_pnn.sh
```

Training from scratch is the default. Outputs are written under
`outputs/pnn_physical_joint_scratch_v2/`.

### Closed-loop evaluation

Install CARLA 0.9.15 separately, then run:

```bash
CARLA_ROOT=/absolute/path/to/CARLA_0.9.15 \
HIPAD_CKPT=checkpoints/hipad_stage2.pth \
PNN_CONTROL_CKPT=checkpoints/pnn_control.pth \
PNN_STATS_PATH=checkpoints/pnn_stats.pt \
bash scripts/eval_closedloop.sh
```

Use `PNN_PREFLIGHT_ONLY=1` first to validate paths without starting CARLA.

## Documentation

- [Installation](docs/INSTALL.md)
- [Data preparation](docs/DATA_PREPARATION.md)
- [Training](docs/TRAINING.md)
- [Evaluation](docs/EVALUATION.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)
- [Release audit](docs/AUDIT_REPORT.md)

## Model artifacts

Weights are not part of the Git repository. Add official release URLs and
checksums here only after the files have been approved for redistribution.

## License and attribution

The HiP-AD-derived code is provided under Apache-2.0; see `LICENSE` and
`NOTICE`. Bench2Drive retains its own license. Review
`THIRD_PARTY_LICENSES.md` before publishing. Please retain the original HiP-AD
paper citation and add the HiPAD-PNN citation when it becomes available.
