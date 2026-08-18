# Installation

## Supported environment

- Linux x86-64
- Python 3.8.20
- CUDA 11.7-compatible driver/toolkit
- PyTorch 1.13.0+cu117
- MMCV Full 1.7.1 and MMDetection 2.28.2
- CARLA 0.9.15 for closed-loop evaluation only

Create the environment and compile the repository-local CUDA extension:

```bash
conda env create -f environment.yml
conda activate hipad-pnn
python -m pip install -e ./projects/mmdet3d_plugin/ops
python -m pytest -q tests
```

If `flash-attn` cannot be built for this legacy stack, install a wheel matching
Python, PyTorch, and CUDA, or omit it when the selected model configuration
does not import it.

CARLA is intentionally not vendored. Install CARLA 0.9.15 and expose its
Python API as described by the CARLA and Bench2Drive documentation.
