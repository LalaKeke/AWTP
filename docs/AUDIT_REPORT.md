# Release audit — 2026-08-18

This report describes the local Git staging tree prepared for publication.

## Included checks

- All tracked shell scripts pass `bash -n`.
- All Python sources pass byte-code compilation.
- All 20 modules under `tools/pnn_pipeline/` import in the project environment.
- The smoke, temporal-alignment, and dense-horizon-controller suites pass
  (13 tests).
- Stage-1 and Stage-2 MMCV configs resolve the repository root and honor
  `HIPAD_PNN_ROOT`.
- Closed-loop preflight validates the agent, config, routes, CARLA path,
  HiP-AD checkpoint, ControlNet checkpoint, and normalization statistics.
- Local links in the root README and project documentation resolve.
- No tracked model weights, generated tensors, evaluation outputs, private
  keys, machine-specific user paths, or files larger than 50 MiB were found.
- The only tracked file under `data/` is the public Bench2Drive split metadata.

## Data-preparation closure fixes

The public pipeline now includes the previously missing info splitter,
static-v1 dataset builder, static normalization-statistics builder, and
metric-aligned supervision dependency. Data-preparation and training defaults
use one consistent layout:

```text
data/infos/chunks/
outputs/inference_chunks/
data/pnn/static_v1/
data/pnn/time_aligned_v3/
data/pnn/static_v3/
data/pnn/static_v31/
```

## Manual publication gates

- Confirm contributor ownership and redistribution permission for the PNN
  sources.
- Add approved weight/data download URLs and SHA256 checksums.
- Configure Git author identity and a GitHub remote before committing/pushing.
- Run full training and open-/closed-loop evaluation after weights and datasets
  are installed; those large artifacts are intentionally absent here.

Whitespace warnings inherited from the vendored Bench2Drive/HiP-AD sources
were not mechanically rewritten, to preserve upstream code provenance.
