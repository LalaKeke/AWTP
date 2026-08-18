# Data preparation

Datasets and generated tensors must stay outside Git history.

## Bench2Drive

Download Bench2Drive Base using the official instructions, then link it:

```bash
mkdir -p data
ln -s /absolute/path/to/Bench2Drive data/bench2drive
python tools/data_converter/bench2drive_converter.py
bash tools/kmeans/kemans.sh
```

The public Base train/validation split required by the converter is included
at `data/splits/bench2drive_base_train_val_split.json`. It is lightweight
metadata from Bench2DriveZoo, not a dataset sample. Do not replace it silently
when reporting comparable results.

## PNN tensors

Build the current training inputs in dependency order. This is an expensive
GPU/CPU pipeline and requires the Bench2Drive annotations plus a trained
HiP-AD checkpoint at `checkpoints/hipad_stage2.pth`.

```bash
bash tools/pnn_pipeline/prepare_pnn_static_v1_dataset.sh
bash tools/pnn_pipeline/prepare_pnn_time_aligned_v3_dataset.sh
bash tools/pnn_pipeline/prepare_pnn_static_v3_data.sh
bash tools/pnn_pipeline/prepare_pnn_static_v31_lane_data.sh
```

The commands split the converted training annotations under
`data/infos/chunks/`, cache HiP-AD inference under
`outputs/inference_chunks/`, then produce the current training interface:

```text
data/pnn/time_aligned_v3/train_old.pt
data/pnn/static_v3/train_new_with_hipad_plan.pt
data/pnn/static_v31/solid_lane_supervision.pt
```

The conversion and temporal-alignment-v3 utilities are under
`tools/pnn_pipeline/`. They preserve the `pnn_xy` coordinate convention and
record alignment metadata. Do not use any file under `evaluation/` as a
training input.

Record checksums for every externally distributed artifact:

```bash
find data/pnn checkpoints -type f -print0 | sort -z | xargs -0 sha256sum > artifacts.sha256
```

Do not commit `artifacts.sha256` if it reveals private filenames; publish a
reviewed manifest alongside the corresponding release assets.
