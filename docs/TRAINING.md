# Training

The release recipe trains the physical joint scratch v2 ControlNet using
temporal-alignment-v3 actor inputs and solid-lane supervision. WeightNet and
Theseus are disabled in the deployed path.

```bash
PNN_GPUS=0,1,2,3 \
PNN_TOTAL_EPOCHS=200 \
PNN_BATCH_SIZE=48 \
bash scripts/train_pnn.sh
```

Important overrides:

```text
PNN_OLD_DATA              time-aligned HiP-AD source tensor
PNN_NEW_DATA              static/GT-augmented tensor
PNN_SUPERVISION_DATA      solid-lane supervision tensor
PNN_SAVE_DIR              output run directory
PNN_ALLOW_RESUME=1        allow an existing last.pth to resume
PYTHON_BIN                 Python interpreter
```

The wrapper refuses a missing input and prevents accidental resume unless
explicitly enabled. Training outputs, TensorBoard files, and checkpoints are
ignored by Git.

Per-epoch SparseDrive L2 evaluation is disabled by default because its
standalone evaluator and model artifacts are not part of this release. Set
`PNN_EVAL_EACH_EPOCH=true` and `PNN_L2_EVAL_SCRIPT=/path/to/run_sparse_l2_eval.py`
only when that external evaluation environment is available.
