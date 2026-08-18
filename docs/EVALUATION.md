# Evaluation

## Open loop

`scripts/eval_openloop.sh` evaluates HiP-AD with the PNN planning replacement.
It defaults to ControlNet-only inference, `pnn_xy`, robust q0.005-q0.995 stats,
and planning-only collection.

Set `PNN_MAX_BATCHES=1` for a short integration check before a complete run.

## Closed loop

`scripts/eval_closedloop.sh` uses the Bench2Drive multi-route evaluator and the
horizon-v2.3 controller. Required artifacts are CARLA 0.9.15, the HiP-AD
checkpoint, PNN ControlNet checkpoint, and matching normalization statistics.

Run path validation first:

```bash
PNN_PREFLIGHT_ONLY=1 CARLA_ROOT=/path/to/carla \
bash scripts/eval_closedloop.sh
```

Generated JSON, logs, videos, and route outputs stay under `evaluation/` and
must not be committed. Only reviewed aggregate metrics should be copied into a
paper or release note.
