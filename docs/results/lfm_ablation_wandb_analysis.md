# LFM W&B ablation snapshot audit

- Evidence status: `preliminary_direction_only`
- Promotion eligible: `false`
- Run quality: 5 evaluated of 10 observed
- Evaluated A0 baselines: 0

## Comparable preliminary pair

`prh5gy29` (vision) versus `zivt0ner` (connector), both on `A1_spotting_on`.

| Heldout metric | Vision minus connector |
| --- | ---: |
| `score` | +0.0486 |
| `grounding` | +0.0607 |
| `L1-locate` | +0.0234 |
| `L1-region` | -0.0581 |
| `kie` | +0.0000 |
| `ocr-full` | -0.0466 |
| `H-comprehension` | +0.0627 |

## Evidence limits

- Finished without evaluation: 4
- Crashed runs: 1
- Missing promotion controls: `base_model_revision`, `data_seed`, `heldout_sample_ids_fingerprint`, `max_steps`, `optimizer_seed`, `training_sample_count`

The result is direction-only. Execute [`configs/lora_vision_connector_sweep.yaml`](../../configs/lora_vision_connector_sweep.yaml) before promoting a placement.
