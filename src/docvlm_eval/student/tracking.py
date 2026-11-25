"""Optional external metric tracking for native student training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


_STEP_METRICS = {
    "pretrain": "train/global_step",
    "sft": "train/global_step",
    "preference": "preference/preference_step",
    "rlvr": "rlvr/rollout_step",
}

_METRIC_PREFIXES = {
    "pretrain": ("train/*", "eval/*", "adaptive/*", "gradient_probe/*"),
    "sft": ("train/*",),
    "preference": ("preference/*", "reward/*", "reward_diagnostic/*"),
    "rlvr": ("rlvr/*", "reward/*", "reward_diagnostic/*"),
}


@dataclass
class WandbMetricTracker:
    """Log numeric stage metrics and finish one W&B run."""

    run: Any
    stage: str

    def __call__(self, payload: dict[str, Any]) -> None:
        numeric = {
            key: value
            for key, value in payload.items()
            if isinstance(value, (bool, int, float))
        }
        if numeric:
            self.run.log(numeric)

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if summary:
            for key, value in summary.items():
                self.run.summary[key] = value
        self.run.finish()


def start_wandb_metric_tracker(
    *,
    stage: str,
    project: str | None,
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    run_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> WandbMetricTracker | None:
    """Create one optional rank-zero W&B stage run."""

    if not project or int(os.environ.get("RANK", "0")) != 0:
        return None
    if stage not in _STEP_METRICS:
        raise ValueError(f"unsupported training tracking stage {stage!r}")
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "native training W&B tracking requires `pip install wandb`"
        ) from error
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        tags=tags,
        id=run_id,
        resume="allow" if run_id else None,
        job_type=stage,
        config=config,
    )
    step_metric = _STEP_METRICS[stage]
    run.define_metric(step_metric)
    for pattern in _METRIC_PREFIXES[stage]:
        run.define_metric(pattern, step_metric=step_metric)
    return WandbMetricTracker(run=run, stage=stage)
