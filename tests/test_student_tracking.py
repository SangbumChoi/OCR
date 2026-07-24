import sys
from types import SimpleNamespace

from docvlm_eval.student.tracking import start_wandb_metric_tracker


class _Run:
    def __init__(self):
        self.defined = []
        self.logged = []
        self.summary = {}
        self.finished = False

    def define_metric(self, name, **kwargs):
        self.defined.append((name, kwargs))

    def log(self, payload):
        self.logged.append(payload)

    def finish(self):
        self.finished = True


def test_wandb_tracker_defines_stage_axes_and_logs_only_numeric_values(
    monkeypatch,
):
    run = _Run()
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    tracker = start_wandb_metric_tracker(
        stage="rlvr",
        project="docvlm-native",
        entity="sbdc",
        name="trial--rlvr",
        group="trial",
        tags=["native-student", "rlvr"],
        run_id="stable-run-id",
        config={"seed": 7},
    )

    assert tracker is not None
    tracker(
        {
            "kind": "rlvr",
            "sample_id": "sample-1",
            "rlvr/rollout_step": 3.0,
            "rlvr/loss": 0.25,
            "rlvr/replay": True,
        }
    )
    tracker.finish({"stage_status": "completed"})

    assert calls[0]["id"] == "stable-run-id"
    assert calls[0]["resume"] == "allow"
    assert calls[0]["job_type"] == "rlvr"
    assert (
        "rlvr/*",
        {"step_metric": "rlvr/rollout_step"},
    ) in run.defined
    assert run.logged == [
        {
            "rlvr/rollout_step": 3.0,
            "rlvr/loss": 0.25,
            "rlvr/replay": True,
        }
    ]
    assert run.summary["stage_status"] == "completed"
    assert run.finished is True


def test_wandb_tracker_is_disabled_without_project_or_off_rank(monkeypatch):
    monkeypatch.setenv("RANK", "1")

    assert (
        start_wandb_metric_tracker(
            stage="pretrain",
            project="docvlm-native",
        )
        is None
    )
    monkeypatch.setenv("RANK", "0")
    assert start_wandb_metric_tracker(stage="pretrain", project=None) is None
