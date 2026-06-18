"""The model wrapper exposes resource measurement (time is in the pipeline; memory here)."""

from docvlm_eval.models import build_model


def test_peak_cpu_mb_positive():
    m = build_model("dummy-echo", device="cpu")
    mb = m.peak_cpu_mb()
    assert mb is None or mb > 0  # on Linux returns RSS in MB


def test_peak_gpu_none_on_cpu():
    m = build_model("dummy-echo", device="cpu")
    assert m.peak_gpu_mb() is None


def test_reset_peak_memory_noop_on_cpu():
    m = build_model("dummy-echo", device="cpu")
    m.reset_peak_memory()  # must not raise on CPU


def test_pipeline_records_efficiency(tiny_benchmark, tmp_path):
    from docvlm_eval.pipeline import run_evaluation

    samples, _ = tiny_benchmark
    summary = run_evaluation("dummy-echo", samples, str(tmp_path / "r"), device="cpu")
    for k in ("load_seconds", "avg_latency_s", "p90_latency_s", "total_infer_s",
              "peak_cpu_mb", "peak_gpu_mb", "device"):
        assert k in summary
