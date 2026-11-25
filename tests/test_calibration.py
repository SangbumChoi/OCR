"""Calibration: reliability binning + ECE."""

from docvlm_eval.metrics.calibration import expected_calibration_error, reliability_table


def test_reliability_table_bins():
    conf = [0.05, 0.15, 0.95]
    corr = [0.0, 0.0, 1.0]
    bins = reliability_table(conf, corr, n_bins=10)
    assert len(bins) == 10
    assert sum(b.count for b in bins) == 3
    # last bin holds the 0.95 sample
    assert bins[-1].count == 1
    assert bins[-1].accuracy == 1.0


def test_ece_zero_when_calibrated():
    ece = expected_calibration_error([0.95, 0.95, 0.05, 0.05], [1, 1, 0, 0], n_bins=10)
    assert ece is not None and ece < 0.06


def test_ece_none_without_confidence():
    assert expected_calibration_error([None, None], [1, 0]) is None


def test_ece_ignores_none_entries():
    # mix of None and real confidences -> uses only the real ones
    ece = expected_calibration_error([None, 0.9, 0.9], [1, 1, 0], n_bins=10)
    assert ece is not None


def test_temperature_scaling_fits_overconfident_binary_predictions():
    from docvlm_eval.metrics.calibration import (
        fit_temperature_scaling,
        temperature_scale_confidence,
    )

    result = fit_temperature_scaling(
        [0.9] * 20,
        [1.0] * 10 + [0.0] * 10,
        min_samples=20,
    )

    assert result.status == "fitted"
    assert result.temperature is not None
    assert result.calibrated_nll < result.raw_nll
    assert temperature_scale_confidence(0.9, result.temperature) < 0.53


def test_temperature_scaling_requires_both_outcomes_and_enough_rows():
    from docvlm_eval.metrics.calibration import fit_temperature_scaling

    too_few = fit_temperature_scaling(
        [0.8] * 4,
        [1.0, 0.0, 1.0, 0.0],
        min_samples=5,
    )
    one_class = fit_temperature_scaling(
        [0.8] * 5,
        [1.0] * 5,
        min_samples=5,
    )

    assert too_few.status == "insufficient_evidence"
    assert one_class.status == "insufficient_evidence"
