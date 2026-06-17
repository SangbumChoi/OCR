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
