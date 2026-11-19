import csv

import matplotlib.pyplot as plt
import numpy as np
import pytest

from export_utils import save_plot, save_signal_csv


def test_csv_roundtrip(tmp_path):
    time = np.linspace(0, 1, 50)
    original = np.sin(time)
    filtered = original * 0.5
    path = tmp_path / "signals.csv"

    save_signal_csv(path, time, original, filtered)

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    assert data.shape == (50, 3)
    np.testing.assert_allclose(data[:, 0], time)
    np.testing.assert_allclose(data[:, 1], original, atol=1e-6)
    np.testing.assert_allclose(data[:, 2], filtered, atol=1e-6)
    with path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == ["Time (s)", "Original", "Filtered"]


def test_csv_rejects_mismatched_array_lengths(tmp_path):
    with pytest.raises(ValueError, match="same number of samples"):
        save_signal_csv(tmp_path / "mismatch.csv", [0, 1], [1, 2], [1])


@pytest.mark.parametrize("extension", ["png", "pdf"])
def test_plot_export_writes_nonempty_file(tmp_path, extension):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    path = tmp_path / f"plot.{extension}"

    try:
        save_plot(path, figure)
    finally:
        plt.close(figure)

    assert path.exists()
    assert path.stat().st_size > 0
