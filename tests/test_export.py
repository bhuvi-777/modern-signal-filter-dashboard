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


def test_load_two_column_csv_uses_header_and_derives_fs(tmp_path):
    path = tmp_path / "two.csv"
    path.write_text("Time (s),Signal\n0,0\n0.01,1\n0.02,0\n", encoding="utf-8")
    time, signal, fs = __import__("export_utils").load_signal_csv(path)
    np.testing.assert_allclose(time, [0, 0.01, 0.02], atol=1e-12)
    np.testing.assert_allclose(signal, [0, 1, 0])
    assert fs == pytest.approx(100)


def test_load_one_column_csv_with_fs(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("Signal\n0\n1\n0\n", encoding="utf-8")
    time, signal, fs = __import__("export_utils").load_signal_csv(path, fs=100)
    np.testing.assert_allclose(time, [0, 0.01, 0.02])
    assert fs == 100


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "Signal\n",
        "abc\n",
        "0\nnan\n",
        "Time,Signal\n0,0\n0.01,1\n0.03,0.5\n",
    ],
)
def test_load_csv_invalid_cases_raise_value_error(tmp_path, contents):
    path = tmp_path / "bad.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        __import__("export_utils").load_signal_csv(path, fs=100)


def test_session_save_load_roundtrip(tmp_path):
    settings = {
        "waveform": "Square",
        "filter_family": "Elliptic",
        "filter_type": "Bandpass",
        "cutoff1": 80.0,
        "cutoff2": 120.0,
        "order": 6,
        "effects": {"noise": 0.1, "distortion": 0.2},
        "theme": "dark",
        "presets": {"My preset": {"waveform": "Sine"}},
    }
    path = tmp_path / "session.json"
    __import__("export_utils").save_session(path, settings)
    loaded = __import__("export_utils").load_session(path)
    assert loaded == settings


@pytest.mark.parametrize(
    "contents",
    [
        "{bad json",
        "{}",
        '{"version":2,"settings":{}}',
        '{"version":"1","settings":{}}',
        '{"version":1,"settings":[]}',
    ],
)
def test_session_invalid_json_or_schema_raises(tmp_path, contents):
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        __import__("export_utils").load_session(path)


def test_session_extra_keys_are_ignored(tmp_path):
    path = tmp_path / "extra.json"
    path.write_text(
        '{"version":1,"settings":{"waveform":"Sine"},"future":123}',
        encoding="utf-8",
    )
    loaded = __import__("export_utils").load_session(path)
    assert loaded == {"waveform": "Sine"}
