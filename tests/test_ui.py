import ast
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

import numpy as np
import pyqt_dashboard
from dsp_utils import SIGNAL_TYPES
from pyqt_dashboard import SignalDashboard


@pytest.fixture
def window(qtbot):
    dashboard = SignalDashboard()
    qtbot.addWidget(dashboard)
    dashboard.show()
    return dashboard


def test_app_starts_with_tabs_toolbar_and_initial_plot(window):
    assert [window.control_tabs.tabText(i) for i in range(window.control_tabs.count())] == [
        "Signal",
        "Filter",
        "Effects",
        "Export",
        "Help",
    ]
    assert window.toolbar is not None
    assert len(window.cursors) == 2
    assert window.last_export_data is not None


def test_every_preset_can_be_applied_and_plotted(window):
    for name in SignalDashboard.PRESETS:
        window.apply_preset(name)
        window.update_timer.stop()
        window.update_plot()
        assert not window.status_label.text().startswith("Invalid input"), name
        assert "Filter configuration error" not in window.status_label.text(), name
        assert window.last_export_data is not None

    assert "Aliasing risk" in window.status_label.text()


def test_every_waveform_and_filter_combination_plots(window):
    window.samp_freq_entry.setText("100")
    window.duration_entry.setText("1")
    window.cutoff1_entry.setText("15")
    window.cutoff2_entry.setText("25")

    for signal_type, filter_type in product(
        SIGNAL_TYPES, ["None", "Lowpass", "Highpass", "Bandpass", "Bandstop"]
    ):
        window.sig_type_cb.setCurrentText(signal_type)
        window.filter_cb.setCurrentText(filter_type)
        window.update_timer.stop()
        window.update_plot()

        assert "Invalid input" not in window.status_label.text(), (
            signal_type,
            filter_type,
        )
        assert "Filter configuration error" not in window.status_label.text(), (
            signal_type,
            filter_type,
        )
        assert any(
            f"{signal_type} (primary)" == line.get_label()
            for line in window.ax1.lines
        )
        assert window.last_export_data[0].size == 100


@pytest.mark.parametrize("order", ["2", "4", "6", "8"])
def test_filter_order_choices_update_response(window, order):
    window.samp_freq_entry.setText("200")
    window.filter_cb.setCurrentText("Bandpass")
    window.order_combo.setCurrentText(order)
    window.update_timer.stop()

    window.update_plot()

    assert "Filter configuration error" not in window.status_label.text()
    assert window.ax3.lines


def test_reversed_band_cutoffs_are_accepted(window):
    window.samp_freq_entry.setText("100")
    window.filter_cb.setCurrentText("Bandpass")
    window.cutoff1_entry.setText("25")
    window.cutoff2_entry.setText("15")
    window.update_timer.stop()

    window.update_plot()

    assert "Filter configuration error" not in window.status_label.text()
    assert window.ax3.lines


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sig_freq_entry", "abc", "must be a number"),
        ("sig_freq_entry", "NaN", "finite number"),
        ("samp_freq_entry", "0", "greater than 0"),
        ("duration_entry", "-1", "greater than 0"),
        ("duration_entry", "3601", "must not exceed"),
        ("noise_entry", "2", "between 0 and 1"),
        ("distortion_entry", "101", "between 0% and 100%"),
    ],
)
def test_invalid_input_reports_hint_and_keeps_previous_plot(
    window, field, value, message
):
    original_line = window.ax1.lines[0]
    getattr(window, field).setText(value)
    window.update_timer.stop()

    window.update_plot()

    assert message in window.status_label.text()
    assert window.ax1.lines[0] is original_line


def test_filter_cutoff_above_nyquist_reports_error(window):
    window.samp_freq_entry.setText("100")
    window.filter_cb.setCurrentText("Lowpass")
    window.cutoff1_entry.setText("50")
    window.update_timer.stop()

    window.update_plot()

    assert "below the Nyquist frequency" in window.status_label.text()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sig_freq_entry", "-1", "0 Hz or higher"),
        ("sig_freq_entry", "1000001", "must not exceed 1,000,000 Hz"),
        ("samp_freq_entry", "1000001", "must not exceed 1,000,000 Sa/s"),
    ],
)
def test_frequency_limits_show_actionable_errors(window, field, value, message):
    getattr(window, field).setText(value)
    window.update_timer.stop()

    window.update_plot()

    assert message in window.status_label.text()


def test_filter_requires_enough_samples(window):
    window.samp_freq_entry.setText("100")
    window.duration_entry.setText("0.1")
    window.filter_cb.setCurrentText("Lowpass")
    window.update_timer.stop()

    window.update_plot()

    assert "needs more than" in window.status_label.text()


def test_sample_limit_is_validated(window):
    window.samp_freq_entry.setText("1000000")
    window.duration_entry.setText("2")
    window.update_timer.stop()

    window.update_plot()

    assert "more than 1,000,000 samples" in window.status_label.text()


def test_filter_selection_enables_relevant_cutoffs(window):
    window.filter_cb.setCurrentText("Lowpass")
    assert window.cutoff1_entry.isEnabled()
    assert not window.cutoff2_entry.isEnabled()

    window.filter_cb.setCurrentText("Bandpass")
    assert window.cutoff1_entry.isEnabled()
    assert window.cutoff2_entry.isEnabled()

    window.filter_cb.setCurrentText("None")
    assert not window.cutoff1_entry.isEnabled()
    assert not window.cutoff2_entry.isEnabled()


@pytest.mark.parametrize(
    ("entry_name", "slider_name", "slider_value", "expected_text", "typed_text", "expected_slider"),
    [
        ("sig_freq_entry", "sig_freq_slider", 80, "8", "12.3", 123),
        ("samp_freq_entry", "samp_freq_slider", 650, "65", "123.4", 1234),
        ("duration_entry", "duration_slider", 125, "1.25", "2.5", 250),
        ("cutoff1_entry", "cutoff1_slider", 200, "20", "22.3", 223),
        ("cutoff2_entry", "cutoff2_slider", 300, "30", "35.7", 357),
        ("noise_entry", "noise_slider", 50, "0.5", "0.75", 75),
        ("distortion_entry", "distortion_slider", 65, "65", "72", 72),
    ],
)
def test_sliders_and_text_entries_stay_in_sync(
    window, entry_name, slider_name, slider_value, expected_text, typed_text, expected_slider
):
    entry = getattr(window, entry_name)
    slider = getattr(window, slider_name)
    slider.setValue(slider_value)
    assert entry.text() == expected_text

    entry.setText(typed_text)
    assert slider.value() == expected_slider


def test_theme_toggle_updates_theme(window):
    assert window.dark_theme

    window.toggle_theme()

    assert not window.dark_theme
    assert window.fig.get_facecolor()[:3] == pytest.approx(
        (244 / 255, 246 / 255, 248 / 255)
    )

    window.toggle_theme()
    assert window.dark_theme


def test_cursor_readout_displays_plot_coordinates(window):
    window.inspect_plot_position(
        SimpleNamespace(inaxes=window.ax1, xdata=0.25, ydata=0.75)
    )

    assert "time = 0.25 s" in window.cursor_label.text()
    assert "amplitude = 0.75" in window.cursor_label.text()


@pytest.mark.parametrize(
    "event",
    [
        SimpleNamespace(inaxes=None, xdata=0.25, ydata=0.75),
        SimpleNamespace(inaxes="not-an-axis", xdata=0.25, ydata=0.75),
        SimpleNamespace(inaxes="target-axis", xdata=None, ydata=0.75),
        SimpleNamespace(inaxes="target-axis", xdata=0.25, ydata=None),
    ],
)
def test_cursor_ignores_outside_or_incomplete_coordinates(window, event):
    if event.inaxes == "target-axis":
        event.inaxes = window.ax1
    before = window.cursor_label.text()

    window.inspect_plot_position(event)

    assert window.cursor_label.text() == before


def test_parameter_edits_trigger_debounced_plot_update(window, qtbot):
    window.sig_type_cb.setCurrentText("Cosine")
    qtbot.waitUntil(
        lambda: any("Cosine (primary)" == line.get_label() for line in window.ax1.lines),
        timeout=2000,
    )


@pytest.mark.parametrize("extension", ["png", "pdf"])
def test_ui_exports_plot(window, tmp_path, monkeypatch, extension):
    path = tmp_path / f"dashboard.{extension}"
    monkeypatch.setattr(
        pyqt_dashboard.QFileDialog,
        "getSaveFileName",
        lambda *args: (str(path), ""),
    )

    window.export_plot()

    assert path.exists() and path.stat().st_size > 0
    assert "Plot saved" in window.status_label.text()


def test_ui_exports_csv(window, tmp_path, monkeypatch):
    path = tmp_path / "dashboard.csv"
    monkeypatch.setattr(
        pyqt_dashboard.QFileDialog,
        "getSaveFileName",
        lambda *args: (str(path), ""),
    )

    window.export_csv()

    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("Time (s),Original,Filtered")
    assert "Signal data exported" in window.status_label.text()


def test_export_failure_is_reported(window, monkeypatch, tmp_path):
    monkeypatch.setattr(
        pyqt_dashboard.QFileDialog,
        "getSaveFileName",
        lambda *args: (str(tmp_path / "failure.png"), ""),
    )

    def fail_save(*args):
        raise OSError("permission denied")

    monkeypatch.setattr(pyqt_dashboard, "save_plot", fail_save)
    window.export_plot()

    assert "Could not save plot: permission denied" in window.status_label.text()


@pytest.mark.parametrize("method_name", ["export_plot", "export_csv"])
def test_export_buttons_do_nothing_before_a_plot_exists(
    window, monkeypatch, method_name
):
    window.last_export_data = None
    status_before = window.status_label.text()

    def unexpected_dialog(*args):
        raise AssertionError("The save dialog should not open without plot data.")

    monkeypatch.setattr(pyqt_dashboard.QFileDialog, "getSaveFileName", unexpected_dialog)

    getattr(window, method_name)()

    assert window.status_label.text() == status_before


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (OSError("disk full"), "Could not export CSV: disk full"),
        (ValueError("invalid signal data"), "Could not export CSV: invalid signal data"),
    ],
)
def test_csv_export_failures_are_reported(window, monkeypatch, tmp_path, error, status):
    monkeypatch.setattr(
        pyqt_dashboard.QFileDialog,
        "getSaveFileName",
        lambda *args: (str(tmp_path / "signals.csv"), ""),
    )

    def fail_export(*args):
        raise error

    monkeypatch.setattr(pyqt_dashboard, "save_signal_csv", fail_export)

    window.export_csv()

    assert status in window.status_label.text()


def test_script_entrypoint_creates_shows_and_runs_application():
    source_path = Path(pyqt_dashboard.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    main_block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    calls = []

    class FakeApplication:
        def __init__(self, arguments):
            calls.append(("application", arguments))

        def exec(self):
            calls.append(("event-loop",))
            return 0

    class FakeDashboard:
        def show(self):
            calls.append(("show",))

    fake_sys = SimpleNamespace(
        argv=["pyqt_dashboard.py"],
        exit=lambda status: calls.append(("exit", status)),
    )
    module = ast.Module(body=[main_block], type_ignores=[])

    exec(
        compile(module, str(source_path), "exec"),
        {
            "__name__": "__main__",
            "sys": fake_sys,
            "QApplication": FakeApplication,
            "SignalDashboard": FakeDashboard,
        },
    )

    assert calls == [
        ("application", ["pyqt_dashboard.py"]),
        ("show",),
        ("event-loop",),
        ("exit", 0),
    ]


def test_spectrum_plot_is_created_at_startup(window):
    assert window.ax4 in window.fig.axes
    assert window.ax4.lines
    assert window.ax4.get_xlabel() == "Frequency (Hz)"
    assert window.ax4.get_ylabel() == "Amplitude"
    assert "Peak:" in window.ax4.get_title()


def test_spectrum_peak_tracks_signal_frequency(window):
    window.samp_freq_entry.setText("1000")
    window.sig_freq_entry.setText("80")
    window.update_timer.stop()

    window.update_plot()

    frequencies, amplitude = window.ax4.lines[0].get_data()
    peak_index = np.argmax(amplitude)
    assert frequencies[peak_index] == pytest.approx(80)
    assert "Peak: 80 Hz" in window.ax4.get_title()


def test_invalid_input_keeps_previous_spectrum(window):
    previous_line = window.ax4.lines[0]
    previous_title = window.ax4.get_title()
    window.sig_freq_entry.setText("invalid")
    window.update_timer.stop()

    window.update_plot()

    assert window.ax4.lines[0] is previous_line
    assert window.ax4.get_title() == previous_title


def test_spectrum_redraws_with_both_themes(window):
    window.toggle_theme()

    assert window.ax4.lines
    assert window.ax4.get_facecolor()[:3] == pytest.approx(
        (244 / 255, 246 / 255, 248 / 255)
    )
    assert window.ax4.xaxis.label.get_color() == "#20252b"

    window.toggle_theme()

    assert window.ax4.lines
    assert window.ax4.get_facecolor()[:3] == pytest.approx(
        (43 / 255, 43 / 255, 43 / 255)
    )
    assert window.ax4.xaxis.label.get_color() == "#ffffff"


def test_sampled_output_markers_match_the_output_trace(window):
    window.samp_freq_entry.setText("100")
    window.sig_freq_entry.setText("5")
    window.update_timer.stop()

    window.update_plot()

    trace = window.ax2.lines[0]
    markers = window.ax2.lines[1]
    np.testing.assert_array_equal(trace.get_xdata(), markers.get_xdata())
    np.testing.assert_array_equal(trace.get_ydata(), markers.get_ydata())


def test_sidebar_and_plot_pane_can_be_resized(window):
    splitter = window.main_splitter
    assert splitter.count() == 2

    splitter.setSizes([300, 800])
    narrow_sidebar_width = splitter.sizes()[0]
    splitter.setSizes([500, 600])
    wide_sidebar_width = splitter.sizes()[0]

    assert wide_sidebar_width > narrow_sidebar_width


def test_individual_plots_can_be_resized_and_reset(window):
    upper_axis, lower_axis = window.ax1, window.ax2
    initial_upper_height = upper_axis.get_position().height
    initial_lower_height = lower_axis.get_position().height
    upper_position = upper_axis.get_position()
    lower_position = lower_axis.get_position()
    figure_width, figure_height = window.fig.bbox.width, window.fig.bbox.height
    event_x = (upper_position.x0 + upper_position.x1) * figure_width / 2
    divider_y = (upper_position.y0 + lower_position.y1) * figure_height / 2

    window._start_plot_resize(
        SimpleNamespace(x=event_x, y=divider_y, button=1, dblclick=False)
    )
    window._update_plot_resize(
        SimpleNamespace(x=event_x, y=divider_y - 60, button=1)
    )
    window._finish_plot_resize(
        SimpleNamespace(x=event_x, y=divider_y - 60, button=1)
    )

    assert upper_axis.get_position().height != pytest.approx(initial_upper_height)
    assert lower_axis.get_position().height != pytest.approx(initial_lower_height)
    resized_upper_height = upper_axis.get_position().height
    window.update_plot()
    assert upper_axis.get_position().height == pytest.approx(resized_upper_height)

    upper_position = upper_axis.get_position()
    lower_position = lower_axis.get_position()
    divider_y = (upper_position.y0 + lower_position.y1) * figure_height / 2
    window._start_plot_resize(
        SimpleNamespace(x=event_x, y=divider_y, button=1, dblclick=True)
    )

    assert upper_axis.get_position().height == pytest.approx(
        lower_axis.get_position().height
    )


def test_plot_resize_handles_are_visible_and_legends_are_draggable(window):
    assert len(window._plot_divider_artists) == 3
    for divider_line, grip_marks in window._plot_divider_artists:
        assert divider_line.get_visible()
        assert divider_line.get_linewidth() <= 1
        assert not grip_marks.get_visible()

    legend = window.ax1.get_legend()
    assert legend is not None
    assert legend.get_draggable()

    window._set_plot_divider_hover(1)
    assert window._plot_divider_artists[1][1].get_visible()


def test_measurements_panel_is_present_and_updated(window):
    assert hasattr(window, "measurement_label")
    window.update_plot()
    text = window.measurement_label.text()
    assert "RMS:" in text
    assert "Peak:" in text
    assert "Peak-to-peak:" in text
    assert "Mean:" in text
    assert "Crest factor:" in text
    assert "Peak frequency:" in text


def test_measurements_panel_shows_snr_when_noise_is_on(window):
    window.noise_entry.setText("0.10")
    window.update_timer.stop()
    window.update_plot()

    assert "SNR:" in window.measurement_label.text()


def test_measurements_panel_shows_original_filtered_comparison(window):
    window.update_timer.stop()
    window.update_plot()
    text = window.measurement_label.text()

    assert "Original RMS:" in text
    assert "Filtered RMS:" in text
    assert "Attenuation:" in text
    assert "RMS reduction:" in text
