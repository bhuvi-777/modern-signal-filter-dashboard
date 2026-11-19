import sys
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.lines import Line2D
from matplotlib.widgets import Cursor
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from dsp_utils import (
    SIGNAL_TYPES,
    add_noise_and_distortion,
    apply_digital_filter,
    calculate_frequency_response,
    compute_measurements,
    compute_snr_db,
    compute_spectrum,
    generate_signal,
)
from export_utils import save_plot, save_signal_csv


class SignalDashboard(QMainWindow):
    PRESETS = {
        "Custom": None,
        "Clean 5 Hz Sine": {
            "signal": "Sine",
            "frequency": "5.0",
            "sampling": "50.0",
            "duration": "1.0",
            "filter": "None",
            "cutoff1": "15.0",
            "cutoff2": "25.0",
            "noise": 0,
            "distortion": 0,
        },
        "Lowpass 10 Hz Demo": {
            "signal": "Square",
            "frequency": "5.0",
            "sampling": "100.0",
            "duration": "1.0",
            "filter": "Lowpass",
            "cutoff1": "10.0",
            "cutoff2": "25.0",
            "noise": 0,
            "distortion": 0,
        },
        "Noisy Bandpass": {
            "signal": "Sine",
            "frequency": "20.0",
            "sampling": "200.0",
            "duration": "1.0",
            "filter": "Bandpass",
            "cutoff1": "10.0",
            "cutoff2": "30.0",
            "noise": 0.03,
            "distortion": 5,
        },
        "Nyquist Warning": {
            "signal": "Sine",
            "frequency": "30.0",
            "sampling": "40.0",
            "duration": "1.0",
            "filter": "None",
            "cutoff1": "15.0",
            "cutoff2": "25.0",
            "noise": 0,
            "distortion": 0,
        },
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern Signal & Filter Dashboard (PyQt6)")
        self.resize(1250, 950)
        self.last_export_data = None
        self.dark_theme = True

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(20)

        sidebar_frame = QFrame()
        sidebar_frame.setMinimumWidth(300)
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)

        header_layout = QHBoxLayout()
        title_label = QLabel("Signal Dashboard")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        self.theme_btn = QPushButton("Switch to Light Theme")
        self.theme_btn.setCheckable(True)
        self.theme_btn.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.theme_btn)
        sidebar_layout.addLayout(header_layout)

        self.control_tabs = QTabWidget()
        sidebar_layout.addWidget(self.control_tabs, stretch=1)
        signal_tab = QWidget()
        signal_layout = QVBoxLayout(signal_tab)
        signal_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.control_tabs.addTab(signal_tab, "Signal")
        filter_tab = QWidget()
        filter_layout = QVBoxLayout(filter_tab)
        filter_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.control_tabs.addTab(filter_tab, "Filter")
        effects_tab = QWidget()
        effects_layout = QVBoxLayout(effects_tab)
        effects_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.control_tabs.addTab(effects_tab, "Effects")
        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)
        export_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.control_tabs.addTab(export_tab, "Export")
        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)
        self.control_tabs.addTab(help_tab, "Help")

        signal_layout.addWidget(QLabel("Preset:"))
        self.preset_cb = QComboBox()
        self.preset_cb.addItems(list(self.PRESETS))
        signal_layout.addWidget(self.preset_cb)

        signal_layout.addWidget(QLabel("Primary Signal Type:"))
        self.sig_type_cb = QComboBox()
        self.sig_type_cb.addItems(SIGNAL_TYPES)
        signal_layout.addWidget(self.sig_type_cb)

        self.sig_freq_entry, self.sig_freq_slider = self._add_labeled_slider(
            signal_layout, "Signal Frequency (Hz)", "5.0", 0, 1000, 50
        )
        self.samp_freq_entry, self.samp_freq_slider = self._add_labeled_slider(
            signal_layout, "Sampling Rate (Sa/s)", "50.0", 10, 50000, 500
        )
        self.duration_entry, self.duration_slider = self._add_labeled_slider(
            signal_layout, "Time Duration (s)", "1.0", 1, 6000, 100
        )

        filter_layout.addWidget(QLabel("Filter Type:"))
        self.filter_cb = QComboBox()
        self.filter_cb.addItems(["None", "Lowpass", "Highpass", "Bandpass", "Bandstop"])
        filter_layout.addWidget(self.filter_cb)

        cutoff_layout = QGridLayout()
        cutoff_layout.addWidget(QLabel("Cutoff 1 (Hz):"), 0, 0)
        self.cutoff1_entry, self.cutoff1_slider = self._make_slider("15.0", 1, 40000, 150)
        cutoff_layout.addWidget(self.cutoff1_entry, 1, 0)
        cutoff_layout.addWidget(self.cutoff1_slider, 2, 0)
        cutoff_layout.addWidget(QLabel("Cutoff 2 (Hz):"), 0, 1)
        self.cutoff2_entry, self.cutoff2_slider = self._make_slider("25.0", 1, 40000, 250)
        cutoff_layout.addWidget(self.cutoff2_entry, 1, 1)
        cutoff_layout.addWidget(self.cutoff2_slider, 2, 1)
        filter_layout.addLayout(cutoff_layout)
        filter_layout.addWidget(QLabel("Filter Order:"))
        self.order_combo = QComboBox()
        self.order_combo.addItems(["2", "4", "6", "8"])
        self.order_combo.setCurrentText("4")
        filter_layout.addWidget(self.order_combo)
        filter_layout.addStretch()

        effects_layout.addWidget(QLabel("Signal Overlays:"))
        overlay_layout = QGridLayout()
        self.overlay_checks = {}
        for index, signal_type in enumerate(SIGNAL_TYPES):
            check = QCheckBox(signal_type)
            check.setChecked(signal_type == "Sine")
            check.toggled.connect(self.schedule_update)
            self.overlay_checks[signal_type] = check
            overlay_layout.addWidget(check, index // 3, index % 3)
        effects_layout.addLayout(overlay_layout)
        effects_layout.addWidget(QLabel("Noise (RMS amplitude):"))
        self.noise_entry, self.noise_slider = self._make_slider("0.0", 0, 100, 0)
        self.noise_entry.setText("0.00")
        effects_layout.addWidget(self.noise_entry)
        effects_layout.addWidget(self.noise_slider)
        effects_layout.addWidget(QLabel("Distortion (0–100%):"))
        self.distortion_entry, self.distortion_slider = self._make_slider("0", 0, 100, 0)
        effects_layout.addWidget(self.distortion_entry)
        effects_layout.addWidget(self.distortion_slider)
        effects_layout.addStretch()

        self.update_btn = QPushButton("Update Plot")
        signal_layout.addWidget(self.update_btn)
        export_buttons_layout = QHBoxLayout()
        self.export_plot_btn = QPushButton("Save Plot")
        self.export_csv_btn = QPushButton("Export CSV")
        export_buttons_layout.addWidget(self.export_plot_btn)
        export_buttons_layout.addWidget(self.export_csv_btn)
        export_layout.addLayout(export_buttons_layout)
        export_layout.addWidget(QLabel("Save the current local plot as PNG/PDF or sampled data as CSV."))

        help_browser = QTextBrowser()
        help_browser.setOpenExternalLinks(False)
        help_browser.setHtml("""
            <h3>Using the dashboard</h3>
            <p><b>Signal</b>: choose a preset or waveform, then set its frequency,
            sample rate, and duration. Select Update Plot to refresh immediately;
            edits also update automatically.</p>
            <p><b>Filter</b>: choose a filter, set its cutoff(s), and adjust the order.
            The response plot shows magnitude and phase.</p>
            <p><b>Effects</b>: toggle waveform overlays and adjust noise/distortion.</p>
            <p><b>Plot navigation</b>: use the Matplotlib toolbar to pan, zoom, or
            restore the original view. Move the pointer over either time plot to see
            its time and amplitude coordinates.</p>
            <p><b>Validation</b>: warnings below the tabs explain how to resolve
            invalid settings, aliasing, or filter constraints.</p>
            <p>All signals are generated locally. The app does not connect to
            external devices or services.</p>
        """)
        help_layout.addWidget(help_browser)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(self.status_label)
        self.cursor_label = QLabel("Move over a time plot to inspect coordinates.")
        self.cursor_label.setWordWrap(True)
        sidebar_layout.addWidget(self.cursor_label)
        self.measurement_label = QLabel("Measurements:")
        self.measurement_label.setWordWrap(True)
        self.measurement_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        sidebar_layout.addWidget(self.measurement_label)

        plot_frame = QFrame()
        plot_frame.setMinimumWidth(400)
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(
            4, 1, figsize=(7, 12)
        )
        self.ax3_phase = self.ax3.twinx()

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        self.plot_resize_hint = QLabel(
            "Drag a gap between plots to resize; double-click to reset adjacent plots."
        )
        self.plot_resize_hint.setAlignment(Qt.AlignmentFlag.AlignRight)
        plot_layout.addWidget(self.plot_resize_hint)
        plot_layout.addWidget(self.canvas)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(sidebar_frame)
        self.main_splitter.addWidget(plot_frame)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([380, 850])
        main_layout.addWidget(self.main_splitter)
        self.canvas.mpl_connect("motion_notify_event", self.inspect_plot_position)
        self.canvas.mpl_connect("button_press_event", self._start_plot_resize)
        self.canvas.mpl_connect("motion_notify_event", self._update_plot_resize)
        self.canvas.mpl_connect("button_release_event", self._finish_plot_resize)
        self.canvas.mpl_connect("resize_event", self._resize_plot_canvas)
        self._active_plot_divider = None
        self._plot_layout_initialized = False
        self._plot_divider_artists = []
        self._hovered_plot_divider = None
        self._plot_divider_color = "#8090a0"
        self.cursors = []
        self._apply_theme()

        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(180)
        self.update_timer.timeout.connect(self.update_plot)

        self.preset_cb.currentTextChanged.connect(self.apply_preset)
        self.filter_cb.currentTextChanged.connect(self.on_filter_change)
        self.update_btn.clicked.connect(self.update_plot)
        self.export_plot_btn.clicked.connect(self.export_plot)
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.sig_type_cb.currentTextChanged.connect(self.schedule_update)
        self.order_combo.currentTextChanged.connect(self.schedule_update)

        for entry in (
            self.sig_freq_entry,
            self.samp_freq_entry,
            self.duration_entry,
            self.cutoff1_entry,
            self.cutoff2_entry,
            self.noise_entry,
            self.distortion_entry,
        ):
            entry.textChanged.connect(self.schedule_update)
            entry.returnPressed.connect(self.update_plot)

        self.sig_freq_slider.valueChanged.connect(
            lambda value: self._set_slider_entry(self.sig_freq_entry, value / 10.0)
        )
        self.samp_freq_slider.valueChanged.connect(
            lambda value: self._set_slider_entry(self.samp_freq_entry, value / 10.0)
        )
        self.duration_slider.valueChanged.connect(
            lambda value: self._set_slider_entry(self.duration_entry, value / 100.0)
        )
        self.cutoff1_slider.valueChanged.connect(
            lambda value: self._set_slider_entry(self.cutoff1_entry, value / 10.0)
        )
        self.cutoff2_slider.valueChanged.connect(
            lambda value: self._set_slider_entry(self.cutoff2_entry, value / 10.0)
        )
        self.noise_slider.valueChanged.connect(
            lambda value: self._set_slider_entry(self.noise_entry, value / 100.0)
        )
        self.distortion_slider.valueChanged.connect(
            lambda value: self._set_slider_entry(self.distortion_entry, value)
        )

        self.slider_bindings = (
            (self.sig_freq_entry, self.sig_freq_slider, 10),
            (self.samp_freq_entry, self.samp_freq_slider, 10),
            (self.duration_entry, self.duration_slider, 100),
            (self.cutoff1_entry, self.cutoff1_slider, 10),
            (self.cutoff2_entry, self.cutoff2_slider, 10),
            (self.noise_entry, self.noise_slider, 100),
            (self.distortion_entry, self.distortion_slider, 1),
        )
        for entry, slider, scale in self.slider_bindings:
            entry.textChanged.connect(
                lambda text, entry=entry, slider=slider, scale=scale:
                self._sync_slider_from_entry(text, slider, scale)
            )

        self.update_plot()

    def _make_slider(self, initial_text, minimum, maximum, initial):
        entry = QLineEdit(initial_text)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(initial)
        slider.valueChanged.connect(lambda: self.schedule_update())
        return entry, slider

    def _set_slider_entry(self, entry, value):
        if "." in str(value) or isinstance(value, float):
            entry.setText(f"{value:.2f}".rstrip("0").rstrip("."))
        else:
            entry.setText(str(value))

    def _sync_slider_from_entry(self, text, slider, scale):
        try:
            value = float(text)
            if not math.isfinite(value):
                return
            slider.blockSignals(True)
            slider.setValue(round(value * scale))
        except ValueError:
            return
        finally:
            slider.blockSignals(False)

    def _add_labeled_slider(self, layout, label, initial, minimum, maximum, initial_value):
        layout.addWidget(QLabel(label))
        entry, slider = self._make_slider(initial, minimum, maximum, initial_value)
        layout.addWidget(entry)
        layout.addWidget(slider)
        return entry, slider

    def schedule_update(self, *args):
        self.update_timer.start()

    def _apply_theme(self):
        if self.dark_theme:
            background, foreground = "#2b2b2b", "#ffffff"
            field_background, disabled_background = "#3b3b3b", "#222222"
            grid_color = "#b0b0b0"
        else:
            background, foreground = "#f4f6f8", "#20252b"
            field_background, disabled_background = "#ffffff", "#e5e7eb"
            grid_color = "#78828c"

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {background}; color: {foreground}; }}
            QLabel {{ font-size: 13px; }}
            QLineEdit {{
                background-color: {field_background}; color: {foreground};
                border: 1px solid #777; border-radius: 4px; padding: 5px;
            }}
            QLineEdit:disabled {{ background-color: {disabled_background}; color: #777; }}
            QComboBox {{
                background-color: {field_background}; color: {foreground};
                border: 1px solid #777; border-radius: 4px; padding: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {field_background}; color: {foreground};
                selection-background-color: #3daee9;
            }}
            QPushButton {{
                background-color: #3daee9; color: white;
                border-radius: 5px; padding: 8px; font-weight: bold; font-size: 13px;
            }}
            QPushButton:hover {{ background-color: #5bc0de; }}
            QSlider::groove:horizontal {{
                border: 1px solid #777; height: 6px; background: {disabled_background}; margin: 2px 0;
            }}
            QSlider::handle:horizontal {{
                background: #3daee9; width: 14px; margin: -5px 0; border-radius: 7px;
            }}
            QTabWidget::pane {{ border: 1px solid #777; }}
        """)
        self.theme_btn.setText("Switch to Light Theme" if self.dark_theme else "Switch to Dark Theme")
        if hasattr(self, "fig"):
            self.fig.patch.set_facecolor(background)
            self._plot_divider_color = "#8090a0" if self.dark_theme else "#65727e"
            for axis in self.fig.axes:
                axis.set_facecolor(background)
                axis.tick_params(colors=foreground)
                axis.xaxis.label.set_color(foreground)
                axis.yaxis.label.set_color(foreground)
                axis.title.set_color(foreground)
                axis.grid(True, color=grid_color, linestyle="--", alpha=0.3)
            self.canvas.draw_idle()
            self._set_plot_divider_hover(self._hovered_plot_divider)

    def toggle_theme(self):
        self.dark_theme = not self.dark_theme
        self._apply_theme()
        self.update_plot()

    def inspect_plot_position(self, event):
        if event.inaxes not in (self.ax1, self.ax2) or event.xdata is None or event.ydata is None:
            return
        plot_name = "Overlay" if event.inaxes is self.ax1 else "Output"
        self.cursor_label.setText(
            f"{plot_name}: time = {event.xdata:.5g} s, amplitude = {event.ydata:.5g}"
        )

    def _find_plot_divider(self, event):
        figure_width, figure_height = self.fig.bbox.width, self.fig.bbox.height
        if not figure_width or not figure_height or event.x is None or event.y is None:
            return None

        axes = sorted(
            (self.ax1, self.ax2, self.ax3, self.ax4),
            key=lambda axis: axis.get_position().y0,
            reverse=True,
        )
        first_position = axes[0].get_position()
        x_position = event.x / figure_width
        if not first_position.x0 <= x_position <= first_position.x1:
            return None

        y_position = event.y / figure_height
        for index, (upper_axis, lower_axis) in enumerate(zip(axes, axes[1:])):
            upper_position = upper_axis.get_position()
            lower_position = lower_axis.get_position()
            divider = (upper_position.y0 + lower_position.y1) / 2
            if abs(y_position - divider) * figure_height <= 16:
                return index, upper_axis, lower_axis
        return None

    def _create_plot_resize_handles(self):
        if self._plot_divider_artists:
            return
        for _ in range(3):
            divider_line = Line2D(
                [],
                [],
                transform=self.fig.transFigure,
                color=self._plot_divider_color,
                linewidth=0.8,
                alpha=0.7,
                solid_capstyle="round",
                clip_on=False,
                zorder=10,
            )
            grip_marks = Line2D(
                [],
                [],
                transform=self.fig.transFigure,
                color="#3daee9",
                linestyle="None",
                marker="|",
                markersize=6,
                markeredgewidth=1.2,
                visible=False,
                zorder=11,
            )
            self.fig.add_artist(divider_line)
            self.fig.add_artist(grip_marks)
            self._plot_divider_artists.append((divider_line, grip_marks))
        self._update_plot_resize_handles()

    def _update_plot_resize_handles(self):
        if not self._plot_divider_artists:
            return
        axes = sorted(
            (self.ax1, self.ax2, self.ax3, self.ax4),
            key=lambda axis: axis.get_position().y0,
            reverse=True,
        )
        for index, (upper_axis, lower_axis) in enumerate(zip(axes, axes[1:])):
            upper_position = upper_axis.get_position()
            lower_position = lower_axis.get_position()
            y_position = (upper_position.y0 + lower_position.y1) / 2
            left, right = upper_position.x0, upper_position.x1
            divider_line, grip_marks = self._plot_divider_artists[index]
            divider_line.set_data([left, right], [y_position, y_position])
            center = (left + right) / 2
            grip_width = min(0.02, (right - left) * 0.04)
            grip_marks.set_data(
                [center - grip_width, center, center + grip_width],
                [y_position] * 3,
            )
        self._set_plot_divider_hover(self._hovered_plot_divider)

    def _set_plot_divider_hover(self, index):
        self._hovered_plot_divider = index
        for handle_index, (divider_line, grip_marks) in enumerate(
            self._plot_divider_artists
        ):
            is_hovered = handle_index == index
            divider_line.set_color(
                "#3daee9"
                if is_hovered
                else ("#484f57" if self.dark_theme else "#c9ced3")
            )
            divider_line.set_linewidth(1.4 if is_hovered else 0.8)
            grip_marks.set_visible(is_hovered)
            grip_marks.set_color("#3daee9")

    def _apply_responsive_plot_sizing(self):
        for axis in (self.ax1, self.ax2, self.ax3, self.ax3_phase, self.ax4):
            bounds = axis.get_window_extent()
            scale = min(1.0, max(0.65, min(bounds.width / 500, bounds.height / 160)))
            axis.title.set_fontsize(10 * scale)
            axis.xaxis.label.set_size(9 * scale)
            axis.yaxis.label.set_size(9 * scale)
            axis.tick_params(labelsize=8 * scale)
            legend = axis.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(8 * scale)

    @staticmethod
    def _style_legend(legend, background, foreground, grid_color):
        legend.get_frame().set_facecolor(background)
        legend.get_frame().set_edgecolor(grid_color)
        for text in legend.get_texts():
            text.set_color(foreground)
        legend.set_draggable(True, use_blit=False)

    def _resize_plot_canvas(self, event):
        if not self._plot_layout_initialized:
            return
        self._update_plot_resize_handles()
        self._apply_responsive_plot_sizing()

    def _set_plot_divider(self, upper_axis, lower_axis, upper_position, lower_position, delta):
        minimum_height = 0.08
        minimum_delta = lower_position.y0 + minimum_height - lower_position.y1
        maximum_delta = upper_position.y1 - minimum_height - upper_position.y0
        delta = min(max(delta, minimum_delta), maximum_delta)

        upper_bottom = upper_position.y0 + delta
        lower_top = lower_position.y1 + delta
        upper_axis.set_position(
            (
                upper_position.x0,
                upper_bottom,
                upper_position.width,
                upper_position.y1 - upper_bottom,
            ),
            which="both",
        )
        lower_axis.set_position(
            (
                lower_position.x0,
                lower_position.y0,
                lower_position.width,
                lower_top - lower_position.y0,
            ),
            which="both",
        )
        if upper_axis is self.ax3 or lower_axis is self.ax3:
            self.ax3_phase.set_position(self.ax3.get_position(), which="both")
        self._update_plot_resize_handles()
        self._apply_responsive_plot_sizing()

    def _start_plot_resize(self, event):
        if event.button != 1 or getattr(self.toolbar, "mode", ""):
            return
        divider = self._find_plot_divider(event)
        if divider is None:
            return

        _, upper_axis, lower_axis = divider
        upper_position = upper_axis.get_position().frozen()
        lower_position = lower_axis.get_position().frozen()
        if event.dblclick:
            gap = upper_position.y0 - lower_position.y1
            equal_height = (
                upper_position.y1 - lower_position.y0 - gap
            ) / 2
            delta = upper_position.y1 - equal_height - upper_position.y0
            self._set_plot_divider(
                upper_axis, lower_axis, upper_position, lower_position, delta
            )
            self.canvas.draw_idle()
            return

        self._active_plot_divider = (
            upper_axis,
            lower_axis,
            upper_position,
            lower_position,
            event.y,
        )

    def _update_plot_resize(self, event):
        if self._active_plot_divider is not None:
            figure_height = self.fig.bbox.height
            if not figure_height or event.y is None:
                return
            (
                upper_axis,
                lower_axis,
                upper_position,
                lower_position,
                start_y,
            ) = self._active_plot_divider
            delta = (event.y - start_y) / figure_height
            self._set_plot_divider(
                upper_axis, lower_axis, upper_position, lower_position, delta
            )
            self.canvas.draw_idle()
            return

        if getattr(self.toolbar, "mode", ""):
            return
        divider = self._find_plot_divider(event)
        divider_index = divider[0] if divider is not None else None
        if divider is not None:
            self.canvas.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        self._set_plot_divider_hover(divider_index)

    def _finish_plot_resize(self, event):
        if self._active_plot_divider is not None:
            self._active_plot_divider = None
            self.canvas.draw_idle()
            self._update_plot_resize(event)

    def on_filter_change(self, choice):
        enabled = choice != "None"
        band = choice in ("Bandpass", "Bandstop")
        self.cutoff1_entry.setEnabled(enabled)
        self.cutoff1_slider.setEnabled(enabled)
        self.cutoff2_entry.setEnabled(band)
        self.cutoff2_slider.setEnabled(band)
        self.schedule_update()

    def apply_preset(self, name):
        preset = self.PRESETS.get(name)
        if preset is None:
            return

        self.sig_type_cb.setCurrentText(preset["signal"])
        self.sig_freq_entry.setText(preset["frequency"])
        self.samp_freq_entry.setText(preset["sampling"])
        self.duration_entry.setText(preset["duration"])
        self.filter_cb.setCurrentText(preset["filter"])
        self.cutoff1_entry.setText(preset["cutoff1"])
        self.cutoff2_entry.setText(preset["cutoff2"])
        self.noise_slider.setValue(round(preset["noise"] * 100))
        self.distortion_slider.setValue(preset["distortion"])
        self.schedule_update()

    @staticmethod
    def _parse_finite_number(text, label):
        try:
            value = float(text)
        except ValueError as err:
            raise ValueError(f"{label} must be a number.") from err
        if not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number.")
        return value

    def set_status(self, message, severity="info"):
        colors = {
            "error": "#ff5555",
            "warning": "#ffaa00",
            "info": "#55ff55" if self.dark_theme else "#237a35",
        }
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            f"color: {colors[severity]}; font-weight: bold;"
        )

    def update_plot(self):
        try:
            signal_frequency = self._parse_finite_number(
                self.sig_freq_entry.text(), "Signal frequency"
            )
            sampling_frequency = self._parse_finite_number(
                self.samp_freq_entry.text(), "Sampling rate"
            )
            duration = self._parse_finite_number(
                self.duration_entry.text(), "Duration"
            )
            noise_amount = self._parse_finite_number(
                self.noise_entry.text(), "Noise amount"
            )
            distortion_percent = self._parse_finite_number(
                self.distortion_entry.text(), "Distortion"
            )

            if signal_frequency < 0:
                raise ValueError("Signal frequency must be 0 Hz or higher.")
            if signal_frequency > 1_000_000:
                raise ValueError("Signal frequency must not exceed 1,000,000 Hz.")
            if sampling_frequency <= 0:
                raise ValueError("Sampling rate must be greater than 0 Sa/s.")
            if sampling_frequency > 1_000_000:
                raise ValueError("Sampling rate must not exceed 1,000,000 Sa/s.")
            if duration <= 0:
                raise ValueError("Duration must be greater than 0 seconds.")
            if duration > 3600:
                raise ValueError("Duration must not exceed 3,600 seconds.")
            if noise_amount < 0 or noise_amount > 1:
                raise ValueError("Noise amount must be between 0 and 1.")
            if distortion_percent < 0 or distortion_percent > 100:
                raise ValueError("Distortion must be between 0% and 100%.")
            distortion_amount = distortion_percent / 100.0

            filter_type = self.filter_cb.currentText()
            cutoff1 = (
                self._parse_finite_number(self.cutoff1_entry.text(), "Cutoff 1")
                if filter_type != "None"
                else 0.0
            )
            cutoff2 = (
                self._parse_finite_number(self.cutoff2_entry.text(), "Cutoff 2")
                if filter_type in ("Bandpass", "Bandstop")
                else 0.0
            )
            filter_order = int(self.order_combo.currentText())
            estimated_sample_count = duration * sampling_frequency
            if not math.isfinite(estimated_sample_count) or estimated_sample_count > 1_000_000:
                raise ValueError(
                    "This setting creates more than 1,000,000 samples. "
                    "Reduce duration or sampling rate."
                )
            sample_count = max(1, int(round(estimated_sample_count)))
            filter_length = (
                2 * filter_order + 1
                if filter_type in ("Bandpass", "Bandstop")
                else filter_order + 1
            )
            if filter_type != "None" and sample_count <= 3 * filter_length:
                raise ValueError(
                    f"Order {filter_order} {filter_type.lower()} filtering needs more "
                    f"than {3 * filter_length} samples. Increase duration or "
                    "sampling rate, or choose no filter."
                )
        except ValueError as err:
            self.set_status(f"Invalid input: {err}", "error")
            return

        continuous_sampling_frequency = 1000.0
        continuous_count = min(
            1_000_000, max(1000, int(duration * continuous_sampling_frequency))
        )
        t_cont = np.linspace(0, duration, continuous_count)
        t_samp = np.arange(sample_count) / sampling_frequency

        primary_type = self.sig_type_cb.currentText()
        y_cont_primary = generate_signal(t_cont, signal_frequency, primary_type)
        y_samp_primary = generate_signal(t_samp, signal_frequency, primary_type)
        y_samp_clean = y_samp_primary.copy()
        y_cont_primary = add_noise_and_distortion(
            y_cont_primary, noise_amount, distortion_amount
        )
        y_samp_primary = add_noise_and_distortion(
            y_samp_primary, noise_amount, distortion_amount
        )

        try:
            y_cont_filtered, _, _ = apply_digital_filter(
                y_cont_primary,
                continuous_sampling_frequency,
                filter_type,
                cutoff1,
                cutoff2,
                filter_order,
            )
            y_samp_filtered, b, a = apply_digital_filter(
                y_samp_primary,
                sampling_frequency,
                filter_type,
                cutoff1,
                cutoff2,
                filter_order,
            )
        except ValueError as err:
            self.set_status(
                f"{err} Review the cutoff, filter order, and sample count.",
                "error",
            )
            return

        try:
            spectrum_frequencies, spectrum_amplitude = compute_spectrum(
                y_samp_filtered, sampling_frequency
            )
        except ValueError as err:
            self.set_status(f"Invalid input: {err}", "error")
            return

        try:
            measurements = compute_measurements(y_samp_filtered, sampling_frequency)
            measurement_text = (
                "Measurements:\n"
                f"RMS: {measurements['rms']:.6g}\n"
                f"Peak: {measurements['peak']:.6g}\n"
                f"Peak-to-peak: {measurements['peak_to_peak']:.6g}\n"
                f"Mean: {measurements['mean']:.6g}\n"
                f"Crest factor: {measurements['crest_factor']:.6g}\n"
                f"Peak frequency: {measurements['peak_frequency_hz']:g} Hz"
            )
            if noise_amount > 0:
                try:
                    snr_db = compute_snr_db(y_samp_clean, y_samp_primary)
                    measurement_text += f"\nSNR: {snr_db:.6g} dB"
                except ValueError:
                    measurement_text += "\nSNR: unavailable"
            self.measurement_label.setText(measurement_text)
        except ValueError as err:
            self.set_status(f"Invalid input: {err}", "error")
            return

        minimum_sampling_frequency = 2 * signal_frequency
        if sampling_frequency <= minimum_sampling_frequency:
            diagnostic = (
                f"Aliasing risk: sampling rate must be greater than "
                f"{minimum_sampling_frequency:g} Sa/s for this fundamental."
            )
            severity = "warning"
        else:
            diagnostic = "Nyquist criterion met for the signal's fundamental."
            severity = "info"
        primary_type = self.sig_type_cb.currentText()
        if primary_type not in ("Sine", "Cosine") and sampling_frequency > minimum_sampling_frequency:
            diagnostic += " Higher harmonics of this waveform may still alias."
            severity = "warning"
        self.set_status(diagnostic, severity)

        background = "#2b2b2b" if self.dark_theme else "#f4f6f8"
        foreground = "#ffffff" if self.dark_theme else "#20252b"
        grid_color = "#b0b0b0" if self.dark_theme else "#78828c"
        self.ax1.clear()
        for axis in (self.ax1, self.ax2, self.ax3, self.ax3_phase, self.ax4):
            axis.set_facecolor(background)
            axis.tick_params(colors=foreground)
            axis.xaxis.label.set_color(foreground)
            axis.yaxis.label.set_color(foreground)
            axis.title.set_color(foreground)
            axis.grid(True, color=grid_color, linestyle="--", alpha=0.3)
        self.fig.patch.set_facecolor(background)
        primary_type = self.sig_type_cb.currentText()
        overlay_colors = {
            "Sine": "#3daee9",
            "Cosine": "#55d98d",
            "Square": "#ffaa00",
            "Sawtooth": "#d65db1",
            "Triangular": "#00b8d9",
        }
        for signal_type, check in self.overlay_checks.items():
            if check.isChecked() or signal_type == primary_type:
                y_overlay = generate_signal(t_cont, signal_frequency, signal_type)
                y_overlay = add_noise_and_distortion(
                    y_overlay, noise_amount, distortion_amount
                )
                label = (
                    f"{signal_type} (primary)"
                    if signal_type == primary_type
                    else signal_type
                )
                self.ax1.plot(
                    t_cont,
                    y_overlay,
                    color=overlay_colors[signal_type],
                    linewidth=1.8,
                    alpha=0.9,
                    label=label,
                )
        if filter_type != "None":
            self.ax1.plot(
                t_cont,
                y_cont_filtered,
                color=foreground,
                linewidth=2.2,
                linestyle="--",
                label=f"{filter_type} filtered",
            )
        self.ax1.set_title("Signal Overlays", fontweight="bold", color=foreground)
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude")
        if self.ax1.get_legend_handles_labels()[0]:
            legend = self.ax1.legend(loc="best", fontsize=8, framealpha=0.8)
            self._style_legend(legend, background, foreground, grid_color)
        self.ax1.grid(True, linestyle="--", alpha=0.3)

        self.ax2.clear()
        self.ax2.plot(t_samp, y_samp_filtered, color="#3daee9", alpha=0.65)
        self.ax2.stem(
            t_samp,
            y_samp_filtered,
            linefmt="#ff5555",
            markerfmt="ro",
            basefmt=foreground,
        )
        self.ax2.set_title(
            f"Output Signal (Sampled @ {sampling_frequency:g} Hz)",
            fontweight="bold",
            color=foreground,
        )
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Amplitude")
        self.ax2.grid(True, linestyle="--", alpha=0.3)

        self.ax3.clear()
        self.ax3_phase.clear()
        if b is not None and a is not None:
            frequencies, magnitude_db, phase_degrees = calculate_frequency_response(
                b, a, sampling_frequency
            )
            line_mag = self.ax3.plot(
                frequencies, magnitude_db, color="#3daee9", linewidth=2, label="Magnitude"
            )
            line_phase = self.ax3_phase.plot(
                frequencies,
                phase_degrees,
                color="#ffaa00",
                linewidth=1.5,
                linestyle="--",
                label="Phase",
            )
            self.ax3.set_ylabel("Magnitude (dB)", color="#3daee9", fontweight="bold")
            self.ax3.tick_params(axis="y", colors="#3daee9")
            self.ax3.set_ylim(
                max(-80, np.min(magnitude_db) - 5),
                max(5, np.max(magnitude_db) + 5),
            )
            self.ax3_phase.set_ylabel("Phase (Degrees)", color="#ffaa00", fontweight="bold")
            self.ax3_phase.yaxis.set_label_position("right")
            self.ax3_phase.tick_params(axis="y", colors="#ffaa00")
            self.ax3.set_title(
                f"{filter_type} Filter Frequency Response (Magnitude & Phase)",
                fontweight="bold",
                color=foreground,
            )
            lines = line_mag + line_phase
            labels = [line.get_label() for line in lines]
            legend = self.ax3.legend(
                lines, labels, loc="best", fontsize=8, framealpha=0.8
            )
            self._style_legend(legend, background, foreground, grid_color)
        else:
            self.ax3.set_title(
                "Filter Frequency Response (No Filter Selected)",
                fontweight="bold",
                color=foreground,
            )
            self.ax3.set_ylabel("Magnitude (dB)", color=foreground)
            self.ax3_phase.set_ylabel("Phase (Degrees)", color=foreground)
            self.ax3_phase.yaxis.set_label_position("right")

        self.ax3.set_xlabel("Frequency (Hz)")
        self.ax3.set_xlim(0, sampling_frequency / 2)
        self.ax3.grid(True, color=grid_color, linestyle="--", alpha=0.3)

        self.ax4.clear()
        self.ax4.set_facecolor(background)
        self.ax4.tick_params(colors=foreground)
        peak_index = np.argmax(spectrum_amplitude)
        peak_frequency = spectrum_frequencies[peak_index]
        self.ax4.plot(
            spectrum_frequencies,
            spectrum_amplitude,
            color="#3daee9",
            linewidth=1.8,
        )
        self.ax4.set_title(
            f"Output Spectrum (Peak: {peak_frequency:g} Hz)",
            fontweight="bold",
            color=foreground,
        )
        self.ax4.set_xlabel("Frequency (Hz)", color=foreground)
        self.ax4.set_ylabel("Amplitude", color=foreground)
        self.ax4.set_xlim(0, sampling_frequency / 2)
        self.ax4.grid(True, color=grid_color, linestyle="--", alpha=0.3)

        self.last_export_data = (t_samp, y_samp_primary, y_samp_filtered)
        for cursor in self.cursors:
            cursor.disconnect_events()
        self.cursors = [
            Cursor(axis, useblit=False, color="#ff5555", linewidth=0.8)
            for axis in (self.ax1, self.ax2)
        ]
        if not self._plot_layout_initialized:
            self.fig.tight_layout(pad=3.0)
            self._plot_layout_initialized = True
            self._create_plot_resize_handles()
        self._update_plot_resize_handles()
        self._apply_responsive_plot_sizing()
        self.canvas.draw()

    def export_plot(self):
        if self.last_export_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "signal_dashboard",
            "PNG Image (*.png);;PDF Document (*.pdf)",
        )
        if path:
            try:
                save_plot(path, self.fig)
            except (OSError, ValueError) as err:
                self.set_status(f"Could not save plot: {err}", "error")
            else:
                self.set_status(f"Plot saved to {path}", "info")

    def export_csv(self):
        if self.last_export_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Signal Data",
            "signal_data.csv",
            "CSV File (*.csv)",
        )
        if path:
            t, original, filtered = self.last_export_data
            try:
                save_signal_csv(path, t, original, filtered)
            except (OSError, ValueError) as err:
                self.set_status(f"Could not export CSV: {err}", "error")
            else:
                self.set_status(f"Signal data exported to {path}", "info")

    def closeEvent(self, event):
        for cursor in self.cursors:
            cursor.disconnect_events()
        plt.close(self.fig)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalDashboard()
    window.show()
    sys.exit(app.exec())
