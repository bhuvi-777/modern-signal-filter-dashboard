import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import butter, filtfilt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QComboBox, 
                             QPushButton, QMessageBox, QFrame, QGridLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class SignalDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Modern Signal & Filter Dashboard (PyQt6)")
        self.resize(1100, 850)

        # Apply Dark Theme Stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: white; }
            QLabel { font-size: 13px; }
            QLineEdit { 
                background-color: #3b3b3b; color: white; 
                border: 1px solid #555; border-radius: 4px; padding: 5px; 
            }
            QLineEdit:disabled { background-color: #222; color: #777; }
            QComboBox { 
                background-color: #3b3b3b; color: white; 
                border: 1px solid #555; border-radius: 4px; padding: 5px; 
            }
            QComboBox QAbstractItemView {
                background-color: #3b3b3b; color: white; selection-background-color: #3daee9;
            }
            QPushButton { 
                background-color: #3daee9; color: white; 
                border-radius: 5px; padding: 8px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background-color: #5bc0de; }
        """)

        # Main Central Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(20)

        # --- Left Frame: Controls ---
        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setSpacing(12)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Title
        title_label = QLabel("Signal & Filter Params")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        sidebar_layout.addWidget(title_label)
        sidebar_layout.addSpacing(10)

        # 1. Signal Type
        sidebar_layout.addWidget(QLabel("Signal Type:"))
        self.sig_type_cb = QComboBox()
        self.sig_type_cb.addItems(["Sine", "Cosine", "Square", "Sawtooth", "Triangular"])
        sidebar_layout.addWidget(self.sig_type_cb)

        # 2. Signal Frequency
        sidebar_layout.addWidget(QLabel("Signal Frequency (Hz):"))
        self.sig_freq_entry = QLineEdit("5.0")
        sidebar_layout.addWidget(self.sig_freq_entry)

        # 3. Sampling Frequency
        sidebar_layout.addWidget(QLabel("Sampling Rate (Sa/s):"))
        self.samp_freq_entry = QLineEdit("50.0")
        sidebar_layout.addWidget(self.samp_freq_entry)

        # 4. Duration
        sidebar_layout.addWidget(QLabel("Time Duration (s):"))
        self.duration_entry = QLineEdit("1.0")
        sidebar_layout.addWidget(self.duration_entry)
        sidebar_layout.addSpacing(10)

        # 5. Filter Type
        sidebar_layout.addWidget(QLabel("Filter Type:"))
        self.filter_cb = QComboBox()
        self.filter_cb.addItems(["None", "Lowpass", "Highpass", "Bandpass", "Bandstop"])
        sidebar_layout.addWidget(self.filter_cb)

        # 6. Cutoff Frequencies (Grid Layout for side-by-side)
        cutoff_layout = QGridLayout()
        cutoff_layout.addWidget(QLabel("Cutoff 1 (Hz):"), 0, 0)
        self.cutoff1_entry = QLineEdit("15.0")
        self.cutoff1_entry.setEnabled(False)
        cutoff_layout.addWidget(self.cutoff1_entry, 1, 0)

        cutoff_layout.addWidget(QLabel("Cutoff 2 (Hz):"), 0, 1)
        self.cutoff2_entry = QLineEdit("25.0")
        self.cutoff2_entry.setEnabled(False)
        cutoff_layout.addWidget(self.cutoff2_entry, 1, 1)
        sidebar_layout.addLayout(cutoff_layout)

        # Spacer to push button to the bottom
        sidebar_layout.addStretch()

        # Update Button
        self.update_btn = QPushButton("Update Plot")
        sidebar_layout.addWidget(self.update_btn)

        # Nyquist Warning Label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(self.status_label)

        main_layout.addWidget(sidebar_frame)

        # --- Right Frame: Plotting ---
        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Setup dark-themed Matplotlib figure
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(7, 6))
        self.fig.patch.set_facecolor('#2b2b2b') 
        
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor('#2b2b2b')
            ax.grid(True, linestyle='--', alpha=0.3)

        self.fig.tight_layout(pad=4.0)
        
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_frame, stretch=1)

        # --- Event Connections ---
        self.filter_cb.currentTextChanged.connect(self.on_filter_change)
        self.update_btn.clicked.connect(self.update_plot)
        self.sig_type_cb.currentTextChanged.connect(self.update_plot)
        
        for entry in [self.sig_freq_entry, self.samp_freq_entry, self.duration_entry,
                      self.cutoff1_entry, self.cutoff2_entry]:
            entry.returnPressed.connect(self.update_plot)

        # Render initial plot
        self.update_plot()

    def on_filter_change(self, choice):
        """Dynamically enable/disable cutoff inputs based on filter selection."""
        if choice == "None":
            self.cutoff1_entry.setEnabled(False)
            self.cutoff2_entry.setEnabled(False)
        elif choice in ["Lowpass", "Highpass"]:
            self.cutoff1_entry.setEnabled(True)
            self.cutoff2_entry.setEnabled(False)
        else: # Bandpass or Bandstop
            self.cutoff1_entry.setEnabled(True)
            self.cutoff2_entry.setEnabled(True)
        self.update_plot()

    def generate_signal(self, t, sig_freq, sig_type):
        if sig_type == "Sine":
            return np.sin(2 * np.pi * sig_freq * t)
        elif sig_type == "Cosine":
            return np.cos(2 * np.pi * sig_freq * t)
        elif sig_type == "Square":
            return np.sign(np.sin(2 * np.pi * sig_freq * t))
        elif sig_type == "Sawtooth":
            return 2 * (t * sig_freq - np.floor(0.5 + t * sig_freq))
        elif sig_type == "Triangular":
            return 2 * np.abs(2 * (t * sig_freq - np.floor(0.5 + t * sig_freq))) - 1
        return np.zeros_like(t)

    def apply_digital_filter(self, signal, samp_freq, filter_type, cutoff1, cutoff2):
        if filter_type == "None" or len(signal) < 15:
            return signal
            
        nyquist = 0.5 * samp_freq
        order = 4
        
        try:
            if filter_type in ["Lowpass", "Highpass"]:
                if cutoff1 >= nyquist:
                    raise ValueError(f"{filter_type} cutoff ({cutoff1}Hz) exceeds Nyquist ({nyquist}Hz).")
                if cutoff1 <= 0:
                    raise ValueError("Cutoff must be > 0.")
                b, a = butter(order, cutoff1/nyquist, btype=filter_type.lower())
            
            elif filter_type in ["Bandpass", "Bandstop"]:
                low, high = min(cutoff1, cutoff2), max(cutoff1, cutoff2)
                if high >= nyquist:
                    raise ValueError(f"High cutoff ({high}Hz) exceeds Nyquist ({nyquist}Hz).")
                if low <= 0:
                    raise ValueError("Low cutoff must be > 0.")
                b, a = butter(order, [low/nyquist, high/nyquist], btype=filter_type.lower())
                
            return filtfilt(b, a, signal)
            
        except ValueError as err:
            raise ValueError(f"Filter config error for sampling freq={samp_freq}Hz:\n{str(err)}")

    def update_plot(self):
        try:
            signal_frequency = float(self.sig_freq_entry.text())
            sampling_frequency = float(self.samp_freq_entry.text())
            duration = float(self.duration_entry.text())
            sig_type = self.sig_type_cb.currentText()
            
            filter_type = self.filter_cb.currentText()
            cutoff1 = float(self.cutoff1_entry.text()) if filter_type != "None" else 0.0
            cutoff2 = float(self.cutoff2_entry.text()) if filter_type in ["Bandpass", "Bandstop"] else 0.0

            if sampling_frequency <= 0 or duration <= 0:
                raise ValueError("Sampling rate and duration must be positive.")
                
        except ValueError as err:
            QMessageBox.critical(self, "Input Error", f"Invalid input.\n{err}")
            return

        # Continuous data generation
        continuous_sampling_frequency = 1000.0 / duration
        t_cont = np.linspace(0, duration, 1000)
        y_cont = self.generate_signal(t_cont, signal_frequency, sig_type)

        # Sampled data generation
        t_samp = np.arange(0, duration + 1e-9, 1/sampling_frequency)
        y_samp = self.generate_signal(t_samp, signal_frequency, sig_type)

        # Apply Filters
        try:
            y_cont_filt = self.apply_digital_filter(y_cont, continuous_sampling_frequency, filter_type, cutoff1, cutoff2)
            y_samp_filt = self.apply_digital_filter(y_samp, sampling_frequency, filter_type, cutoff1, cutoff2)
        except ValueError as err:
            QMessageBox.critical(self, "Filter Error", str(err))
            return

        # Nyquist Theorem Check Update
        minimum_sampling_frequency = 2 * signal_frequency
        if sampling_frequency <= minimum_sampling_frequency:
            self.status_label.setText(f"WARNING: Aliasing! (Sampling frequency must be > "f"{minimum_sampling_frequency} Hz)")
            self.status_label.setStyleSheet("color: #ff5555; font-weight: bold;")
        else:
            self.status_label.setText("Nyquist criterion met. No aliasing detected.")
            self.status_label.setStyleSheet("color: #55ff55; font-weight: bold;")

        # Update Top Plot
        self.ax1.clear()
        if filter_type != "None":
            self.ax1.plot(t_cont, y_cont, color='gray', alpha=0.3, label="Original")
            self.ax1.plot(t_cont, y_cont_filt, color='#3daee9', linewidth=2, label="Filtered")
            self.ax1.legend(loc="upper right", fontsize=8)
        else:
            self.ax1.plot(t_cont, y_cont, color='#3daee9', linewidth=2)
            
        self.ax1.set_title(f"Input Signal ({sig_type} @ {signal_frequency} Hz)",fontweight='bold',color='white',)
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.set_ylim(-1.5, 1.5)
        self.ax1.grid(True, linestyle='--', alpha=0.3)

        # Update Bottom Plot
        self.ax2.clear()
        self.ax2.plot(t_cont, y_cont_filt, color='#3daee9', alpha=0.2)
        self.ax2.stem(t_samp, y_samp_filt, linefmt='#ff5555', markerfmt='ro', basefmt='white')
        self.ax2.set_title(f"Output Signal (Sampled @ {sampling_frequency} Hz) + Digital Filter", fontweight='bold', color='white',)
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Amplitude")
        self.ax2.set_ylim(-1.5, 1.5)
        self.ax2.grid(True, linestyle='--', alpha=0.3)

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalDashboard()
    window.show()
    sys.exit(app.exec())