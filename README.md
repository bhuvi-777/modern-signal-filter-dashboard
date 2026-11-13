# Modern Signal & Filter Dashboard

A PyQt6-based dashboard for generating common signal waveforms and applying digital filters in real time.

## Overview

This project lets you:

- Generate waveforms such as sine, cosine, square, sawtooth, and triangular signals
- Set signal frequency, sampling rate, and duration
- Apply digital filters including lowpass, highpass, bandpass, and bandstop
- Visualize the original versus filtered signal in the time and frequency domains
- Compare input and filtered one-sided FFT amplitude spectra
- Check whether the selected sampling rate satisfies the Nyquist criterion

## Features

- Dark-themed modern UI
- Interactive parameter input panel
- Dynamic filter cutoff controls based on selected filter type
- Real-time waveform plotting using Matplotlib and PyQt6
- Frequency-domain plotting with frequency bins in Hz and amplitude-scaled FFT
- Alias warning when the sampling rate is too low

## Requirements

Install the required Python packages:

```bash
pip install PyQt6 numpy matplotlib scipy
```

## Run the Application

From the project directory, run:

```bash
python pyqt_dashboard.py
```

## Controls

### Signal Type
Choose the waveform used for signal generation.

### Signal Frequency (Hz)
Sets the frequency of the generated signal.

### Sampling Rate (Sa/s)
Defines how many samples are taken per second.

### Time Duration (s)
Determines the total time length of the generated signal.

### Frequency-Domain Spectrum
The third plot shows a one-sided, Hann-windowed FFT amplitude spectrum of the sampled signal.

The Hann window reduces spectral leakage, and its coherent gain is accounted for in the
amplitude scaling. Zero-padding smooths the plotted curve but does not increase the true
frequency resolution, which is shown in the plot title. When a filter is selected, the
input and filtered spectra are overlaid.

### Filter Type
Select the filter to apply:

- None
- Lowpass
- Highpass
- Bandpass
- Bandstop

### Cutoff 1 / Cutoff 2 (Hz)
Used depending on the filter type selected.

## Notes

- The Nyquist theorem is checked automatically.
- Filter cutoff values must stay below the Nyquist frequency.
- A warning is shown when aliasing may occur.

## Potential Improvements

The current app is already functional, but the following enhancements would make it more powerful and useful:

1. Filter response visualization (magnitude and phase)
2. Live sliders for parameters
3. Noise and distortion controls
4. Multiple signal overlays
5. Preset configurations for common DSP use cases
6. Save/export of plots and signal data
7. Real-time updates while editing values
8. Filter order adjustment
9. Better validation and diagnostic hints
10. Tabbed layout for more organized controls
11. User-friendly help/instruction panel
12. Light/dark theme toggle
13. Zoom and cursor inspection on plots
14. Export to CSV/PNG/PDF

## File Structure

```text
pyqt_git/
├── pyqt_dashboard.py
├── README.md
```

## License

This project is provided as a learning/demo project for signal processing and PyQt6 GUI development.


Learning Git: Understanding commit and push.