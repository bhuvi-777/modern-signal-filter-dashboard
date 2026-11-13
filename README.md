# Modern Signal & Filter Dashboard

A PyQt6-based dashboard for generating common signal waveforms and applying digital filters in real time.

## Overview

This project lets you:

- Generate waveforms such as sine, cosine, square, sawtooth, and triangular signals
- Set signal frequency, sampling rate, and duration
- Apply digital filters including lowpass, highpass, bandpass, and bandstop
- Visualize the original versus filtered signal
- Check whether the selected sampling rate satisfies the Nyquist criterion

## Features

- Dark-themed modern UI
- Interactive parameter input panel
- Dynamic filter cutoff controls based on selected filter type
- Real-time waveform plotting using Matplotlib and PyQt6
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

1. FFT / frequency-domain plotting
2. Filter response visualization (magnitude and phase)
3. Live sliders for parameters
4. Noise and distortion controls
5. Multiple signal overlays
6. Preset configurations for common DSP use cases
7. Save/export of plots and signal data
8. Real-time updates while editing values
9. Filter order adjustment
10. Better validation and diagnostic hints
11. Tabbed layout for more organized controls
12. User-friendly help/instruction panel
13. Light/dark theme toggle
14. Zoom and cursor inspection on plots
15. Export to CSV/PNG/PDF

## File Structure

```text
pyqt_git/
├── pyqt_dashboard.py
├── README.md
```

## License

This project is provided as a learning/demo project for signal processing and PyQt6 GUI development.
