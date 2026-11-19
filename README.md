# Modern Signal & Filter Dashboard

A PyQt6-based dashboard for generating common signal waveforms and applying digital filters in real time.

## Overview

This project lets you:

- Generate waveforms such as sine, cosine, square, sawtooth, and triangular signals
- Set signal frequency, sampling rate, and duration
- Apply digital filters including lowpass, highpass, bandpass, and bandstop
- Visualize generated and filtered signals in the time domain
- Compare original and filtered signals
- Inspect filter magnitude and phase response
- Check whether the selected sampling rate satisfies the Nyquist criterion

## Features

- Dark-themed modern UI
- Interactive parameter input panel
- Dynamic filter cutoff controls based on selected filter type
- Real-time waveform plotting using Matplotlib and PyQt6
- Alias warning when the sampling rate is too low
- Filter magnitude and phase response visualization
- Live sliders and debounced real-time updates
- Noise and distortion controls
- Multiple signal overlays
- Preset configurations for common DSP scenarios
- PNG/PDF plot export and CSV signal-data export
- Adjustable digital filter order
- Tabbed controls with an in-app help panel
- Light/dark theme toggle
- Plot pan/zoom toolbar, crosshair, and coordinate readout
- Input validation with actionable diagnostic hints
- Local-only signal generation; no external connections

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

### Filter Frequency Response
The bottom plot shows the selected filter's magnitude in dB and phase in degrees
across frequencies up to the Nyquist limit. It updates when filter type, cutoff,
or filter order changes.

### Filter Type
Select the filter to apply:

- None
- Lowpass
- Highpass
- Bandpass
- Bandstop

### Cutoff 1 / Cutoff 2 (Hz)
Used depending on the filter type selected.

### Plot navigation
Use the Matplotlib toolbar to pan, zoom, and reset the plot view. A crosshair is
available on the time-domain plots; move the pointer over a plot to inspect time
and amplitude in the coordinate readout.

### Tabs and theme
Signal, Filter, Effects, Export, and Help controls are organized in tabs. Use the
button above the tabs to switch between dark and light themes.

## Notes

- The Nyquist theorem is checked automatically.
- Filter cutoff values must stay below the Nyquist frequency.
- A warning is shown when aliasing may occur.

## Improvements

1. ~~Filter response visualization (magnitude and phase)~~
2. ~~Live sliders for parameters~~
3. ~~Noise and distortion controls~~
4. ~~Multiple signal overlays~~
5. ~~Preset configurations for common DSP use cases~~
6. ~~Save/export of plots and signal data~~
7. ~~Real-time updates while editing values~~
8. ~~Filter order adjustment~~
9. ~~Better validation and diagnostic hints~~
10. ~~Tabbed layout for more organized controls~~
11. ~~User-friendly help/instruction panel~~
12. ~~Light/dark theme toggle~~
13. ~~Zoom and cursor inspection on plots~~
14. ~~Plot and signal-data export (CSV/PNG/PDF)~~

The dashboard works with locally generated signals only. It does not connect to
external devices or services.

## File Structure

```text
├── pyqt_dashboard.py
├── dsp_utils.py
├── export_utils.py
├── README.md
```

## License

This project is provided as a learning/demo project for signal processing and PyQt6 GUI development.


Learning Git: Understanding commit and push.