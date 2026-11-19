import numpy as np
from scipy.signal import (
    bessel,
    butter,
    cheby1,
    cheby2,
    ellip,
    filtfilt,
    firwin,
    freqz,
    lfilter,
)
from scipy.signal import spectrogram


SIGNAL_TYPES = ("Sine", "Cosine", "Square", "Sawtooth", "Triangular")


def generate_signal(t, frequency, signal_type):
    if signal_type == "Sine":
        return np.sin(2 * np.pi * frequency * t)
    if signal_type == "Cosine":
        return np.cos(2 * np.pi * frequency * t)
    if signal_type == "Square":
        return np.sign(np.sin(2 * np.pi * frequency * t))
    if signal_type == "Sawtooth":
        return 2 * (t * frequency - np.floor(0.5 + t * frequency))
    if signal_type == "Triangular":
        return 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1
    return np.zeros_like(t)


def add_noise_and_distortion(signal, noise_amount=0.0, distortion_amount=0.0, seed=42):
    if not np.isfinite(noise_amount) or noise_amount < 0:
        raise ValueError("Noise amount must be non-negative.")
    if not np.isfinite(distortion_amount) or not 0 <= distortion_amount <= 1:
        raise ValueError("Distortion amount must be between 0 and 1.")
    result = np.asarray(signal, dtype=float).copy()
    if noise_amount > 0:
        rng = np.random.default_rng(seed)
        result += rng.normal(0.0, noise_amount, size=result.shape)
    if distortion_amount > 0:
        gain = 1.0 + 9.0 * distortion_amount
        driven = result * gain
        result = np.tanh(driven) / np.tanh(gain)
    return result


def apply_digital_filter(
    signal,
    sampling_frequency,
    filter_type,
    cutoff1,
    cutoff2,
    order=4,
    family="Butterworth",
    ripple_db=1.0,
    stopband_db=40.0,
    fir_window="Hamming",
    fir_taps=101,
):
    if filter_type == "None":
        return np.asarray(signal, dtype=float).copy(), None, None

    if family not in ("Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic", "Bessel", "FIR"):
        raise ValueError(f"Unknown filter family: {family}.")
    if ripple_db is not None and (not np.isfinite(ripple_db) or ripple_db <= 0):
        if family in ("Chebyshev I", "Elliptic"):
            raise ValueError("Ripple must be greater than 0 dB.")
    if stopband_db is not None and (not np.isfinite(stopband_db) or stopband_db <= 0):
        if family in ("Chebyshev II", "Elliptic"):
            raise ValueError("Stopband attenuation must be greater than 0 dB.")
    if family == "FIR":
        if not isinstance(fir_taps, (int, np.integer)):
            raise ValueError("FIR taps must be an integer.")

    nyquist = 0.5 * sampling_frequency
    try:
        if filter_type in ("Lowpass", "Highpass"):
            if cutoff1 <= 0:
                raise ValueError("Cutoff must be greater than 0 Hz.")
            if cutoff1 >= nyquist:
                raise ValueError(
                    f"{filter_type} cutoff ({cutoff1:g} Hz) must be below "
                    f"the Nyquist frequency ({nyquist:g} Hz)."
                )
            normalized_cutoff = cutoff1 / nyquist
            if family == "Butterworth":
                b, a = butter(order, normalized_cutoff, btype=filter_type.lower())
            elif family == "Chebyshev I":
                if ripple_db <= 0:
                    raise ValueError("Chebyshev I ripple must be greater than 0 dB.")
                b, a = cheby1(order, ripple_db, normalized_cutoff, btype=filter_type.lower())
            elif family == "Chebyshev II":
                if stopband_db <= 0:
                    raise ValueError("Chebyshev II stopband attenuation must be greater than 0 dB.")
                b, a = cheby2(order, stopband_db, normalized_cutoff, btype=filter_type.lower())
            elif family == "Elliptic":
                if ripple_db <= 0:
                    raise ValueError("Elliptic ripple must be greater than 0 dB.")
                if stopband_db <= 0:
                    raise ValueError("Elliptic stopband attenuation must be greater than 0 dB.")
                b, a = ellip(order, ripple_db, stopband_db, normalized_cutoff, btype=filter_type.lower())
            elif family == "Bessel":
                b, a = bessel(order, normalized_cutoff, btype=filter_type.lower())
            elif family == "FIR":
                if fir_taps < 3:
                    raise ValueError("FIR taps must be at least 3.")
                if fir_taps % 2 == 0:
                    fir_taps += 1
                if fir_window.lower() not in ("hamming", "hann", "blackman"):
                    raise ValueError("Unknown FIR window; use Hamming, Hann, or Blackman.")
                b = firwin(fir_taps, normalized_cutoff, window=fir_window.lower(),
                           pass_zero=filter_type.lower() == "lowpass")
                a = np.array([1.0])
            else:
                raise ValueError(f"Unknown filter family: {family}.")
        elif filter_type in ("Bandpass", "Bandstop"):
            low, high = sorted((cutoff1, cutoff2))
            if low <= 0:
                raise ValueError("Low cutoff must be greater than 0 Hz.")
            if high <= low:
                raise ValueError("Cutoff 2 must be different from Cutoff 1.")
            if high >= nyquist:
                raise ValueError(
                    f"High cutoff ({high:g} Hz) must be below "
                    f"the Nyquist frequency ({nyquist:g} Hz)."
                )
            normalized = [low / nyquist, high / nyquist]
            if family == "Butterworth":
                b, a = butter(order, normalized, btype=filter_type.lower())
            elif family == "Chebyshev I":
                if ripple_db <= 0:
                    raise ValueError("Chebyshev I ripple must be greater than 0 dB.")
                b, a = cheby1(order, ripple_db, normalized, btype=filter_type.lower())
            elif family == "Chebyshev II":
                if stopband_db <= 0:
                    raise ValueError("Chebyshev II stopband attenuation must be greater than 0 dB.")
                b, a = cheby2(order, stopband_db, normalized, btype=filter_type.lower())
            elif family == "Elliptic":
                if ripple_db <= 0:
                    raise ValueError("Elliptic ripple must be greater than 0 dB.")
                if stopband_db <= 0:
                    raise ValueError("Elliptic stopband attenuation must be greater than 0 dB.")
                b, a = ellip(order, ripple_db, stopband_db, normalized, btype=filter_type.lower())
            elif family == "Bessel":
                b, a = bessel(order, normalized, btype=filter_type.lower())
            elif family == "FIR":
                if fir_taps < 3:
                    raise ValueError("FIR taps must be at least 3.")
                if fir_taps % 2 == 0:
                    fir_taps += 1
                if fir_window.lower() not in ("hamming", "hann", "blackman"):
                    raise ValueError("Unknown FIR window; use Hamming, Hann, or Blackman.")
                b = firwin(fir_taps, normalized, window=fir_window.lower(),
                           pass_zero=filter_type.lower() == "bandstop")
                a = np.array([1.0])
            else:
                raise ValueError(f"Unknown filter family: {family}.")
        else:
            return np.asarray(signal, dtype=float).copy(), None, None

        pad_length = 3 * max(len(a), len(b))
        if len(signal) <= pad_length:
            raise ValueError(
                f"Order {order} requires more than {pad_length} samples for "
                f"{filter_type} filtering; increase duration or sampling rate."
            )
        if family in ("Chebyshev I", "Chebyshev II", "Elliptic", "Bessel"):
            return lfilter(b, a, signal), b, a
        return filtfilt(b, a, signal), b, a
    except ValueError as err:
        raise ValueError(f"Filter configuration error:\n{err}") from err


def calculate_frequency_response(b, a, sampling_frequency, points=2000):
    w, h = freqz(b, a, worN=points)
    frequencies = w * sampling_frequency / (2 * np.pi)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    phase_degrees = np.unwrap(np.angle(h)) * 180 / np.pi
    return frequencies, magnitude_db, phase_degrees


def compute_spectrum(signal, fs):
    try:
        samples = np.asarray(signal, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("Signal must contain finite numeric samples.") from err

    if samples.ndim != 1:
        raise ValueError("Signal must be a one-dimensional array.")
    if samples.size == 0:
        raise ValueError("Signal must not be empty.")
    if samples.size < 2:
        raise ValueError("Signal must contain at least 2 samples.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Signal samples must be finite (no NaN or infinity).")

    try:
        sampling_frequency = float(fs)
    except (TypeError, ValueError) as err:
        raise ValueError(
            "Sampling frequency must be a positive finite number."
        ) from err
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be a positive finite number.")

    amplitude = np.abs(np.fft.rfft(samples)) * (2.0 / samples.size)
    amplitude[0] /= 2.0
    if samples.size % 2 == 0:
        amplitude[-1] /= 2.0

    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sampling_frequency)
    return frequencies, amplitude


def _validate_measurement_signal(signal, fs):
    try:
        samples = np.asarray(signal, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("Signal must contain finite numeric samples.") from err
    if samples.ndim != 1:
        raise ValueError("Signal must be a one-dimensional array.")
    if samples.size == 0:
        raise ValueError("Signal must not be empty.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Signal samples must be finite (no NaN or infinity).")
    try:
        sampling_frequency = float(fs)
    except (TypeError, ValueError) as err:
        raise ValueError("Sampling frequency must be a positive finite number.") from err
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be a positive finite number.")
    return samples, sampling_frequency


def compute_measurements(signal, fs):
    samples, sampling_frequency = _validate_measurement_signal(signal, fs)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    peak_to_peak = float(np.max(samples) - np.min(samples))
    mean = float(np.mean(samples))
    crest_factor = float(peak / rms) if rms > 0 else 0.0

    frequencies, amplitude = compute_spectrum(samples, sampling_frequency)
    peak_frequency = float(frequencies[int(np.argmax(amplitude))])
    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "mean": mean,
        "crest_factor": crest_factor,
        "peak_frequency_hz": peak_frequency,
    }


def compute_snr_db(clean, noisy):
    clean_samples, _ = _validate_measurement_signal(clean, 1.0)
    noisy_samples, _ = _validate_measurement_signal(noisy, 1.0)
    if clean_samples.size != noisy_samples.size:
        raise ValueError("Clean and noisy signals must have the same number of samples.")

    clean_power = float(np.mean(clean_samples * clean_samples))
    noise = noisy_samples - clean_samples
    noise_power = float(np.mean(noise * noise))
    if clean_power <= 0:
        raise ValueError("Clean signal must have non-zero power.")
    if noise_power <= 0:
        raise ValueError("Noise power must be greater than zero.")
    return float(10.0 * np.log10(clean_power / noise_power))


def compute_thd(signal, fs, fundamental_hz, n_harmonics=10):
    samples, sampling_frequency = _validate_measurement_signal(signal, fs)
    if not np.isfinite(fundamental_hz) or fundamental_hz <= 0:
        raise ValueError("Fundamental frequency must be a positive finite number.")
    if fundamental_hz >= sampling_frequency / 2:
        raise ValueError("Fundamental frequency must be below the Nyquist frequency.")
    if not isinstance(n_harmonics, (int, np.integer)) or n_harmonics < 1:
        raise ValueError("Number of harmonics must be a positive integer.")

    frequencies, amplitude = compute_spectrum(samples, sampling_frequency)
    fundamental_index = int(np.argmin(np.abs(frequencies - fundamental_hz)))
    fundamental_amplitude = amplitude[fundamental_index]
    if fundamental_amplitude <= 0:
        raise ValueError("Fundamental signal amplitude must be greater than zero.")

    harmonic_power = 0.0
    max_harmonic = int(sampling_frequency / (2 * fundamental_hz))
    for harmonic in range(2, max_harmonic + 1):
        harmonic_frequency = harmonic * fundamental_hz
        if harmonic_frequency > sampling_frequency / 2:
            break
        harmonic_index = int(np.argmin(np.abs(frequencies - harmonic_frequency)))
        harmonic_power += float(amplitude[harmonic_index] ** 2)
    return float(np.sqrt(harmonic_power) / fundamental_amplitude)


def compute_comparison(original, filtered):
    try:
        original_samples = np.asarray(original, dtype=float)
        filtered_samples = np.asarray(filtered, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("Original and filtered signals must contain numeric samples.") from err
    if original_samples.ndim != 1 or filtered_samples.ndim != 1:
        raise ValueError("Original and filtered signals must be one-dimensional.")
    if original_samples.size == 0 or filtered_samples.size == 0:
        raise ValueError("Original and filtered signals must not be empty.")
    if original_samples.size != filtered_samples.size:
        raise ValueError("Original and filtered signals must have the same number of samples.")
    if not np.all(np.isfinite(original_samples)) or not np.all(np.isfinite(filtered_samples)):
        raise ValueError("Original and filtered signals must contain only finite values.")

    rms_original = float(np.sqrt(np.mean(original_samples * original_samples)))
    rms_filtered = float(np.sqrt(np.mean(filtered_samples * filtered_samples)))
    if rms_original <= 0:
        raise ValueError("Original signal RMS must be greater than zero.")
    if rms_filtered <= 0:
        raise ValueError("Filtered signal RMS must be greater than zero.")

    attenuation_db = float(20.0 * np.log10(rms_filtered / rms_original))
    rms_reduction_percent = float((1.0 - rms_filtered / rms_original) * 100.0)
    return {
        "rms_original": rms_original,
        "rms_filtered": rms_filtered,
        "attenuation_db": attenuation_db,
        "rms_reduction_percent": rms_reduction_percent,
    }


def compute_spectrogram(signal, fs, nperseg=256, overlap=0.5):
    try:
        samples = np.asarray(signal, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("Spectrogram signal must contain finite numeric samples.") from err
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Spectrogram signal must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Spectrogram signal must contain only finite values.")
    try:
        sampling_frequency = float(fs)
    except (TypeError, ValueError) as err:
        raise ValueError("Sampling frequency must be a positive finite number.") from err
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be a positive finite number.")
    if not isinstance(nperseg, (int, np.integer)) or nperseg < 2:
        raise ValueError("nperseg must be an integer of at least 2.")
    if nperseg > samples.size:
        raise ValueError("nperseg must not be larger than the signal.")
    if not np.isfinite(overlap) or overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be in the range [0, 1).")

    noverlap = int(round(nperseg * overlap))
    frequencies, times, power = spectrogram(
        samples,
        fs=sampling_frequency,
        nperseg=nperseg,
        noverlap=noverlap,
        mode="psd",
    )
    power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))
    return frequencies, times, power_db
