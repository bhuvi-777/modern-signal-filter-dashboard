import numpy as np
from scipy.signal import butter, filtfilt, freqz


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


def apply_digital_filter(signal, sampling_frequency, filter_type, cutoff1, cutoff2, order=4):
    if filter_type == "None":
        return np.asarray(signal, dtype=float).copy(), None, None

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
            b, a = butter(order, cutoff1 / nyquist, btype=filter_type.lower())
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
            b, a = butter(order, [low / nyquist, high / nyquist], btype=filter_type.lower())
        else:
            return np.asarray(signal, dtype=float).copy(), None, None

        pad_length = 3 * max(len(a), len(b))
        if len(signal) <= pad_length:
            raise ValueError(
                f"Order {order} requires more than {pad_length} samples for "
                f"{filter_type} filtering; increase duration or sampling rate."
            )
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
