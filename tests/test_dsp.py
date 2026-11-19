import numpy as np
import pytest
from scipy.signal import butter

from dsp_utils import (
    SIGNAL_TYPES,
    add_noise_and_distortion,
    apply_digital_filter,
    calculate_frequency_response,
    generate_signal,
)


FS = 1000


def band_power(signal, sampling_frequency, frequency, width=3):
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(len(signal), 1 / sampling_frequency)
    band = (frequencies > frequency - width) & (frequencies < frequency + width)
    return spectrum[band].sum()


def test_sine_has_expected_frequency():
    time = np.arange(0, 2, 1 / FS)
    signal = generate_signal(time, 50, "Sine")
    frequencies = np.fft.rfftfreq(len(signal), 1 / FS)
    peak_frequency = frequencies[np.argmax(np.abs(np.fft.rfft(signal)))]

    assert abs(peak_frequency - 50) < 1


@pytest.mark.parametrize("signal_type", SIGNAL_TYPES)
def test_waveforms_are_finite_and_bounded(signal_type):
    time = np.arange(0, 1, 1 / FS)
    signal = generate_signal(time, 10, signal_type)

    assert signal.shape == time.shape
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) <= 1.5


def test_unknown_waveform_returns_zeros():
    time = np.arange(10, dtype=float)

    np.testing.assert_array_equal(generate_signal(time, 10, "unknown"), np.zeros_like(time))


def test_lowpass_rejects_high_frequency_and_keeps_low_frequency():
    time = np.arange(0, 2, 1 / FS)
    signal = np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 200 * time)
    filtered, numerator, denominator = apply_digital_filter(
        signal, FS, "Lowpass", 30, 0, order=4
    )

    assert band_power(filtered, FS, 200) < 0.01 * band_power(signal, FS, 200)
    assert band_power(filtered, FS, 5) > 0.9 * band_power(signal, FS, 5)
    assert numerator is not None
    assert denominator is not None


def test_highpass_rejects_low_frequency_and_keeps_high_frequency():
    time = np.arange(0, 2, 1 / FS)
    signal = np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 200 * time)
    filtered, _, _ = apply_digital_filter(
        signal, FS, "Highpass", 100, 0, order=4
    )

    assert band_power(filtered, FS, 5) < 0.01 * band_power(signal, FS, 5)
    assert band_power(filtered, FS, 200) > 0.9 * band_power(signal, FS, 200)


def test_bandpass_keeps_passband_and_rejects_outside_frequencies():
    time = np.arange(0, 2, 1 / FS)
    signal = (
        np.sin(2 * np.pi * 5 * time)
        + np.sin(2 * np.pi * 100 * time)
        + np.sin(2 * np.pi * 250 * time)
    )
    filtered, _, _ = apply_digital_filter(
        signal, FS, "Bandpass", 80, 120, order=4
    )

    assert band_power(filtered, FS, 100) > 0.9 * band_power(signal, FS, 100)
    assert band_power(filtered, FS, 5) < 0.01 * band_power(signal, FS, 5)
    assert band_power(filtered, FS, 250) < 0.01 * band_power(signal, FS, 250)


def test_bandstop_rejects_stopband_and_keeps_outside_frequencies():
    time = np.arange(0, 2, 1 / FS)
    signal = (
        np.sin(2 * np.pi * 5 * time)
        + np.sin(2 * np.pi * 100 * time)
        + np.sin(2 * np.pi * 250 * time)
    )
    filtered, _, _ = apply_digital_filter(
        signal, FS, "Bandstop", 80, 120, order=4
    )

    assert band_power(filtered, FS, 100) < 0.01 * band_power(signal, FS, 100)
    assert band_power(filtered, FS, 5) > 0.9 * band_power(signal, FS, 5)
    assert band_power(filtered, FS, 250) > 0.9 * band_power(signal, FS, 250)


def test_band_filter_accepts_cutoffs_in_either_order():
    time = np.arange(0, 1, 1 / FS)
    signal = np.sin(2 * np.pi * 100 * time)

    ascending, _, _ = apply_digital_filter(signal, FS, "Bandpass", 80, 120)
    descending, _, _ = apply_digital_filter(signal, FS, "Bandpass", 120, 80)

    np.testing.assert_allclose(ascending, descending)


@pytest.mark.parametrize("filter_type", ["Lowpass", "Highpass", "Bandpass", "Bandstop"])
@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_supported_filter_orders(filter_type, order):
    signal = np.ones(1000)
    cutoff2 = 120 if filter_type in ("Bandpass", "Bandstop") else 0

    filtered, numerator, denominator = apply_digital_filter(
        signal, FS, filter_type, 80, cutoff2, order
    )

    assert filtered.shape == signal.shape
    assert numerator is not None and denominator is not None


@pytest.mark.parametrize(
    ("filter_type", "cutoff1", "cutoff2"),
    [
        ("Lowpass", 0, 0),
        ("Lowpass", 500, 0),
        ("Highpass", -1, 0),
        ("Bandpass", 0, 100),
        ("Bandpass", 100, 100),
        ("Bandstop", 100, 500),
    ],
)
def test_invalid_filter_cutoffs_are_rejected(filter_type, cutoff1, cutoff2):
    with pytest.raises(ValueError, match="Filter configuration error"):
        apply_digital_filter(
            np.ones(1000), FS, filter_type, cutoff1, cutoff2, order=4
        )


def test_too_few_samples_for_filtering_are_rejected():
    with pytest.raises(ValueError, match="more than"):
        apply_digital_filter(np.zeros(5), FS, "Lowpass", 30, 0, order=8)


def test_no_filter_returns_a_copy_and_no_coefficients():
    signal = np.arange(20, dtype=float)

    filtered, numerator, denominator = apply_digital_filter(
        signal, FS, "None", 0, 0
    )

    np.testing.assert_array_equal(filtered, signal)
    assert filtered is not signal
    assert numerator is None and denominator is None


def test_noise_is_repeatable_and_does_not_mutate_input():
    signal = np.zeros(100)
    first = add_noise_and_distortion(signal, 0.5, 0)
    second = add_noise_and_distortion(signal, 0.5, 0)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(signal, np.zeros(100))


@pytest.mark.parametrize(
    ("noise_amount", "distortion_amount"),
    [(-0.1, 0), (float("nan"), 0), (0, -0.1), (0, 1.1), (0, float("nan"))],
)
def test_invalid_noise_and_distortion_amounts_are_rejected(
    noise_amount, distortion_amount
):
    with pytest.raises(ValueError):
        add_noise_and_distortion(np.zeros(10), noise_amount, distortion_amount)


def test_distortion_soft_clips_signal_without_mutating_input():
    signal = np.linspace(-1, 1, 100)
    original = signal.copy()
    distorted = add_noise_and_distortion(signal, 0, 1)

    assert np.max(np.abs(distorted)) <= 1
    np.testing.assert_array_equal(signal, original)


def test_frequency_response_has_expected_shape_and_cutoff_attenuation():
    numerator, denominator = butter(4, 30 / (FS / 2), btype="lowpass")
    frequencies, magnitude_db, phase_degrees = calculate_frequency_response(
        numerator, denominator, FS
    )

    cutoff_index = np.argmin(np.abs(frequencies - 30))
    high_frequency_index = np.argmin(np.abs(frequencies - 200))
    assert frequencies.shape == magnitude_db.shape == phase_degrees.shape == (2000,)
    assert np.all(np.isfinite(magnitude_db))
    assert np.all(np.isfinite(phase_degrees))
    assert magnitude_db[cutoff_index] == pytest.approx(-3, abs=0.2)
    assert magnitude_db[high_frequency_index] < -40
