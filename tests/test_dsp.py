import numpy as np
import pytest
from scipy.signal import butter

import dsp_utils
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


def test_spectrum_finds_sine_frequency_and_amplitude():
    time = np.arange(0, 2, 1 / FS)
    signal = np.sin(2 * np.pi * 50 * time)

    frequencies, amplitude = dsp_utils.compute_spectrum(signal, FS)
    peak_index = np.argmax(amplitude)

    assert frequencies[peak_index] == pytest.approx(50, abs=1)
    assert amplitude[peak_index] == pytest.approx(1, rel=0.02)


def test_spectrum_scales_amplitude():
    time = np.arange(0, 2, 1 / FS)
    signal = 2.5 * np.sin(2 * np.pi * 50 * time)

    _, amplitude = dsp_utils.compute_spectrum(signal, FS)

    assert np.max(amplitude) == pytest.approx(2.5, rel=0.02)


def test_spectrum_preserves_constant_signal_as_dc():
    signal = np.full(1000, 3.0)

    _, amplitude = dsp_utils.compute_spectrum(signal, FS)

    assert amplitude[0] == pytest.approx(3.0)
    np.testing.assert_allclose(amplitude[1:], 0, atol=1e-12)


def test_square_wave_spectrum_has_expected_odd_harmonics():
    time = np.arange(0, 1, 1 / FS)
    signal = np.sign(np.sin(2 * np.pi * 5 * time))

    frequencies, amplitude = dsp_utils.compute_spectrum(signal, FS)
    fundamental = amplitude[np.argmin(np.abs(frequencies - 5))]
    third_harmonic = amplitude[np.argmin(np.abs(frequencies - 15))]
    fifth_harmonic = amplitude[np.argmin(np.abs(frequencies - 25))]
    second_harmonic = amplitude[np.argmin(np.abs(frequencies - 10))]
    fourth_harmonic = amplitude[np.argmin(np.abs(frequencies - 20))]

    assert third_harmonic / fundamental == pytest.approx(1 / 3, rel=0.02)
    assert fifth_harmonic / fundamental == pytest.approx(1 / 5, rel=0.02)
    assert second_harmonic == pytest.approx(0, abs=0.01)
    assert fourth_harmonic == pytest.approx(0, abs=0.01)


def test_sum_of_sines_has_two_dominant_spectrum_peaks():
    time = np.arange(0, 2, 1 / FS)
    signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 120 * time)

    frequencies, amplitude = dsp_utils.compute_spectrum(signal, FS)
    peak_indices = np.argsort(amplitude[1:])[-2:] + 1

    np.testing.assert_allclose(np.sort(frequencies[peak_indices]), [10, 120])


def test_spectrum_frequency_axis_includes_nyquist():
    signal = np.zeros(1000)

    frequencies, amplitude = dsp_utils.compute_spectrum(signal, FS)

    assert frequencies[0] == 0
    assert frequencies[-1] == FS / 2
    assert len(frequencies) == len(amplitude)


@pytest.mark.parametrize(
    ("signal", "sampling_frequency", "message"),
    [
        (np.array([]), FS, "empty"),
        (np.array([1.0]), FS, "2|two|short"),
        (np.ones(2), 0, "positive|greater"),
        (np.ones(2), -1, "positive|greater"),
        (np.ones(2), np.nan, "finite"),
        (np.ones(2), np.inf, "finite"),
        (np.array([0.0, np.nan]), FS, "finite|nan"),
        (np.array([0.0, np.inf]), FS, "finite|inf"),
    ],
)
def test_spectrum_rejects_invalid_input(signal, sampling_frequency, message):
    with pytest.raises(ValueError, match=message):
        dsp_utils.compute_spectrum(signal, sampling_frequency)


def test_spectrum_does_not_modify_input():
    time = np.arange(0, 1, 1 / FS)
    signal = np.sin(2 * np.pi * 50 * time)
    original = signal.copy()

    dsp_utils.compute_spectrum(signal, FS)

    np.testing.assert_array_equal(signal, original)
