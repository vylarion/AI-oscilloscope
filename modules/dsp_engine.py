"""
DSP Measurement Engine — NumPy / SciPy signal processing.

All measurements use standard, verifiable DSP formulas — the same
methods real oscilloscopes employ internally.

Entry point:
    report = DSPEngine().compute(times, voltages)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
import json

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MeasurementReport:
    """All computed measurements for a single waveform."""

    # Identity
    waveform_type: str = "Unknown"

    # Timing
    frequency_hz: Optional[float] = None
    period_s: Optional[float] = None
    duty_cycle_pct: Optional[float] = None
    rise_time_s: Optional[float] = None
    fall_time_s: Optional[float] = None
    settling_time_s: Optional[float] = None

    # Voltage
    amplitude_v: float = 0.0
    peak_v: float = 0.0
    peak_to_peak_v: float = 0.0
    rms_v: float = 0.0
    average_v: float = 0.0

    # Quality
    overshoot_pct: Optional[float] = None
    undershoot_pct: Optional[float] = None
    noise_level: str = "Low"        # "Low" | "Medium" | "High"
    noise_std_mv: float = 0.0
    ripple_mv: Optional[float] = None
    jitter_ns: Optional[float] = None

    # Metadata
    num_cycles: int = 0
    num_samples: int = 0
    duration_s: float = 0.0

    # ----------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize to a clean dictionary (None → 'N/A' for display)."""
        def _fmt(v):
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return round(v, 6)
            return v

        return {
            "waveform_type": self.waveform_type,
            "frequency_hz": _fmt(self.frequency_hz),
            "period_s": _fmt(self.period_s),
            "duty_cycle_pct": _fmt(self.duty_cycle_pct),
            "rise_time_s": _fmt(self.rise_time_s),
            "fall_time_s": _fmt(self.fall_time_s),
            "settling_time_s": _fmt(self.settling_time_s),
            "amplitude_v": round(self.amplitude_v, 6),
            "peak_v": round(self.peak_v, 6),
            "peak_to_peak_v": round(self.peak_to_peak_v, 6),
            "rms_v": round(self.rms_v, 6),
            "average_v": round(self.average_v, 6),
            "overshoot_pct": _fmt(self.overshoot_pct),
            "undershoot_pct": _fmt(self.undershoot_pct),
            "noise_level": self.noise_level,
            "noise_std_mv": round(self.noise_std_mv, 4),
            "ripple_mv": _fmt(self.ripple_mv),
            "jitter_ns": _fmt(self.jitter_ns),
            "num_cycles": self.num_cycles,
            "num_samples": self.num_samples,
            "duration_s": round(self.duration_s, 9),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DSPEngine:
    """
    Computes all standard oscilloscope measurements from a (times, voltages)
    NumPy array pair using deterministic DSP algorithms.
    """

    # Rise/fall time thresholds (industry standard 10–90%)
    RISE_LOW_PCT = 0.10
    RISE_HIGH_PCT = 0.90

    # Settling band: ±2% of final value
    SETTLING_BAND_PCT = 0.02

    # Minimum number of samples for reliable FFT
    MIN_SAMPLES_FFT = 32

    def compute(self, times: np.ndarray, voltages: np.ndarray) -> MeasurementReport:
        """
        Run all measurements and return a MeasurementReport.

        Args:
            times:    1-D array of time values in seconds (monotonically increasing)
            voltages: 1-D array of voltage values in volts, same length as times

        Returns:
            MeasurementReport with all available measurements populated.
        """
        report = MeasurementReport()

        # --- Guard ---
        if len(times) < 4 or len(voltages) < 4:
            logger.warning("Too few samples for DSP measurements.")
            return report

        times = np.asarray(times, dtype=float)
        voltages = np.asarray(voltages, dtype=float)

        # Ensure monotonically increasing time
        if not np.all(np.diff(times) > 0):
            order = np.argsort(times)
            times, voltages = times[order], voltages[order]

        report.num_samples = len(times)
        report.duration_s = float(times[-1] - times[0])

        # Basic voltage measurements
        report.peak_v = float(np.max(voltages))
        report.peak_to_peak_v = float(np.ptp(voltages))
        report.amplitude_v = report.peak_to_peak_v / 2.0
        report.rms_v = float(np.sqrt(np.mean(voltages ** 2)))
        report.average_v = float(np.mean(voltages))

        # --- Resample to uniform time grid for FFT ---
        n_uniform = 512
        t_uniform = np.linspace(times[0], times[-1], n_uniform)
        v_uniform = np.interp(t_uniform, times, voltages)

        dt = (times[-1] - times[0]) / (n_uniform - 1)
        fs = 1.0 / dt if dt > 0 else 1.0

        # --- Frequency / period ---
        freq, period, num_cycles = self._measure_frequency(t_uniform, v_uniform, fs)
        report.frequency_hz = freq
        report.period_s = period
        report.num_cycles = num_cycles

        # --- Waveform type ---
        report.waveform_type = self._classify_waveform(v_uniform, fs, freq)

        # --- Duty cycle ---
        if freq is not None:
            report.duty_cycle_pct = self._measure_duty_cycle(v_uniform)

        # --- Rise / Fall time ---
        if freq is not None:
            rt, ft, st = self._measure_edge_times(t_uniform, v_uniform, report.amplitude_v)
            report.rise_time_s = rt
            report.fall_time_s = ft
            report.settling_time_s = st

        # --- Overshoot / Undershoot ---
        if freq is not None and report.amplitude_v > 1e-9:
            ov, un = self._measure_overshoot_undershoot(v_uniform, report.amplitude_v)
            report.overshoot_pct = ov
            report.undershoot_pct = un

        # --- Noise ---
        noise_std, noise_level = self._measure_noise(v_uniform)
        report.noise_std_mv = float(noise_std * 1000)
        report.noise_level = noise_level

        # --- Ripple ---
        ripple = self._measure_ripple(v_uniform, fs, freq)
        report.ripple_mv = ripple

        # --- Jitter ---
        if freq is not None and num_cycles >= 3:
            jitter = self._measure_jitter(t_uniform, v_uniform)
            report.jitter_ns = jitter

        return report

    # ------------------------------------------------------------------
    # Frequency / Period
    # ------------------------------------------------------------------

    def _measure_frequency(
        self, times: np.ndarray, voltages: np.ndarray, fs: float
    ) -> tuple[Optional[float], Optional[float], int]:
        """
        Estimate frequency via zero-crossing analysis (primary)
        and FFT peak (secondary cross-check).

        Returns: (frequency_hz, period_s, num_cycles)
        """
        midpoint = (np.max(voltages) + np.min(voltages)) / 2.0
        centered = voltages - midpoint

        # Rising zero crossings
        sign_changes = np.where(np.diff(np.sign(centered)) > 0)[0]

        if len(sign_changes) < 2:
            # No clear zero crossings → try FFT
            return self._freq_from_fft(times, voltages, fs)

        # Interpolate crossing times for precision
        crossing_times = []
        for idx in sign_changes:
            if idx + 1 < len(times):
                # Linear interpolation
                t0, t1 = times[idx], times[idx + 1]
                v0, v1 = centered[idx], centered[idx + 1]
                if v1 != v0:
                    t_cross = t0 + (0 - v0) / (v1 - v0) * (t1 - t0)
                    crossing_times.append(t_cross)

        if len(crossing_times) < 2:
            return self._freq_from_fft(times, voltages, fs)

        crossing_times = np.array(crossing_times)
        periods = np.diff(crossing_times) * 2  # two half-periods = one full period
        # Keep only consistent periods (within 20% of median)
        med = np.median(periods)
        good = periods[np.abs(periods - med) < 0.2 * med]

        if len(good) == 0:
            good = periods

        period = float(np.mean(good))
        if period <= 0:
            return None, None, 0

        freq = 1.0 / period
        num_cycles = max(1, len(sign_changes) // 2)
        return freq, period, num_cycles

    def _freq_from_fft(
        self, times: np.ndarray, voltages: np.ndarray, fs: float
    ) -> tuple[Optional[float], Optional[float], int]:
        """Estimate frequency from the dominant FFT peak."""
        if len(voltages) < self.MIN_SAMPLES_FFT:
            return None, None, 0

        n = len(voltages)
        window = np.hanning(n)
        spectrum = np.abs(rfft(voltages * window))
        freqs = rfftfreq(n, d=1.0 / fs)

        # Skip DC component
        if len(freqs) < 2:
            return None, None, 0
        spectrum[0] = 0

        peak_idx = int(np.argmax(spectrum))
        peak_freq = float(freqs[peak_idx])

        if peak_freq <= 0:
            return None, None, 0

        period = 1.0 / peak_freq
        duration = float(times[-1] - times[0])
        num_cycles = max(1, int(duration / period))
        return peak_freq, period, num_cycles

    # ------------------------------------------------------------------
    # Waveform Classification
    # ------------------------------------------------------------------

    def _classify_waveform(
        self,
        voltages: np.ndarray,
        fs: float,
        freq: Optional[float],
    ) -> str:
        """
        Classify waveform type using crest factor + FFT harmonic analysis.
        """
        vpp = float(np.ptp(voltages))
        if vpp < 1e-9:
            return "DC / Flat Signal"

        # DC detection: std dev is tiny relative to amplitude
        std = float(np.std(voltages))
        if std / vpp < 0.02:
            return "DC Signal"

        # Noise detection: no clear frequency, high entropy
        if freq is None:
            return "Noise / Unclassified"

        # Crest factor: peak / RMS (peak-to-peak / 2) / RMS-AC
        v_ac = voltages - np.mean(voltages)
        rms_ac = float(np.sqrt(np.mean(v_ac ** 2)))
        peak_ac = float(np.max(np.abs(v_ac)))
        crest = peak_ac / rms_ac if rms_ac > 1e-12 else 1.0

        # FFT harmonic ratios
        n = len(voltages)
        window = np.hanning(n)
        spectrum = np.abs(rfft(v_ac * window))
        freqs_fft = rfftfreq(n, d=1.0 / fs)

        def _harmonic_amp(harmonic: int) -> float:
            """Amplitude of the nth harmonic relative to fundamental."""
            target = freq * harmonic
            idx = int(np.argmin(np.abs(freqs_fft - target)))
            bandwidth = max(1, int(n * freq / (fs * 4)))
            lo = max(0, idx - bandwidth)
            hi = min(len(spectrum), idx + bandwidth + 1)
            return float(np.max(spectrum[lo:hi]))

        fund = _harmonic_amp(1)
        if fund < 1e-12:
            return "Noise / Unclassified"

        h2 = _harmonic_amp(2) / fund   # significant for sawtooth
        h3 = _harmonic_amp(3) / fund   # 1/3 for square, 1/9 for triangle
        h5 = _harmonic_amp(5) / fund

        # Classification rules
        if crest < 1.05:                        # near 1.0
            return "Square Wave"
        elif crest > 1.7 and h2 < 0.05:        # triangle: crest ≈ 1.73, no even harmonics
            return "Triangle Wave"
        elif h2 > 0.1:                          # sawtooth: strong 2nd harmonic
            return "Sawtooth Wave"
        elif h3 < 0.08 and h5 < 0.05:          # clean — mostly fundamental
            return "Sine Wave"
        elif h3 > 0.15:                         # strong 3rd harmonic — square-like
            # Check duty cycle to distinguish PWM from square
            dc = self._measure_duty_cycle(voltages)
            if dc is not None and abs(dc - 50.0) > 8.0:
                return "PWM Signal"
            return "Square Wave"
        else:
            return "Complex / Mixed Signal"

    # ------------------------------------------------------------------
    # Duty Cycle
    # ------------------------------------------------------------------

    def _measure_duty_cycle(self, voltages: np.ndarray) -> Optional[float]:
        """
        Duty cycle = fraction of time the signal is in the 'high' state.
        Threshold = midpoint between 10th and 90th percentile levels.
        """
        v_low = float(np.percentile(voltages, 10))
        v_high = float(np.percentile(voltages, 90))
        if abs(v_high - v_low) < 1e-9:
            return None
        threshold = (v_low + v_high) / 2.0
        high_count = int(np.sum(voltages > threshold))
        return float(high_count / len(voltages) * 100.0)

    # ------------------------------------------------------------------
    # Rise / Fall / Settling Time
    # ------------------------------------------------------------------

    def _measure_edge_times(
        self,
        times: np.ndarray,
        voltages: np.ndarray,
        amplitude: float,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Industry-standard 10%–90% method for rise and fall times.
        Settling time uses ±2% band around final settled value.
        """
        if amplitude < 1e-9:
            return None, None, None

        v_low = float(np.min(voltages))
        v_high = float(np.max(voltages))
        v10 = v_low + 0.10 * (v_high - v_low)
        v90 = v_low + 0.90 * (v_high - v_low)
        midpoint = (v_low + v_high) / 2.0

        # Detect rising and falling edges via sign change of (v - midpoint)
        centered = voltages - midpoint
        sign = np.sign(centered)
        sign[sign == 0] = 1  # tie-break

        changes = np.where(np.diff(sign) != 0)[0]
        if len(changes) == 0:
            return None, None, None

        rise_times: list[float] = []
        fall_times: list[float] = []

        for idx in changes:
            is_rising = centered[idx] < 0  # was below midpoint → rising
            # Find nearest crossing of v10 and v90 around this index
            search_width = max(5, len(times) // 20)
            lo = max(0, idx - search_width)
            hi = min(len(times) - 1, idx + search_width)

            region_t = times[lo:hi + 1]
            region_v = voltages[lo:hi + 1]

            if is_rising:
                t10 = self._crossing_time(region_t, region_v, v10, rising=True)
                t90 = self._crossing_time(region_t, region_v, v90, rising=True)
                if t10 is not None and t90 is not None and t90 > t10:
                    rise_times.append(t90 - t10)
            else:
                t90 = self._crossing_time(region_t, region_v, v90, rising=False)
                t10 = self._crossing_time(region_t, region_v, v10, rising=False)
                if t90 is not None and t10 is not None and t10 > t90:
                    fall_times.append(t10 - t90)

        rise_time = float(np.median(rise_times)) if rise_times else None
        fall_time = float(np.median(fall_times)) if fall_times else None

        # Settling time: time after first edge until signal stays within ±2%
        settling_time = self._measure_settling(times, voltages, v_high, amplitude)

        return rise_time, fall_time, settling_time

    @staticmethod
    def _crossing_time(
        times: np.ndarray,
        voltages: np.ndarray,
        threshold: float,
        rising: bool,
    ) -> Optional[float]:
        """Linear-interpolated time when voltage crosses threshold in given direction."""
        for i in range(len(voltages) - 1):
            v0, v1 = voltages[i], voltages[i + 1]
            if rising and v0 <= threshold <= v1:
                if v1 != v0:
                    return float(times[i] + (threshold - v0) / (v1 - v0) * (times[i + 1] - times[i]))
            elif not rising and v0 >= threshold >= v1:
                if v1 != v0:
                    return float(times[i] + (threshold - v0) / (v1 - v0) * (times[i + 1] - times[i]))
        return None

    @staticmethod
    def _measure_settling(
        times: np.ndarray,
        voltages: np.ndarray,
        v_high: float,
        amplitude: float,
    ) -> Optional[float]:
        """
        Time from the first rising edge to when the signal stays within
        ±2% of the settled high value permanently.
        """
        band = 0.02 * amplitude
        target = v_high
        midpoint = v_high - amplitude

        # Find first rising edge crossing
        edge_indices = np.where(np.diff(np.sign(voltages - midpoint)) > 0)[0]
        if len(edge_indices) == 0:
            return None
        edge_idx = edge_indices[0]
        edge_time = float(times[edge_idx])

        # From this edge forward, find when signal stays within ±band of target
        post_edge_v = voltages[edge_idx:]
        post_edge_t = times[edge_idx:]
        in_band = np.abs(post_edge_v - target) <= band

        # Find first position where all subsequent samples are in band
        for i in range(len(in_band)):
            if np.all(in_band[i:]):
                return float(post_edge_t[i]) - edge_time

        return None

    # ------------------------------------------------------------------
    # Overshoot / Undershoot
    # ------------------------------------------------------------------

    def _measure_overshoot_undershoot(
        self, voltages: np.ndarray, amplitude: float
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Overshoot = (peak - settled_high) / amplitude * 100%
        Undershoot = (settled_low - trough) / amplitude * 100%
        Settled levels estimated from most-populated histogram bins.
        """
        if amplitude < 1e-9:
            return None, None

        v_low = float(np.min(voltages))
        v_high = float(np.max(voltages))

        # Estimate settled levels using mode of upper and lower halves
        midpoint = (v_low + v_high) / 2.0
        upper = voltages[voltages > midpoint]
        lower = voltages[voltages <= midpoint]

        if len(upper) == 0 or len(lower) == 0:
            return None, None

        # Use percentile to estimate "settled" levels (robust to peaks)
        settled_high = float(np.percentile(upper, 25))   # lower quartile of upper = settled high
        settled_low = float(np.percentile(lower, 75))    # upper quartile of lower = settled low

        peak = v_high
        trough = v_low

        overshoot = max(0.0, (peak - settled_high) / amplitude * 100.0)
        undershoot = max(0.0, (settled_low - trough) / amplitude * 100.0)

        return (
            float(overshoot) if overshoot > 0.01 else None,
            float(undershoot) if undershoot > 0.01 else None,
        )

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def _measure_noise(
        self, voltages: np.ndarray
    ) -> tuple[float, str]:
        """
        Estimate noise by computing the standard deviation in the flattest
        10% of samples (the settled regions with least slope).

        Returns: (std_volts, level_string)
        """
        if len(voltages) < 8:
            return 0.0, "Low"

        # Find low-slope (settled) regions
        diff = np.abs(np.gradient(voltages))
        threshold_slope = np.percentile(diff, 20)  # bottom 20% by slope = settled
        flat_mask = diff <= threshold_slope

        flat_voltages = voltages[flat_mask]
        if len(flat_voltages) < 4:
            std = float(np.std(voltages)) * 0.2  # rough estimate
        else:
            std = float(np.std(flat_voltages))

        vpp = float(np.ptp(voltages))
        noise_pct = (std / vpp * 100) if vpp > 1e-9 else 0.0

        if noise_pct > 5.0 or std * 1000 > 50:
            level = "High"
        elif noise_pct > 2.0 or std * 1000 > 20:
            level = "Medium"
        else:
            level = "Low"

        return std, level

    # ------------------------------------------------------------------
    # Ripple
    # ------------------------------------------------------------------

    def _measure_ripple(
        self,
        voltages: np.ndarray,
        fs: float,
        freq: Optional[float],
    ) -> Optional[float]:
        """
        Measure ripple: AC amplitude on a signal after removing high-frequency content.
        Ripple = amplitude of low-frequency variation that is NOT the signal itself.

        Returns: ripple in millivolts.
        """
        vpp = float(np.ptp(voltages))
        if vpp < 1e-9 or len(voltages) < 16:
            return None

        # Define ripple as components below signal_frequency / 10
        # (or below 1 Hz if no frequency is detected)
        cutoff = (freq / 10.0) if freq is not None else 1.0
        nyquist = fs / 2.0

        if cutoff <= 0 or cutoff >= nyquist:
            # Cannot filter — use overall AC variation of DC-like signals
            if freq is None:
                ac = voltages - np.mean(voltages)
                return float(np.ptp(ac) * 1000)
            return None

        # Low-pass Butterworth filter
        try:
            b, a = sp_signal.butter(
                N=4,
                Wn=cutoff / nyquist,
                btype='low',
            )
            ripple_signal = sp_signal.filtfilt(b, a, voltages)
            ripple_mv = float(np.ptp(ripple_signal) * 1000)
            # Only return if ripple is significant (> 0.5% of Vpp)
            return ripple_mv if ripple_mv > vpp * 0.5 * 10 else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Jitter
    # ------------------------------------------------------------------

    def _measure_jitter(
        self, times: np.ndarray, voltages: np.ndarray
    ) -> Optional[float]:
        """
        Cycle-to-cycle jitter = std-dev of consecutive periods (in ns).
        Periods measured via rising-edge zero crossings.
        """
        midpoint = (np.max(voltages) + np.min(voltages)) / 2.0
        centered = voltages - midpoint

        # Rising zero crossings with linear interpolation
        crossing_times = []
        for i in range(len(centered) - 1):
            if centered[i] <= 0 < centered[i + 1]:
                t0, t1 = times[i], times[i + 1]
                v0, v1 = centered[i], centered[i + 1]
                if v1 != v0:
                    t_cross = t0 - v0 / (v1 - v0) * (t1 - t0)
                    crossing_times.append(t_cross)

        if len(crossing_times) < 3:
            return None

        crossing_times = np.array(crossing_times)
        periods = np.diff(crossing_times)

        # Filter out outliers (keep within 3σ of median)
        med = np.median(periods)
        std = np.std(periods)
        good = periods[np.abs(periods - med) < 3 * std]

        if len(good) < 2:
            return None

        jitter_ns = float(np.std(good) * 1e9)
        return jitter_ns if jitter_ns > 0 else None
