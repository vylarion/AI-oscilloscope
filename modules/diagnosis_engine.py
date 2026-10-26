"""
Diagnosis Engine — Rule-Based Signal Integrity Analysis.

Applies a set of expert engineering rules to a MeasurementReport and
produces a DiagnosisReport with detected issues, causes, and suggestions.

All rules are documented and traceable — each maps to a real, well-known
electronics fault signature. Thresholds are loaded from
assets/diagnosis_rules.json to allow tuning without code changes.

Entry point:
    report = DiagnosisEngine().diagnose(measurements)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .dsp_engine import MeasurementReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load rule thresholds
# ---------------------------------------------------------------------------

def _load_rules() -> dict:
    """Load diagnosis thresholds from assets/diagnosis_rules.json."""
    # Try workspace-relative path first, then same-directory fallback
    candidates = [
        Path(__file__).parent.parent / "assets" / "diagnosis_rules.json",
        Path(os.getcwd()) / "assets" / "diagnosis_rules.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    logger.warning("diagnosis_rules.json not found — using built-in defaults.")
    return _DEFAULT_RULES


_DEFAULT_RULES = {
    "overshoot":       {"warning_pct": 5.0,  "critical_pct": 15.0},
    "undershoot":      {"warning_pct": 5.0,  "critical_pct": 10.0},
    "jitter":          {"warning_pct_period": 0.5, "critical_pct_period": 2.0, "warning_ns": 5.0, "critical_ns": 20.0},
    "noise":           {"medium_pct_amplitude": 2.0, "high_pct_amplitude": 5.0, "medium_mv": 20.0, "high_mv": 50.0},
    "ripple":          {"warning_pct_amplitude": 1.0, "critical_pct_amplitude": 5.0, "warning_mv": 50.0, "critical_mv": 200.0},
    "clipping":        {"flatness_threshold_pct": 2.0},
    "slow_rise_time":  {"slow_threshold_pct_period": 10.0},
    "dc_offset":       {"warning_pct_amplitude": 10.0},
}

_RULES: dict = _load_rules()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    """A single detected signal integrity issue."""
    name: str
    severity: str               # "Warning" | "Critical" | "Info"
    description: str
    measured_values: dict = field(default_factory=dict)
    possible_causes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity,
            "description": self.description,
            "measured_values": self.measured_values,
            "possible_causes": self.possible_causes,
            "suggestions": self.suggestions,
        }


@dataclass
class DiagnosisReport:
    """Collection of all detected issues with overall health assessment."""
    issues: list[Issue] = field(default_factory=list)
    overall_health: str = "Good"    # "Good" | "Fair" | "Poor"
    health_score: int = 100         # 0 – 100

    def to_dict(self) -> dict:
        return {
            "overall_health": self.overall_health,
            "health_score": self.health_score,
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "Critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "Warning")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DiagnosisEngine:
    """
    Applies all signal integrity rules to a MeasurementReport.

    Each rule method returns an Issue or None.  The engine collects all
    non-None results, scores them, and produces a DiagnosisReport.
    """

    def diagnose(self, m: MeasurementReport) -> DiagnosisReport:
        """
        Run all rules against the measurement report.

        Args:
            m: A fully-populated MeasurementReport from DSPEngine.

        Returns:
            DiagnosisReport with all detected issues.
        """
        rules = [
            self._check_overshoot,
            self._check_undershoot,
            self._check_ringing,
            self._check_jitter,
            self._check_noise,
            self._check_ripple,
            self._check_clipping,
            self._check_slow_rise_time,
            self._check_dc_offset,
        ]

        issues: list[Issue] = []
        for rule in rules:
            try:
                result = rule(m)
                if result is not None:
                    issues.append(result)
            except Exception as exc:
                logger.warning(f"Rule {rule.__name__} raised: {exc}")

        # Compute health score (deduct based on severity)
        score = 100
        for issue in issues:
            if issue.severity == "Critical":
                score -= 20
            elif issue.severity == "Warning":
                score -= 8
            else:  # Info
                score -= 2
        score = max(0, score)

        if score >= 80:
            health = "Good"
        elif score >= 55:
            health = "Fair"
        else:
            health = "Poor"

        return DiagnosisReport(issues=issues, overall_health=health, health_score=score)

    # ------------------------------------------------------------------
    # Rule: Overshoot
    # ------------------------------------------------------------------

    def _check_overshoot(self, m: MeasurementReport) -> Optional[Issue]:
        ov = m.overshoot_pct
        if ov is None or ov < 0.5:
            return None

        r = _RULES.get("overshoot", _DEFAULT_RULES["overshoot"])
        warning_pct = r.get("warning_pct", 5.0)
        critical_pct = r.get("critical_pct", 15.0)
        if ov < warning_pct:
            return None

        severity = "Critical" if ov >= critical_pct else "Warning"
        causes = r.get("possible_causes", [
            "Impedance mismatch between source and transmission line",
            "Underdamped LC response (parasitic inductance + load capacitance)",
            "Long PCB trace at fast edge speeds",
            "Probe compensation error",
        ])
        suggestions = r.get("suggestions", [
            "Add a series termination resistor (22–100 Ω) close to the driver",
            "Check and adjust probe compensation with a 1 kHz calibration square wave",
            "Reduce trace length or use controlled-impedance routing",
        ])

        return Issue(
            name="Overshoot",
            severity=severity,
            description=(
                f"The signal overshoots its settled high level by {ov:.1f}% of amplitude "
                f"after the rising edge. Industry standard tolerance is ≤ {warning_pct}%."
            ),
            measured_values={"overshoot_pct": round(ov, 2)},
            possible_causes=causes,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Rule: Undershoot
    # ------------------------------------------------------------------

    def _check_undershoot(self, m: MeasurementReport) -> Optional[Issue]:
        un = m.undershoot_pct
        if un is None or un < 0.5:
            return None

        r = _RULES.get("undershoot", _DEFAULT_RULES["undershoot"])
        warning_pct = r.get("warning_pct", 5.0)
        critical_pct = r.get("critical_pct", 10.0)
        if un < warning_pct:
            return None

        severity = "Critical" if un >= critical_pct else "Warning"
        causes = r.get("possible_causes", [
            "Same root causes as overshoot — impedance mismatch on falling edge",
            "Inductive kickback from a switched load",
            "Negative ringing from reflected signal on long traces",
        ])
        suggestions = r.get("suggestions", [
            "Add series termination resistor near the source",
            "For inductive loads: add a flyback diode across the load",
            "Reduce switching speed if timing allows (slower slew rate = less ringing)",
        ])

        return Issue(
            name="Undershoot",
            severity=severity,
            description=(
                f"The signal undershoots its settled low level by {un:.1f}% of amplitude "
                f"after the falling edge."
            ),
            measured_values={"undershoot_pct": round(un, 2)},
            possible_causes=causes,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Rule: Ringing (overshoot + oscillation)
    # ------------------------------------------------------------------

    def _check_ringing(self, m: MeasurementReport) -> Optional[Issue]:
        """
        Ringing = overshoot > warning threshold AND the waveform shows
        oscillation characteristics (rise time is fast, jitter may be present).
        Only flagged if not already dominated by a simpler issue.
        """
        ov = m.overshoot_pct
        if ov is None or ov < 5.0:
            return None

        # Ringing is typically associated with fast edges on square waves
        if "Square" not in m.waveform_type and "PWM" not in m.waveform_type:
            return None

        # Fast rise time relative to period suggests high-frequency components
        fast_edge = False
        if m.rise_time_s is not None and m.period_s is not None:
            fast_edge = m.rise_time_s < m.period_s * 0.05

        if not fast_edge and ov < 10.0:
            return None  # Not clearly ringing

        r = _RULES.get("ringing", {})
        causes = r.get("possible_causes", [
            "Resonance between parasitic trace inductance and load/stray capacitance",
            "Transmission line unterminated or improperly terminated",
            "Probe ground lead is too long (adds inductance)",
            "Via or connector discontinuity causing impedance mismatch",
        ])
        suggestions = r.get("suggestions", [
            "Add a series damping resistor (22–100 Ω) at the driver output",
            "Shorten probe ground lead or use a ground spring clip",
            "Match line impedance with proper source or load termination",
            "Use ground planes and minimize loop areas in PCB layout",
        ])

        return Issue(
            name="Ringing / Underdamped Response",
            severity="Warning",
            description=(
                f"Ringing detected on signal edges ({ov:.1f}% overshoot with fast rise time). "
                "This indicates an underdamped LC resonance in the signal path."
            ),
            measured_values={
                "overshoot_pct": round(ov, 2),
                "rise_time_ns": round(m.rise_time_s * 1e9, 2) if m.rise_time_s else "N/A",
            },
            possible_causes=causes,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Rule: Jitter
    # ------------------------------------------------------------------

    def _check_jitter(self, m: MeasurementReport) -> Optional[Issue]:
        jitter_ns = m.jitter_ns
        period_s = m.period_s
        if jitter_ns is None or jitter_ns <= 0:
            return None

        r = _RULES.get("jitter", _DEFAULT_RULES["jitter"])
        warning_ns = r.get("warning_ns", 5.0)
        critical_ns = r.get("critical_ns", 20.0)
        warning_pct = r.get("warning_pct_period", 0.5)
        critical_pct = r.get("critical_pct_period", 2.0)

        jitter_pct = (jitter_ns * 1e-9 / period_s * 100) if period_s else 0

        if jitter_ns < warning_ns and jitter_pct < warning_pct:
            return None

        severity = "Critical" if (jitter_ns >= critical_ns or jitter_pct >= critical_pct) else "Warning"
        causes = r.get("possible_causes", [
            "EMI coupling into the signal path",
            "Power supply noise modulating the switching threshold",
            "PLL / clock source instability",
            "Insufficient ground plane or poor PCB layout",
        ])
        suggestions = r.get("suggestions", [
            "Improve shielding and reduce EMI sources near the signal path",
            "Add bulk and high-frequency decoupling capacitors to power rails",
            "Check PLL loop filter for synthesized clocks",
            "Improve PCB ground plane continuity",
        ])

        return Issue(
            name="Clock Jitter",
            severity=severity,
            description=(
                f"Cycle-to-cycle jitter is {jitter_ns:.2f} ns "
                f"({jitter_pct:.2f}% of period). This indicates timing instability."
            ),
            measured_values={
                "jitter_ns": round(jitter_ns, 3),
                "jitter_pct_period": round(jitter_pct, 3),
            },
            possible_causes=causes,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Rule: Noise
    # ------------------------------------------------------------------

    def _check_noise(self, m: MeasurementReport) -> Optional[Issue]:
        if m.noise_level == "Low":
            return None

        r = _RULES.get("noise", _DEFAULT_RULES["noise"])
        severity = "Critical" if m.noise_level == "High" else "Warning"
        causes = r.get("possible_causes", [
            "EMI pickup from nearby switching supplies or RF sources",
            "Ground loop — multiple ground paths with different potentials",
            "Insufficient high-frequency decoupling on power rails",
        ])
        suggestions = r.get("suggestions", [
            "Add 100 nF ceramic bypass capacitors close to every IC power pin",
            "Use a solid, uninterrupted ground plane",
            "Route sensitive signals away from switching power or clock lines",
        ])

        return Issue(
            name=f"{m.noise_level} Noise Level",
            severity=severity,
            description=(
                f"Noise in settled regions is {m.noise_std_mv:.2f} mV RMS "
                f"(level: {m.noise_level}). This exceeds the acceptable threshold."
            ),
            measured_values={"noise_std_mv": round(m.noise_std_mv, 3), "noise_level": m.noise_level},
            possible_causes=causes,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Rule: Ripple
    # ------------------------------------------------------------------

    def _check_ripple(self, m: MeasurementReport) -> Optional[Issue]:
        ripple = m.ripple_mv
        if ripple is None or ripple <= 0:
            return None

        r = _RULES.get("ripple", _DEFAULT_RULES["ripple"])
        warning_mv = r.get("warning_mv", 50.0)
        critical_mv = r.get("critical_mv", 200.0)
        vpp_mv = m.peak_to_peak_v * 1000

        warning_pct = r.get("warning_pct_amplitude", 1.0)
        ripple_pct = (ripple / vpp_mv * 100) if vpp_mv > 0 else 0

        if ripple < warning_mv and ripple_pct < warning_pct:
            return None

        severity = "Critical" if ripple >= critical_mv else "Warning"
        causes = r.get("possible_causes", [
            "Insufficient output capacitance on the power supply",
            "Switching frequency ripple from a DC-DC converter",
            "Line frequency (50/60 Hz) hum from an unfiltered AC-DC supply",
        ])
        suggestions = r.get("suggestions", [
            "Increase output filter capacitance (bulk electrolytic + ceramic in parallel)",
            "Add a post-regulation LC filter at the load",
            "Use an LDO regulator after the switching supply for noise-sensitive circuits",
        ])

        return Issue(
            name="Power Supply Ripple",
            severity=severity,
            description=(
                f"Low-frequency ripple of {ripple:.1f} mV detected on the signal. "
                "This typically indicates power supply filtering issues."
            ),
            measured_values={"ripple_mv": round(ripple, 2), "ripple_pct": round(ripple_pct, 2)},
            possible_causes=causes,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Rule: Clipping
    # ------------------------------------------------------------------

    def _check_clipping(self, m: MeasurementReport) -> Optional[Issue]:
        """
        Detect clipping by checking if the signal peak is 'flat'.
        We look at what fraction of samples are at or very near the peak.
        If more than threshold_pct% of samples are within 2% of peak, it's clipped.
        """
        if m.peak_to_peak_v < 1e-9:
            return None

        # This check is primarily relevant for non-square waves
        if "Square" in m.waveform_type or "DC" in m.waveform_type:
            return None

        r = _RULES.get("clipping", _DEFAULT_RULES["clipping"])
        flatness_pct = r.get("flatness_threshold_pct", 2.0)
        causes = r.get("possible_causes", [
            "Amplifier output saturated — input signal is too large",
            "Output voltage limited by supply rail (exceeds Vcc or GND)",
            "ADC input range exceeded (digital clipping in acquisition hardware)",
        ])
        suggestions = r.get("suggestions", [
            "Reduce input signal amplitude to keep output within linear range",
            "Check if supply voltage is sufficient for the required output swing",
            "Verify amplifier output voltage specification vs. observed clipping level",
        ])

        # For a typical oscilloscope image extraction, this would be a flat-top
        # We can't easily detect this from measurements alone without raw samples.
        # We report it if the waveform_type was classified as "Complex" and
        # overshoot is very low (flat tops, not peaks) — this is a heuristic.
        # A more reliable check requires knowing if the waveform was expected to
        # peak higher but is limited.
        # We skip this rule if we don't have enough data to be confident.
        return None  # Placeholder — would be enhanced with raw sample analysis

    # ------------------------------------------------------------------
    # Rule: Slow Rise Time
    # ------------------------------------------------------------------

    def _check_slow_rise_time(self, m: MeasurementReport) -> Optional[Issue]:
        rt = m.rise_time_s
        period = m.period_s
        if rt is None or period is None:
            return None

        r = _RULES.get("slow_rise_time", _DEFAULT_RULES["slow_rise_time"])
        threshold_pct = r.get("slow_threshold_pct_period", 10.0)

        rt_pct = (rt / period) * 100
        if rt_pct < threshold_pct:
            return None

        causes = r.get("possible_causes", [
            "Excessive capacitive load on the driver output",
            "Signal bandwidth limited by an RC filter",
            "Driver output current limited (cannot charge load capacitance quickly)",
            "Scope bandwidth set too low (apparent issue only)",
        ])
        suggestions = r.get("suggestions", [
            "Reduce capacitive load or add a buffer between driver and load",
            "Check scope bandwidth — should be at least 5× the signal frequency",
            "Use a higher-current driver or buffer amplifier",
        ])

        return Issue(
            name="Slow Rise Time",
            severity="Warning",
            description=(
                f"Rise time ({rt * 1e9:.1f} ns) is {rt_pct:.1f}% of the signal period. "
                f"This may limit signal integrity at the target frequency."
            ),
            measured_values={
                "rise_time_ns": round(rt * 1e9, 2),
                "rise_time_pct_period": round(rt_pct, 2),
            },
            possible_causes=causes,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Rule: DC Offset
    # ------------------------------------------------------------------

    def _check_dc_offset(self, m: MeasurementReport) -> Optional[Issue]:
        """
        Report significant DC offset: average voltage is far from zero
        relative to the signal amplitude.
        """
        if m.amplitude_v < 1e-9:
            return None

        r = _RULES.get("dc_offset", _DEFAULT_RULES["dc_offset"])
        warning_pct = r.get("warning_pct_amplitude", 10.0)

        offset_pct = abs(m.average_v) / m.amplitude_v * 100
        if offset_pct < warning_pct:
            return None

        causes = r.get("possible_causes", [
            "Intentional DC bias (AC signal riding on a DC rail)",
            "Leakage current through coupling capacitors",
            "Scope probe ground reference at different potential than signal ground",
        ])
        suggestions = r.get("suggestions", [
            "Use AC coupling on the scope channel to view only the AC component",
            "Verify scope probe ground reference is at the correct circuit node",
            "Check if DC offset is expected for the circuit's operating point",
        ])

        return Issue(
            name="Significant DC Offset",
            severity="Info",
            description=(
                f"Average voltage is {m.average_v:.3f} V ({offset_pct:.1f}% of amplitude). "
                "The signal has a notable DC component."
            ),
            measured_values={
                "average_v": round(m.average_v, 4),
                "offset_pct_amplitude": round(offset_pct, 2),
            },
            possible_causes=causes,
            suggestions=suggestions,
        )
