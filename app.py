"""
AI Oscilloscope Copilot — Main Streamlit Application
=====================================================

Sections:
  1. Upload oscilloscope image
  2. Analyze Waveform (triggers full pipeline)
  3. Measurement Results (auto-shown after analysis)
  4. Signal Integrity Diagnosis (on-demand)
  5. AI Chat (conversational assistant grounded in measured data)
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from modules.image_processor import ImageProcessor, ScaleInfo, WaveformTrace
from modules.dsp_engine import DSPEngine, MeasurementReport
from modules.diagnosis_engine import DiagnosisEngine, DiagnosisReport
from modules.ai_assistant import AIAssistant, AVAILABLE_MODELS, DEFAULT_MODEL, PROVIDERS, DEFAULT_PROVIDER

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Oscilloscope Copilot",
    page_icon="~",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI Oscilloscope Copilot — Intelligent waveform analysis and conversational assistant.",
        "Report a bug": None,
        "Get help": None,
    },
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def _load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_ocr_reader():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:
        logger.warning(f"EasyOCR unavailable: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def _get_image_processor() -> ImageProcessor:
    return ImageProcessor()


@st.cache_resource(show_spinner=False)
def _get_dsp_engine() -> DSPEngine:
    return DSPEngine()


@st.cache_resource(show_spinner=False)
def _get_diagnosis_engine() -> DiagnosisEngine:
    return DiagnosisEngine()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "uploaded_hash": None,
        "trace": None,
        "measurements": None,
        "diagnosis": None,
        "chat_history": [],
        "analysis_done": False,
        "diagnosis_done": False,
        "image_bytes": None,
        "api_key": "",
        "provider": DEFAULT_PROVIDER,
        "model": DEFAULT_MODEL,
        "base_url": PROVIDERS[DEFAULT_PROVIDER]["base_url"],
        "use_ocr": False,
        "user_v_per_div": 1.0,
        "user_t_per_div": 1.0,
        "user_v_unit": "V",
        "user_t_unit": "ms",
        "override_scale": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _image_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _fmt_freq(hz: Optional[float]) -> str:
    if hz is None:
        return "N/A"
    if hz >= 1e6:
        return f"{hz/1e6:.3f} MHz"
    if hz >= 1e3:
        return f"{hz/1e3:.3f} kHz"
    return f"{hz:.2f} Hz"


def _fmt_time(s: Optional[float]) -> str:
    if s is None:
        return "N/A"
    if s < 1e-9:
        return f"{s*1e12:.2f} ps"
    if s < 1e-6:
        return f"{s*1e9:.2f} ns"
    if s < 1e-3:
        return f"{s*1e6:.2f} us"
    if s < 1.0:
        return f"{s*1e3:.3f} ms"
    return f"{s:.4f} s"


def _fmt_v(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    if abs(v) < 1e-3:
        return f"{v*1e6:.2f} uV"
    if abs(v) < 1.0:
        return f"{v*1e3:.2f} mV"
    return f"{v:.4f} V"


def _fmt_pct(p: Optional[float]) -> str:
    if p is None:
        return "N/A"
    return f"{p:.2f}%"


def _severity_color(severity: str) -> str:
    return {"Critical": "#ff4444", "Warning": "#ffaa00", "Info": "#00ccff"}.get(severity, "#7dbd9a")


def _health_color(health: str) -> str:
    return {"Good": "#00ff88", "Fair": "#ffaa00", "Poor": "#ff4444"}.get(health, "#7dbd9a")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1.2rem;">
                <div style="font-size:1.1rem; font-weight:700; color:#00ff88; letter-spacing:0.04em;">
                    AI Oscilloscope Copilot
                </div>
                <div style="font-size:0.72rem; color:#5a8a6a; margin-top:0.2rem;">
                    Powered by Google Gemini
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # --- Provider selector ---
        st.markdown("**AI Provider**")
        provider_names = list(PROVIDERS.keys())
        current_provider_idx = provider_names.index(st.session_state.provider) if st.session_state.provider in provider_names else 0
        chosen_provider = st.selectbox(
            "Provider",
            options=provider_names,
            index=current_provider_idx,
            label_visibility="collapsed",
        )
        if chosen_provider != st.session_state.provider:
            # Provider changed: reset model to first in new provider's list
            st.session_state.provider = chosen_provider
            st.session_state.base_url = PROVIDERS[chosen_provider]["base_url"]
            st.session_state.model = PROVIDERS[chosen_provider]["models"][0]

        provider_info = PROVIDERS[st.session_state.provider]
        st.caption(provider_info["notes"])

        # --- API Key ---
        st.markdown("**API Key**")
        default_key = ""
        try:
            default_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        except Exception:
            default_key = os.getenv("GOOGLE_API_KEY", "")

        if default_key and not st.session_state.api_key:
            st.session_state.api_key = default_key

        key_input = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder=provider_info["key_hint"],
            label_visibility="collapsed",
        )
        if key_input != st.session_state.api_key:
            st.session_state.api_key = key_input

        # --- Model selector (provider-specific) ---
        st.markdown("**Model**")
        provider_models = provider_info["models"]
        model_idx = provider_models.index(st.session_state.model) if st.session_state.model in provider_models else 0
        chosen_model = st.selectbox(
            "Model",
            options=provider_models,
            index=model_idx,
            label_visibility="collapsed",
        )
        st.session_state.model = chosen_model

        # --- Test button ---
        if st.button("Test API Key", use_container_width=True, key="btn_test_key"):
            if st.session_state.api_key:
                with st.spinner("Testing connection..."):
                    _test_assistant = AIAssistant(
                        api_key=st.session_state.api_key,
                        model=st.session_state.model,
                        base_url=st.session_state.base_url,
                    )
                    ok, msg = _test_assistant.validate_key()
                if ok:
                    st.success(f"Connection OK")
                else:
                    st.error(f"Failed: {msg}")
            else:
                st.warning("Enter an API key first.")

        st.divider()

        # Analysis options
        st.markdown("**Analysis Options**")
        use_ocr = st.checkbox(
            "Use OCR for scale detection",
            value=st.session_state.use_ocr,
            help=(
                "EasyOCR reads V/div and time/div text from the image. "
                "Downloads ~1.5 GB on first use. "
                "Disable to enter scale values manually."
            ),
        )
        st.session_state.use_ocr = use_ocr

        override = st.checkbox(
            "Override scale manually",
            value=st.session_state.override_scale,
            help="Enter V/div and time/div if OCR is disabled or inaccurate.",
        )
        st.session_state.override_scale = override

        if override:
            with st.expander("Manual Scale", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    v_val = st.number_input(
                        "V/div", value=st.session_state.user_v_per_div,
                        min_value=0.001, step=0.1, format="%.4f"
                    )
                    st.session_state.user_v_per_div = v_val
                with col2:
                    v_unit = st.selectbox(
                        "Unit##v", ["V", "mV"],
                        index=0 if st.session_state.user_v_unit == "V" else 1
                    )
                    st.session_state.user_v_unit = v_unit

                col3, col4 = st.columns(2)
                with col3:
                    t_val = st.number_input(
                        "Time/div", value=st.session_state.user_t_per_div,
                        min_value=0.001, step=0.1, format="%.4f"
                    )
                    st.session_state.user_t_per_div = t_val
                with col4:
                    t_unit = st.selectbox("Unit##t", ["ms", "us", "ns", "s"], index=0)
                    st.session_state.user_t_unit = t_unit

        st.divider()

        if st.button("Reset Analysis", use_container_width=True):
            for key in ["trace", "measurements", "diagnosis", "uploaded_hash", "image_bytes"]:
                st.session_state[key] = None
            st.session_state.analysis_done = False
            st.session_state.diagnosis_done = False
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.markdown(
            """
            <div style="font-size:0.72rem; color:#3a6a4a; line-height:1.7;">
            Upload an oscilloscope image.<br>
            The system extracts, measures, and diagnoses<br>
            the waveform, then lets you discuss it with AI.<br><br>
            <b>Measurements</b> — deterministic DSP<br>
            <b>Diagnosis</b> — rule-based expert logic<br>
            <b>Chat</b> — Gemini grounded in measured data
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_title():
    st.markdown(
        """
        <div class="app-title">
            <h1>AI Oscilloscope Copilot</h1>
            <p>Upload an oscilloscope image — get measurements, signal diagnosis, and an AI assistant that knows your exact waveform.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_upload_section() -> Optional[bytes]:
    st.markdown(
        '<div class="section-header"><h2>Upload Oscilloscope Image</h2>'
        '<span class="section-badge">Step 1</span></div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose an oscilloscope screenshot or photograph",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        help="Accepts screenshots (PNG/JPEG), scope USB exports, and photographs of oscilloscope screens.",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image_bytes = uploaded.read()
        img_hash = _image_hash(image_bytes)

        if img_hash != st.session_state.uploaded_hash:
            st.session_state.uploaded_hash = img_hash
            st.session_state.image_bytes = image_bytes
            st.session_state.trace = None
            st.session_state.measurements = None
            st.session_state.diagnosis = None
            st.session_state.diagnosis_done = False
            st.session_state.analysis_done = False
            st.session_state.chat_history = []

        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.markdown('<div class="scope-image-container">', unsafe_allow_html=True)
            st.image(image_bytes, caption="Uploaded image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_info:
            size_kb = len(image_bytes) / 1024
            st.markdown(
                f"""
                <div style="padding: 1rem; background: rgba(0,255,136,0.04);
                     border: 1px solid rgba(0,255,136,0.15); border-radius: 8px;">
                    <div style="color:#7dbd9a; font-size:0.75rem; text-transform:uppercase;
                         letter-spacing:0.06em; font-weight:600; margin-bottom:0.6rem;">
                        Image Info
                    </div>
                    <div style="font-size:0.85rem; color:#c0d4c8; line-height:1.8;">
                        Name: <b style="color:#e0f0e8;">{uploaded.name}</b><br>
                        Size: {size_kb:.1f} KB<br>
                        Type: {uploaded.type or "image"}<br>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.session_state.analysis_done:
                st.markdown(
                    '<div class="no-issues-banner" style="margin-top:0.8rem;">Analysis complete</div>',
                    unsafe_allow_html=True,
                )

        return image_bytes
    else:
        st.info(
            "Drag and drop an oscilloscope image above, or click Browse. "
            "Accepts PNG, JPEG, BMP from any oscilloscope brand."
        )
        return None


def _run_analysis(image_bytes: bytes):
    processor = _get_image_processor()
    dsp = _get_dsp_engine()

    user_scale = None
    if st.session_state.override_scale:
        multipliers = {"V": 1.0, "mV": 1e-3, "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
        v_per_div = st.session_state.user_v_per_div * multipliers.get(st.session_state.user_v_unit, 1.0)
        t_per_div = st.session_state.user_t_per_div * multipliers.get(st.session_state.user_t_unit, 1e-3)
        user_scale = ScaleInfo(
            v_per_div=v_per_div,
            t_per_div=t_per_div,
            v_unit=st.session_state.user_v_unit,
            t_unit=st.session_state.user_t_unit,
            source="user",
        )

    ocr_reader = None
    if st.session_state.use_ocr and not st.session_state.override_scale:
        with st.spinner("Loading OCR engine (first-run downloads ~1.5 GB, subsequent runs are instant)..."):
            ocr_reader = _get_ocr_reader()

    with st.spinner("Processing image and extracting waveform trace..."):
        trace = processor.process(image_bytes, ocr_reader=ocr_reader, user_scale=user_scale)

    with st.spinner("Computing DSP measurements..."):
        measurements = dsp.compute(trace.times, trace.voltages)

    st.session_state.trace = trace
    st.session_state.measurements = measurements
    st.session_state.analysis_done = True
    st.session_state.diagnosis = None
    st.session_state.diagnosis_done = False


def _render_measurements(m: MeasurementReport):
    st.markdown(
        '<div class="section-header"><h2>Measurement Results</h2>'
        '<span class="section-badge">Auto</span></div>',
        unsafe_allow_html=True,
    )

    # Waveform type badge
    st.markdown(
        f'<div style="margin-bottom:1.2rem;">'
        f'<span style="color:#7dbd9a; font-size:0.8rem; font-weight:600; '
        f'text-transform:uppercase; letter-spacing:0.06em; margin-right:0.8rem;">Waveform Type</span>'
        f'<span class="waveform-badge">{m.waveform_type}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Timing
    st.markdown(
        '<div style="color:#7dbd9a; font-size:0.75rem; font-weight:700; '
        'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Timing</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Frequency", _fmt_freq(m.frequency_hz))
    c2.metric("Period", _fmt_time(m.period_s))
    c3.metric("Duty Cycle", _fmt_pct(m.duty_cycle_pct))
    c4.metric("Rise Time", _fmt_time(m.rise_time_s))
    c5.metric("Fall Time", _fmt_time(m.fall_time_s))

    c6, c7, _, _, _ = st.columns(5)
    c6.metric("Settling Time", _fmt_time(m.settling_time_s))
    c7.metric("Cycles Detected", str(m.num_cycles) if m.num_cycles else "N/A")

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # Voltage
    st.markdown(
        '<div style="color:#7dbd9a; font-size:0.75rem; font-weight:700; '
        'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem; margin-top:0.4rem;">Voltage</div>',
        unsafe_allow_html=True,
    )
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("Amplitude", _fmt_v(m.amplitude_v))
    v2.metric("Peak (Vp)", _fmt_v(m.peak_v))
    v3.metric("Peak-to-Peak", _fmt_v(m.peak_to_peak_v))
    v4.metric("RMS", _fmt_v(m.rms_v))
    v5.metric("Average", _fmt_v(m.average_v))

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # Signal Quality
    st.markdown(
        '<div style="color:#7dbd9a; font-size:0.75rem; font-weight:700; '
        'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem; margin-top:0.4rem;">Signal Quality</div>',
        unsafe_allow_html=True,
    )
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Overshoot", _fmt_pct(m.overshoot_pct))
    q2.metric("Undershoot", _fmt_pct(m.undershoot_pct))
    q3.metric("Noise Level", m.noise_level)
    q4.metric("Noise (RMS)", f"{m.noise_std_mv:.2f} mV")
    q5.metric("Jitter", _fmt_time(m.jitter_ns * 1e-9 if m.jitter_ns else None))

    q6, _, _, _, _ = st.columns(5)
    q6.metric("Ripple", f"{m.ripple_mv:.1f} mV" if m.ripple_mv else "N/A")

    # Annotated image
    trace = st.session_state.trace
    if trace is not None and trace.annotated_image is not None:
        with st.expander("View Detected Waveform Overlay", expanded=False):
            annotated_rgb = cv2.cvtColor(trace.annotated_image, cv2.COLOR_BGR2RGB)
            st.image(
                annotated_rgb,
                caption="Extracted trace overlay (cyan) on detected grid area (green)",
                use_container_width=True,
            )
            scale = trace.scale_info
            st.caption(
                f"Scale source: **{scale.source}** | "
                f"V/div: **{scale.v_per_div*1000:.2f} mV** | "
                f"Time/div: **{_fmt_time(scale.t_per_div)}** | "
                f"Grid: {trace.grid_info.n_vdiv} x {trace.grid_info.n_hdiv} divisions"
            )
            if trace.ocr_texts:
                with st.expander("OCR detected text"):
                    st.code(", ".join(trace.ocr_texts))


def _render_diagnosis_section(m: MeasurementReport):
    st.markdown(
        '<div class="section-header"><h2>Signal Integrity Diagnosis</h2>'
        '<span class="section-badge">On Demand</span></div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Run the rule-based diagnosis engine to detect common signal integrity issues "
        "(overshoot, ringing, jitter, noise, ripple) with causes and suggestions."
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_diag = st.button("Detect Wave Problems", use_container_width=True, key="btn_diagnose")

    if run_diag:
        engine = _get_diagnosis_engine()
        with st.spinner("Running diagnosis engine..."):
            diagnosis = engine.diagnose(m)
        st.session_state.diagnosis = diagnosis
        st.session_state.diagnosis_done = True

    if st.session_state.diagnosis_done and st.session_state.diagnosis is not None:
        _render_diagnosis_results(st.session_state.diagnosis)


def _render_diagnosis_results(d: DiagnosisReport):
    health_color = _health_color(d.overall_health)
    bar_class = d.overall_health.lower()

    st.markdown(
        f"""
        <div style="background: rgba(0,0,0,0.3); border: 1px solid {health_color}33;
             border-radius: 10px; padding: 1rem 1.4rem; margin: 0.8rem 0 1.2rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                <span style="color:{health_color}; font-weight:700; font-size:1rem;">
                    Signal Health: {d.overall_health}
                </span>
                <span style="color:{health_color}; font-family:'JetBrains Mono',monospace;
                      font-size:1.1rem; font-weight:600;">{d.health_score}/100</span>
            </div>
            <div class="health-bar-container">
                <div class="health-bar {bar_class}" style="width:{d.health_score}%"></div>
            </div>
            <div style="color:#5a8a6a; font-size:0.78rem; margin-top:0.4rem;">
                {d.critical_count} critical issue(s) &middot; {d.warning_count} warning(s)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not d.has_issues:
        st.markdown(
            '<div class="no-issues-banner">No significant signal integrity issues detected. '
            'The waveform appears healthy within normal operating parameters.</div>',
            unsafe_allow_html=True,
        )
        return

    for issue in d.issues:
        sev_color = _severity_color(issue.severity)
        label_class = issue.severity.lower()

        causes_html = "".join(f"<li>{c}</li>" for c in issue.possible_causes)
        suggestions_html = "".join(f"<li>{s}</li>" for s in issue.suggestions)

        st.markdown(
            f"""
            <div class="issue-card {'critical' if issue.severity == 'Critical' else ''}">
                <div>
                    <span class="issue-label {label_class}">{issue.severity}</span>
                    <h4 style="color:{sev_color};">{issue.name}</h4>
                </div>
                <p style="margin-bottom:0.5rem;">{issue.description}</p>
                <details style="margin-top:0.5rem;">
                    <summary style="color:#7dbd9a; font-size:0.82rem; cursor:pointer;
                               font-weight:600; outline:none;">
                        Possible Causes &amp; Suggestions
                    </summary>
                    <div style="margin-top:0.4rem;">
                        <div style="color:#9aaa9a; font-size:0.8rem; font-weight:600;
                             margin:0.4rem 0 0.2rem;">Possible Causes:</div>
                        <ul style="color:#c0d4c8; font-size:0.83rem; padding-left:1.2rem;
                              margin:0 0 0.6rem 0;">{causes_html}</ul>
                        <div style="color:#9aaa9a; font-size:0.8rem; font-weight:600;
                             margin:0.4rem 0 0.2rem;">Suggestions:</div>
                        <ul style="color:#c0d4c8; font-size:0.83rem; padding-left:1.2rem;
                              margin:0;">{suggestions_html}</ul>
                    </div>
                </details>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_chat_section(m: MeasurementReport):
    st.markdown(
        '<div class="section-header"><h2>Ask the AI Copilot</h2>'
        '<span class="section-badge">AI Chat</span></div>',
        unsafe_allow_html=True,
    )

    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.warning(
            "**API key required.** Enter your Google AI Studio API key in the sidebar to enable AI chat."
        )
        return

    diagnosis: Optional[DiagnosisReport] = (
        st.session_state.diagnosis if st.session_state.diagnosis_done else None
    )

    diag_note = (
        "Diagnosis data included in AI context."
        if diagnosis is not None
        else "Run **Detect Wave Problems** above to include diagnosis context."
    )
    st.caption(f"The AI has full access to your waveform measurements. {diag_note}")

    # Chat history
    st.markdown('<div class="chat-container" id="chat-area">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        role_label = "You" if role == "user" else "AI Copilot"
        bubble_class = "user" if role == "user" else "assistant"
        st.markdown(
            f"""
            <div class="chat-bubble {bubble_class}">
                <div class="chat-role">{role_label}</div>
                <div>{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Suggested starter questions
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='color:#5a8a6a; font-size:0.78rem; margin-bottom:0.5rem;'>Try asking:</div>",
            unsafe_allow_html=True,
        )
        suggestions = [
            "Why is there ringing on this signal?",
            "Is this waveform suitable for SPI?",
            "Explain what overshoot means here.",
            "How do I improve the rise time?",
            "Compare this to an ideal square wave.",
        ]
        sug_cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            if sug_cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
                _send_chat_message(sug, m, diagnosis)
                st.rerun()

    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Your question",
                placeholder="Ask anything about this waveform...",
                label_visibility="collapsed",
                key="chat_input_field",
            )
        with col_send:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        _send_chat_message(user_input.strip(), m, diagnosis)
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


def _send_chat_message(
    user_text: str,
    m: MeasurementReport,
    diagnosis: Optional[DiagnosisReport],
):
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    st.markdown(
        f'<div class="chat-bubble user"><div class="chat-role">You</div><div>{user_text}</div></div>',
        unsafe_allow_html=True,
    )

    api_key = st.session_state.api_key
    model = st.session_state.model
    base_url = st.session_state.get("base_url", PROVIDERS[DEFAULT_PROVIDER]["base_url"])
    assistant = AIAssistant(api_key=api_key, model=model, base_url=base_url)

    messages_for_api = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.chat_history
    ]

    response_placeholder = st.empty()
    response_text = ""

    with st.spinner("AI Copilot is thinking..."):
        try:
            for chunk in assistant.chat(messages_for_api, m, diagnosis):
                response_text += chunk
                response_placeholder.markdown(
                    f'<div class="chat-bubble assistant">'
                    f'<div class="chat-role">AI Copilot</div>'
                    f'<div>{response_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            response_text = f"Error: {exc}"
            response_placeholder.error(response_text)

    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main():
    _render_sidebar()
    _render_title()
    st.divider()

    # Section 1 — Upload
    image_bytes = _render_upload_section()
    if image_bytes is None:
        st.stop()

    # Section 2 — Analyze
    st.markdown(
        '<div class="section-header"><h2>Analyze Waveform</h2>'
        '<span class="section-badge">Step 2</span></div>',
        unsafe_allow_html=True,
    )

    col_analyze, col_note = st.columns([1, 4])
    with col_analyze:
        analyze_clicked = st.button(
            "Analyze Waveform",
            use_container_width=True,
            key="btn_analyze",
            help="Run the full image processing and DSP measurement pipeline.",
        )
    with col_note:
        if not st.session_state.analysis_done:
            st.markdown(
                '<div style="padding-top:0.65rem; color:#5a8a6a; font-size:0.85rem;">'
                'Click to extract the waveform and compute all measurements.</div>',
                unsafe_allow_html=True,
            )

    if analyze_clicked:
        try:
            _run_analysis(image_bytes)
            st.success("Analysis complete. Measurements are displayed below.")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            logger.exception("Analysis error")
            st.stop()

    if not st.session_state.analysis_done or st.session_state.measurements is None:
        st.info("Upload an image and click Analyze Waveform to begin.")
        st.stop()

    m: MeasurementReport = st.session_state.measurements
    st.divider()

    # Section 3 — Measurements
    _render_measurements(m)
    st.divider()

    # Section 4 — Diagnosis
    _render_diagnosis_section(m)
    st.divider()

    # Section 5 — AI Chat
    _render_chat_section(m)

    _save_report(m)


def _save_report(m: MeasurementReport):
    try:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "latest_measurements.json", "w", encoding="utf-8") as f:
            f.write(m.to_json())
        if st.session_state.diagnosis_done and st.session_state.diagnosis:
            with open(output_dir / "latest_diagnosis.json", "w", encoding="utf-8") as f:
                f.write(st.session_state.diagnosis.to_json())
    except Exception:
        pass


if __name__ == "__main__":
    main()