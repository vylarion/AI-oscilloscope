"""AI Oscilloscope Copilot — Core modules package."""

from .image_processor import ImageProcessor, WaveformTrace, GridInfo, ScaleInfo
from .dsp_engine import DSPEngine, MeasurementReport
from .diagnosis_engine import DiagnosisEngine, DiagnosisReport, Issue
from .ai_assistant import AIAssistant

__all__ = [
    "ImageProcessor", "WaveformTrace", "GridInfo", "ScaleInfo",
    "DSPEngine", "MeasurementReport",
    "DiagnosisEngine", "DiagnosisReport", "Issue",
    "AIAssistant",
]
