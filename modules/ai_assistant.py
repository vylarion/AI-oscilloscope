"""
AI Conversational Assistant — multi-provider, OpenAI-compatible API.

Supports Google AI Studio, OpenRouter, Groq, and any other provider
that implements the OpenAI chat completions API.

Entry point:
    assistant = AIAssistant(api_key=..., model=..., base_url=...)
    for chunk in assistant.chat(messages, measurements, diagnosis):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Optional

from openai import OpenAI

from .dsp_engine import MeasurementReport
from .diagnosis_engine import DiagnosisReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider presets — base URLs, model lists, and key hints
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    "Google AI Studio": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
        ],
        "key_hint": "AIzaSy...  —  aistudio.google.com/app/apikey",
        "notes": "Free tier: 15 req/min, 1500 req/day. Quota resets daily.",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "deepseek/deepseek-r1-0528:free",
            "microsoft/phi-3-mini-128k-instruct:free",
        ],
        "key_hint": "sk-or-...  —  openrouter.ai/keys",
        "notes": "Free models are marked :free. No quota issues on free models.",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "llama3-8b-8192",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "key_hint": "gsk_...  —  console.groq.com/keys",
        "notes": "Very fast. Generous free tier. Great for engineering chat.",
    },
}

DEFAULT_PROVIDER = "Google AI Studio"
DEFAULT_MODEL = "gemini-2.0-flash"

# Flat model list kept for backward compatibility
AVAILABLE_MODELS = PROVIDERS[DEFAULT_PROVIDER]["models"]


class AIAssistant:
    """
    Wraps any OpenAI-compatible API for streaming chat completions.
    Provider is selected by passing the appropriate base_url and api_key.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = PROVIDERS[DEFAULT_PROVIDER]["base_url"],
    ):
        """
        Args:
            api_key:  Provider API key.
            model:    Model ID string (provider-specific).
            base_url: OpenAI-compatible base URL for the provider.
        """
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    # ------------------------------------------------------------------
    # System prompt builder
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        measurements: MeasurementReport,
        diagnosis: Optional[DiagnosisReport] = None,
    ) -> str:
        """
        Build the system prompt injecting waveform measurement and diagnosis JSON.
        """
        meas_json = measurements.to_json()
        diag_section = ""
        if diagnosis is not None:
            diag_json = diagnosis.to_json()
            diag_section = f"""
## Signal Integrity Diagnosis (user has run the diagnosis tool)
```json
{diag_json}
```
"""
        else:
            diag_section = """
## Signal Integrity Diagnosis
The user has not yet run the diagnosis tool. If they ask about problems,
you may infer potential issues from the measurements but clearly note that
a formal diagnosis has not been performed.
"""

        return f"""You are an expert electronics and signal integrity engineer with 20+ years of experience, acting as an AI Oscilloscope Copilot.

The user has uploaded an oscilloscope image which has been analyzed automatically. Here is the complete, verified waveform analysis:

## Waveform Measurements (computed from the uploaded image via DSP)
```json
{meas_json}
```
{diag_section}

## Your Role
1. **Ground every answer in the measured data above.** Reference specific numbers when relevant (e.g., "your rise time of X ns", "given the measured frequency of Y Hz").
2. **Never guess or re-analyze the image yourself.** You only have access to the data provided above, and that is all you need.
3. **Adapt explanation depth** to the complexity of the user's question:
   - Beginner questions → clear analogies, avoid jargon, explain why things happen
   - Advanced/technical questions → use proper terminology, dive into circuit theory, cite relevant standards (IEEE, JEDEC, etc.) where appropriate
4. **Be direct and practical.** Engineers want answers they can act on, not lengthy disclaimers.
5. **If asked about something the measurements can't answer** (e.g., asking about a different channel not analyzed), clearly explain what additional measurement or data would be needed.
6. **If the diagnosis hasn't been run yet** and the user asks about problems, guide them to click the "Detect Wave Problems" button, or you may discuss what the measurements suggest.

## Communication Style
- Use Markdown formatting (bold, code blocks, bullet lists) for clarity.
- Keep answers concise but complete. Don't pad with filler text.
- When citing measured values, use units: "8 ns", "4.98 V", "998 Hz".
- Feel free to explain underlying physics/circuit theory when it helps understanding.
"""

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        measurements: MeasurementReport,
        diagnosis: Optional[DiagnosisReport] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response from the Gemini model.

        Args:
            messages:     List of {"role": "user"|"assistant", "content": str}
                          representing the conversation history (most recent last).
            measurements: Current waveform MeasurementReport.
            diagnosis:    DiagnosisReport if diagnosis has been run, else None.

        Yields:
            Text chunks (str) as they stream from the API.
        """
        system_prompt = self.build_system_prompt(measurements, diagnosis)

        # Build OpenAI-format messages list
        api_messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                stream=True,
                temperature=0.4,    # Slightly lower temperature for factual accuracy
                max_tokens=2048,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        except Exception as exc:
            logger.error(f"AI API error: {exc}")
            error_msg = str(exc)

            if "API_KEY" in error_msg.upper() or "401" in error_msg or "403" in error_msg:
                yield (
                    f"**Authentication Error** — the API key was rejected by Google AI Studio.\n\n"
                    f"**Raw error:** `{error_msg}`\n\n"
                    f"Check that the key is a valid Google AI Studio API key "
                    f"(get one at [aistudio.google.com](https://aistudio.google.com/app/apikey)). "
                    f"Keys look like `AIzaSy...` — not a Cloud service account token."
                )
            elif (
                "quota" in error_msg.lower()
                or "RESOURCE_EXHAUSTED" in error_msg
                or "429" in error_msg
                or "rate" in error_msg.lower()
            ):
                yield (
                    f"**Quota / Rate Limit Error**\n\n"
                    f"**Raw error:** `{error_msg}`\n\n"
                    f"Possible causes:\n"
                    f"- Free tier is limited to **15 requests/min**. Wait ~1 minute and retry.\n"
                    f"- The model `{self.model}` may require billing on your account. "
                    f"Try switching to `gemini-2.0-flash-lite` in the sidebar.\n"
                    f"- Your Google AI Studio account may not have the Gemini API enabled. "
                    f"Visit [aistudio.google.com](https://aistudio.google.com) to verify."
                )
            else:
                yield (
                    f"**API Error**\n\n"
                    f"**Raw error:** `{error_msg}`\n\n"
                    f"Please check your API key and internet connection, "
                    f"or use the **Test API Key** button in the sidebar."
                )

    # ------------------------------------------------------------------
    # Validate API key (quick connectivity test)
    # ------------------------------------------------------------------

    def validate_key(self) -> tuple[bool, str]:
        """
        Send a minimal test request to verify the API key works.

        Returns:
            (is_valid: bool, message: str)
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with just the word: OK"}],
                max_tokens=5,
            )
            return True, "API key validated successfully."
        except Exception as exc:
            return False, str(exc)
