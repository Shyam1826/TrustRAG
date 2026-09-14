r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_3_generation/generator.py
   - Role: Multi-provider synthesis and generation engine.
   - Purpose: Dispatches structured RAG prompts to cloud API providers (Groq, Gemini),
     local transformers (HuggingFace), or offline deterministic mocks with graceful
     credential fallback.

2. INPUT (IP):
   - prompt (str): Formatted RAG prompt containing XML context from `src/pipeline_3_generation/prompt.py`.

3. PROCESS UNDER THE HOOD:
   - BaseGenerator: Abstract Base Class defining standard `generate(prompt: str) -> str`.
   - GroqGenerator:
     * Dispatches HTTP POST to `https://api.groq.com/openai/v1/chat/completions`.
     * Passes system instructions enforcing closed-world rules and inline [Doc-X] citations.
     * Enforces `temperature=0.0` for deterministic, grounded outputs.
   - GeminiGenerator:
     * Dispatches HTTP POST to Google Generative Language API (`/v1beta/models/{model}:generateContent`).
     * Sets `temperature=0.0`.
   - MockGenerator:
     * Offline deterministic rule-based generator for testing without cloud credentials.
   - get_generator:
     * Inspects `config.GENERATOR_PROVIDER` or explicit provider argument.
     * Checks for required API keys; if missing, logs a descriptive notice and falls back
       gracefully to `MockGenerator`.

4. OUTPUT (OP):
   - str: Synthesized draft response containing inline citations.
   - Consumed by: `src/pipeline_3_generation/citation_check.py` and `src/pipeline_4_verification/`.

5. LIBRARIES & DEPENDENCIES:
   - urllib.request, urllib.error, json: Standard library HTTP client for zero-dependency API calls.
   - abc (ABC, abstractmethod): Abstract base class definitions.
   - src.common.config: Provides generator provider selection, API keys, and model names.
================================================================================
"""

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

from src.common.config import config
from src.pipeline_3_generation.prompt import FALLBACK_INSUFFICIENT_INFO


class BaseGenerator(ABC):
    """Abstract interface for text generation models in TrustRAG."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response text given a structured prompt.

        Args:
            prompt: Formatted RAG prompt.

        Returns:
            Generated text string.
        """
        pass


class GroqGenerator(BaseGenerator):
    """Cloud LLM generator using Groq's high-speed inference API (Llama-3, etc.)."""

    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or config.GROQ_API_KEY
        self.model_name = model_name or config.GROQ_MODEL
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for GroqGenerator.")

    def generate(self, prompt: str) -> str:
        """Invoke Groq Chat Completions API with temperature=0.0."""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise AI assistant adhering to strict verification standards. "
                        "Answer strictly using ONLY the provided <context> documents and append inline [Doc-X] "
                        "citation tags to every factual assertion."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TrustRAG/1.0",
        }

        req = urllib.request.Request(
            self.ENDPOINT,
            data=data_bytes,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
                return response_data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq API error ({e.code}): {error_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with Groq API: {e}") from e


class GeminiGenerator(BaseGenerator):
    """Cloud LLM generator using Google Generative Language API (Gemini-1.5, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.GEMINI_MODEL
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiGenerator.")

    def generate(self, prompt: str) -> str:
        """Invoke Google Generative Language API with temperature=0.0."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TrustRAG/1.0",
        }

        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
                candidates = response_data.get("candidates", [])
                if not candidates:
                    return FALLBACK_INSUFFICIENT_INFO
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    return FALLBACK_INSUFFICIENT_INFO
                return parts[0].get("text", "").strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error ({e.code}): {error_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with Gemini API: {e}") from e


class MockGenerator(BaseGenerator):
    """Deterministic, rule-based generator for testing without external LLM dependencies."""

    def generate(self, prompt: str) -> str:
        """Generate response based on deterministic keyword inspection."""
        if not prompt:
            return FALLBACK_INSUFFICIENT_INFO

        # Trigger for simulated invalid / out-of-bounds citations
        if "simulate_bad_citation" in prompt:
            return "The Model-X processor operates at 125W TDP [Doc-9]."

        # Grounded hardware query detection
        if "125W TDP" in prompt and "TDP" in prompt:
            return "The Model-X processor features 16 physical cores and operates at 125W TDP [Doc-1]."

        # Grounded corporate policy detection
        if "20 days of annual paid time off" in prompt and ("PTO" in prompt or "paid time off" in prompt):
            return "All employees are entitled to 20 days of annual paid time off [Doc-2]."

        # Grounded network guide detection
        if "192.168.1.1" in prompt and "gateway" in prompt:
            return "The gateway router IP is 192.168.1.1 with subnet 255.255.255.0 [Doc-3]."

        # Fallback for insufficient context
        return FALLBACK_INSUFFICIENT_INFO


class LocalHFGenerator(BaseGenerator):
    """Local transformer-based text generation using HuggingFace pipelines."""

    def __init__(
        self,
        model_name: str = "google/gemma-2b-it",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> None:
        from transformers import pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            device=device,
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        """Run text generation pipeline on prompt."""
        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return outputs[0]["generated_text"]


def get_generator(generator_type: Optional[str] = None) -> BaseGenerator:
    """Factory creating configured generator instance with graceful offline fallback.

    Args:
        generator_type: Optional provider override ('groq', 'gemini', 'hf', 'mock').

    Returns:
        Instance conforming to BaseGenerator interface.
    """
    target_provider = (generator_type or config.GENERATOR_PROVIDER).lower()

    if target_provider == "groq":
        if config.GROQ_API_KEY:
            return GroqGenerator()
        print("[Notice] GROQ_API_KEY not configured. Falling back to MockGenerator.")
        return MockGenerator()

    if target_provider == "gemini":
        if config.GEMINI_API_KEY:
            return GeminiGenerator()
        print("[Notice] GEMINI_API_KEY not configured. Falling back to MockGenerator.")
        return MockGenerator()

    if target_provider == "hf":
        return LocalHFGenerator()

    return MockGenerator()
