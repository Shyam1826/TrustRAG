r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_3_generation/generator.py
   - Role: Multi-provider synthesis, closed-world generation, and open-world fallback engine.
   - Purpose: Dispatches structured RAG prompts to cloud API providers (Groq, Gemini),
     local transformers (HuggingFace), or offline deterministic mocks with graceful
     credential fallback. Provides dual-mode generation capabilities: closed-world
     vault-grounded answer generation with strict citation constraints, and open-world
     parametric answer generation with standardized enterprise disclaimers. Implements
     exponential/adaptive backoff retry loops on HTTP 429 rate limits.

2. INPUT (IP):
   - prompt (str): Formatted RAG prompt containing XML context from `src/pipeline_3_generation/prompt.py`.
   - query (str): Natural language user inquiry for open-world fallback generation.

3. PROCESS UNDER THE HOOD:
   - BaseGenerator: Abstract Base Class defining `generate()`, `generate_answer()`, and `generate_open_world()`.
   - OPEN_WORLD_DISCLAIMER: Standardized enterprise notice prepended to parametric fallback responses.
   - GroqGenerator:
     * Closed-World: temperature=0.0, system prompt enforcing strict XML document groundings and [Doc-X] tags.
     * Open-World: temperature=0.2, instructs model to answer using general knowledge without citations.
     * Adaptive HTTP 429 retry backoff parsing exact retry durations.
   - MockGenerator:
     * Deterministic offline mock for test suites and credential-free evaluation.
     * Supports both closed-world rule-based synthesis and open-world disclaimed generation.
   - get_generator(): Factory creating configured active generator with automatic fallback to mock.

4. OUTPUT (OP):
   - str: Synthesized answer string (closed-world with [Doc-X] tags or open-world with enterprise disclaimer).
   - Consumed by: `src/main.py` (TrustRAGPipeline) and downstream verification pipelines.

5. LIBRARIES & DEPENDENCIES:
   - urllib.request, urllib.error, json, time, re: Standard library HTTP client, timing, and parsing.
   - abc (ABC, abstractmethod): Abstract base class definitions.
   - src.common.config: Provides generator provider selection, API keys, and model names.
================================================================================
"""

import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

from src.common.config import config
from src.pipeline_3_generation.prompt import FALLBACK_INSUFFICIENT_INFO


def deduplicate_sentences(text: str, overlap_threshold: float = 0.9) -> str:
    """Discard identical or near-duplicate sentences/bullets (>0.9 lexical overlap) from generated text.

    Args:
        text: Raw generated response text.
        overlap_threshold: Jaccard word-overlap ceiling (default 0.9).

    Returns:
        Deduplicated response text.
    """
    if not text or not text.strip():
        return text

    lines = text.split("\n")
    deduped_lines: List[str] = []
    seen_token_sets: List[set] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            deduped_lines.append(line)
            continue

        # Extract tokens for lexical comparison (excluding citations like [Doc-1])
        clean_text = re.sub(r"\[Doc-\d+\]", "", stripped)
        tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", clean_text.lower()))

        if not tokens:
            deduped_lines.append(line)
            continue

        # Check against previously seen sentences
        is_dup = False
        for seen_tokens in seen_token_sets:
            intersection = len(tokens & seen_tokens)
            union = len(tokens | seen_tokens)
            if union > 0 and (intersection / union) >= overlap_threshold:
                is_dup = True
                break

        if not is_dup:
            deduped_lines.append(line)
            seen_token_sets.append(tokens)

    return "\n".join(deduped_lines)


OPEN_WORLD_DISCLAIMER = (
    "⚠️ Notice: The provided documentation does not contain sufficient information to answer this inquiry. "
    "The following response is generated using open-world general knowledge and is not verified against your document vault.\n\n"
)


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

    def generate_answer(self, prompt: str) -> str:
        """Generate response and deduplicate identical or near-duplicate sentences (>0.9 lexical overlap).

        Args:
            prompt: Formatted RAG prompt.

        Returns:
            Deduplicated generated response.
        """
        raw = self.generate(prompt)
        return deduplicate_sentences(raw, overlap_threshold=0.9)

    def generate_open_world(self, query: str) -> str:
        """Generate open-world parametric answer with standardized enterprise disclaimer.

        Args:
            query: User search inquiry.

        Returns:
            Disclaimed open-world response string.
        """
        prompt = (
            "You are an expert AI assistant. Answer the user inquiry clearly and concisely "
            f"using general parametric knowledge. Do not reference or invent document citations like [Doc-X].\n\nQuestion: {query}"
        )
        raw = self.generate(prompt)
        return f"{OPEN_WORLD_DISCLAIMER}{raw.strip()}"



class GroqGenerator(BaseGenerator):
    """Cloud LLM generator using Groq's high-speed inference API with adaptive 429 retries."""

    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 45,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or config.GROQ_API_KEY
        self.model_name = model_name or config.GROQ_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for GroqGenerator.")

    def _send_payload(self, payload: dict) -> str:
        """Send chat payload to Groq API with 429 adaptive retry."""
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TrustRAG/1.0",
        }

        for attempt in range(self.max_retries + 1):
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

                # Handle Rate Limits (TPM/RPM 429) gracefully
                if e.code == 429 and attempt < self.max_retries:
                    wait_seconds = 14.0  # sensible default for Groq TPM resets
                    try:
                        err_json = json.loads(error_body)
                        msg = err_json.get("error", {}).get("message", "")
                        match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", msg)
                        if match:
                            wait_seconds = float(match.group(1)) + 1.0  # add 1s safety buffer
                    except Exception:
                        pass

                    print(
                        f"\n[RateLimit] Groq TPM limit reached (Attempt {attempt + 1}/{self.max_retries}). "
                        f"Pausing {wait_seconds:.1f}s before automatic retry..."
                    )
                    time.sleep(wait_seconds)
                    continue

                raise RuntimeError(f"Groq API error ({e.code}): {error_body}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to communicate with Groq API: {e}") from e

        return FALLBACK_INSUFFICIENT_INFO

    def generate(self, prompt: str) -> str:
        """Invoke Groq Chat Completions API with temperature=0.0 and adaptive 429 backoff."""
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
        return self._send_payload(payload)

    def generate_open_world(self, query: str) -> str:
        """Invoke Groq Chat Completions API with open-world instructions and disclaimer."""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI assistant. Answer the user inquiry accurately and concisely "
                        "using general parametric knowledge. Do not reference or invent document citations like [Doc-X]."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "temperature": 0.2,
        }
        raw_ans = self._send_payload(payload)
        return f"{OPEN_WORLD_DISCLAIMER}{raw_ans.strip()}"



class GeminiGenerator(BaseGenerator):
    """Cloud LLM generator using Google Generative Language API with 429 backoff."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 45,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.GEMINI_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiGenerator.")

    def generate(self, prompt: str) -> str:
        """Invoke Google Generative Language API with temperature=0.0 and retry backoff."""
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

        for attempt in range(self.max_retries + 1):
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
                if e.code == 429 and attempt < self.max_retries:
                    wait_seconds = 10.0 * (attempt + 1)
                    print(
                        f"\n[RateLimit] Gemini rate limit reached (Attempt {attempt + 1}/{self.max_retries}). "
                        f"Pausing {wait_seconds:.1f}s before retry..."
                    )
                    time.sleep(wait_seconds)
                    continue

                raise RuntimeError(f"Gemini API error ({e.code}): {error_body}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to communicate with Gemini API: {e}") from e

        return FALLBACK_INSUFFICIENT_INFO


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

        # Grounded tabular relational row detection
        if "[Section: Table:" in prompt:
            docs = re.findall(
                r'<document id="(?P<doc_id>Doc-\d+)"[^>]*>\s*(?P<doc_text>.*?)\s*</document>',
                prompt,
                re.DOTALL,
            )
            lines = []
            for doc_handle, doc_text in docs:
                clean_text = doc_text.strip()
                if "[Section: Table:" in clean_text:
                    body = re.sub(r"^\[Section:[^\]]+\]\s*", "", clean_text)
                    items = [item.strip() for item in body.split(" | ") if ":" in item]
                    kv = {}
                    for item in items:
                        parts = item.split(":", 1)
                        if len(parts) == 2:
                            kv[parts[0].strip()] = parts[1].strip()

                    name = kv.get("Document Name") or kv.get("Filename") or kv.get("Name") or kv.get("Title") or "the contract"
                    summary_parts = []
                    for k, v in kv.items():
                        if k not in ("Document Name", "Filename", "Name", "Title") and not k.endswith("-Answer"):
                            val = v[:150].rstrip(".")
                            summary_parts.append(f"{k} is '{val}'")

                    attr_str = " and ".join(summary_parts) if summary_parts else "details are recorded"
                    lines.append(f"- Under the {name}, {attr_str} [{doc_handle}].")

            if lines:
                return deduplicate_sentences("\n".join(lines[:5]))

        # Dynamic narrative multi-attribute query extraction from context documents
        docs = re.findall(
            r'<document id="(?P<doc_id>Doc-\d+)"[^>]*>\s*(?P<doc_text>.*?)\s*</document>',
            prompt,
            re.DOTALL,
        )
        if docs:
            q_match = re.search(r"User Question:\s*(.*?)(?:\n\n|\Z)", prompt, re.DOTALL)
            q_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", q_match.group(1).lower())) if q_match else set()

            generated_bullets = []
            for doc_handle, doc_text in docs:
                clean_text = doc_text.strip()
                raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean_text) if s.strip()]
                for sent in raw_sentences:
                    if sent.startswith("[Document:") or sent.startswith("##"):
                        continue
                    sent_clean = re.sub(r"\[(?:Document|Section|Sheet|Table):[^\]]+\]\s*", "", sent).strip()
                    if not sent_clean or len(sent_clean) < 15:
                        continue
                    sent_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", sent_clean.lower()))
                    if not q_words or (sent_words & q_words):
                        bullet = f"{sent_clean.rstrip('.')} [{doc_handle}]."
                        generated_bullets.append(bullet)
                        if len(generated_bullets) >= 4:
                            break
                if len(generated_bullets) >= 4:
                    break

            if generated_bullets:
                return deduplicate_sentences("\n".join(generated_bullets))

        # Fallback for insufficient context
        return FALLBACK_INSUFFICIENT_INFO

    def generate_open_world(self, query: str) -> str:
        """Generate deterministic open-world response with standard disclaimer for offline evaluation."""
        clean_q = query.strip().rstrip("?")
        body = f"Based on general world knowledge, {clean_q} is addressed using parametric facts and domain principles."
        return f"{OPEN_WORLD_DISCLAIMER}{body}"



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


def generate_answer(generator: BaseGenerator, prompt: str) -> str:
    """Generate answer from prompt with multi-attribute coverage and post-generation deduplication."""
    return generator.generate_answer(prompt)


def correct_unverified_claims(
    generator: BaseGenerator,
    correction_prompt: str,
) -> str:
    """Execute corrective rewrite with multi-attribute coverage and sentence deduplication."""
    return generator.generate_answer(correction_prompt)


def generate_open_world(query: str, generator: Optional[BaseGenerator] = None) -> str:
    """Direct the LLM to answer using general parametric knowledge with standard disclaimer."""
    gen = generator if generator is not None else get_generator()
    return gen.generate_open_world(query)