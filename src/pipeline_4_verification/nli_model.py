r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_4_verification/nli_model.py
   - Role: Natural Language Inference (NLI) sequence classification engine.
   - Purpose: Evaluates premise-hypothesis pairs using cross-attention DeBERTa-v3
     architecture to compute probabilistic entailment, contradiction, and neutral scores.

2. INPUT (IP):
   - claims (list[str]): Atomic claim hypothesis strings from `src/pipeline_4_verification/claim_extractor.py`.
   - premises (list[str]): Context premise passages retrieved from source documents.
   - batch_size (int): Batch size for inference execution (default 16).

3. PROCESS UNDER THE HOOD:
   - Accelerator Selection: Selects CUDA, Apple Silicon MPS, or CPU dynamically.
   - Model Loading: Loads AutoTokenizer and AutoModelForSequenceClassification for
     `cross-encoder/nli-deberta-v3-base`, setting the network to inference evaluation mode (`.eval()`).
   - Batch Processing:
     * Chunks (premise, claim) pairs into batches.
     * Tokenizes pairs with padding and truncation to `max_length=512`.
     * Runs forward pass under `@torch.inference_mode()`.
     * Applies softmax activation over class logits: $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$.
     * Maps model output indices to standardized probability dictionary:
       `{"contradiction": float, "entailment": float, "neutral": float}`.

4. OUTPUT (OP):
   - list[dict[str, dict[str, float]]]: List of dictionaries containing 3-class NLI probabilities.
   - Consumed by: `src/pipeline_4_verification/adjudicator.py`.

5. LIBRARIES & DEPENDENCIES:
   - torch: Deep learning tensor execution and GPU/MPS acceleration.
   - transformers (AutoTokenizer, AutoModelForSequenceClassification): HuggingFace model abstractions.
   - src.common.config: Provides centralized NLI model name configuration.
================================================================================
"""

from typing import Dict, List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.config import config


def get_optimal_device() -> str:
    """Detect available hardware accelerator (CUDA, Apple Silicon MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DebertaNLIVerifier:
    """Cross-encoder DeBERTa-v3 Natural Language Inference classifier."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize DeBERTa NLI tokenizer and sequence classification model.

        Args:
            model_name: HuggingFace model identifier (default: config.models.nli_model_name).
            device: Hardware execution device ('cuda', 'mps', 'cpu').
        """
        self.model_name = model_name or config.models.nli_model_name
        self.device = device or get_optimal_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Build label index mapping from model config or standard default
        # Standard: 0 -> contradiction, 1 -> entailment, 2 -> neutral
        self.id2label: Dict[int, str] = {}
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            for idx, raw_label in self.model.config.id2label.items():
                clean_label = str(raw_label).lower()
                if "contra" in clean_label:
                    self.id2label[int(idx)] = "contradiction"
                elif "entail" in clean_label:
                    self.id2label[int(idx)] = "entailment"
                else:
                    self.id2label[int(idx)] = "neutral"
        else:
            self.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    @torch.inference_mode()
    def predict_batch(
        self,
        claims: List[str],
        premises: List[str],
        batch_size: int = 16,
    ) -> List[Dict[str, Dict[str, float]]]:
        """Compute NLI class probabilities for aligned (claim, premise) pairs.

        Args:
            claims: List of claim strings (hypotheses).
            premises: List of corresponding premise context strings.
            batch_size: Maximum batch size for inference pass.

        Returns:
            List of dictionaries formatted as:
            [{"probabilities": {"contradiction": float, "entailment": float, "neutral": float}}, ...]
        """
        if not claims or not premises or len(claims) != len(premises):
            return []

        results: List[Dict[str, Dict[str, float]]] = []

        total_samples = len(claims)
        for i in range(0, total_samples, batch_size):
            batch_claims = claims[i : i + batch_size]
            batch_premises = premises[i : i + batch_size]

            # Cross-encoder premise-hypothesis tokenization: (premise, hypothesis)
            encoded = self.tokenizer(
                batch_premises,
                batch_claims,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            for item_idx in range(probs.shape[0]):
                item_probs: Dict[str, float] = {}
                for class_idx in range(probs.shape[1]):
                    label_name = self.id2label.get(class_idx, f"class_{class_idx}")
                    item_probs[label_name] = float(probs[item_idx, class_idx].item())

                # Ensure all 3 canonical keys exist
                canonical_probs = {
                    "contradiction": item_probs.get("contradiction", 0.0),
                    "entailment": item_probs.get("entailment", 0.0),
                    "neutral": item_probs.get("neutral", 0.0),
                }

                results.append({"probabilities": canonical_probs})

        return results
