r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_p3_generation.py
   - Role: Test suite for Pipeline 3 (Citation-Aware Generation).
   - Purpose: Asserts prompt construction, MockGenerator response grounding, citation
     parsing, Unicode bracket normalization (【Doc-X】 -> [Doc-X]), out-of-bounds
     citation detection, insufficient evidence fallbacks, and cloud generator provider
     factory selection.

2. INPUT (IP):
   - Synthetic queries, RetrievalCandidate instances, and mock HTTP responses.

3. PROCESS UNDER THE HOOD:
   - Tests XML prompt construction and structural integrity.
   - Tests MockGenerator response synthesis with inline citations.
   - Tests citation validation for valid, Unicode bracket, and out-of-bounds texts.
   - Tests fallback behavior under missing evidence conditions.
   - Tests Groq and Gemini generator initialization and factory fallback behaviors.

4. OUTPUT (OP):
   - Pytest assertions and test outcomes.

5. LIBRARIES & DEPENDENCIES:
   - pytest, unittest.mock: Test execution framework and HTTP mock utilities.
   - src.common.schemas: Pydantic schemas.
   - src.pipeline_3_generation.*: Generation components.
================================================================================
"""

from unittest.mock import MagicMock, patch
import pytest

from src.common.schemas import RetrievalCandidate
from src.pipeline_3_generation.prompt import build_rag_prompt, FALLBACK_INSUFFICIENT_INFO
from src.pipeline_3_generation.generator import (
    GeminiGenerator,
    GroqGenerator,
    MockGenerator,
    get_generator,
)
from src.pipeline_3_generation.citation_check import validate_and_parse_citations


def test_build_rag_prompt_structure():
    candidates = [
        RetrievalCandidate(
            parent_id="p1",
            doc_id="doc_1",
            page_number=1,
            text="Hardware Specs: Model-X operates at 125W TDP.",
            score=0.95,
            match_type="cross_encoder_reranked",
        ),
        RetrievalCandidate(
            parent_id="p2",
            doc_id="doc_2",
            page_number=2,
            text="Corporate Policies: 20 days PTO.",
            score=0.80,
            match_type="cross_encoder_reranked",
        ),
    ]
    query = "What is the TDP wattage of Model-X processor?"

    prompt = build_rag_prompt(query, candidates)

    # Assert XML tags and document metadata
    assert "<context>" in prompt
    assert "</context>" in prompt
    assert '<document id="Doc-1" doc_id="doc_1" page="1">' in prompt
    assert '<document id="Doc-2" doc_id="doc_2" page="2">' in prompt
    assert "Hardware Specs: Model-X operates at 125W TDP." in prompt

    # Assert operational rules
    assert "Strict Semantic Grounding & Closed-World Assumption" in prompt
    assert "Strict Attribute Isolation & Anti-Bundle Enforcement" in prompt
    assert "Strict Inventory & Entity Grounding" in prompt
    assert "ASCII Inline Citations" in prompt
    assert "Insufficient Evidence" in prompt

    # Assert query inclusion
    assert f"User Question: {query}" in prompt



def test_generator_grounded_response():
    generator = MockGenerator()
    candidates = [
        RetrievalCandidate(
            parent_id="p1",
            doc_id="doc_1",
            page_number=1,
            text="Hardware Specs: The Model-X processor features 16 physical cores and operates at 125W TDP.",
            score=0.98,
            match_type="cross_encoder_reranked",
        )
    ]
    query = "What is the TDP of Model-X?"
    prompt = build_rag_prompt(query, candidates)

    response = generator.generate(prompt)

    assert "[Doc-1]" in response
    assert "125W TDP" in response
    assert "Model-X" in response


def test_citation_validator_valid():
    text = "The Model-X processor operates at 125W TDP [Doc-1] and has 16 cores [Doc-1]."
    draft = validate_and_parse_citations(text, max_valid_doc_id=2)

    assert draft.raw_text == text
    assert draft.cited_doc_ids == ["Doc-1"]
    assert draft.citations_valid is True

    # Test Unicode full-width brackets normalization
    unicode_text = "Statement one 【Doc-1】. Statement two 【Doc-2】."
    draft_unicode = validate_and_parse_citations(unicode_text, max_valid_doc_id=2)
    assert draft_unicode.cited_doc_ids == ["Doc-1", "Doc-2"]
    assert draft_unicode.citations_valid is True
    assert "[Doc-1]" in draft_unicode.raw_text
    assert "[Doc-2]" in draft_unicode.raw_text


def test_citation_validator_out_of_bounds():
    text = "The Model-X processor operates at 125W TDP [Doc-9]."
    draft = validate_and_parse_citations(text, max_valid_doc_id=3)

    assert draft.cited_doc_ids == ["Doc-9"]
    assert draft.citations_valid is False

    # Also test invalid Doc-0 index
    text_zero = "Invalid citation [Doc-0]."
    draft_zero = validate_and_parse_citations(text_zero, max_valid_doc_id=3)
    assert draft_zero.citations_valid is False


def test_insufficient_evidence_fallback():
    generator = MockGenerator()
    candidates = [
        RetrievalCandidate(
            parent_id="p2",
            doc_id="doc_2",
            page_number=1,
            text="Corporate Policies: All employees are entitled to 20 days PTO.",
            score=0.45,
            match_type="cross_encoder_reranked",
        )
    ]
    query = "What is the chemical composition of lunar regolith?"
    prompt = build_rag_prompt(query, candidates)

    response = generator.generate(prompt)

    assert response == FALLBACK_INSUFFICIENT_INFO
    draft = validate_and_parse_citations(response, max_valid_doc_id=1)
    assert draft.cited_doc_ids == []
    assert draft.citations_valid is True


def test_groq_generator_mocked_http():
    generator = GroqGenerator(api_key="gsk_test_mock_key", model_name="llama-3.1-8b-instant")

    mock_resp_payload = b'{"choices": [{"message": {"content": "Model-X runs at 125W [Doc-1]."}}]}'
    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_resp_payload
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = generator.generate("Test prompt")
        assert result == "Model-X runs at 125W [Doc-1]."


def test_gemini_generator_mocked_http():
    generator = GeminiGenerator(api_key="AIzaSyTestMockKey", model_name="gemini-1.5-flash")

    mock_resp_payload = b'{"candidates": [{"content": {"parts": [{"text": "Model-X runs at 125W [Doc-1]."}]}}]}'
    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_resp_payload
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = generator.generate("Test prompt")
        assert result == "Model-X runs at 125W [Doc-1]."


def test_get_generator_fallback_without_keys():
    # Without api keys configured in environment, factory returns MockGenerator
    with patch("src.common.config.config.generation.groq_api_key", ""):
        gen = get_generator("groq")
        assert isinstance(gen, MockGenerator)

    with patch("src.common.config.config.generation.gemini_api_key", ""):
        gen_gemini = get_generator("gemini")
        assert isinstance(gen_gemini, MockGenerator)
