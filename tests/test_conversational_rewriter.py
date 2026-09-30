r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_conversational_rewriter.py
   - Role: Unit and integration test suite for Conversational Query Reformulation (Phase 3).
   - Purpose: Validates that ambiguous multi-turn follow-ups, elliptical questions,
     and pronominal references are rewritten into standalone search queries before
     retrieval, and tests end-to-end `TrustRAGPipeline.chat()` thread message logging
     and audit persistence.

2. INPUT (IP):
   - Multi-turn conversational query sequences with pronouns and elliptical phrases.
   - Ingested test documents in TrustRAGPipeline.

3. PROCESS UNDER THE HOOD:
   - Tests `ConversationalQueryRewriter.is_conversational_query`:
     * Asserts standalone queries return False.
     * Asserts pronouns and elliptical queries return True.
   - Tests `ConversationalQueryRewriter.rewrite_query`:
     * Standalone query passthrough.
     * Elliptical entity substitution ("What about Chase?" -> "What is the governing law in Chase?").
     * Pronoun coreference resolution ("its warranty period" -> "Model-X processor's warranty period").
   - Tests `TrustRAGPipeline.chat`:
     * Turn 1 execution and database message logging (user & assistant).
     * Turn 2 follow-up execution with coreference rewriting.
     * Database ledger integrity: verifies 4 ordered messages with populated audit reports.

4. OUTPUT (OP):
   - Pytest assertions and verification results.

5. LIBRARIES & DEPENDENCIES:
   - pytest, tempfile, pathlib.Path.
   - src.common.schemas (TrustAuditReport).
   - src.pipeline_2_retrieval.conversational_rewriter (ConversationalQueryRewriter).
   - src.main (TrustRAGPipeline).
================================================================================
"""

import os
from pathlib import Path
import tempfile
from typing import Generator
import pytest

from src.pipeline_2_retrieval.conversational_rewriter import ConversationalQueryRewriter
from src.main import TrustRAGPipeline


def test_is_conversational_query():
    """Verify that is_conversational_query distinguishes standalone queries from conversational follow-ups."""
    rewriter = ConversationalQueryRewriter()

    # Standalone queries (must return False)
    assert rewriter.is_conversational_query("What is the governing law in LinkPlus?") is False
    assert rewriter.is_conversational_query("What is the TDP wattage of Model-X processor?") is False
    assert rewriter.is_conversational_query("List all active enterprise contracts") is False
    assert rewriter.is_conversational_query("Who is the primary vendor in the agreement?") is False

    # Conversational follow-ups with pronouns / demonstratives (must return True)
    assert rewriter.is_conversational_query("What is its warranty period?") is True
    assert rewriter.is_conversational_query("Does it feature 16 physical cores?") is True
    assert rewriter.is_conversational_query("How many days do they offer?") is True
    assert rewriter.is_conversational_query("Can you summarize this?") is True
    assert rewriter.is_conversational_query("What are their terms?") is True

    # Elliptical follow-up queries (must return True)
    assert rewriter.is_conversational_query("What about Chase?") is True
    assert rewriter.is_conversational_query("How about Chase?") is True
    assert rewriter.is_conversational_query("And for Model-Y?") is True
    assert rewriter.is_conversational_query("Compare that to LinkPlus") is True
    assert rewriter.is_conversational_query("Why?") is True
    assert rewriter.is_conversational_query("Tell me more") is True


def test_rewrite_query_standalone_passthrough():
    """Verify that standalone queries or queries without history pass through unchanged."""
    rewriter = ConversationalQueryRewriter()

    # Empty history
    query = "What is the governing law in LinkPlus?"
    assert rewriter.rewrite_query(query, []) == query

    # Standalone query with history
    history = [
        {"role": "user", "content": "What is the TDP wattage of Model-X processor?"},
        {"role": "assistant", "content": "The Model-X processor operates at 125W TDP [Doc-1]."},
    ]
    standalone = "Who is the primary vendor in LinkPlus?"
    assert rewriter.rewrite_query(standalone, history) == standalone


def test_rewrite_query_elliptical_substitution():
    """Verify that elliptical follow-ups substitute new entities into prior query intents."""
    rewriter = ConversationalQueryRewriter()

    history = [
        {"role": "user", "content": "What is the governing law in LinkPlus?"},
        {"role": "assistant", "content": "The agreement is governed by the laws of the State of New York [Doc-1]."},
    ]

    # Follow-up: "What about Chase?"
    rewritten = rewriter.rewrite_query("What about Chase?", history)
    assert "Chase" in rewritten
    assert "governing law" in rewritten.lower()
    assert "linkplus" not in rewritten.lower()

    # Follow-up: "How about Apex Corp?"
    rewritten_apex = rewriter.rewrite_query("How about Apex Corp?", history)
    assert "Apex Corp" in rewritten_apex
    assert "governing law" in rewritten_apex.lower()


def test_rewrite_query_pronoun_resolution():
    """Verify that pronouns (its, it, they, this) are resolved to the prior entity."""
    rewriter = ConversationalQueryRewriter()

    history = [
        {"role": "user", "content": "Can you tell me what is the TDP wattage of Model-X processor?"},
        {"role": "assistant", "content": "The Model-X processor operates at 125W TDP with 16 cores [Doc-1]."},
    ]

    # Follow-up: "What is its warranty period?"
    rewritten_possessive = rewriter.rewrite_query("What is its warranty period?", history)
    assert "Model-X processor" in rewritten_possessive
    assert "warranty period" in rewritten_possessive
    assert "its" not in rewritten_possessive.lower().split()

    # Follow-up: "Does it support 4096-bit?"
    rewritten_subjective = rewriter.rewrite_query("Does it support 4096-bit?", history)
    assert "Model-X processor" in rewritten_subjective
    assert "4096-bit" in rewritten_subjective


def test_pipeline_end_to_end_chat():
    """Verify full TrustRAGPipeline.chat() multi-turn conversation and database ledger persistence."""
    with tempfile.TemporaryDirectory() as temp_dir:
        doc_path = Path(temp_dir) / "hardware_specs.txt"
        doc_path.write_text(
            "Hardware Specs: The Model-X processor features 16 physical cores and operates at 125W TDP.\n"
            "It is engineered for enterprise datacenters.",
            encoding="utf-8",
        )

        pipeline = TrustRAGPipeline(generator_type="mock")

        user_id = "usr_conv_test_1"
        thread_id = "thr_conv_test_1"

        # Ingest document scoped to this tenant
        pipeline.ingest_document(str(doc_path), doc_id="spec_doc", user_id=user_id, thread_id=thread_id)

        # ----------------------------------------------------------------------
        # Turn 1: Standalone initial question
        # ----------------------------------------------------------------------
        report_1 = pipeline.chat(
            user_query="What is the TDP wattage of Model-X processor?",
            user_id=user_id,
            thread_id=thread_id,
        )

        assert report_1.draft_text is not None
        assert "125W TDP" in report_1.draft_text
        assert report_1.action == "PASS"

        # Verify database messages after Turn 1
        msgs_t1 = pipeline.db_repo.get_thread_messages(thread_id, user_id)
        assert len(msgs_t1) == 2
        assert msgs_t1[0].role == "user"
        assert msgs_t1[0].content == "What is the TDP wattage of Model-X processor?"
        assert msgs_t1[1].role == "assistant"
        assert "125W TDP" in msgs_t1[1].content
        assert msgs_t1[1].audit_report is not None
        assert msgs_t1[1].audit_report.get("action") == "PASS"

        # ----------------------------------------------------------------------
        # Turn 2: Conversational follow-up with pronoun coreference
        # ----------------------------------------------------------------------
        report_2 = pipeline.chat(
            user_query="Can you tell me what is its TDP wattage?",
            user_id=user_id,
            thread_id=thread_id,
        )

        assert report_2.draft_text is not None
        assert "125W TDP" in report_2.draft_text
        assert report_2.action == "PASS"

        # Verify database messages after Turn 2 (exactly 4 ordered messages)
        msgs_t2 = pipeline.db_repo.get_thread_messages(thread_id, user_id)
        assert len(msgs_t2) == 4
        assert [m.role for m in msgs_t2] == ["user", "assistant", "user", "assistant"]

        assert msgs_t2[2].role == "user"
        assert msgs_t2[2].content == "Can you tell me what is its TDP wattage?"

        assert msgs_t2[3].role == "assistant"
        assert "125W TDP" in msgs_t2[3].content
        assert msgs_t2[3].audit_report is not None
        assert msgs_t2[3].audit_report.get("action") == "PASS"

        pipeline.close()
