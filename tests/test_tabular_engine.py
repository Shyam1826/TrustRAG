r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_tabular_engine.py
   - Role: Unit and integration test suite for the DuckDB In-Memory Tabular Engine.
   - Purpose: Validates table discovery, column reflection, schema-aware SQL generation,
     deterministic relational filtering (WHERE conditions, boolean flags, multi-column
     constraints), candidate serialization into standard `RetrievalCandidate` objects,
     and end-to-end DeBERTa-v3 NLI verification under zero-hardcoding principles.

2. INPUT (IP):
   - Synthetic CSV and Excel files with diverse column types (boolean, text, numeric).
   - Relational and boolean queries matching various conditions.

3. PROCESS UNDER THE HOOD:
   - Tests `TabularStore`:
     * Ingestion of multi-column CSV files via DuckDB native read_csv_auto.
     * Ingestion of multi-sheet Excel files via pandas.
     * Table schema and column data type reflection.
   - Tests `TabularQueryEngine`:
     * Query classification (`is_tabular_query`) against registered catalogs.
     * Dynamic SQL synthesis handling boolean flags, operators, and projection columns.
     * Safe execution preventing mutating SQL and enforcing `LIMIT 15`.
     * Candidate serialization formatting rows into structured `RetrievalCandidate` models.
   - Tests DeBERTa-v3 Verification Gate:
     * Audits row-derived factual assertions against candidate premise text.
     * Asserts entailment confidence >= 0.85 (GATE VERDICT: PASS).
   - Tests `TrustRAGPipeline` Integration:
     * Validates `retrieve()` routing and `ask()` report generation.

4. OUTPUT (OP):
   - Pytest assertions and test outcomes.

5. LIBRARIES & DEPENDENCIES:
   - pandas, duckdb, pytest, pathlib.Path, tempfile.
   - src.common.schemas (RetrievalCandidate, AtomicClaim).
   - src.pipeline_1_ingestion.tabular_store (TabularStore).
   - src.pipeline_2_retrieval.tabular_engine (TabularQueryEngine).
   - src.pipeline_4_verification.adjudicator (AuditAdjudicator).
   - src.pipeline_4_verification.nli_model (DebertaNLIVerifier).
   - src.main (TrustRAGPipeline).
================================================================================
"""

from pathlib import Path
import tempfile
from typing import Generator
import pandas as pd
import pytest

from src.common.schemas import AtomicClaim, RetrievalCandidate
from src.main import TrustRAGPipeline
from src.pipeline_1_ingestion.tabular_store import TabularStore
from src.pipeline_2_retrieval.tabular_engine import TabularQueryEngine
from src.pipeline_4_verification.adjudicator import AuditAdjudicator
from src.pipeline_4_verification.nli_model import DebertaNLIVerifier


@pytest.fixture
def synthetic_csv_file() -> Generator[Path, None, None]:
    """Create a temporary synthetic multi-column CSV file for relational tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "synthetic_agreements.csv"
        df = pd.DataFrame({
            "Agreement ID": ["AGR_001", "AGR_002", "AGR_003", "AGR_004"],
            "Entity Name": ["Alpha Corp", "Beta LLC", "Gamma Inc", "Delta Co"],
            "Liability Cap": ["Yes", "Yes", "No", "No"],
            "Indemnity Flag": ["No", "Yes", "Yes", "No"],
            "Term Months": [12, 24, 36, 6],
            "Governing Law": ["Delaware", "New York", "California", "Delaware"],
        })
        df.to_csv(csv_path, index=False)
        yield csv_path


@pytest.fixture
def synthetic_excel_file() -> Generator[Path, None, None]:
    """Create a temporary synthetic multi-sheet Excel file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        xlsx_path = Path(tmp_dir) / "enterprise_catalog.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_hardware = pd.DataFrame({
                "Server Model": ["Rack-A1", "Blade-B2", "Tower-T3"],
                "CPU Cores": [64, 128, 32],
                "ECC Memory": [True, True, False],
            })
            df_hardware.to_excel(writer, sheet_name="Hardware", index=False)

            df_network = pd.DataFrame({
                "Switch Model": ["NetSwitch-X", "NetSwitch-Y"],
                "Bandwidth Gbps": [100, 400],
                "Redundant PSU": [True, True],
            })
            df_network.to_excel(writer, sheet_name="Network", index=False)

        yield xlsx_path


def test_tabular_store_csv_ingestion_and_schema_reflection(synthetic_csv_file: Path) -> None:
    """Verify TabularStore registers CSV files and accurately reflects schemas and data types."""
    store = TabularStore()
    tables = store.register_table_from_file(synthetic_csv_file, doc_id="synthetic_agreements")

    assert len(tables) == 1
    assert tables[0] == "synthetic_agreements"

    schemas = store.get_table_schemas()
    assert "synthetic_agreements" in schemas
    cols = schemas["synthetic_agreements"]
    assert "Agreement ID" in cols
    assert "Entity Name" in cols
    assert "Liability Cap" in cols
    assert "Term Months" in cols

    cols_types = store.get_table_columns_with_types("synthetic_agreements")
    assert "Agreement ID" in cols_types
    assert cols_types["Term Months"] in ("BIGINT", "INTEGER")

    rows = store.execute_query('SELECT * FROM synthetic_agreements WHERE "Agreement ID" = \'AGR_001\';')
    assert len(rows) == 1
    assert rows[0]["Entity Name"] == "Alpha Corp"
    store.close()


def test_tabular_store_excel_multi_sheet_registration(synthetic_excel_file: Path) -> None:
    """Verify TabularStore registers all sheets of an Excel workbook as discrete DuckDB tables."""
    store = TabularStore()
    tables = store.register_table_from_file(synthetic_excel_file, doc_id="enterprise_catalog")

    assert len(tables) == 2
    assert "enterprise_catalog_Hardware" in tables
    assert "enterprise_catalog_Network" in tables

    schemas = store.get_table_schemas()
    assert "enterprise_catalog_Hardware" in schemas
    assert "Server Model" in schemas["enterprise_catalog_Hardware"]
    assert "Switch Model" in schemas["enterprise_catalog_Network"]

    hw_rows = store.execute_query('SELECT * FROM enterprise_catalog_Hardware WHERE "CPU Cores" > 60;')
    assert len(hw_rows) == 2
    store.close()


def test_tabular_query_engine_detection(synthetic_csv_file: Path) -> None:
    """Verify TabularQueryEngine detects queries targeting registered tables with relational intent."""
    store = TabularStore()
    store.register_table_from_file(synthetic_csv_file, doc_id="synthetic_agreements")
    engine = TabularQueryEngine(tabular_store=store)

    # Relational queries with table mention
    assert engine.is_tabular_query("Find records in synthetic_agreements where Liability Cap is 'Yes'")
    assert engine.is_tabular_query("List agreements in synthetic_agreements where Term Months > 12")

    # Non-tabular generic query
    assert not engine.is_tabular_query("What is the history of computer networks?")
    store.close()


def test_tabular_query_engine_sql_generation_and_execution(synthetic_csv_file: Path) -> None:
    """Verify dynamic schema-aware SQL generation and deterministic safe execution."""
    store = TabularStore()
    store.register_table_from_file(synthetic_csv_file, doc_id="synthetic_agreements")
    engine = TabularQueryEngine(tabular_store=store)

    query = "Find the records in synthetic_agreements where Liability Cap is 'Yes' and Indemnity Flag is 'No'"
    sql = engine.generate_sql(query, target_table="synthetic_agreements")

    assert "FROM synthetic_agreements" in sql
    assert "LIMIT 15" in sql
    assert "WHERE" in sql

    rows = engine.execute_safe(sql)
    assert len(rows) == 1
    assert rows[0]["Agreement ID"] == "AGR_001"
    assert rows[0]["Entity Name"] == "Alpha Corp"
    store.close()


def test_tabular_query_engine_safety_bounds() -> None:
    """Verify TabularQueryEngine rejects mutating statements and enforces limit."""
    store = TabularStore()
    engine = TabularQueryEngine(tabular_store=store)

    with pytest.raises(ValueError, match="Mutating keywords detected"):
        engine.execute_safe("DROP TABLE synthetic_agreements;")

    with pytest.raises(ValueError, match="Mutating keywords detected"):
        engine.execute_safe("DELETE FROM synthetic_agreements WHERE 1=1;")
    store.close()


def test_candidate_serialization(synthetic_csv_file: Path) -> None:
    """Verify tabular row serialization produces valid RetrievalCandidate objects."""
    store = TabularStore()
    store.register_table_from_file(synthetic_csv_file, doc_id="synthetic_agreements")
    engine = TabularQueryEngine(tabular_store=store)

    rows = store.execute_query('SELECT * FROM synthetic_agreements WHERE "Agreement ID" = \'AGR_002\';')
    candidates = engine.serialize_rows_to_candidates(rows, table_name="synthetic_agreements")

    assert len(candidates) == 1
    cand = candidates[0]
    assert isinstance(cand, RetrievalCandidate)
    assert cand.match_type == "tabular_sql"
    assert "Table: synthetic_agreements | Row: 1" in cand.section_name
    assert "[Section: Table: synthetic_agreements | Row: 1]" in cand.text
    assert "Entity Name: Beta LLC" in cand.text
    assert "Liability Cap: Yes" in cand.text
    store.close()


def test_deberta_verification_on_tabular_candidate(synthetic_csv_file: Path) -> None:
    """Verify DeBERTa-v3 successfully verifies factual claims derived from tabular candidates."""
    store = TabularStore()
    store.register_table_from_file(synthetic_csv_file, doc_id="synthetic_agreements")
    engine = TabularQueryEngine(tabular_store=store)

    candidates = engine.query("Show clauses in synthetic_agreements where Liability Cap is 'Yes' and Indemnity Flag is 'No'")
    assert len(candidates) == 1
    cand = candidates[0]

    adjudicator = AuditAdjudicator()
    nli_verifier = DebertaNLIVerifier()

    claim = AtomicClaim(
        claim_id="claim_1",
        claim_text="The agreement AGR_001 for Alpha Corp specifies a Liability Cap of Yes and an Indemnity Flag of No.",
        cited_doc_ids=["Doc-1"],
    )

    context_map = {"Doc-1": cand}
    draft = "The agreement AGR_001 for Alpha Corp specifies a Liability Cap of Yes and an Indemnity Flag of No [Doc-1]."

    report = adjudicator.adjudicate(
        claims=[claim],
        context_map=context_map,
        nli_verifier=nli_verifier,
        draft_text=draft,
    )

    assert len(report.audits) == 1
    audit = report.audits[0]
    assert audit.verdict == "ENTAILED"
    assert audit.confidence >= 0.85
    assert report.faithfulness_score >= 0.85
    assert report.action == "PASS"
    store.close()


def test_orchestrator_retrieve_tabular_integration(synthetic_csv_file: Path) -> None:
    """Verify TrustRAGPipeline.retrieve routes tabular queries and returns RetrievalCandidates."""
    pipeline = TrustRAGPipeline(generator_type="mock", qdrant_location=":memory:")
    pipeline.tabular_store.register_table_from_file(synthetic_csv_file, doc_id="synthetic_agreements")

    query = "List the records in synthetic_agreements where Liability Cap is 'Yes' and Term Months = 24"
    candidates = pipeline.retrieve(query)

    assert len(candidates) == 1
    assert candidates[0].match_type == "tabular_sql"
    assert "Beta LLC" in candidates[0].text
    pipeline.close()
