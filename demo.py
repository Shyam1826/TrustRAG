r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: demo.py
   - Role: Interactive Terminal CLI and demonstration runner for TrustRAG.
   - Purpose: Discovers raw PDF documents in `data/raw/`, orchestrates automatic
     ingestion and indexing, showcases active LLM provider configuration (Groq, Gemini,
     Local, or Mock), displays retrieved parent context passages, and executes
     claim-level NLI verification and automated safety gating.

2. INPUT (IP):
   - PDF files placed inside the `data/raw/` directory.
   - Interactive user query strings entered via terminal standard input (stdin).

3. PROCESS UNDER THE HOOD:
   - Scans `data/raw/` directory for `.pdf` files (generates synthetic demo PDF if empty).
   - Identifies and prints active LLM provider (Groq, Gemini, Local HF, or Mock).
   - Instantiates `TrustRAGPipeline` and ingests all discovered PDF documents.
   - Enters interactive command-line evaluation loop (`while True`):
     * Prompts user for natural language queries.
     * Executes `pipeline.ask(query)`.
     * Prints retrieved context passages ([Doc-1], [Doc-2], ...) used by the generator.
     * Formats and renders the synthesized answer with inline citations.
     * Renders per-claim NLI verification verdicts with confidence scores.
     * Displays overall faithfulness score and safety gate actions (PASS / TRIGGER_REWRITE / WARN).

4. OUTPUT (OP):
   - Terminal stdout: Beautifully formatted, structured, and auditable trust reports.

5. LIBRARIES & DEPENDENCIES:
   - sys, os: Standard library system utilities.
   - pathlib.Path: File path manipulation.
   - fitz (PyMuPDF): Automatic sample creation if directory is empty.
   - src.common.config: Provides configuration parameters and active model settings.
   - src.main.TrustRAGPipeline: Complete system orchestrator.
================================================================================
"""

import sys
from pathlib import Path
import fitz  # PyMuPDF

from src.common.config import config
from src.main import TrustRAGPipeline


def create_sample_pdf_if_empty(raw_dir: Path) -> Path:
    """Create a sample hardware reference PDF in data/raw if no PDFs exist.

    Args:
        raw_dir: Path to data/raw directory.

    Returns:
        Path to the sample PDF document.
    """
    sample_path = raw_dir / "sample_hardware_specs.pdf"
    if sample_path.exists():
        return sample_path

    doc = fitz.open()

    page1 = doc.new_page()
    page1_text = (
        "Enterprise Hardware Specification - Model-X Series\n\n"
        "The Model-X processor features 16 physical cores and operates at 125W TDP.\n"
        "It utilizes a 4nm fabrication process and supports PCIe Gen 5.0 high-speed interconnects.\n\n"
        "Thermal throttling initiates automatically at 95 degrees Celsius to protect core integrity."
    )
    page1.insert_text((50, 72), page1_text, fontsize=11)

    page2 = doc.new_page()
    page2_text = (
        "Corporate Operational Guidelines\n\n"
        "All engineering personnel are entitled to 20 days of annual paid time off.\n"
        "Standard laboratory gateway router IP is configured at 192.168.1.1 with subnet 255.255.255.0."
    )
    page2.insert_text((50, 72), page2_text, fontsize=11)

    doc.save(str(sample_path))
    doc.close()
    return sample_path


def run_cli() -> None:
    """Run interactive terminal CLI."""
    print("=" * 60)
    print("           TrustRAG — Enterprise Verification RAG Engine")
    print("=" * 60)

    # 1. Display active generator provider
    provider = config.GENERATOR_PROVIDER.upper()
    if provider == "GROQ":
        model_info = config.GROQ_MODEL
        status = "Active" if config.GROQ_API_KEY else "Missing API Key -> Fallback to MOCK"
    elif provider == "GEMINI":
        model_info = config.GEMINI_MODEL
        status = "Active" if config.GEMINI_API_KEY else "Missing API Key -> Fallback to MOCK"
    else:
        model_info = "Deterministic Rules"
        status = "Active"

    print(f"\n[Generator] Active Provider: {provider} (Model: {model_info}) [{status}]")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 2. Discover PDFs in data/raw
    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        print("\n[Notice] No PDF documents found in 'data/raw/'.")
        print("Generating a sample demonstration document: 'data/raw/sample_hardware_specs.pdf'...")
        sample_file = create_sample_pdf_if_empty(raw_dir)
        pdf_files = [sample_file]

    print(f"\n[Ingestion] Discovered {len(pdf_files)} document(s) in 'data/raw/'.")

    # 3. Initialize TrustRAG pipeline
    print("\n[Initializing] Loading models and vector index...")
    pipeline = TrustRAGPipeline()

    # 4. Ingest documents
    for pdf_file in pdf_files:
        print(f"Ingesting: {pdf_file.name}")
        pipeline.ingest_pdf(str(pdf_file))

    print("\n" + "=" * 60)
    print(" System Ready. Type 'quit' or 'exit' to terminate.")
    print("=" * 60 + "\n")

    # 5. Interactive Query Loop
    while True:
        try:
            user_query = input("Ask a question about your documents: ").strip()
            if not user_query:
                continue

            if user_query.lower() in ("quit", "exit", "q"):
                print("\nExiting TrustRAG CLI. Goodbye!")
                break

            print("\nProcessing query through verification pipeline...")
            report = pipeline.ask(user_query)

            # Print Retrieved Context Passages
            if pipeline.last_retrieved_contexts:
                print("\n" + "-" * 50)
                print("RETRIEVED CONTEXT PASSAGES:")
                for idx, ctx in enumerate(pipeline.last_retrieved_contexts, start=1):
                    preview = ctx.text.replace("\n", " ").strip()
                    if len(preview) > 120:
                        preview = preview[:117] + "..."
                    print(f'  [Doc-{idx}] (doc_id="{ctx.doc_id}", page={ctx.page_number}, score={ctx.score:.3f}):')
                    print(f'    "{preview}"')

            # Format Terminal Output
            print("-" * 50)
            print("ANSWER:")
            print(report.draft_text)
            print("-" * 50)
            print("CLAIM-LEVEL AUDIT:")

            if not report.audits:
                print("  (No verifiable atomic claims extracted)")
            else:
                for audit in report.audits:
                    print(f'- Claim: "{audit.claim_text}"')
                    print(f"  Verdict: {audit.verdict} (Confidence: {audit.confidence:.2f})")

            print("-" * 50)
            print(f"FAITHFULNESS SCORE: {report.faithfulness_score:.2f} / 1.00")
            print(f"GATE VERDICT: {report.action}")
            print("-" * 50 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error] An error occurred while processing query: {e}\n")


if __name__ == "__main__":
    run_cli()
