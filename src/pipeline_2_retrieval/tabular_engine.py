r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/tabular_engine.py
   - Role: Relational query detection, schema-aware SQL generation, and candidate serialization.
   - Purpose: Identifies natural language queries targeting structured tabular documents
     (CSV, TSV, Excel), constructs safe, syntax-valid DuckDB SQL queries under zero-hardcoding
     principles, executes them with safety limits, and serializes output rows into standard
     RetrievalCandidate objects consumable by generation and verification pipelines.

2. INPUT (IP):
   - query (str): Natural language search query or decomposed sub-query.
   - tabular_store (TabularStore): Initialized DuckDB tabular store with registered tables.
   - generator (Optional[BaseGenerator]): Optional LLM generator for schema-aware SQL synthesis.

3. PROCESS UNDER THE HOOD:
   - Relational Query Detection (`is_tabular_query`):
     * Checks if query references any registered table name / stem in the catalog.
     * Identifies relational/boolean query markers (`where`, `is '...'`, `equals`, `count`,
       `sum`, `filter`, `having both`, `>`, `<`, `!=`, `clauses where`, `rows where`, boolean flags).
   - Schema-Aware SQL Generation (`generate_sql`):
     * Reflects the target table's columns and data types dynamically.
     * Tier 1 (LLM-Assisted Synthesis): When an external LLM is available, formats schema context
       and rules to generate valid DuckDB SQL.
     * Tier 2 (Universal Deterministic Parser): If LLM is unavailable or fails, extracts relational
       conditions using domain-agnostic regex, matches column names via canonical normalization
       and boolean `-Answer` preference, resolves projection columns based on query tokens and
       table identifier columns, and synthesizes syntactically valid SQL.
   - Safe Execution (`execute_safe`):
     * Enforces read-only statements (rejects `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, etc.).
     * Enforces a strict safety limit (`LIMIT 15`).
   - Candidate Serialization (`serialize_rows_to_candidates`):
     * Formats each result row into key-value relational text lines:
       `[Section: Table: <table_name> | Row: <row_idx>] col1: val1 | col2: val2 ...`
     * Wraps each row in a standard `RetrievalCandidate` model with `match_type="tabular_sql"`.

4. OUTPUT (OP):
   - List[RetrievalCandidate]: Standard candidate models ready for generation and verification.
   - Consumed by: `src/main.py`, `src/pipeline_3_generation/prompt.py`, `src/pipeline_4_verification/adjudicator.py`.

5. LIBRARIES & DEPENDENCIES:
   - difflib, re, typing (Dict, List, Optional, Tuple, Any, Union).
   - src.common.schemas (RetrievalCandidate).
   - src.pipeline_1_ingestion.tabular_store (TabularStore).
================================================================================
"""

import difflib
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from src.common.schemas import ProvenanceCoordinate, RetrievalCandidate
from src.pipeline_1_ingestion.tabular_store import TabularStore


class TabularQueryEngine:
    """Detects relational tabular queries, synthesizes schema-aware SQL, and emits RetrievalCandidates."""

    _RELATIONAL_MARKERS = [
        re.compile(r"\b(?:where|whose|having|with\s+both|having\s+both)\b", re.IGNORECASE),
        re.compile(r"\b(?:is|equals|=|equal\s+to|is\s+not|!=)\s+['\"]?(?:yes|no|true|false|[a-zA-Z0-9_.-]+)['\"]?", re.IGNORECASE),
        re.compile(r"\b(?:filter(?:ed)?\s+by|rows?\s+where|clauses?\s+where|records?\s+where)\b", re.IGNORECASE),
        re.compile(r"\b(?:count|sum|average|avg|minimum|min|maximum|max|total)\s+(?:of|for|across)?\b", re.IGNORECASE),
        re.compile(r"\b(?:greater\s+than|less\s+than|above|below|at\s+least|at\s+most|[><]=?)\s+[0-9.]+", re.IGNORECASE),
    ]

    _MUTATING_KEYWORDS_REGEX = re.compile(
        r"\b(?:DROP|DELETE|UPDATE|INSERT|ALTER|ATTACH|DETACH|COPY|PRAGMA|EXECUTE|CREATE)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        tabular_store: TabularStore,
        generator: Optional[Any] = None,
    ) -> None:
        """Initialize TabularQueryEngine with store reference and optional LLM generator.

        Args:
            tabular_store: Initialized TabularStore instance.
            generator: Optional generator instance for LLM-assisted SQL synthesis.
        """
        self.tabular_store = tabular_store
        self.generator = generator

    def is_tabular_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> bool:
        """Determine if a query targets structured tabular data and relational operations.

        Args:
            query: User search query or sub-query string.
            user_id: Optional tenant user_id.
            thread_id: Optional tenant thread_id.

        Returns:
            True if query targets a registered table and contains relational filtering markers.
        """
        if not query or not query.strip():
            return False

        table_names = self.tabular_store.get_table_names(user_id=user_id, thread_id=thread_id)
        if not table_names:
            return False

        target_table = self.detect_target_table(query, user_id=user_id, thread_id=thread_id)
        if not target_table:
            return False

        has_relational_marker = any(pat.search(query) for pat in self._RELATIONAL_MARKERS)
        if has_relational_marker:
            return True

        # Check if table is explicitly named and query mentions columns
        schemas = self.tabular_store.get_table_schemas(user_id=user_id, thread_id=thread_id)
        cols = schemas.get(target_table, [])
        q_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        col_tokens = {t for c in cols for t in re.findall(r"[a-zA-Z0-9]+", c.lower()) if len(t) >= 3}
        if len(q_words & col_tokens) >= 2:
            return True

        return False

    def detect_target_table(
        self,
        query: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[str]:
        """Identify which registered table best matches the query string.

        Args:
            query: Natural language query string.
            user_id: Optional tenant user_id.
            thread_id: Optional tenant thread_id.

        Returns:
            Matching table name or None if no candidate matches.
        """
        schemas = self.tabular_store.get_table_schemas(user_id=user_id, thread_id=thread_id)
        if not schemas:
            return None

        q_clean = query.lower()
        canonical_q = re.sub(r"[^a-zA-Z0-9]", "", q_clean)

        # 1. Exact or substring match on table name / stem
        best_table: Optional[str] = None
        best_len = 0

        for table_name in schemas.keys():
            t_canonical = re.sub(r"[^a-zA-Z0-9]", "", table_name.lower())
            if len(t_canonical) >= 3 and t_canonical in canonical_q:
                if len(t_canonical) > best_len:
                    best_len = len(t_canonical)
                    best_table = table_name

            # Check original doc_id
            doc_id = self.tabular_store.get_doc_for_table(table_name)
            if doc_id:
                doc_canonical = re.sub(r"[^a-zA-Z0-9]", "", doc_id.lower())
                if len(doc_canonical) >= 3 and doc_canonical in canonical_q:
                    if len(doc_canonical) > best_len:
                        best_len = len(doc_canonical)
                        best_table = table_name

        if best_table:
            return best_table

        # 2. Token overlap match on table name tokens (len >= 3)
        q_tokens = set(re.findall(r"[a-zA-Z0-9]+", q_clean))
        max_overlap = 0

        for table_name in schemas.keys():
            t_tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", table_name.lower()) if len(t) >= 3}
            overlap = len(q_tokens & t_tokens)
            if overlap > max_overlap and overlap >= 1:
                max_overlap = overlap
                best_table = table_name

        if best_table:
            return best_table

        # 3. If query contains relational markers AND columns of a single table match:
        has_relational_marker = any(pat.search(query) for pat in self._RELATIONAL_MARKERS)
        if has_relational_marker and len(schemas) == 1:
            tbl = next(iter(schemas.keys()))
            cols = schemas[tbl]
            col_tokens = {t for c in cols for t in re.findall(r"[a-zA-Z0-9]+", c.lower()) if len(t) >= 3}
            if len(q_tokens & col_tokens) >= 1:
                return tbl

        return None

    def generate_sql(
        self,
        query: str,
        target_table: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """Synthesize a safe, schema-aware DuckDB SQL query matching user conditions.

        Args:
            query: User query string.
            target_table: Optional target table name.
            user_id: Optional tenant user_id.
            thread_id: Optional tenant thread_id.

        Returns:
            DuckDB SQL query string with LIMIT 15 bound.
        """
        table_name = target_table or self.detect_target_table(query, user_id=user_id, thread_id=thread_id)
        if not table_name:
            raise ValueError("No matching registered table found for tabular query.")

        # Attempt Tier 1: LLM-Assisted Generation if generator is functional
        if self.generator and not self._is_mock_generator(self.generator):
            try:
                llm_sql = self._generate_sql_with_llm(query, table_name)
                if llm_sql:
                    llm_sql = self._ensure_condition_columns_projected(llm_sql, table_name)
                    # Validate by explaining query in DuckDB
                    self.tabular_store.con.execute(f"EXPLAIN {llm_sql}")
                    return llm_sql
            except Exception as e:
                print(f"[TabularQueryEngine] LLM SQL generation failed ({e}), falling back to deterministic parser.")

        # Tier 2: Universal Deterministic Parser
        sql = self._generate_sql_deterministic(query, table_name)
        return self._ensure_condition_columns_projected(sql, table_name)

    @staticmethod
    def _is_mock_generator(gen: Any) -> bool:
        """Detect if the configured generator is an offline MockGenerator."""
        return gen.__class__.__name__ == "MockGenerator"

    def _ensure_condition_columns_projected(self, sql: str, table_name: str) -> str:
        """Ensure all columns referenced in the WHERE clause are included in the SELECT projection."""
        cols_with_types = self.tabular_store.get_table_columns_with_types(table_name)
        all_cols = set(cols_with_types.keys())

        where_match = re.search(r"\bWHERE\b([\s\S]+?)(?:\bLIMIT\b|;|$)", sql, re.IGNORECASE)
        if not where_match:
            return sql

        where_text = where_match.group(1)
        quoted_cols = set(re.findall(r'"([^"]+)"', where_text))
        referenced_cols = {c for c in quoted_cols if c in all_cols}

        select_match = re.search(r"SELECT\b([\s\S]+?)\bFROM\b", sql, re.IGNORECASE)
        if not select_match:
            return sql

        select_text = select_match.group(1).strip()
        if select_text == "*":
            return sql

        existing_cols = set(re.findall(r'"([^"]+)"', select_text))
        missing_cols = [c for c in referenced_cols if c not in existing_cols]

        # Also check for base columns (e.g. "X" if "X-Answer" is referenced)
        for c in list(missing_cols):
            base_c = c[:-7] if c.endswith("-Answer") else c
            if base_c in all_cols and base_c not in existing_cols and base_c not in missing_cols:
                missing_cols.append(base_c)

        if not missing_cols:
            return sql

        new_select_cols = select_text + ", " + ", ".join(f'"{c}"' for c in missing_cols)
        new_sql = sql[:select_match.start(1)] + " " + new_select_cols + " " + sql[select_match.end(1):]
        return new_sql

    def _generate_sql_with_llm(self, query: str, table_name: str) -> Optional[str]:
        """Prompt active LLM generator to construct schema-aware DuckDB SQL."""
        cols_with_types = self.tabular_store.get_table_columns_with_types(table_name)
        schema_lines = [f'- "{col}" ({col_type})' for col, col_type in cols_with_types.items()]
        schema_str = "\n".join(schema_lines)

        sql_prompt = (
            "You are an expert SQL engineer. Generate a single, valid DuckDB SQL query to answer the user question.\n\n"
            f"TABLE NAME: {table_name}\n"
            "COLUMNS:\n"
            f"{schema_str}\n\n"
            "RULES:\n"
            "1. Return ONLY executable SQL in a ```sql ... ``` block or plain text without explanations.\n"
            "2. Always double-quote column names (e.g., \"Document Name\") as columns contain spaces.\n"
            "3. For BOOLEAN columns, use = true or = false (do NOT compare boolean columns to 'Yes' or 'No').\n"
            "4. If a concept has both a text column (e.g. \"X\") and a boolean answer column (e.g. \"X-Answer\") and the query checks for Yes/No or presence/absence, filter on the boolean answer column.\n"
            "5. In the SELECT clause, ALWAYS include the filter condition columns (and their corresponding base/answer columns) along with the identifier columns (Document Name, Filename, etc.) and any requested attributes, so that the retrieved rows explicitly substantiate the filter conditions.\n"
            "6. Always append LIMIT 15.\n\n"
            f"USER QUESTION: {query.strip()}\n\n"
            "SQL Query:"
        )

        response = self.generator.generate(sql_prompt)
        if not response or response.strip() == "":
            return None

        # Extract SQL from code block if present
        sql_match = re.search(r"```(?:sql)?\s*([\s\S]+?)\s*```", response, re.IGNORECASE)
        sql = sql_match.group(1).strip() if sql_match else response.strip()

        # Sanitize and ensure LIMIT 15
        sql = self._enforce_limit(sql)
        return sql

    def _generate_sql_deterministic(self, query: str, table_name: str) -> str:
        """Universal, domain-agnostic deterministic SQL generator matching conditions and columns."""
        cols_with_types = self.tabular_store.get_table_columns_with_types(table_name)
        cols = list(cols_with_types.keys())
        canonical_cols = {re.sub(r"[^a-zA-Z0-9]", "", c).lower(): c for c in cols}

        # Regex pattern matching relational condition clauses
        cond_pattern = re.compile(
            r"\b(?P<conj>where|and|but|or|with|having|whose|for\s+which)\s+"
            r"(?P<col>[A-Za-z0-9_\s/-]+?)\s+"
            r"(?P<op>is\s+not|does\s+not\s+equal|is|equals|=|equal\s+to|contains?|>|>=|<|<=)\s+"
            r"['\"]?(?P<val>[A-Za-z0-9_.-]+)['\"]?",
            re.IGNORECASE,
        )

        where_parts: List[Tuple[str, str]] = []
        condition_cols: List[str] = []

        for m in cond_pattern.finditer(query):
            d = m.groupdict()
            raw_col = d["col"].strip()
            val = d["val"].strip()
            conj = "OR" if d["conj"].lower() == "or" else "AND"
            op = d["op"].lower().strip()

            c_target = re.sub(r"[^a-zA-Z0-9]", "", raw_col).lower()
            matched_col: Optional[str] = None

            # Check boolean condition preference: e.g. 'Cap On Liability' is 'Yes' -> 'Cap On Liability-Answer' = true
            if val.lower() in ("yes", "no", "true", "false"):
                ans_cand = f"{c_target}answer"
                if ans_cand in canonical_cols and cols_with_types.get(canonical_cols[ans_cand]) == "BOOLEAN":
                    matched_col = canonical_cols[ans_cand]
                elif c_target in canonical_cols and cols_with_types.get(canonical_cols[c_target]) == "BOOLEAN":
                    matched_col = canonical_cols[c_target]

            if not matched_col:
                if c_target in canonical_cols:
                    matched_col = canonical_cols[c_target]
                else:
                    best = difflib.get_close_matches(c_target, list(canonical_cols.keys()), n=1, cutoff=0.7)
                    if best:
                        matched_col = canonical_cols[best[0]]

            if matched_col:
                condition_cols.append(matched_col)
                col_type = cols_with_types.get(matched_col, "VARCHAR")

                if col_type == "BOOLEAN":
                    b_val = "true" if val.lower() in ("yes", "true", "1") else "false"
                    if "not" in op or "!=" in op:
                        b_val = "false" if b_val == "true" else "true"
                    cond_str = f'"{matched_col}" = {b_val}'

                elif col_type in ("INTEGER", "BIGINT", "FLOAT", "DOUBLE", "HUGEINT") and val.replace(".", "", 1).isdigit():
                    num_op = op if op in (">", ">=", "<", "<=", "=") else "="
                    cond_str = f'"{matched_col}" {num_op} {val}'

                else:
                    # Text column matching
                    if "not" in op or "!=" in op:
                        cond_str = f'LOWER("{matched_col}") != \'{val.lower()}\''
                    elif "contain" in op:
                        cond_str = f'"{matched_col}" ILIKE \'%{val}%\''
                    else:
                        cond_str = f'LOWER("{matched_col}") = \'{val.lower()}\''

                where_parts.append((conj, cond_str))

        # Build WHERE SQL
        where_sql = ""
        for idx, (conj, c_str) in enumerate(where_parts):
            if idx == 0:
                where_sql += c_str
            else:
                where_sql += f" {conj} {c_str}"

        # Resolve Projection Columns
        select_cols: List[str] = []

        # 1. Preferred identifier columns
        for c in cols:
            c_l = c.lower()
            if any(id_term in c_l for id_term in ("document name", "document_name", "title", "name", "filename", "id")) and not c_l.endswith("-answer"):
                if c not in select_cols:
                    select_cols.append(c)

        # 2. Query words matching columns (e.g., 'renewal or term periods' matches 'Renewal Term')
        q_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        for c in cols:
            c_tokens = set(re.findall(r"[a-zA-Z0-9]+", c.lower()))
            overlap = len(c_tokens & q_tokens)
            if overlap >= 2 or (len(c_tokens) == 1 and overlap == 1 and len(next(iter(c_tokens))) >= 4):
                if c not in select_cols and not c.endswith("-Answer"):
                    select_cols.append(c)

        # 3. Add base condition columns and their answer companions
        for cc in condition_cols:
            base_cc = cc[:-7] if cc.endswith("-Answer") else cc
            if base_cc in cols and base_cc not in select_cols:
                select_cols.append(base_cc)
            if cc in cols and cc not in select_cols:
                select_cols.append(cc)

        if not select_cols:
            select_str = "*"
        else:
            select_str = ", ".join(f'"{c}"' for c in select_cols)

        where_clause = f"WHERE {where_sql}" if where_sql else "WHERE 1=1"
        sql = f'SELECT {select_str}\nFROM {table_name}\n{where_clause}\nLIMIT 15;'
        return sql

    def _enforce_limit(self, sql: str, max_limit: int = 15) -> str:
        """Ensure SQL query contains a safe LIMIT <= max_limit."""
        clean_sql = sql.rstrip("; \t\n")
        limit_match = re.search(r"\bLIMIT\s+(\d+)", clean_sql, re.IGNORECASE)
        if limit_match:
            existing_limit = int(limit_match.group(1))
            if existing_limit > max_limit:
                clean_sql = re.sub(r"\bLIMIT\s+\d+", f"LIMIT {max_limit}", clean_sql, flags=re.IGNORECASE)
        else:
            clean_sql = f"{clean_sql}\nLIMIT {max_limit}"
        return f"{clean_sql};"

    def execute_safe(
        self,
        sql: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Validate and execute read-only SQL query safely via TabularStore.

        Args:
            sql: SQL statement to execute.
            user_id: Optional requesting tenant user_id.
            thread_id: Optional requesting tenant thread_id.

        Returns:
            List of row dictionaries.

        Raises:
            ValueError: If query contains mutating or unsafe statements.
        """
        if self._MUTATING_KEYWORDS_REGEX.search(sql):
            raise ValueError(f"Unsafe SQL rejected: Mutating keywords detected in query: {sql}")

        safe_sql = self._enforce_limit(sql, max_limit=15)
        return self.tabular_store.execute_query(safe_sql, user_id=user_id, thread_id=thread_id)

    def serialize_rows_to_candidates(
        self,
        rows: List[Dict[str, Any]],
        table_name: str,
        doc_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[RetrievalCandidate]:
        """Serialize returned tabular rows into standard RetrievalCandidate objects.

        Args:
            rows: List of column-value row dictionaries returned by DuckDB.
            table_name: Name of queried table.
            doc_id: Originating document identifier.
            user_id: Optional tenant user_id.
            thread_id: Optional tenant thread_id.

        Returns:
            List of RetrievalCandidate models formatted for prompt and verification.
        """
        effective_doc_id = doc_id or self.tabular_store.get_doc_for_table(table_name, user_id=user_id, thread_id=thread_id) or table_name
        source_type = self.tabular_store.get_source_type_for_table(table_name)
        sheet_name = self.tabular_store.get_sheet_name_for_table(table_name)
        candidates: List[RetrievalCandidate] = []

        for idx, row in enumerate(rows, start=1):
            row_items: List[str] = []
            for col_name, val in row.items():
                if val is None or str(val).strip() in ("", "[]"):
                    continue

                val_str = str(val).strip()

                # Clean up stringified python lists e.g. "['content']" -> "content"
                if (val_str.startswith("['") and val_str.endswith("']")) or (val_str.startswith('["') and val_str.endswith('"]')):
                    val_str = val_str[2:-2].strip().replace("\\'", "'")

                # Format boolean values cleanly
                if isinstance(val, bool):
                    val_str = "Yes" if val else "No"
                elif val_str.lower() == "true":
                    val_str = "Yes"
                elif val_str.lower() == "false":
                    val_str = "No"

                row_items.append(f"{col_name}: {val_str}")

            if not row_items:
                continue

            row_line = " | ".join(row_items)
            text = f"[Section: Table: {table_name} | Row: {idx}] {row_line}"

            prov = ProvenanceCoordinate(
                doc_id=effective_doc_id,
                source_type=source_type,
                page=1,
                section_name=f"Table: {table_name} | Row: {idx}",
                bbox=None,
                sheet_name=sheet_name,
                row_index=idx,
                matched_columns=list(row.keys()),
                snippet=row_line,
            )

            candidate = RetrievalCandidate(
                parent_id=f"{effective_doc_id}_row_{idx}",
                doc_id=effective_doc_id,
                page_number=1,
                chunk_index=idx - 1,
                section_name=f"Table: {table_name} | Row: {idx}",
                text=text,
                score=1.0,
                match_type="tabular_sql",
                user_id=user_id,
                thread_id=thread_id,
                provenance=prov,
            )
            candidates.append(candidate)

        return candidates

    def query(
        self,
        query: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[RetrievalCandidate]:
        """End-to-end execution of natural language relational query to RetrievalCandidates with tenant isolation.

        Args:
            query: User search query string.
            user_id: Optional tenant user_id.
            thread_id: Optional tenant thread_id.

        Returns:
            List of RetrievalCandidate models representing matching table rows.
        """
        table_name = self.detect_target_table(query, user_id=user_id, thread_id=thread_id)
        if not table_name:
            return []

        sql = self.generate_sql(query, target_table=table_name, user_id=user_id, thread_id=thread_id)
        print(f"[TabularEngine] Executing generated DuckDB SQL on '{table_name}':\n{sql}")

        rows = self.execute_safe(sql, user_id=user_id, thread_id=thread_id)
        print(f"[TabularEngine] Query returned {len(rows)} row(s).")

        doc_id = self.tabular_store.get_doc_for_table(table_name, user_id=user_id, thread_id=thread_id) or table_name
        return self.serialize_rows_to_candidates(
            rows,
            table_name=table_name,
            doc_id=doc_id,
            user_id=user_id,
            thread_id=thread_id,
        )
