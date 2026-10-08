r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/tabular_store.py
   - Role: In-memory relational database and tabular schema catalog for TrustRAG.
   - Purpose: Manages an in-process DuckDB instance for deterministic SQL execution
     across structured CSV, TSV, and Excel workbook sheets. Registers tables with
     sanitized identifiers, reflects column schemas and data types, and executes
     relational filters under zero-hardcoding principles.

2. INPUT (IP):
   - file_path (Path | str): Path to CSV, TSV, or Excel file.
   - doc_id (str): Document identifier.
   - sql (str): Relational SQL query string to execute.

3. PROCESS UNDER THE HOOD:
   - Manages an in-process DuckDB connection (`duckdb.connect(database=":memory:")`).
   - Identifier Sanitization:
     * Removes non-alphanumeric characters (except underscores).
     * Replaces slashes, hyphens, and spaces with underscores.
     * Ensures table name starts with a letter (prefixes with `t_` if digit).
   - Ingestion:
     * CSV/TSV: Executes `CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{path}', header=True)`.
       Falls back to pandas loading if needed.
     * Excel: Parses each sheet via pandas into a DataFrame, registers it into DuckDB, and saves as
       `{table_name}_{sheet_name}` (and `{table_name}` if single sheet).
     * Records metadata: table names, column names, column types, row counts, and source doc_id mappings.
   - Schema Reflection & Query Execution:
     * `get_table_schemas() -> Dict[str, List[str]]`: Returns `{table_name: [columns]}`.
     * `get_table_columns_with_types(table_name: str) -> Dict[str, str]`: Returns `{col: type}`.
     * `execute_query(sql: str) -> List[Dict[str, Any]]`: Executes SQL query safely via cursor.

4. OUTPUT (OP):
   - Table schemas, metadata catalogs, and relational row dictionaries.
   - Consumed by: `src/pipeline_2_retrieval/tabular_engine.py` and `src/main.py`.

5. LIBRARIES & DEPENDENCIES:
   - duckdb: In-process SQL OLAP query engine.
   - pandas: Tabular file loading for Excel workbooks.
   - pathlib.Path, re, typing (Dict, List, Optional, Any, Union).
================================================================================
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import duckdb


class TabularStore:
    """Manages an in-process DuckDB relational store and schema catalog for tabular files."""

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize in-process DuckDB connection and metadata registries.

        Args:
            db_path: Path to database or ':memory:' for transient in-process execution.
        """
        self.db_path = db_path
        self.con = duckdb.connect(database=db_path)
        self._table_schemas: Dict[str, List[str]] = {}
        self._table_types: Dict[str, Dict[str, str]] = {}
        self._table_to_doc: Dict[str, str] = {}
        self._doc_to_tables: Dict[str, List[str]] = {}
        self._table_to_tenant: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._table_source_types: Dict[str, str] = {}
        self._table_sheets: Dict[str, Optional[str]] = {}

    @staticmethod
    def sanitize_identifier(name: str) -> str:
        """Sanitize an arbitrary string into a valid SQL table or column identifier.

        Args:
            name: Raw string identifier.

        Returns:
            Sanitized identifier consisting only of alphanumeric characters and underscores.
        """
        if not name:
            return "t_table"
        sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_")
        if not sanitized:
            return "t_table"
        if sanitized[0].isdigit():
            sanitized = f"t_{sanitized}"
        return sanitized

    @staticmethod
    def normalize_column_headers(columns: List[Any]) -> List[str]:
        """Normalize tabular column headers: strip trailing spaces, replace special characters with underscores, deduplicate.

        Args:
            columns: Iterable of raw column names.

        Returns:
            List of normalized, unique, valid SQL column identifiers.
        """
        clean_cols: List[str] = []
        seen: Dict[str, int] = {}
        for idx, col in enumerate(columns):
            raw = str(col).strip() if col is not None else ""
            if not raw or raw.startswith("Unnamed:"):
                base = f"column_{idx + 1}"
            else:
                norm = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_")
                base = norm if norm else f"column_{idx + 1}"
                if base[0].isdigit():
                    base = f"col_{base}"

            count = seen.get(base, 0)
            if count == 0:
                final_name = base
            else:
                final_name = f"{base}_{count}"
            seen[base] = count + 1
            clean_cols.append(final_name)
        return clean_cols

    def register_table_from_file(
        self,
        file_path: Union[str, Path],
        doc_id: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[str]:
        """Ingest and register a structured CSV or Excel file as one or more DuckDB tables with tenant scoping.

        Args:
            file_path: Filesystem path to CSV, TSV, or Excel file.
            doc_id: Document identifier.
            user_id: Optional tenant user_id.
            thread_id: Optional tenant thread_id.

        Returns:
            List of registered table names in DuckDB.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Tabular file not found at: {file_path}")

        ext = path.suffix.lower()
        clean_doc_id = self.sanitize_identifier(doc_id)
        if user_id or thread_id:
            prefix = f"{self.sanitize_identifier(user_id or 'anon')}_{self.sanitize_identifier(thread_id or 'main')}_"
            base_table_name = f"{prefix}{clean_doc_id}"
        else:
            base_table_name = clean_doc_id

        registered_tables: List[str] = []

        if ext in (".csv", ".tsv"):
            import pandas as pd
            sep = "\t" if ext == ".tsv" else ","
            try:
                # Load CSV via pandas to normalize headers and handle up to 100+ columns seamlessly
                df = pd.read_csv(str(path), sep=sep)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if not df.empty:
                    df.columns = self.normalize_column_headers(df.columns)
                    temp_name = f"_temp_df_{base_table_name}"
                    self.con.register(temp_name, df)
                    self.con.execute(f"CREATE OR REPLACE TABLE {base_table_name} AS SELECT * FROM {temp_name}")
                    self.con.unregister(temp_name)
                    self._record_table_metadata(base_table_name, doc_id, source_type="tabular_csv", sheet_name=None, user_id=user_id, thread_id=thread_id)
                    registered_tables.append(base_table_name)
            except Exception as e:
                # Fallback via native DuckDB CSV loader
                quoted_path = str(path.resolve()).replace("'", "''")
                self.con.execute(
                    f"CREATE OR REPLACE TABLE {base_table_name} AS "
                    f"SELECT * FROM read_csv_auto('{quoted_path}', header=True)"
                )
                self._record_table_metadata(base_table_name, doc_id, source_type="tabular_csv", sheet_name=None, user_id=user_id, thread_id=thread_id)
                registered_tables.append(base_table_name)

        elif ext in (".xlsx", ".xls"):
            import pandas as pd
            engine = "openpyxl" if ext == ".xlsx" else None
            excel_file = pd.ExcelFile(str(path), engine=engine)

            sheet_names = excel_file.sheet_names
            for sheet_name in sheet_names:
                df = excel_file.parse(sheet_name)
                # Discard completely empty rows/columns
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue

                # Normalize column headers with underscore substitutions and trailing space removal
                df.columns = self.normalize_column_headers(df.columns)

                sanitized_sheet = self.sanitize_identifier(sheet_name)
                sheet_table_name = f"{base_table_name}_{sanitized_sheet}"

                temp_name = f"_temp_df_{sheet_table_name}"
                self.con.register(temp_name, df)
                self.con.execute(f"CREATE OR REPLACE TABLE {sheet_table_name} AS SELECT * FROM {temp_name}")
                self.con.unregister(temp_name)

                self._record_table_metadata(sheet_table_name, doc_id, source_type="tabular_excel", sheet_name=sheet_name, user_id=user_id, thread_id=thread_id)
                registered_tables.append(sheet_table_name)

            # If only 1 non-empty sheet was registered, also alias base_table_name
            if len(registered_tables) == 1 and registered_tables[0] != base_table_name:
                self.con.execute(
                    f"CREATE OR REPLACE TABLE {base_table_name} AS SELECT * FROM {registered_tables[0]}"
                )
                self._record_table_metadata(
                    base_table_name,
                    doc_id,
                    source_type="tabular_excel",
                    sheet_name=sheet_names[0] if sheet_names else None,
                    user_id=user_id,
                    thread_id=thread_id,
                )
                registered_tables.append(base_table_name)

        if registered_tables:
            self._doc_to_tables[doc_id] = registered_tables
            for t in registered_tables:
                self._table_to_doc[t] = doc_id
            print(f"[TabularStore] Registered {len(registered_tables)} table(s) for '{doc_id}': {registered_tables}")

        return registered_tables

    def register_table_from_dataframe(
        self,
        df: Any,
        table_name: str,
        doc_id: str,
        source_type: str = "pdf_table",
        sheet_name: Optional[str] = None,
        page_no: Optional[int] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """Register a pandas DataFrame directly into DuckDB with schema caching and tenant scoping.

        Args:
            df: pandas DataFrame containing table rows.
            table_name: Desired table name identifier.
            doc_id: Originating document identifier.
            source_type: Category identifier ('pdf_table', 'tabular_csv', etc.).
            sheet_name: Optional sheet name.
            page_no: Optional 1-indexed page number.
            user_id: Optional tenant user_id.
            thread_id: Optional tenant thread_id.

        Returns:
            Registered table name in DuckDB, or empty string if registration fails.
        """
        import pandas as pd
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return ""

        clean_table_name = self.sanitize_identifier(table_name)

        # Make copy of DataFrame and ensure clean, non-empty, unique column names
        df_clean = df.copy()
        df_clean.columns = self.normalize_column_headers(df_clean.columns)

        if user_id or thread_id:
            prefix = f"{self.sanitize_identifier(user_id or 'anon')}_{self.sanitize_identifier(thread_id or 'main')}_"
            effective_table_name = f"{prefix}{clean_table_name}"
        else:
            effective_table_name = clean_table_name

        try:
            temp_name = f"_temp_df_{clean_table_name}"
            self.con.register(temp_name, df_clean)
            self.con.execute(f"CREATE OR REPLACE TABLE {effective_table_name} AS SELECT * FROM {temp_name}")
            self.con.unregister(temp_name)

            # If tenant-prefixed, also register non-prefixed alias view if not colliding
            if effective_table_name != clean_table_name:
                try:
                    self.con.execute(f"CREATE OR REPLACE VIEW {clean_table_name} AS SELECT * FROM {effective_table_name}")
                except Exception:
                    pass

            self._record_table_metadata(
                effective_table_name,
                doc_id,
                source_type=source_type,
                sheet_name=sheet_name or (f"Page_{page_no}" if page_no is not None else None),
                user_id=user_id,
                thread_id=thread_id,
            )
            if effective_table_name != clean_table_name:
                self._record_table_metadata(
                    clean_table_name,
                    doc_id,
                    source_type=source_type,
                    sheet_name=sheet_name or (f"Page_{page_no}" if page_no is not None else None),
                    user_id=user_id,
                    thread_id=thread_id,
                )

            if doc_id not in self._doc_to_tables:
                self._doc_to_tables[doc_id] = []
            if clean_table_name not in self._doc_to_tables[doc_id]:
                self._doc_to_tables[doc_id].append(clean_table_name)
            self._table_to_doc[clean_table_name] = doc_id
            if effective_table_name != clean_table_name:
                self._table_to_doc[effective_table_name] = doc_id

            print(f"[TabularStore] Registered DataFrame table '{clean_table_name}' for doc '{doc_id}' ({len(df_clean)} rows).")
            return clean_table_name
        except Exception as e:
            print(f"[TabularStore] Warning: Failed to register DataFrame table '{clean_table_name}': {e}")
            return ""

    def register_table_from_rows(
        self,
        headers: List[str],
        rows: List[List[Any]],
        table_name: str,
        doc_id: str,
        source_type: str = "pdf_table",
        sheet_name: Optional[str] = None,
        page_no: Optional[int] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """Register raw tabular rows with headers into DuckDB."""
        import pandas as pd
        if not rows:
            return ""
        df = pd.DataFrame(rows, columns=headers)
        return self.register_table_from_dataframe(
            df=df,
            table_name=table_name,
            doc_id=doc_id,
            source_type=source_type,
            sheet_name=sheet_name,
            page_no=page_no,
            user_id=user_id,
            thread_id=thread_id,
        )

    def _record_table_metadata(
        self,
        table_name: str,
        doc_id: str,
        source_type: str = "tabular_csv",
        sheet_name: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Inspect and cache table schema column names, data types, and tenant mapping."""
        cols_info = self.con.execute(f"DESCRIBE {table_name}").fetchall()
        col_names = [r[0] for r in cols_info]
        col_types = {r[0]: str(r[1]).upper() for r in cols_info}
        self._table_schemas[table_name] = col_names
        self._table_types[table_name] = col_types
        self._table_to_tenant[table_name] = (user_id, thread_id)
        self._table_source_types[table_name] = source_type
        self._table_sheets[table_name] = sheet_name

    def get_source_type_for_table(self, table_name: str) -> str:
        """Return the source type ('tabular_csv' or 'tabular_excel') for a table."""
        return self._table_source_types.get(table_name, "tabular_csv")

    def get_sheet_name_for_table(self, table_name: str) -> Optional[str]:
        """Return the source sheet name for an Excel table, if applicable."""
        return self._table_sheets.get(table_name, None)

    def get_table_schemas(
        self,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Return registered table names and their column headers filtered by tenant.

        Args:
            user_id: Optional tenant user_id.
            thread_id: Optional tenant thread_id.

        Returns:
            Dictionary mapping table name to list of column names.
        """
        if user_id is not None:
            return {
                t: cols for t, cols in self._table_schemas.items()
                if (self._table_to_tenant.get(t, (None, None))[0] is None or self._table_to_tenant.get(t, (None, None))[0] == user_id)
                and (thread_id is None or self._table_to_tenant.get(t, (None, None))[1] is None or self._table_to_tenant.get(t, (None, None))[1] == thread_id)
            }
        else:
            unscoped = {
                t: cols for t, cols in self._table_schemas.items()
                if self._table_to_tenant.get(t, (None, None))[0] is None
            }
            return unscoped if unscoped or not self._table_schemas else dict(self._table_schemas)

    def get_table_columns_with_types(self, table_name: str) -> Dict[str, str]:
        """Return column names and SQL types for a specific table.

        Args:
            table_name: Name of registered DuckDB table.

        Returns:
            Dictionary mapping column name to uppercase SQL data type.
        """
        return dict(self._table_types.get(table_name, {}))

    def get_table_names(
        self,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[str]:
        """Return registered table names scoped by tenant."""
        return list(self.get_table_schemas(user_id=user_id, thread_id=thread_id).keys())

    def get_doc_for_table(
        self,
        table_name: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[str]:
        """Return originating doc_id for a given table name verifying tenant ownership."""
        if user_id is not None:
            tenant = self._table_to_tenant.get(table_name, (None, None))
            if tenant[0] is not None and tenant[0] != user_id:
                return None
            if thread_id is not None and tenant[1] is not None and tenant[1] != thread_id:
                return None
        return self._table_to_doc.get(table_name)

    def get_tables_for_doc(
        self,
        doc_id: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[str]:
        """Return table names registered for a given doc_id verifying tenant ownership."""
        tables = self._doc_to_tables.get(doc_id, [])
        if user_id is not None:
            return [
                t for t in tables
                if (self._table_to_tenant.get(t, (None, None))[0] is None or self._table_to_tenant.get(t, (None, None))[0] == user_id)
                and (thread_id is None or self._table_to_tenant.get(t, (None, None))[1] is None or self._table_to_tenant.get(t, (None, None))[1] == thread_id)
            ]
        return tables

    def execute_query(
        self,
        sql: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a read-only SQL query safely via cursor with cross-tenant access enforcement.

        Args:
            sql: SQL statement to execute.
            user_id: Optional requesting tenant user_id.
            thread_id: Optional requesting tenant thread_id.

        Returns:
            List of row dictionaries with column-value mappings.

        Raises:
            PermissionError: If SQL references a table registered to another tenant.
        """
        if user_id is not None:
            for t, (t_user, t_thr) in self._table_to_tenant.items():
                if t_user is not None and t_user != user_id:
                    if re.search(r"\b" + re.escape(t) + r"\b", sql, re.IGNORECASE):
                        raise PermissionError(
                            f"Cross-tenant access violation: table '{t}' does not belong to user '{user_id}'."
                        )

        cursor = self.con.cursor()
        try:
            df = cursor.execute(sql).df()
            return df.to_dict(orient="records")
        finally:
            cursor.close()

    def close(self) -> None:
        """Close DuckDB connection and free resources."""
        if hasattr(self, "con") and self.con is not None:
            try:
                self.con.close()
            except Exception:
                pass
