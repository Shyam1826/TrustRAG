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
from typing import Any, Dict, List, Optional, Union
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

    def register_table_from_file(
        self,
        file_path: Union[str, Path],
        doc_id: str,
    ) -> List[str]:
        """Ingest and register a structured CSV or Excel file as one or more DuckDB tables.

        Args:
            file_path: Filesystem path to CSV, TSV, or Excel file.
            doc_id: Document identifier.

        Returns:
            List of registered table names in DuckDB.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Tabular file not found at: {file_path}")

        ext = path.suffix.lower()
        base_table_name = self.sanitize_identifier(doc_id)
        registered_tables: List[str] = []

        if ext in (".csv", ".tsv"):
            try:
                # Fast native DuckDB CSV loader
                quoted_path = str(path.resolve()).replace("'", "''")
                self.con.execute(
                    f"CREATE OR REPLACE TABLE {base_table_name} AS "
                    f"SELECT * FROM read_csv_auto('{quoted_path}', header=True)"
                )
                self._record_table_metadata(base_table_name, doc_id)
                registered_tables.append(base_table_name)
            except Exception as e:
                # Fallback via pandas if native loader hits delimiter or encoding quirks
                import pandas as pd
                sep = "\t" if ext == ".tsv" else ","
                df = pd.read_csv(str(path), sep=sep)
                temp_name = f"_temp_df_{base_table_name}"
                self.con.register(temp_name, df)
                self.con.execute(f"CREATE OR REPLACE TABLE {base_table_name} AS SELECT * FROM {temp_name}")
                self.con.unregister(temp_name)
                self._record_table_metadata(base_table_name, doc_id)
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

                # Clean column headers
                clean_cols = []
                for c_idx, c in enumerate(df.columns):
                    c_str = str(c).strip()
                    if not c_str or c_str.startswith("Unnamed:"):
                        clean_cols.append(f"Column_{c_idx + 1}")
                    else:
                        clean_cols.append(c_str)
                df.columns = clean_cols

                sanitized_sheet = self.sanitize_identifier(sheet_name)
                sheet_table_name = f"{base_table_name}_{sanitized_sheet}"

                temp_name = f"_temp_df_{sheet_table_name}"
                self.con.register(temp_name, df)
                self.con.execute(f"CREATE OR REPLACE TABLE {sheet_table_name} AS SELECT * FROM {temp_name}")
                self.con.unregister(temp_name)

                self._record_table_metadata(sheet_table_name, doc_id)
                registered_tables.append(sheet_table_name)

            # If only 1 non-empty sheet was registered, also alias base_table_name
            if len(registered_tables) == 1 and registered_tables[0] != base_table_name:
                self.con.execute(
                    f"CREATE OR REPLACE TABLE {base_table_name} AS SELECT * FROM {registered_tables[0]}"
                )
                self._record_table_metadata(base_table_name, doc_id)
                registered_tables.append(base_table_name)

        if registered_tables:
            self._doc_to_tables[doc_id] = registered_tables
            for t in registered_tables:
                self._table_to_doc[t] = doc_id
            print(f"[TabularStore] Registered {len(registered_tables)} table(s) for '{doc_id}': {registered_tables}")

        return registered_tables

    def _record_table_metadata(self, table_name: str, doc_id: str) -> None:
        """Inspect and cache table schema column names and data types."""
        cols_info = self.con.execute(f"DESCRIBE {table_name}").fetchall()
        col_names = [r[0] for r in cols_info]
        col_types = {r[0]: str(r[1]).upper() for r in cols_info}
        self._table_schemas[table_name] = col_names
        self._table_types[table_name] = col_types

    def get_table_schemas(self) -> Dict[str, List[str]]:
        """Return registered table names and their column headers.

        Returns:
            Dictionary mapping table name to list of column names.
        """
        return dict(self._table_schemas)

    def get_table_columns_with_types(self, table_name: str) -> Dict[str, str]:
        """Return column names and SQL types for a specific table.

        Args:
            table_name: Name of registered DuckDB table.

        Returns:
            Dictionary mapping column name to uppercase SQL data type.
        """
        return dict(self._table_types.get(table_name, {}))

    def get_table_names(self) -> List[str]:
        """Return all registered table names."""
        return list(self._table_schemas.keys())

    def get_doc_for_table(self, table_name: str) -> Optional[str]:
        """Return originating doc_id for a given table name."""
        return self._table_to_doc.get(table_name)

    def get_tables_for_doc(self, doc_id: str) -> List[str]:
        """Return table names registered for a given doc_id."""
        return self._doc_to_tables.get(doc_id, [])

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a read-only SQL query safely via cursor.

        Args:
            sql: SQL statement to execute.

        Returns:
            List of row dictionaries with column-value mappings.
        """
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
