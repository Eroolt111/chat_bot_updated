import psycopg2
import oracledb
import pandas as pd
from sqlalchemy import create_engine, MetaData, inspect, text, bindparam
from sqlalchemy.engine import Engine
from typing import List, Dict, Optional, Tuple
import logging
from contextlib import contextmanager
from sqlalchemy.exc import SQLAlchemyError
from .config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection and management class"""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self._connect()
    
    def _connect(self) -> None:
        """Create SQLAlchemy engine for ORACLE database"""
        try:
            self.engine = create_engine(
                config.DATABASE_URL,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT banner FROM v$version"))
                version = result.fetchone()[0]
                logger.info(f"Connected to Oracle DB: {version}")
                
        except Exception as e:
            logger.error(f"Error creating database engine: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1 FROM dual"))
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_table_names(self) -> List[str]:
        """Get list of table names in the database"""
        try:
            if config.TEST_MODE:
                # Return only test tables
                logger.info(f"TEST_MODE enabled, using tables: {config.TEST_TABLES}")
                return [table.strip() for table in config.TEST_TABLES if table.strip()]
            
            # Normal mode - get all tables from Oracle
            inspector = inspect(self.engine)
            # For Oracle, you might want to specify schema
            all_tables = inspector.get_table_names(schema="DBM")
            # Add schema prefix if needed
            return [f"DBM.{table}" for table in all_tables]
            
        except Exception as e:
            logger.error(f"Error getting table names: {e}")
            return []
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get detailed information about a table"""
        try:
            # Handle  schema.table format
            if '.' in table_name:
                schema_name, table_only = table_name.split('.', 1)
            else:
                schema_name, table_only = 'DBM', table_name
            
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_only, schema=schema_name)
            
            # Get row count.
            with self.get_connection() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM {table_name}'))
                row_count = result.fetchone()[0]
            
            return {
                'table_name': table_name,
                'columns': [f"{col['name']} ({col['type']})" for col in columns],
                'column_names': [col['name'] for col in columns],
                'row_count': row_count
            }
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            return {}
    
    def load_table_data(self, table_name: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Load data from a table"""
        try:
            # Use test row limit if in test mode
            if config.TEST_MODE:
                limit = min(limit, config.TEST_ROW_LIMIT)
                logger.info(f"TEST_MODE: limiting {table_name} to {limit} rows")
            
            # Remove quotes for Oracle
            query = f'SELECT * FROM {table_name} FETCH FIRST {limit} ROWS ONLY'
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} rows from table '{table_name}'")
            return df
        except Exception as e:
            logger.error(f"Error loading data from table {table_name}: {e}")
            return None
    


    # -------------------------------------------------------------------------
    # INCREMENTAL INDEXING HELPERS
    # -------------------------------------------------------------------------
    def get_new_rows_since_id(
        self,
        table_name: str,
        last_id: int,
        id_column: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch new rows from `table_name` where id_column > last_id.
        """
        with self.get_connection() as conn:
            # Auto-detect id column if needed using Oracle system tables
            if id_column is None:
                if '.' in table_name:
                    schema_name, table_only = table_name.split('.', 1)
                else:
                    schema_name, table_only = 'DBM', table_name
                
                cols = conn.execute(text("""
                    SELECT column_name 
                    FROM all_tab_columns 
                    WHERE owner = :schema 
                    AND table_name = :tbl 
                    AND column_name IN ('ID', 'id', 'Id')
                    ORDER BY column_id
                """), {"schema": schema_name.upper(), "tbl": table_only.upper()}).fetchall()
            
                if not cols:
                    raise ValueError(f"No ID column found for table {table_name}")
                id_column = cols[0][0]

            # Build and execute query
            sql = f'SELECT * FROM {table_name} WHERE {id_column} > :last_id ORDER BY {id_column}'
            if limit:
                sql += f' FETCH FIRST {limit} ROWS ONLY'

            result = conn.execute(text(sql), {"last_id": last_id})
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    def get_new_rows_by_offset(
        self,
        table_name: str,
        offset: int,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch rows from `table_name` starting at `offset`.
        Returns list of dicts.
        """
        with self.get_connection() as conn:
            sql = f'SELECT * FROM "{table_name}" OFFSET :offset'
            if limit:
                sql += f' LIMIT {limit}'

            result = conn.execute(text(sql), {"offset": offset})
            # Use RowMapping to get dict-like rows
            rows = result.mappings().all()
            return [dict(r) for r in rows]
        
    def detect_date_column(self, table_name: str) -> Optional[str]:
        """
        Auto-detect a suitable date column for ordering latest records.
        Prioritizes columns of DATE/TIMESTAMP type or with 'date' in name.
        Returns best match or None if no candidates found.
        """
        try:
            # Handle schema.table format
            if '.' in table_name:
                schema_name, table_only = table_name.split('.', 1)
            else:
                schema_name, table_only = 'DBM', table_name
            
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_only, schema=schema_name)
            
            date_candidates = []
            priority_names = ['txn_date', 'created_datetime', 'start_date', 'end_date', 'approv_date', 'modified_datetime','last_txn_date']  # Add more common ones as needed
            
            for col in columns:
                col_type = str(col['type']).lower()
                col_name = col['name'].lower()
                
                # Check if likely a date column
                if 'date' in col_type or 'timestamp' in col_type or 'date' in col_name:
                    priority = 0  # Default low priority
                    for idx, pri in enumerate(priority_names):
                        if pri in col_name:
                            priority = -idx  # Negative for sorting: higher priority = lower number
                            break
                    date_candidates.append((col['name'], priority))  # Original case for query
            
            if date_candidates:
                # Sort: highest priority first
                date_candidates.sort(key=lambda x: x[1])
                best_date_col = date_candidates[0][0]
                logger.info(f"Detected date column for {table_name}: {best_date_col}")
                return best_date_col
            else:
                logger.warning(f"No suitable date column detected for {table_name}")
                return None
        
        except Exception as e:
            logger.error(f"Error detecting date column for {table_name}: {e}")
            return None

    def load_unique_table_data(self, table_name: str, unique_columns: List[str] = None, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Load unique data from a table based on specified columns, selecting the latest row per group.
        Auto-detects date column for ordering; falls back if none found.
        """
        try:
            # Apply test mode limit if enabled
            if config.TEST_MODE:
                limit = min(limit, config.TEST_ROW_LIMIT)
                logger.info(f"TEST_MODE: limiting {table_name} to {limit} rows")
            
            # Fallback to regular load if no unique columns
            if not unique_columns:
                return self.load_table_data(table_name, limit)
            
            # Validate unique columns
            valid_columns = self._get_valid_unique_columns(table_name, unique_columns)
            if not valid_columns:
                logger.warning(f"No valid unique columns found for {table_name}, falling back to regular load")
                return self.load_table_data(table_name, limit)
            
            # Auto-detect date column
            date_column = self.detect_date_column(table_name)
            
            if date_column:
                # Build query with ordering for latest
                partition_cols = ", ".join(valid_columns)
                query = f"""
                SELECT * FROM (
                    SELECT t.*,
                        ROW_NUMBER() OVER (PARTITION BY {partition_cols} ORDER BY {date_column} DESC) AS rn
                    FROM {table_name} t
                ) sub
                WHERE sub.rn = 1
                ORDER BY {date_column} DESC
                FETCH FIRST {limit} ROWS ONLY
                """
            else:
                # No date column: fallback to regular load (avoid arbitrary unique selection)
                logger.info(f"No date column for {table_name}, falling back to regular load without unique filtering")
                return self.load_table_data(table_name, limit)
            
            logger.info(f"Loading unique data (latest per group) from {table_name} based on columns: {valid_columns}")
            logger.debug(f"Unique query: {query}")
            
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} unique rows from table '{table_name}' (based on {valid_columns})")
            
            return df
        
        except SQLAlchemyError as e:
            logger.error(f"SQL error loading unique data from {table_name}: {e}")
            logger.info(f"Falling back to regular table load for {table_name}")
            return self.load_table_data(table_name, limit)
        except Exception as e:  # Broad catch for unexpected
            logger.error(f"Unexpected error loading unique data from {table_name}: {e}")
            return None
    def _get_valid_unique_columns(self, table_name: str, unique_columns: List[str]) -> List[str]:
        """Validate which unique columns actually exist in the table"""
        if not unique_columns:
            return []
        
        try:
            if '.' in table_name:
                schema_name, table_only = table_name.split('.', 1)
            else:
                # Default schema if not provided
                schema_name, table_only = 'DBM', table_name

            # Use SQLAlchemy's 'expanding' bindparam for cross-database compatibility.
            # This fixes the "DPY-3002: Python value of type 'tuple' is not supported" error with oracledb.
            stmt = text(
                """
                SELECT column_name
                FROM all_tab_columns
                WHERE owner = :schema
                AND table_name = :tbl
                AND column_name IN :unique_cols_list
                ORDER BY column_id
                """
            ).bindparams(
                bindparam('unique_cols_list', expanding=True)
            )

            with self.get_connection() as conn:
                result = conn.execute(
                    stmt,
                    {
                        "schema": schema_name.upper(),
                        "tbl": table_only.upper(),
                        "unique_cols_list": [c.upper() for c in unique_columns]
                    }
                )
                valid_columns_upper = [row[0] for row in result.fetchall()]
            
            # Return in original (config) case for consistency
            existing = [col for col in unique_columns if col.upper() in valid_columns_upper]
            
            if len(existing) < len(unique_columns):
                not_found = set(unique_columns) - set(existing)
                logger.warning(f"Some unique columns not found for {table_name}: {not_found}")

            logger.info(f"Validated unique columns for {table_name}: {existing}")
            return existing
        
        except Exception as e:
            logger.error(f"Error validating unique columns for {table_name}: {e}")
            return []


    def get_table_unique_info(self, table_name: str, unique_columns: List[str] = None) -> Dict:
        """Get information about unique values in a table"""
        try:
            info = self.get_table_info(table_name)
            
            if unique_columns:
                valid_columns = self._get_valid_unique_columns(table_name, unique_columns)
                if valid_columns:
                    # Get count of unique combinations
                    distinct_cols = ", ".join(valid_columns)
                    with self.get_connection() as conn:
                        unique_query = f'SELECT COUNT(DISTINCT {distinct_cols}) FROM {table_name}'
                        unique_count = conn.execute(text(unique_query)).fetchone()[0]
                        
                    info.update({
                        'unique_columns': valid_columns,
                        'unique_count': unique_count,
                        'uniqueness_ratio': round(unique_count / max(info.get('row_count', 1), 1), 4)
                    })
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting unique table info for {table_name}: {e}")
            return self.get_table_info(table_name)
        
    def get_unique_names(
            self, table_name: str, name_columns: List[str] = None, limit: int = 500
    ) -> List[Dict[str, str]]:
        """Get unique names from specified columns in a table"""
        if name_columns is None:
            name_columns = config.NAME_COLUMNS
        try:
            if '.' in table_name:
                schema_name, table_only = table_name.split('.', 1)
            else:
                schema_name, table_only = 'DBM', table_name

            inspector = inspect(self.engine)
            existing_columns = [col['name'] for col in inspector.get_columns(table_only, schema=schema_name)]
            valid_name_columns = [col for col in name_columns if col in existing_columns]

            if not valid_name_columns:
                logger.warning(f"No valid name columns found for {table_name}")
                return []
            #BUILD
            # ========== ORACLE-COMPATIBLE QUERY ==========
            # Index by FREQUENCY (most common names first) to ensure important customers are indexed
            columns_str = ", ".join(valid_name_columns)
            where_clause = " OR ".join([f"{col} IS NOT NULL" for col in valid_name_columns])
            
            # Use ORDER BY frequency (most loans = more important) instead of alphabetical
            # This ensures commonly queried customers are indexed first
            query = f"""
            SELECT {columns_str}, COUNT(*) as freq
            FROM {table_name}
            WHERE {where_clause}
            GROUP BY {columns_str}
            ORDER BY freq DESC
            FETCH FIRST {limit} ROWS ONLY
            """
            # ============================================
            logger.info(f"Extracting unique names from {table_name} using columns: {valid_name_columns}")
            with self.get_connection() as conn:
                result = conn.execute(text(query))
                rows = result.mappings().all()
            
            # Remove 'freq' column and filter out rows where ALL name columns are None
            unique_names = []
            for r in rows:
                row_dict = dict(r)
                row_dict.pop('freq', None)
                # Skip rows where every name value is None
                if all(v is None for v in row_dict.values()):
                    continue
                unique_names.append(row_dict)
            logger.info(f"Extracted {len(unique_names)} unique names from {table_name}")
            return unique_names
        except Exception as e:
            logger.error(f"Error extracting unique names from {table_name}: {e}")
            return []
        
    def get_name_count(self, table_name:str, name_columns: List[str] = None) -> int:

        if name_columns is None:
            name_columns = config.NAME_COLUMNS

        try:
            if '.' in table_name:
                schema_name, table_only = table_name.split('.', 1)
            else:
                schema_name, table_only = 'DBM', table_name

            inspector = inspect(self.engine)
            existing_columns = [col['name'] for col in inspector.get_columns(table_only, schema=schema_name)]
            valid_name_columns = [col for col in name_columns if col in existing_columns]

            if not valid_name_columns:
                logger.warning(f"No valid name columns found for {table_name}")
                return 0

            columns_str = ", ".join(valid_name_columns)
            where_clause = " OR ".join([f"{col} IS NOT NULL" for col in valid_name_columns])
            
            # Correct Oracle syntax
            query = f"""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT {columns_str}
                    FROM {table_name}
                    WHERE {where_clause}
            )
            """
            with self.get_connection() as conn:
                result = conn.execute(text(query)).scalar()
    

            logger.info(f"Counted {result} unique names in {table_name}")
            return result
        except Exception as e:  
            logger.error(f"Error counting unique names in {table_name}: {e}")
            return 0




db_manager = DatabaseManager()
