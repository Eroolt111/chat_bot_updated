import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import ast
import pandas as pd
import decimal
from datetime import datetime 
import sqlfluff 
import unicodedata
import string
from threading import Lock
import threading
# ============================================================================
# LLAMAINDEX IMPORTS (Updated for LlamaIndex v0.11+)
# ============================================================================
# CRITICAL CHANGES:
# 1. Pydantic v2: Use 'from pydantic import BaseModel, Field' instead of bridge
# 2. structured_predict: Replaces LLMTextCompletionProgram.from_defaults()
# 3. ChatPromptTemplate: Required for structured_predict
# ============================================================================

from llama_index.core import SQLDatabase, VectorStoreIndex, load_index_from_storage, Document
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.query_pipeline import QueryPipeline as QP, InputComponent, FnComponent
from llama_index.core.prompts import ChatPromptTemplate  # NEW: For structured_predict
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.core.llms import ChatResponse, ChatMessage  # NEW: ChatMessage added
from pydantic import BaseModel, Field  # CHANGED: Direct pydantic v2 import (was bridge.pydantic)
from sqlalchemy import text, schema
from sqlalchemy.exc import SQLAlchemyError
from llama_index.core.memory.types import BaseMemory

from .example_retriever import example_retriever 
from .config import config
from .db import db_manager
from .llm import llm_manager
from .prompts import prompt_manager
from .sandbox import SyntheticDataGenerator
from .user_input_processor import UserInputPIIProcessor
from .context_builder import FastContextBuilder
from .query_logger import query_logger
logger = logging.getLogger(__name__)

sqlfluff_logger = logging.getLogger('sqlfluff')
sqlfluff_logger.setLevel(logging.WARNING)

class TableInfo(BaseModel):
    """Information regarding a structured table."""
    table_name: str = Field(..., description="table name (must be underscores and NO spaces)")
    table_summary: str = Field(..., description="short, concise summary/caption of the table")
    column_descriptions: dict = Field(..., description="dictionary of column_name: description")

class QueryAnalysis(BaseModel):
    """Data model for the output of the query analyzer."""
    complexity: str = Field(..., description="Either 'SIMPLE' or 'COMPLEX'.")
    needs_chat_history: bool = Field(default=False, description="Whether this query depends on previous conversation context.")  # ← NEW
    chat_history_reasoning: str = Field(default="", description="Explanation of how previous context is used (if applicable).")  # ← NEW
    needs_deduplication: bool = Field(default=False, description="Whether this query needs deduplication logic.")  # ← NEW
    explanation: str = Field(..., description="A brief explanation of the user's goal and the plan to achieve it. MUST include table selection reasoning.")
    required_tables: List[str] = Field(..., description="A list of the exact table names required to answer the query.")
    required_columns: Dict[str, List[str]] = Field(..., description="A dictionary where keys are table names and values are lists of the exact column names required from that table.")
    sub_questions: List[str] = Field(default_factory=list, description="A list of sub-questions if the query is complex.")


class IndexTracker:
    """Track which database rows have been indexed"""
    def __init__(self, tracker_file: str = None):
        if tracker_file is None:
            tracker_file = Path(config.TABLE_INDEX_DIR) / "index_tracker.json"
        self.tracker_file = Path(tracker_file)
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        self.load_tracker()
    def load_tracker(self):
        try:
            with open(self.tracker_file, 'r', encoding='utf-8') as f:
                self.tracked = json.load(f)
        except FileNotFoundError:
            self.tracked = {}

    def save_tracker(self):
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(self.tracked, f, indent=2, ensure_ascii=False)

    def get_last_indexed_id(self, table_name: str) -> int:
        return self.tracked.get(table_name, {}).get('last_id', 0)

    def get_last_indexed_count(self, table_name: str) -> int:
        return self.tracked.get(table_name, {}).get('last_count', 0)

    def update_last_indexed(self, table_name: str, last_id: int = None, last_count: int = None):
        if table_name not in self.tracked:
            self.tracked[table_name] = {}
        if last_id is not None:
            self.tracked[table_name]['last_id'] = last_id
        if last_count is not None:
            self.tracked[table_name]['last_count'] = last_count
        self.tracked[table_name]['last_update'] = datetime.now().isoformat()
        self.save_tracker()


class ChatbotPipeline:
    """Main chatbot pipeline for text-to-SQL and response generation with PII masking"""
    def __init__(self):
        self.sql_database = None
        self.query_pipeline = None
        self.table_infos = {}
        self.vector_index_dict = {}
        self.name_index_dict = {}
        self.obj_retriever = None
        self.index_tracker = IndexTracker()
        self.user_input_processor = UserInputPIIProcessor()
        self.context_builder = FastContextBuilder()
        #self._stage_outputs: Dict[str, Any] = {}
        #self._stage_outputs_lock = Lock()
        self._thread_local = threading.local()
        self.chat_history: Dict[str, List[Dict]] = {}
        self.chat_history_lock = Lock()
        self._initialize()

    def _initialize(self):
        logger.info("Initializing chatbot pipeline...")
        self.sql_database = SQLDatabase(db_manager.engine)
        self._generate_table_summaries()
        if config.ENABLE_NAME_INDEX:
            self._create_name_indices()
        self._create_vector_indices()
        self._setup_query_pipeline() 
        logger.info("✅ Chatbot pipeline initialized successfully")

    @property
    def _stage_outputs(self) -> Dict[str, Any]:
        """Get thread-local stage outputs"""
        if not hasattr(self._thread_local, 'stage_outputs'):
            self._thread_local.stage_outputs = {}
        return self._thread_local.stage_outputs
    
    @_stage_outputs.setter
    def _stage_outputs(self, value: Dict[str, Any]):
        """Set thread-local stage outputs"""
        self._thread_local.stage_outputs = value
        
    def is_first_run(self) -> bool:
        return len(self.index_tracker.tracked) == 0

    def incremental_update(self) -> int:
        """Enhanced incremental update with unique filtering support"""
        total_new_docs = 0
        
        for table_name in self.sql_database.get_usable_table_names():
            # Check if table uses unique filtering
            unique_columns = config.get_unique_columns_for_table(table_name)
            
            if unique_columns:
                logger.info(f"Updating unique index for {table_name} (columns: {unique_columns})")
                total_new_docs += self._update_unique_table_index(table_name, unique_columns)
            else:
                logger.info(f"Updating regular index for {table_name}")
                total_new_docs += self._update_table_index(table_name)
        
        logger.info(f"Incremental update complete. Added {total_new_docs} new documents")
        return total_new_docs

    def _update_table_index(self, table_name: str) -> int:
        try:
            logger.debug(f"[DEBUG] Entering _update for {table_name}")
            with db_manager.get_connection() as conn:
                current_count = conn.execute(text(f'SELECT COUNT(*) FROM {table_name}')).scalar()
                last_count = self.index_tracker.get_last_indexed_count(table_name)
                last_id = self.index_tracker.get_last_indexed_id(table_name)
                
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
                
            id_col = cols[0][0] if cols else None
            
            if current_count <= last_count:
                return 0
                
            # Create safe filename for schema.table
            safe_table_name = table_name.replace('.', '_')
            idx_path = Path(config.TABLE_INDEX_DIR) / safe_table_name
            
            if not idx_path.exists():
                return self._create_full_table_index(table_name)
            
            # Load existing index
            idx = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(idx_path)), 
                index_id="vector_index"
            )
            
            # Get new rows
            if id_col:
                new_rows = db_manager.get_new_rows_since_id(
                    table_name, last_id, id_column=id_col, 
                    limit=config.TEST_ROW_LIMIT if config.TEST_MODE else config.MAX_ROWS_PER_TABLE
                )
            else:
                new_rows = db_manager.get_new_rows_by_offset(
                    table_name, offset=last_count, 
                    limit=config.TEST_ROW_LIMIT if config.TEST_MODE else config.MAX_ROWS_PER_TABLE
                )
            
            if not new_rows:
                return 0
            
            # Add new documents
            for row in new_rows:
                idx.insert(Document(text=str(row)))
            
            idx.storage_context.persist(str(idx_path))
            self.vector_index_dict[table_name] = idx
            
            # Update tracker
            self.index_tracker.update_last_indexed(table_name, last_count=current_count)
            if id_col:
                max_id = max(row[id_col] for row in new_rows)
                self.index_tracker.update_last_indexed(table_name, last_id=max_id)
            
            return len(new_rows)
            
        except Exception as e:
            logger.error(f"Error updating index for {table_name}: {e}")
            return 0

    # #############################################################################
    # NEW HELPER FUNCTION TO SANITIZE DATA
    # #############################################################################
    def _sanitize_row_for_indexing(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Converts special data types (Timestamp, NaT, nan) to JSON-safe formats."""
        sanitized_row = {}
        for key, value in row.items():
            if pd.isna(value):
                sanitized_row[key] = None
            elif isinstance(value, datetime):
                sanitized_row[key] = value.isoformat()
            else:
                sanitized_row[key] = value
        return sanitized_row
    
    def _create_name_indices(self):
        logger.info("Creating name indices for tables...")

        Path(config.NAME_INDEX_DIR).mkdir(exist_ok=True)
        table_names = db_manager.get_table_names()
        for table_name in table_names:
            logger.info(f"Indexing names in table: {table_name}")
            safe_name = table_name.replace('.', '_')
            idx_path = Path(config.NAME_INDEX_DIR) / safe_name

            if idx_path.exists():
                try:
                    ctx = StorageContext.from_defaults(persist_dir=str(idx_path))
                    idx = load_index_from_storage(ctx, index_id="name_index")
                    self.name_index_dict[table_name] = idx
                    logger.info(f"Loaded existing name index for {table_name}")
                    continue
                except Exception as e:
                    logger.error(f"Error loading name index for {table_name}: {e}")
                    self._create_name_index_for_table(table_name)
            
            else:
                self._create_name_index_for_table(table_name)
            logger.info(f"Created name index for {len(self.name_index_dict)} tables")

    def _create_name_index_for_table(self, table_name: str) -> int:
        """Create name-only index for a single table"""
        try:
            # Extract unique names
            unique_names = db_manager.get_unique_names(
                table_name, 
                name_columns=config.NAME_COLUMNS,
                limit=config.MAX_NAMES_PER_TABLE
            )
            
            if not unique_names:
                logger.warning(f"No unique names found for {table_name}")
                return 0
            
            # Create documents (one per unique name combination)
            documents = []
            for name_dict in unique_names:
                # Store as simple dict string for easy parsing
                doc = Document(
                    text=str(name_dict),
                    metadata={
                        'table': table_name,
                        'type': 'name_only'
                    }
                )
                documents.append(doc)
            
            # Create and persist index
            idx = VectorStoreIndex(documents)
            idx.set_index_id("name_index")
            
            safe_name = table_name.replace('.', '_')
            idx_path = Path(config.NAME_INDEX_DIR) / safe_name
            idx.storage_context.persist(str(idx_path))
            
            self.name_index_dict[table_name] = idx
            
            # Track in index tracker
            current_name_count = db_manager.get_name_count(table_name, config.NAME_COLUMNS)
            self.index_tracker.update_last_indexed(
                f"{table_name}_names",  # ← Separate tracker key
                last_count=current_name_count
            )
            
            logger.info(f"✅ Created name index for {table_name} with {len(documents)} unique names")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error creating name index for {table_name}: {e}")
            return 0
    def _create_full_table_index(self, table_name: str, unique_columns: List[str] = None) -> int:
        try:
            safe_table_name = table_name.replace('.', '_')
            idx_path = Path(config.TABLE_INDEX_DIR) / safe_table_name
            
            limit = config.TEST_ROW_LIMIT if config.TEST_MODE else getattr(config, 'MAX_UNIQUE_ROWS_PER_TABLE', 1000)
            
            if unique_columns:
                df = db_manager.load_unique_table_data(table_name, unique_columns, limit)
            else:
                df = db_manager.load_table_data(table_name, limit)

            if df is None or df.empty:
                logger.warning(f"No data loaded for {table_name}, skipping indexing.")
                return 0

            rows = df.to_dict('records')

            # FIX: Sanitize each row before converting it to a string for the index.
            # This ensures the string is always a valid, parsable dictionary.
            sanitized_rows = [self._sanitize_row_for_indexing(row) for row in rows]
            nodes = [Document(text=str(row)) for row in sanitized_rows]
            
            if nodes:
                idx = VectorStoreIndex(nodes)
                idx.set_index_id("vector_index")
                idx.storage_context.persist(str(idx_path))
                self.vector_index_dict[table_name] = idx
                
                # Update tracker
                unique_info = db_manager.get_table_unique_info(table_name, unique_columns) if unique_columns else {}
                self.index_tracker.update_last_indexed(
                    table_name, 
                    last_count=unique_info.get('unique_count', len(rows))
                )
                
                logger.info(f"✅ Created index for {table_name} with {len(nodes)} documents")
                return len(nodes)
            return 0
        except Exception as e:
            logger.error(f"Error creating full table index for {table_name}: {e}", exc_info=True)
            return 0
    

    def _update_table_index_with_unique(self, table_name: str, unique_columns: List[str] = None) -> int:
        """Update table index with unique filtering support"""
        try:
            logger.debug(f"Updating index for {table_name} with unique columns: {unique_columns}")
            
            # For tables with unique filtering, we need different update logic
            if unique_columns:
                return self._update_unique_table_index(table_name, unique_columns)
            else:
                # Use the existing regular update logic
                return self._update_table_index(table_name)
                
        except Exception as e:
            logger.error(f"Error updating index for {table_name}: {e}")
            return 0

    def _update_unique_table_index(self, table_name: str, unique_columns: List[str]) -> int:
        """Update index for tables with unique filtering"""
        try:
            # For unique tables, we check if new unique combinations have been added
            current_unique_info = db_manager.get_table_unique_info(table_name, unique_columns)
            current_unique_count = current_unique_info.get('unique_count', 0)
            
            last_unique_count = self.index_tracker.get_last_indexed_count(table_name)
            
            if current_unique_count <= last_unique_count:
                logger.debug(f"No new unique combinations for {table_name}")
                return 0
            
            logger.info(f"Found {current_unique_count - last_unique_count} new unique combinations for {table_name}")
            
            # For unique tables, it's easier to recreate the index rather than trying to find incremental unique rows
            # This is because we need to ensure we're not duplicating unique combinations
            logger.info(f"Recreating unique index for {table_name} to include new unique combinations")
            
            # Remove old index
            safe_table_name = table_name.replace('.', '_')
            idx_path = Path(config.TABLE_INDEX_DIR) / safe_table_name
            
            if idx_path.exists():
                import shutil
                shutil.rmtree(idx_path)
            
            # Create new index with all unique combinations
            return self._create_full_table_index(table_name, unique_columns)
            
        except Exception as e:
            logger.error(f"Error updating unique index for {table_name}: {e}")
            return 0
    
    def _generate_table_summaries(self):
        logger.info("Generating table summaries...")
        
        synthetic_data_generator = SyntheticDataGenerator()
        
        table_names = db_manager.get_table_names()
        if not table_names:
            logger.error("No tables found!")
            return
        
        logger.info(f"Processing {len(table_names)} tables: {table_names}")
        
        dfs = []
        table_info_list = []
        
        for table_name in table_names:
            df = synthetic_data_generator.generate_synthetic_table_data(
                table_name, num_rows=config.TEST_ROW_LIMIT if config.TEST_MODE else 1000
            )
            table_info = db_manager.get_table_info(table_name)
            
            if df is not None and not df.empty and table_info:
                dfs.append(df)
                table_info_list.append(table_info)
            else:
                logger.warning(f"Skipping table {table_name} due to empty synthetic data or missing info.")
        
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=TableInfo,
            llm=llm_manager.get_llm(),
            prompt_template_str=prompt_manager.get_table_info_prompt().template,
        )
        table_names_set = set()
        for table_info, df in zip(table_info_list, dfs):
            original = table_info['table_name']
            existing = self._get_existing_table_info(original)
            if existing:
                # FIX: Store as a dictionary with original_table_name as the key
                self.table_infos[original] = {
                    'original_table_name': original,
                    'table_name': existing.table_name,
                    'table_summary': existing.table_summary,
                    'column_descriptions': existing.column_descriptions
                }
                logger.info(f"Loaded existing info for table: {existing.table_name}")
            else:
                table_structure = ", ".join(table_info['columns'])
                sample_data = df.head(16).to_string()
                print(sample_data)
                attempts = 0
                while attempts < 3:
                    try:
                        gen = program(
                            table_name=original,
                            table_structure=table_structure,
                            table_data=sample_data,
                            exclude_table_name_list=str(list(table_names_set)),
                        )
                        normalized_name = re.sub(r'\W+', '_', gen.table_name).lower().strip('_')
                        if normalized_name not in table_names_set:
                            gen.table_name = normalized_name
                            table_names_set.add(normalized_name)
                            break
                        attempts += 1
                        logger.warning(f"Duplicate table_name {gen.table_name}, retrying...")
                    except Exception as e:
                        logger.error(f"Error generating table summary: {e}")
                        gen = TableInfo(
                            table_name=re.sub(r'\W+', '_', original).lower().strip('_'),
                            table_summary=f"Database table with {table_info['row_count']} rows",
                            column_descriptions={}
                        )
                        break
                self._save_table_info(original, gen)
                # FIX: Store as a dictionary with original_table_name as the key
                self.table_infos[original] = {
                    'original_table_name': original,
                    'table_name': gen.table_name,
                    'table_summary': gen.table_summary,
                    'column_descriptions': gen.column_descriptions
                }
                logger.info(f"Generated summary for table: {gen.table_name}")

    def _get_existing_table_info(self, original_table_name: str) -> Optional[TableInfo]:
        safe_table = original_table_name.replace('.', '_')
        results = list(Path(config.TABLE_INFO_DIR).glob(f"{safe_table}_*.json"))
        if not results:
            return None
        if len(results) > 1:
            logger.warning(f"Multiple table info files for {original_table_name}: {results}")
        path = results[0]
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get("original_table_name") != original_table_name:
                logger.warning(f"Mismatched original_table_name in {path}: expected {original_table_name}, got {data.get('original_table_name')}. Ignoring.")
                return None
            data.setdefault("column_descriptions", {})
            return TableInfo.model_validate(data)
        except Exception as e:
            logger.error(f"Error loading table info from {path}: {e}")
            return None

    def _save_table_info(self, original_table_name: str, info: TableInfo):
        safe_table = original_table_name.replace('.', '_')
        out = Path(config.TABLE_INFO_DIR) / f"{safe_table}_{info.table_name}.json"
        data = {"original_table_name": original_table_name, **info.model_dump()}
        try:
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            logger.error(f"Error saving table info to {out}: {e}")

    def _create_vector_indices(self):
        """Create vector indices for tables with unique filtering support"""
        logger.info("Creating vector indices for tables with unique filtering...")
        
        # Get tables from database manager (wth TEST_MODE)
        table_names = db_manager.get_table_names()
        
        for tbl in table_names:
            logger.info(f"Indexing rows in table: {tbl}")
            
            # Check if this table has unique filtering rules
            unique_columns = config.get_unique_columns_for_table(tbl)
            
            if unique_columns:
                logger.info(f"Using unique filtering for {tbl} based on columns: {unique_columns}")
            
            # Create safe filename for schema.table
            safe_name = tbl.replace('.', '_')
            idx_path = Path(config.TABLE_INDEX_DIR) / safe_name
            
            if not idx_path.exists():
                self._create_full_table_index(tbl, unique_columns)
            else:
                try:
                    ctx = StorageContext.from_defaults(persist_dir=str(idx_path))
                    idx = load_index_from_storage(ctx, index_id="vector_index")
                    self.vector_index_dict[tbl] = idx
                    logger.info(f"Loaded existing index for {tbl}")
                except Exception as e:
                    logger.error(f"Error loading index for {tbl}: {e}")
                    self._create_full_table_index(tbl, unique_columns)
        
        logger.info(f"Created vector indices for {len(self.vector_index_dict)} tables")


    def _parse_response_to_sql(self, response: ChatResponse) -> Dict[str, str]:
        """
        Extract BOTH explanation and SQL from the LLM's response.
        
        Returns:
            dict with keys:
                - "explanation": The generator's reasoning
                - "sql": Clean SQL statement
        """
        print(f"\n--- DEBUG: LLM Response BEFORE SQL Parsing ---\n{response.message.content.strip()}\n--- END DEBUG ---\n")
        self._stage_outputs['text2sql_llm_raw'] = response  # ← ADD THIS

        txt = response.message.content.strip()
        
        # ========== EXTRACT EXPLANATION FIRST (before any modifications) ==========
        explanation_match = re.search(
            r'<explanation>(.*?)</explanation>', 
            txt, 
            re.DOTALL | re.IGNORECASE
        )
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        
        # ========== EXTRACT SQL FROM <sql> TAGS (PRIORITY) ==========
        sql_match = re.search(r'<sql>(.*?)</sql>', txt, re.DOTALL | re.IGNORECASE)
        if sql_match:
            txt = sql_match.group(1).strip()
            print(f"\n--- DEBUG: Extracted from <sql> tags ---\n{txt}\n--- END DEBUG ---\n")
        else:
            # ========== FALLBACK: YOUR ORIGINAL CLEANING LOGIC ==========
            # Remove the <think> block
            txt = re.sub(r'<think>.*?</think>', '', txt, flags=re.DOTALL).strip()
            
            # Remove <explanation> block from SQL extraction
            txt = re.sub(r'<explanation>.*?</explanation>', '', txt, flags=re.DOTALL | re.IGNORECASE).strip()

            # Drop leading "assistant:" if present
            if txt.lower().startswith("assistant:"):
                txt = txt[len("assistant:"):].strip()

            # If there's a fenced sql block, grab its contents
            fence = re.search(r"```(?:sql)?\s*([\s\S]*?)```", txt, re.IGNORECASE)
            if fence:
                txt = fence.group(1).strip()
            else:
                # Otherwise just remove stray backticks
                txt = txt.replace("```", "").replace("`", "")

            # Remove any SQLQuery: or SQLResult: sections
            txt = re.sub(r"(?i)sqlquery:.*", "", txt)
            txt = re.sub(r"(?i)sqlresult:.*", "", txt)

            # Truncate at the last semicolon (keep the semicolon)
            if ";" in txt:
                txt = txt[: txt.rfind(";") + 1]

            # Remove any trailing "Answer:" or similar explanation
            txt = re.split(r"(?i)\banswer\s*:", txt)[0].strip()

            # Finally drop any leading dialect label (sql:, postgresql:)
            txt = re.sub(r"^(?:sql|postgresql)[:\s]*", "", txt, flags=re.IGNORECASE).strip()

        # ========== COMMON CLEANING (applies to both paths) ==========
        # "DBM.LOAN_BALANCE" -> DBM.LOAN_BALANCE
        txt = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)"', r'\1', txt)
        
        # Remove quotes from column names
        txt = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r'\1', txt)
        
        # Semicolon cleanup
        txt = txt.strip()
        if not txt.endswith(';'):
            txt += ';'
        
        # Clean up any double semicolons
        txt = re.sub(r';+', ';', txt)

        print(f"\n--- DEBUG: Extracted Explanation ---\n{explanation}\n--- END DEBUG ---\n")
        print(f"\n--- DEBUG: PARSED SQL (After Cleaning) ---\n{txt}\n--- END DEBUG ---\n")
        
        # ========== RETURN DICT (CRITICAL!) ==========
        parsed =  {
            "explanation": explanation,
            "sql": txt
        }
        self._stage_outputs['sql_parser'] = parsed  # ← ADD THIS
        return parsed

    def _review_and_correct_sql(
        self, 
        sql_result: Dict[str, str],  # ← Changed from str to Dict
        context_str: str, 
        masked_query_str: str, 
        analysis: QueryAnalysis, 
        retrieved_examples: List[Dict]
    ) -> str:  # ← Still returns str (just the SQL)
        """
        Review SQL with access to generator's explanation.
        Returns the final SQL (corrected or original).
        """
        if not config.SELF_ERROR_CORRECTION_ENABLED:
            logger.info("❌ Self-correction is DISABLED. Skipping review.")
            result = sql_result["sql"]  # ← Extract SQL from dict
            return result
        if not config.ALWAYS_REVIEW_SQL and analysis.complexity == "SIMPLE":
            logger.info("--- [REVIEWER] Query is SIMPLE. Skipping review.")
            result = sql_result["sql"]  # ← Extract SQL from dict
            return result
        
        # Format examples
        examples_str = "\n\n".join([
            f"Q: {ex['user_question']}\nSQL: {ex['sql_query']}"
            for ex in retrieved_examples
        ]) if retrieved_examples else "No examples available"
        
        max_attempts = getattr(config, "MAX_REVIEW_ATTEMPTS", 1)
        current_sql = sql_result["sql"]  # ← Extract SQL from dict
        current_explanation = sql_result["explanation"]  # ← Extract explanation from dict
        
        for attempt in range(max_attempts):
            logger.info(f"--- [REVIEWER] Review attempt {attempt + 1}/{max_attempts} ---")
            
            # Build review prompt WITH explanation
            review_prompt = prompt_manager.get_sql_review_prompt().format(
                context_str=context_str,
                masked_query_str=masked_query_str,
                generator_explanation=current_explanation,  # ← NEW: Pass explanation
                sql_query=current_sql,
                examples=examples_str
            )
            
            try:
                response = llm_manager.get_aux_llm().complete(review_prompt)
                self._stage_outputs['sql_reviewer_raw'] = response  # ← ADD THIS
                review_text = response.text.strip()
                
                # Parse JSON response
                json_match = re.search(r'\{.*\}', review_text, re.DOTALL)
                if not json_match:
                    logger.warning("Review didn't return valid JSON. Using current SQL.")
                    break
                
                review_json = json.loads(json_match.group(0))
                
                if review_json.get("is_correct"):
                    logger.info(f"✅ Reviewer approved SQL. Reasoning: {review_json.get('reasoning')}")
                    self._stage_outputs['sql_reviewer'] = current_sql
                    return current_sql
                
                # Get corrected SQL
                corrected_sql = review_json.get("corrected_query", "").strip()
                if corrected_sql:
                    logger.warning(f"⚠️ Reviewer suggested correction: {review_json.get('reasoning')}")
                    
                    # Parse the corrected SQL using the SAME parsing logic
                    # Create a mock ChatResponse to reuse your parser
                    mock_response = type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': f"<sql>{corrected_sql}</sql>"
                        })()
                    })()
                    
                    parsed_result = self._parse_response_to_sql(mock_response)
                    current_sql = parsed_result["sql"]
                    # Keep old explanation since we don't have a new one
                    continue
                
                logger.warning("Review failed but no correction provided. Keeping current SQL.")
                break
                
            except Exception as e:
                logger.error(f"Review error: {e}", exc_info=True)
                break
        self._stage_outputs['sql_reviewer'] = current_sql
        return current_sql  # Returns just the SQL string

    
    def _lint_and_fix_sql(self, sql_query: str) -> str:

        """
        Uses sqlfluff to lint and automatically fix common SQL syntax issues.
        This is a fast, final syntax check before execution.
        """

        if not config.SELF_ERROR_CORRECTION_ENABLED:
            logger.info("❌ Self-correction is DISABLED. Skipping SQL linting.")
            return sql_query
        else:
            try:
                # Use the 'oracle' dialect. sqlfluff will parse and fix the query.
                # The rules are conservative by default and safe to apply.
                fixed_sql = sqlfluff.fix(sql_query, dialect="oracle")
                
                if fixed_sql.strip() != sql_query.strip():
                    logger.info("SQLFluff applied automatic syntax corrections.")
                    if config.DEBUG:
                        print(f"--- [SQLFLUFF] Original SQL: {sql_query}")
                        print(f"--- [SQLFLUFF] Fixed SQL: {fixed_sql}")
                self._stage_outputs['sql_linter'] = fixed_sql.strip()
                return fixed_sql.strip()
            except Exception as e:
                # If sqlfluff fails for any reason, just log it and return the original query.
                logger.warning(f"SQLFluff linting failed with error: {e}. Proceeding with unlinted query.")
                self._stage_outputs['sql_linter'] = sql_query
                return sql_query

    def _execute_sql(self, sql_query: str) -> List[dict]:
        """Execute SQL against DB and return list of row dicts with a hard limit."""
        sql_query = sql_query.strip()
        if not sql_query:
            return []

        print(f"\n--- DEBUG: _execute_sql input: {sql_query} ---")

        # Define a hard limit for rows to prevent context overflow
        HARD_ROW_LIMIT = 500

        try:
            # Clean Query
            cleaned_sql = ' '.join(sql_query.split())
            if cleaned_sql.endswith(';'):
                cleaned_sql = cleaned_sql[:-1]

            print(f"--- DEBUG: Cleaned SQL for execution: {cleaned_sql} ---")
            with db_manager.get_connection() as conn:
                result = conn.execute(text(cleaned_sql))
                
                # Use mappings() which returns an iterable of dict-like objects
                # This is the modern and correct way in SQLAlchemy 2.0
                # We fetch one more than the limit to check if the result was truncated.
                rows = result.mappings().fetchmany(HARD_ROW_LIMIT + 1)

                # Check if the results were truncated
                if len(rows) > HARD_ROW_LIMIT:
                    logger.warning(f"Query returned more than the hard limit of {HARD_ROW_LIMIT} rows. Results have been truncated.")
                    # Slice the list to return only the limited number of rows
                    rows = rows[:HARD_ROW_LIMIT]

                print(f"--- DEBUG: _execute_sql returning {len(rows)} rows. ---")
                # The rows are already dict-like from .mappings(), but we ensure they are plain dicts
                result_rows = [dict(row) for row in rows]
                self._stage_outputs['sql_executor'] = result_rows
                return result_rows
            
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            print(f"--- DEBUG: _execute_sql error: {e} ---\n")
            error =  [{"error": str(e)}]
            self._stage_outputs['sql_executor'] = error
            return error
            
    def _process_user_input(self, input_query: str) -> Dict:
        """Process user input for PII masking"""
        print(f"\n--- DEBUG: _process_user_input ---")
        print(f"Input query: {input_query}")
        
        if not hasattr(self, 'user_input_processor') or self.user_input_processor is None:
            self.user_input_processor = UserInputPIIProcessor()
        
        if config.MASK_USER_INPUT:
            print("--- [PII] MASK_USER_INPUT is True. Masking user query.")
            masked_query, user_mapping = self.user_input_processor.mask_user_input(input_query)
        else:
            print("--- [PII] MASK_USER_INPUT is False. Skipping user query masking.")
            masked_query = input_query
            user_mapping = {}

        result = {
            'masked_query': masked_query,
            'user_mapping': user_mapping,
            'original_query': input_query
        }
        self._stage_outputs['user_input_processor'] = result  # ← ADD THIS
        print(f"--- DEBUG: Processed input result: {result}")
        return result



# --- PII Masking Functions ---
    def _is_sensitive_column(self, col_name: str, sample_value=None) -> bool:
        """
        ULTRA SIMPLE: Mask ALL columns automatically.
        No checking needed - just mask everything for maximum privacy.
        """

        NEVER_MASK_COLUMNS = {
        'acnt_name',
        'customer_name',
        'project_name',
        'cur_code'
     }
    
        col_lower = col_name.lower()
    
        if col_lower in NEVER_MASK_COLUMNS:
            return False 
    
        return True

    def _mask_results(self, rows: List[dict]) -> dict:
        """
        Mask sensitive data in query results and return masked data with mapping.
        """
        # Debug: Check what we're receiving
        print(f"\n--- DEBUG: _mask_results input type: {type(rows)} ---")
        print(f"--- DEBUG: _mask_results input value: {rows} ---\n")
    
        # Handle different input types
        if not rows:
            return {"context_str": "SQLResult: []", "mapping": {}}
    
        # If rows is not a list, try to handle it
        if not isinstance(rows, list):
            print(f"--- DEBUG: Expected list, got {type(rows)}, converting to list ---")
            if isinstance(rows, dict):
                rows = [rows]
            else:
                # Fallback: return empty result
                return {"context_str": f"SQLResult: {str(rows)}", "mapping": {}}

        if not config.ENABLE_PII_MASKING:
            print("--- [PII] ENABLE_PII_MASKING is False. Skipping database result masking. ---")
            return {"context_str": f"SQLResult: {rows}", "mapping": {}}
        
        mapping = {}
        placeholder_counter = {}
        masked_rows = []

        for row in rows:
            # Ensure row is a dictionary
            if not isinstance(row, dict):
                print(f"--- DEBUG: Expected dict row, got {type(row)}: {row} ---")
                # Try to convert or skip
                if hasattr(row, '_mapping'):
                    row = dict(row._mapping)
                elif hasattr(row, '_asdict'):
                    row = row._asdict()
                else:
                    continue
                
            masked_row = {}
            for col, val in row.items():
                # Simple check - just mask columns with 'name' or 'нэр'
                sensitive = self._is_sensitive_column(col, sample_value=val)

                if sensitive and val is not None:
                    col_key = col.upper().replace(" ", "_")
                    idx = placeholder_counter.get(col_key, 0) + 1
                    placeholder = f"[MASKED_{col_key}_{idx}]"
                    placeholder_counter[col_key] = idx
                    mapping[placeholder] = val
                    masked_row[col] = placeholder
                    logger.debug(f"Masked column '{col}' value -> {placeholder}")
                else:
                    masked_row[col] = val
            masked_rows.append(masked_row)

        context_str = f"SQLResult: {masked_rows}"
        print(f"\n--- DEBUG: _mask_results output context_str: {context_str} ---")
        print(f"--- DEBUG: _mask_results output mapping: {mapping} ---\n")

        output = {"context_str": context_str, "mapping": mapping}
        self._stage_outputs['db_result_masker'] = output
        return output

    def _extract_mapping(self, masked_data: dict) -> dict:
        """Extract mapping from the masked data dictionary."""
        return masked_data.get("mapping", {})

    def _mask_retrieved_row_content(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Masks all values in a retrieved row dictionary to send to the LLM for context.
        This is a security function to prevent real data leakage into the LLM prompt.
        """
        masked_row = {}
        for key, value in row_dict.items():
            # Keep column names (keys) unmasked for structural context
            key_lower = key.lower()
            if value is None:
                masked_row[key] = None
            # Mask strings, preserving empty strings
            elif isinstance(value, str):
                 masked_row[key] = f"[MASKED_{key.upper()}]" if value else ""
            # Mask numbers that are not common codes or years
            elif isinstance(value, (int, float)):
                if 'year' in key_lower or 'code' in key_lower or (1000 < value < 2100):
                     masked_row[key] = value # Assume it's a year or code, keep it
                else:
                     masked_row[key] = f"[MASKED_{key.upper()}]"
            # Mask other types like dates
            else:
                masked_row[key] = f"[MASKED_{key.upper()}]"
        return masked_row

    def _setup_query_pipeline(self):
        """Sets up the full query pipeline with the secure double-blind architecture."""
        logger.info("Setting up secure query pipeline...")

        # --- Define all components ---
        input_component = InputComponent()
        #EXTRACTORS:
        enriched_query_extractor = FnComponent(fn=lambda x: x.get("enriched_query", ""))
        original_user_query_extractor = FnComponent(fn=lambda x: x.get("original_user_query", ""))
        user_input_processor = FnComponent(fn=self._process_user_input)
        
        original_query_extractor = FnComponent(fn=lambda x: x.get("original_query", ""))
        masked_query_extractor = FnComponent(fn=lambda x: x.get("masked_query", ""))
        user_mapping_extractor = FnComponent(fn=lambda x: x.get("user_mapping", {}))

        example_retriever_component = FnComponent(fn=example_retriever.retrieve_examples)

        query_analyzer = FnComponent(fn=self._analyze_and_decompose_query)

        def build_minimal_schema(analysis: QueryAnalysis) -> str:
            """Build a schema string using only columns selected by analyzer."""
            if not analysis or not analysis.required_tables:
                logger.warning("Analyzer returned no required tables. Cannot build minimal schema.")
                return "No schema available."
            schema_parts = []
            for table_name in analysis.required_tables:
                table_info = self.table_infos.get(table_name)
                if not table_info:
                    logger.warning(f"No table info found for {table_name}. Skipping.")
                    continue

                schema_parts.append(f"TABLE: {table_name}")
                schema_parts.append(f"SUMMARY: {table_info.get('table_summary', 'N/A')}")

                cols_to_include = analysis.required_columns.get(table_name, [])
                col_descs = table_info.get('column_descriptions', {})

                schema_parts.append("COLUMNS:")
                if not cols_to_include:
                    schema_parts.append("  - No specific columns identified.")
                else:
                    for col in cols_to_include:
                        desc = col_descs.get(col, "No description available.")
                        schema_parts.append(f"  - {col}: {desc}")
                schema_parts.append("")  # Blank line between tables
            minimal_schema = "\n".join(schema_parts)
            logger.debug(f"Built minimal schema: Length: {len(minimal_schema)}\n{minimal_schema}")
            self._stage_outputs['minimal_schema'] = minimal_schema
            return minimal_schema
        minimal_schema_builder = FnComponent(fn=build_minimal_schema)
        # --- Stage 1 Retriever Setup ---
        table_schemas = [
            SQLTableSchema(table_name=t_info['original_table_name'], context_str=t_info['table_summary'])
            for t_info in self.table_infos.values() # Use .values() as it's a dict now
        ]
        obj_index = ObjectIndex.from_objects(
            table_schemas,
            SQLTableNodeMapping(self.sql_database),
            VectorStoreIndex
        )
        # Store the retriever as a class attribute so other methods can access it
        self.obj_retriever = obj_index.as_retriever(similarity_top_k=config.MAX_TABLE_RETRIEVAL)
        
        def select_schema(minimal_schema: str, full_context: Dict) -> str:
            """If minimal schema is empty/invalid, use the full schema."""
            if minimal_schema and "No schema available" not in minimal_schema:
                logger.info("Using MINIMAL schema for SQL generation.")
                return minimal_schema
            else:
                logger.warning("Minimal schema is invalid or empty. Falling back to FULL schema for SQL generation.")
                return full_context.get("context_str", "No schema available.")
        schema_selector = FnComponent(fn=select_schema)
        # --- Context Builder (Now handles both stages of retrieval internally) ---
        table_context_builder = FnComponent(fn=self._get_table_context_and_rows_str)
        def format_text2sql_prompt(query_str: str, schema: str, analysis: QueryAnalysis, retrieved_examples: List[Dict], extracted_names: List[Dict]) -> str:
            """
            Formats the text-to-SQL prompt, providing the analyzer's plan and explanation as a suggestion.
            """ 
            if not retrieved_examples:
                dynamic_examples_str = "No relevant examples found."
            else:
                example_texts = [f" - Query: \"{ex['user_question']}\"\n  - Generate: {ex['sql_query']}" for ex in retrieved_examples]
                dynamic_examples_str = "\n".join(example_texts)

            plan_str = "No plan was generated."
            explanation_str = "No explanation was generated."
            chat_history_note = ""
            if analysis:
                if analysis.sub_questions: 
                    sub_q_str = "\n".join([f"- {sq}" for sq in analysis.sub_questions])
                    plan_str = f"The suggested steps are:\n{sub_q_str}"
                if analysis.explanation:
                    explanation_str = analysis.explanation
            
            if analysis.needs_chat_history and analysis.chat_history_reasoning:
                chat_history_note = f"""
                **📜 CHAT HISTORY CONTEXT:**
                This query references previous conversation.
                Reasoning: {analysis.chat_history_reasoning}

                **IMPORTANT:** You must re-compute the entity reference using a subquery based on the previous SQL pattern.
                """
             # Format entity names
            if not extracted_names:
                entity_names_str = "No specific entity names were retrieved."
            else:
                name_texts = [", ".join([f"{k}={v}" for k, v in name_dict.items()]) for name_dict in extracted_names]
                entity_names_str = "\n".join([f"  - {name_str}" for name_str in name_texts])

            prompt_template = prompt_manager.get_text2sql_prompt()
            
            final_prompt = prompt_template.format(
                query_str=query_str,
                schema=schema,
                analyzer_explanation=explanation_str, # <-- Pass the explanation
                plan=plan_str,
                entity_names=entity_names_str,
                dynamic_examples=dynamic_examples_str
            )
            
            if chat_history_note:
                final_prompt = final_prompt.replace(
                    "** USER QUESTION **:",
                    f"{chat_history_note}\n** USER QUESTION **:"
                )

            # ADD THIS BLOCK TO PRINT THE FINAL PROMPT
            print("\n" + "="*50)
            print("--- FINAL TEXT-TO-SQL PROMPT (to be sent to MAIN LLM) ---")
            print(final_prompt)
            print("="*50 + "\n")
            
            return final_prompt
        # --- SQL Generation ---
        text2sql_prompt_formatter = FnComponent(fn=format_text2sql_prompt)
        text2sql_llm = llm_manager.get_llm()
        sql_parser = FnComponent(fn=self._parse_response_to_sql)

        sql_reviewer = FnComponent(fn=self._review_and_correct_sql)
        sql_linter = FnComponent(fn=self._lint_and_fix_sql) 
        
        # --- SQL Execution (Unmask -> Execute) ---
        def unmask_and_execute(sql: str, mapping: dict) -> list:
            if config.MASK_USER_INPUT:
                unmasked_sql = self.user_input_processor.unmask_sql_query(sql, mapping)
                print(f"--- [SQL EXEC] Executing UNMASKED SQL: {unmasked_sql}")
            else:
                unmasked_sql = sql
                print(f"--- [SQL EXEC] Executing plain SQL (input masking was off): {unmasked_sql}")
            return self._execute_sql(unmasked_sql)
        sql_executor = FnComponent(fn=unmask_and_execute)
        
        # --- Result Masking & Final Response ---
        db_result_masker = FnComponent(fn=self._mask_results)
        masked_context_extractor = FnComponent(fn=lambda x: x.get("context_str", ""))
        db_mapping_extractor = FnComponent(fn=lambda x: x.get("mapping", {}))
        response_prompt = prompt_manager.get_response_synthesis_prompt()
        response_llm = llm_manager.get_llm()


        # --- Final Unmasking ---
        def final_unmasker(response: ChatResponse, user_map: dict, db_map: dict) -> str:
            text = response.message.content
            logger.info(f"--- [UNMASK] Final response BEFORE unmasking: {text}")
            
            # ========== UNESCAPE MARKDOWN FIRST ==========
            # Replace escaped underscores with normal underscores
            text = text.replace(r'\_', '_')
            # ==============================================
            
            # Combine mappings, giving database results priority
            full_mapping = {**user_map, **db_map}

            for placeholder, original_value in full_mapping.items():
                formatted_value = original_value
                # Format numbers with commas
                if isinstance(original_value, (decimal.Decimal, float, int)):
                    formatted_value = f"{original_value:,}"
                # Format dates as YYYY-MM-DD
                elif isinstance(original_value, datetime):
                    formatted_value = original_value.strftime('%Y-%m-%d')
                # Format lists into a clean string
                elif isinstance(original_value, list):
                    formatted_value = ", ".join(map(str, original_value))
                
                # Now replace with normal underscores
                text = text.replace(placeholder, str(formatted_value))

            logger.info(f"--- [UNMASK] Final response AFTER unmasking: {text}")
            self._stage_outputs['final_unmasker'] = text
            return text
        final_unmask_component = FnComponent(fn=final_unmasker)
        entity_names_extractor = FnComponent(fn=lambda x: x.get("extracted_names", []))
        # --- Build the pipeline (Simplified Wiring) ---
        qp = QP(verbose=config.DEBUG)
        qp.add_modules({
            "input": input_component, #ENRICHED QUERY INPUT
            "enriched_query_extractor": enriched_query_extractor,
            "original_user_query_extractor": original_user_query_extractor, #THIS IS FOR EXAMPLE_RETREIVER ONLY
            "user_input_processor": user_input_processor,
            "original_query_extractor": original_query_extractor,
            "masked_query_extractor": masked_query_extractor,
            "user_mapping_extractor": user_mapping_extractor,
            "example_retriever": example_retriever_component,
            "query_analyzer": query_analyzer,
            "minimal_schema_builder": minimal_schema_builder,
            "schema_selector": schema_selector,
            "entity_names_extractor": entity_names_extractor,
            "table_context_builder": table_context_builder, 
            "text2sql_prompt_formatter": text2sql_prompt_formatter,
            "text2sql_llm": text2sql_llm,
            "sql_parser": sql_parser,
            "sql_reviewer": sql_reviewer,
            "sql_linter": sql_linter,
            "sql_executor": sql_executor,
            "db_result_masker": db_result_masker,
            "masked_context_extractor": masked_context_extractor,
            "db_mapping_extractor": db_mapping_extractor,
            "response_prompt": response_prompt,
            "response_llm": response_llm,
            "final_unmasker": final_unmask_component,
        })

        # 1. Process user input
        qp.add_link("input", "enriched_query_extractor")
        qp.add_link("input", "original_user_query_extractor")

        qp.add_link("enriched_query_extractor", "user_input_processor")
        qp.add_link("user_input_processor", "original_query_extractor")
        qp.add_link("user_input_processor", "masked_query_extractor")
        qp.add_link("user_input_processor", "user_mapping_extractor")

        # 2. Retrieve examples using the original query
        qp.add_link("original_user_query_extractor", "example_retriever")
        
        # 3. Build schema context (now independent)
        qp.add_link("original_query_extractor", "table_context_builder", dest_key="query")

        # 4. Analyze the query, now with examples AND schema context
        qp.add_link("masked_query_extractor", "query_analyzer", dest_key="query")
        qp.add_link("table_context_builder", "query_analyzer", dest_key="context_str")
        qp.add_link("example_retriever", "query_analyzer", dest_key="retrieved_examples")

        qp.add_link("query_analyzer", "minimal_schema_builder", dest_key="analysis")
        qp.add_link("minimal_schema_builder", "schema_selector", dest_key="minimal_schema")
        qp.add_link("table_context_builder", "schema_selector", dest_key="full_context")
        qp.add_link("table_context_builder", "entity_names_extractor")

        # 5. Format the final prompt, merging everything
        qp.add_link("masked_query_extractor", "text2sql_prompt_formatter", dest_key="query_str")
        qp.add_link("schema_selector", "text2sql_prompt_formatter", dest_key="schema")
        qp.add_link("query_analyzer", "text2sql_prompt_formatter", dest_key="analysis")
        qp.add_link("entity_names_extractor", "text2sql_prompt_formatter", dest_key="extracted_names")
        qp.add_link("example_retriever", "text2sql_prompt_formatter", dest_key="retrieved_examples")
        
        # 6. Generate SQL
        qp.add_chain(["text2sql_prompt_formatter", "text2sql_llm", "sql_parser"])

        # 7. Review and Correct SQL
        qp.add_link("sql_parser", "sql_reviewer", dest_key="sql_result")
        qp.add_link("table_context_builder", "sql_reviewer", dest_key="context_str")
        qp.add_link("masked_query_extractor", "sql_reviewer", dest_key="masked_query_str")
        qp.add_link("query_analyzer", "sql_reviewer", dest_key="analysis")
        qp.add_link("example_retriever", "sql_reviewer", dest_key="retrieved_examples") 
        qp.add_link("sql_reviewer", "sql_linter")

        # 8. Execute SQL
        qp.add_link("sql_linter", "sql_executor", dest_key="sql")
        qp.add_link("user_mapping_extractor", "sql_executor", dest_key="mapping")

        # 9. Mask results
        qp.add_link("sql_executor", "db_result_masker")
        qp.add_link("db_result_masker", "masked_context_extractor")
        qp.add_link("db_result_masker", "db_mapping_extractor")

        # 10. Synthesize response
        qp.add_link("masked_query_extractor", "response_prompt", dest_key="query_str")
        qp.add_link("sql_linter", "response_prompt", dest_key="sql_query")
        qp.add_link("masked_context_extractor", "response_prompt", dest_key="context_str")
        qp.add_link("response_prompt", "response_llm")

        # 11. Final unmasking
        qp.add_link("response_llm", "final_unmasker", dest_key="response")
        qp.add_link("user_mapping_extractor", "final_unmasker", dest_key="user_map")
        qp.add_link("db_mapping_extractor", "final_unmasker", dest_key="db_map")
        

        self.query_pipeline = qp

        # ========== ADD THIS BLOCK ==========
        logger.info("="*80)
        logger.info("🤖 LLM MODEL CONFIGURATION")
        logger.info("="*80)
        logger.info(f"Main LLM (Text-to-SQL):     {llm_manager.get_llm().model}")
        logger.info(f"Auxiliary LLM (Analyzer):   {llm_manager.get_aux_llm().model}")
        logger.info(f"Reviewer LLM:               {llm_manager.get_aux_llm().model}")
        logger.info(f"Response LLM:               {llm_manager.get_llm().model}")
        logger.info(f"Embedding Model:            {llm_manager.get_embed_model().model_name if hasattr(llm_manager.get_embed_model(), 'model_name') else 'N/A'}")
        logger.info("="*80)
        # ====================================
        
        logger.info("✅ Secure query pipeline with self-correction constructed.")
        


    def select_relevant_tables(self, query: str, candidate_tables: list) -> list:
        """
        Second stage of retrieval - uses reranker to select relevant tables.
        MUCH faster and more reliable than LLM.
        """
        if not candidate_tables:
            return []
        
        if len(candidate_tables) == 1:
            return [candidate_tables[0]["name"]]
        
        # Use the fast reranker
        logger.info(f"--- [CONTEXT BUILDER] Using RERANKER for table selection.")
        return self.context_builder.rerank_tables(
            query=query,
            candidate_tables=candidate_tables,
            top_k=3  # or 3
        )
        
    def _select_relevant_columns(self, query: str, table_name: str) -> List[str]:
        """
        Third stage of retrieval: Use semantic similarity to select relevant columns.
        NO LLM needed - uses embeddings only.
        """
        table_info = self.table_infos.get(table_name)
        if not table_info or not table_info.get('column_descriptions'):
            logger.warning(f"No column descriptions found for {table_name}")
            return []

        all_columns = table_info['column_descriptions']
        
        
        # Option 1: Pure semantic
        selected = self.context_builder.select_columns_reranker(
            query=query,
            table_name=table_name,
            all_columns=all_columns,
        )
        
        # Option 2: Hybrid (faster, still accurate)
        # selected = self.context_builder.select_columns_hybrid(
        #     query=query,
        #     table_name=table_name,
        #     all_columns=all_columns,
        #     max_columns=15
        # )
        
        logger.info(f"--- [CONTEXT BUILDER] Selected {len(selected)} columns for '{table_name}'")
        return selected


    def _extract_entity_name(self, query:str):
        entity_query = query
        keywords_to_remove = [
            "төлбөр", "хуваарь", "данс", "дугаар", "үлдэгдэл", 
        "мэдээлэл", "харуул", "гарга", "зээл", "ийн", "ны", "хуваарыг", "төлбөрийн"
        ]
        for keyword in keywords_to_remove:
            entity_query = entity_query.replace(keyword, "")
        return entity_query.strip()
    
    def _get_table_context_and_rows_str(self, query: str) -> str:
        """
        Builds a detailed context string for a given query.
        MODIFIED: When SEND_SAMPLE_ROWS is False, only sends name columns from retrieved nodes.
        """
        logger.info(f"--- [CONTEXT BUILDER] START ---")
        logger.info(f"--- [CONTEXT BUILDER] Building context for query: '{query}'")
        
        empty_context = {"context_str": "", "extracted_names": []}
        context_parts: List[str] = []
        all_extracted_names: List[Dict[str, Any]] = []
        
        context_parts.append("=" * 80)
        context_parts.append("DATABASE SCHEMA CONTEXT")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        # STAGE 1: Get candidate tables using embeddings (Always runs)
        candidate_schemas = []
        if self.obj_retriever:
            retrieved_schemas = self.obj_retriever.retrieve(query)
            logger.info(f"--- [CONTEXT BUILDER] Stage 1: Retrieved {len(retrieved_schemas)} candidate tables. Table name is: {[s.table_name for s in retrieved_schemas]}")
            candidate_schemas = retrieved_schemas
        else:
            logger.warning("--- [CONTEXT BUILDER] obj_retriever not found. This should not happen.")
            self._stage_outputs["table_context_builder"] = empty_context
            return empty_context

        if not candidate_schemas:
            logger.warning("--- [CONTEXT BUILDER] No candidate tables found after initial retrieval.")
            self._stage_outputs["table_context_builder"] = empty_context
            return empty_context

        # ALWAYS RERANK TABLES
        logger.info("--- [CONTEXT BUILDER] Reranking candidate tables for best match.")
        
        candidate_tables_for_reranker = []
        for schema_obj in candidate_schemas:
            table_name = schema_obj.table_name
            table_info = self.table_infos.get(table_name)
            if table_info:
                candidate_tables_for_reranker.append({
                    "name": table_name,
                    "summary": table_info.get("table_summary", "No summary available")
                })
        
        selected_tables = self.select_relevant_tables(query, candidate_tables_for_reranker)
        logger.info(f"--- [CONTEXT BUILDER] Reranker selected {len(selected_tables)} final tables: {selected_tables}")

        # ========== FIX: SORT TABLES ALPHABETICALLY ==========
        selected_tables.sort()
        logger.info(f"--- [CONTEXT BUILDER] ✅ Tables sorted alphabetically for caching: {selected_tables}")
        # ====================================================

        # Now build context with the reranked tables
        for table_name in selected_tables:
            table_info = self.table_infos.get(table_name)
            
            if not table_info:
                logger.warning(f"Skipping table {table_name} in context builder: missing info.")
                continue
            
            # ========== CHANGED: Add visual separators ==========
            context_parts.append("-" * 80)
            context_parts.append(f"📊 TABLE: {table_name}")
            context_parts.append("-" * 80)
            context_parts.append("")
            
            all_columns = table_info.get('column_descriptions', {})
            relevant_columns = []

            # CONDITIONALLY SELECT COLUMNS based on the TWO_STAGE_RETRIEVAL flag
            if config.TWO_STAGE_RETRIEVAL:
                logger.info("--- [CONTEXT BUILDER] TWO_STAGE_RETRIEVAL is True. Selecting relevant columns.")
                relevant_columns = self._select_relevant_columns(query, table_name)
            else:
                logger.info("--- [CONTEXT BUILDER] TWO_STAGE_RETRIEVAL is False. Using all columns for selected tables.")
            
            if not relevant_columns:
                relevant_columns = list(all_columns.keys())

            if all_columns:
                logger.info(f"--- [CONTEXT BUILDER] Adding {len(relevant_columns)} relevant column descriptions for {table_name}.")
                
                # ========== CHANGED: Use structured formatting ==========
                context_parts.append(f"**Description:**")
                context_parts.append(f"  {table_info.get('table_summary', '')}")
                context_parts.append("")
                context_parts.append(f"**Columns:** {', '.join(relevant_columns)}")
                context_parts.append("")
                context_parts.append("**Column Details:**")
                
                for col_name in relevant_columns:
                    if col_name in all_columns:
                        # ========== CHANGED: Add bullet points and indentation ==========
                        context_parts.append(f"  • {col_name}:")
                        context_parts.append(f"    {all_columns[col_name]}")
                context_parts.append("")
            
            # ========== MODIFIED SAMPLE ROW LOGIC ==========
            if config.SEND_SAMPLE_ROWS: 
                # ========== FULL SAMPLE ROWS MODE  ==========
                table_index = self.vector_index_dict.get(table_name)
                if not table_index:
                    logger.warning(f"Skipping sample rows for {table_name}: missing index.")
                else:
                    try:
                        retriever = table_index.as_retriever(similarity_top_k=config.MAX_ROW_RETRIEVAL)
                        nodes = retriever.retrieve(query)
                        
                        if nodes:
                            logger.info(f"--- [CONTEXT BUILDER] Table '{table_name}': Retrieved {len(nodes)} example rows.")
                            context_parts.append("**🔍 Sample Rows:**")
                            example_rows = []
                            for i, node in enumerate(nodes, 1):
                                row_content_str = node.get_content()
                                try:
                                    row_dict = ast.literal_eval(row_content_str)
                                    final_row_dict = self._mask_retrieved_row_content(row_dict) if config.MASK_SAMPLE_ROWS else self._sanitize_row_for_indexing(row_dict)
                                    example_rows.append(json.dumps(final_row_dict))
                                except (ValueError, SyntaxError) as e:
                                    logger.warning(f"Could not parse row content for {table_name}: {e}")

                            if example_rows:
                                for i, row_str in enumerate(example_rows, 1):
                                    # ========== CHANGED: Add indentation ==========
                                    context_parts.append(f"  {i}. {row_str}")
                            context_parts.append("")
                        else:
                            logger.info(f"--- [CONTEXT BUILDER] No relevant example rows found for {table_name}.")

                    except Exception as e:
                        logger.error(f"Error retrieving sample rows for {table_name}: {e}")
            
            else:
                        # ========== NAMES ONLY MODE ==========
                logger.info(f"--- [CONTEXT BUILDER] SEND_SAMPLE_ROWS is False. Using NAME_INDEX for '{table_name}'.")
                
                name_index = self.name_index_dict.get(table_name)
                
                if name_index:
                    try:
                        entity_query = self._extract_entity_name(query)
                        retriever = name_index.as_retriever(
                            similarity_top_k=config.MAX_ROW_RETRIEVAL
                        )
                        nodes = retriever.retrieve(entity_query)

                        if nodes:
                            extracted_names = []
                            for node in nodes:
                                try:
                                    name_dict = ast.literal_eval(node.get_content())
                                    extracted_names.append({
                                        **name_dict,
                                        '_table': table_name
                                    })
                                except (ValueError, SyntaxError) as e:
                                    logger.warning(f"Could not parse name content for {table_name}: {e}")
                            
                            if extracted_names:
                                all_extracted_names.extend(extracted_names)
                                logger.info(f"✅ Retrieved {len(extracted_names)} UNIQUE names from '{table_name}'.")
                        else:
                            logger.info(f"No relevant names found in '{table_name}' for query: {query}")
                    
                    except Exception as e:
                        logger.error(f"Error retrieving names for {table_name}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.warning(f"❌ No name index found for {table_name}")
                    logger.info(f"Available name indices: {list(self.name_index_dict.keys())}")

            context_parts.append("")  # Space between tables

        # ========== MOVED: Add ALL names AFTER the loop completes ==========
        if all_extracted_names:
            context_parts.append("=" * 80)
            context_parts.append("📋 AVAILABLE ENTITY NAMES (All Tables)")
            context_parts.append("=" * 80)
            context_parts.append("")
            
            # Group by table for clarity
            from collections import defaultdict
            names_by_table = defaultdict(list)
            for name_dict in all_extracted_names:
                table = name_dict.pop('_table', 'unknown')
                names_by_table[table].append(name_dict)
            
            for table_name, names in names_by_table.items():
                context_parts.append(f"**From {table_name}:**")
                for i, name_dict in enumerate(names, 1):
                    name_str = ", ".join([f"{k}={v}" for k, v in name_dict.items()])
                    context_parts.append(f"  {i}. {name_str}")
                context_parts.append("")

        # ========== MOVED: Build final context AFTER the loop ==========
        context_str = "\n".join(context_parts)
        print(f"Length of context string: {len(context_str)}")
        logger.info(f"--- [CONTEXT BUILDER] Completed context building.")
        context = {
            "context_str": context_str,
            "extracted_names": all_extracted_names
        }
        self._stage_outputs['table_context_builder'] = context
        return context
     # --- NEW: THE QUERY ANALYZER COMPONENT ---
    def _analyze_and_decompose_query(self, query: str, context_str: str, retrieved_examples: List[Dict]) -> QueryAnalysis:
        """
        MODIFIED: Uses an LLM program to analyze query complexity, now with examples for context.
        """
        if not config.QUERY_ANALYZER_ENABLED:
            logger.info("--- [ANALYZER] Disabled by config. Defaulting to SIMPLE.")
            analysis_result = QueryAnalysis(
                complexity="SIMPLE", 
                needs_chat_history=False,  # 
                chat_history_reasoning="Analyzer disabled",  # 
                needs_deduplication=False,  # 
                explanation="Query analyzer is disabled.",
                required_tables=[],
                required_columns={},
                sub_questions=[]
            )
            self._stage_outputs['query_analyzer'] = analysis_result
            return analysis_result


        logger.info("--- [ANALYZER] Starting query analysis (SINGLE LLM CALL)...")
        
        # Format examples for the prompt
        if not retrieved_examples:
            examples_str = "No relevant examples found."
        else:
            example_texts = [f"  - Query: \"{ex['user_question']}\"\n  - Generate: {ex['sql_query']}" for ex in retrieved_examples]
            examples_str = "\n".join(example_texts)

        analyzer_program = LLMTextCompletionProgram.from_defaults(
            output_cls=QueryAnalysis,
            prompt=prompt_manager.get_query_analyzer_prompt(examples=examples_str), # Pass examples to prompt
            llm=llm_manager.get_aux_llm(),
            verbose=config.DEBUG
        )
        try:
            analysis_result = analyzer_program(query_str=query, context_str=context_str)
            # ========== LOG TABLE SELECTION REASONING ==========
            logger.info(f"--- [ANALYZER] Analysis complete. Complexity: {analysis_result.complexity}")
            logger.info(f"--- [ANALYZER] Selected tables: {analysis_result.required_tables}")
            logger.info(f"--- [ANALYZER] Reasoning: {analysis_result.explanation[:500]}...") 
            # ===================================================
            self._stage_outputs['query_analyzer'] = analysis_result
            return analysis_result
        except Exception as e:
            logger.error(f"--- [ANALYZER] Failed to analyze query: {e}. Defaulting to SIMPLE.")
            return QueryAnalysis(
                complexity="SIMPLE", 
                needs_chat_history=False,
                chat_history_reasoning="", 
                needs_deduplication=False,
                explanation="Analysis failed",
                required_tables=[],
                required_columns={},
                sub_questions=[]
            )
        
    def get_index_status(self) -> Dict[str, Dict]:   
        """Get enhanced status information including unique filtering info"""
        status = {}
        
        for table_name in self.sql_database.get_usable_table_names():
            try:
                unique_columns = config.get_unique_columns_for_table(table_name)
                
                if unique_columns:
                    # Get unique count info
                    unique_info = db_manager.get_table_unique_info(table_name, unique_columns)
                    current_unique_count = unique_info.get('unique_count', 0)
                    total_rows = unique_info.get('row_count', 0)
                    
                    last_indexed_unique = self.index_tracker.get_last_indexed_count(table_name)
                    
                    status[table_name] = {
                        'total_db_rows': total_rows,
                        'unique_combinations': current_unique_count,
                        'unique_columns': unique_columns,
                        'uniqueness_ratio': unique_info.get('uniqueness_ratio', 0),
                        'last_indexed_unique': last_indexed_unique,
                        'needs_update': current_unique_count > last_indexed_unique,
                        'pending_unique_combinations': max(0, current_unique_count - last_indexed_unique),
                        'filtering_type': 'unique'
                    }
                else:
                    # Regular table status
                    with db_manager.get_connection() as conn:
                        current_count = conn.execute(text(f'SELECT COUNT(*) FROM {table_name}')).scalar()
                    
                    last_count = self.index_tracker.get_last_indexed_count(table_name)
                    
                    status[table_name] = {
                        'current_db_count': current_count,
                        'last_indexed_count': last_count,
                        'needs_update': current_count > last_count,
                        'pending_rows': max(0, current_count - last_count),
                        'filtering_type': 'regular'
                    }
                
                # Common fields
                safe_table_name = table_name.replace('.', '_')
                idx_path = Path(config.TABLE_INDEX_DIR) / safe_table_name
                status[table_name]['index_exists'] = idx_path.exists()
                
            except Exception as e:
                status[table_name] = {'error': str(e), 'filtering_type': 'error'}
        
        return status

    def _is_greeting(self, query: str) -> bool:
        def normalize(text: str) -> str:
            text = unicodedata.normalize("NFKC", text).lower()
            return text.translate(str.maketrans("", "", string.punctuation)).strip()

        normalized = normalize(query)
        greeting_tokens = {
            "сайн байна уу",
            "сайн уу",
            "сайнуу",
            "сайн байнуу",
            "сайна уу",
            "сайн байнуу",
            "hello",
            "hi",
            "hey",
            "good morning",
            "Сайна байна уу?",
            "сайна уу",
            "good evening"
        }
        return normalized in greeting_tokens

    def run_query(self, user_query: str, memory: Optional[BaseMemory] = None, session_id: str = "default") -> str:
        start_time = time.time()
        self._stage_outputs = {}  # ← EACH REQUEST GETS ITS OWN DICT
        
        with self.chat_history_lock:
            if session_id not in self.chat_history:
                self.chat_history[session_id] = []
            history = self.chat_history[session_id].copy()  # ← Copy to avoid mutation issues
        
        # ========== BUILD STRUCTURED HISTORY STRING ==========
        history_str = ""
        if history:
            recent_turns = history[-2:]  # Last 2 turns only
            history_parts = []
            
            for i, turn in enumerate(recent_turns, 1):
                history_parts.append(f"Turn {i}:")
                history_parts.append(f"  Question: {turn['question']}")
                history_parts.append(f"  SQL Pattern: {turn.get('sql', 'N/A')[:500]}...")  # Truncated
                history_parts.append("")
            
            history_str = "\n".join(history_parts)

        # ========== ENRICH QUERY WITH HISTORY ==========
        if history_str:
            enriched_query = f"[Previous Conversation Metadata]\n{history_str}\n[Current Question]\n{user_query}"
        else:
            enriched_query = user_query
        # Default values
        original_query = user_query
        masked_query = user_query
        user_mapping = {}
        selected_tables = []
        table_reranker_scores = {}  
        extracted_names = []
        query_complexity = "SIMPLE"
        sub_questions = []
        generated_sql = "N/A"
        generator_explanation = "No explanation provided"
        was_reviewed = False  
        reviewed_sql = None 
        review_reason = None 
        final_sql = "N/A"
        sql_results = None
        execution_error = None
        
        try:

            if self._is_greeting(user_query):
                execution_time = time.time() - start_time
                greeting_reply = "Сайн байна уу? Танд ямар мэдээлэл хэрэгтэй байна вэ?"
                query_logger.log_complete_query(
                    original_query=user_query,
                    masked_query=user_query,
                    user_mapping={},
                    selected_tables=[],
                    table_reranker_scores={},
                    extracted_names=[],
                    query_complexity="SIMPLE",
                    sub_questions=[],
                    generated_sql="N/A",
                    generator_explanation="Greeting handled without SQL.",
                    was_reviewed=False,
                    reviewed_sql=None,
                    review_reason=None,
                    final_sql="N/A",
                    sql_results=[],
                    execution_error=None,
                    final_answer=greeting_reply,
                    execution_time=execution_time,
                    llm_backend=config.LLM_BACKEND,
                    embedding_backend=config.EMBEDDING_BACKEND,
                    session_id=session_id
                )
                return greeting_reply
            
            result = self.query_pipeline.run(input= {
                "enriched_query": enriched_query, 
                "original_user_query": user_query
            } )  
            
            self._log_cache_performance()
            # ========== READ FROM CACHE ==========
            stage = self._stage_outputs
            
            
            # User input
            processed = stage.get("user_input_processor", {})
            original_query = processed.get("original_query", user_query)
            masked_query = processed.get("masked_query", user_query)
            user_mapping = processed.get("user_mapping", {})
            
            # Context builder
            context_result = stage.get("table_context_builder", {})
            if isinstance(context_result, dict):
                context_str = context_result.get("context_str", "")
                extracted_names = context_result.get("extracted_names", [])
                selected_tables = list(set(re.findall(r'Table: ([\w\.]+)', context_str)))
            
            # Query analyzer
            analysis = stage.get("query_analyzer")
            if analysis:
                query_complexity = getattr(analysis, "complexity", "SIMPLE")
                sub_questions = getattr(analysis, "sub_questions", [])
            
            # SQL generation
            sql_parser_out = stage.get("sql_parser", {})
            generated_sql = sql_parser_out.get("sql", "N/A")
            generator_explanation = sql_parser_out.get("explanation", "No explanation provided")
            
            # SQL execution
            final_sql = stage.get("sql_linter", generated_sql)
            sql_results = stage.get("sql_executor")
            actual_entities = []
            if sql_results and isinstance(sql_results, list) and sql_results:
                for row in sql_results[:5]:
                    if isinstance(row, dict):
                        for col_name in config.NAME_COLUMNS:
                            if col_name in row and row[col_name]:
                                actual_entities.append(
                                    {
                                        'type': col_name,
                                        'value': row[col_name]
                                    }
                                )
            seen = set()
            unique_entities = []
            for entity in actual_entities:
                if entity['value'] not in seen:
                    seen.add(entity['value'])
                    unique_entities.append(entity)
            # Check for errors
            if isinstance(sql_results, list) and sql_results and "error" in sql_results[0]:
                execution_error = sql_results[0]["error"]

            masked_context = stage.get("masked_context_extractor", "")
            # Final answer
            final_answer = stage.get("final_unmasker", result if isinstance(result, str) else str(result))
            
            execution_time = time.time() - start_time
            
            with self.chat_history_lock:
                self.chat_history[session_id].append({
                    "question": user_query,
                    "sql": final_sql,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Keep only last 3 exchanges (reduced from 6)
                if len(self.chat_history[session_id]) > 3:
                    self.chat_history[session_id] = self.chat_history[session_id][-3:]

            # ========== LOG EVERYTHING ==========
            query_logger.log_complete_query(
                original_query=original_query,
                masked_query=masked_query,
                user_mapping=user_mapping,
                selected_tables=selected_tables,
                table_reranker_scores={},
                extracted_names=extracted_names,
                query_complexity=query_complexity,
                sub_questions=sub_questions,
                generated_sql=generated_sql,
                generator_explanation=generator_explanation,
                was_reviewed=False,
                reviewed_sql=None,
                review_reason=None,
                final_sql=final_sql,
                sql_results=sql_results,
                execution_error=execution_error,
                final_answer=final_answer,
                execution_time=execution_time,
                llm_backend=config.LLM_BACKEND,
                embedding_backend=config.EMBEDDING_BACKEND,
                session_id=session_id
            )
            
            return final_answer
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution_error = str(e)
            logger.error(f"❌ Pipeline error: {execution_error}", exc_info=True)
            
            # Log the failure
            query_logger.log_complete_query(
                original_query=original_query,
                masked_query=masked_query,
                user_mapping=user_mapping,
                selected_tables=selected_tables,
                table_reranker_scores=table_reranker_scores,
                extracted_names=extracted_names,
                query_complexity=query_complexity,
                sub_questions=sub_questions,
                generated_sql=generated_sql or "N/A",
                generator_explanation=generator_explanation,
                was_reviewed=was_reviewed,
                reviewed_sql=reviewed_sql,
                review_reason=review_reason,
                final_sql=final_sql or "N/A",
                sql_results=None,
                execution_error=execution_error,
                final_answer="Таны хүсэлтийн дагуу мэдээлэл авах боломжгүй байна. Асуултаа өөрөөр асууна уу.",
                execution_time=execution_time,
                llm_backend=config.LLM_BACKEND,
                embedding_backend=config.EMBEDDING_BACKEND,
                session_id=session_id
            )
            
            return "Таны хүсэлтийн дагуу мэдээлэл авах боломжгүй байна. Асуултаа өөрөөр асууна уу."

    def _log_cache_performance(self):
        logger.info("\n" + "="*80)
        logger.info("💾 PROMPT CACHE PERFORMANCE")
        logger.info("="*80)

        total_input_tokens = 0
        total_cached_tokens = 0

        llm_stages = [
            'text2sql_llm_raw',
            'sql_reviewer_raw',
            'response_llm',
        ]

        for stage_name in llm_stages:
            stage_output = self._stage_outputs.get(stage_name)
            if not stage_output:
                continue
                
            usage = None

            # ========== FIXED: Handle different response types ==========
            try:
                # LlamaIndex ChatResponse objects
                if hasattr(stage_output, "raw"):
                    raw = stage_output.raw
                    # raw is a ChatCompletion object, access usage directly
                    if hasattr(raw, "usage"):
                        usage = raw.usage  # ← Direct attribute access, not .get()
                        # Convert to dict for consistent handling
                        usage = {
                            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(usage, "completion_tokens", 0),
                            "prompt_tokens_details": {
                                "cached_tokens": getattr(
                                    getattr(usage, "prompt_tokens_details", None), 
                                    "cached_tokens", 
                                    0
                                )
                            }
                        }
                
                # Direct dict (shouldn't happen with OpenAI but just in case)
                elif isinstance(stage_output, dict):
                    usage = stage_output.get("usage")
                
                # Alternative path for some response types
                elif hasattr(stage_output, "message"):
                    usage = getattr(stage_output.message, "additional_kwargs", {}).get("usage")
            
            except Exception as e:
                logger.debug(f"Could not extract usage from {stage_name}: {e}")
                continue

            if not usage:
                continue

            prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else getattr(usage, "prompt_tokens", 0)
            cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) if isinstance(usage, dict) else 0
            completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else getattr(usage, "completion_tokens", 0)

            if prompt_tokens == 0:
                continue

            cache_rate = (cached_tokens / prompt_tokens) * 100
            total_input_tokens += prompt_tokens
            total_cached_tokens += cached_tokens

            # Clean stage name for display
            display_name = stage_name.replace('_raw', '')
            
            logger.info(f"[{display_name}]")
            logger.info(f"  Input:      {prompt_tokens:,} tokens")
            logger.info(f"  Cached:     {cached_tokens:,} tokens ({cache_rate:.1f}%)")
            logger.info(f"  Output:     {completion_tokens:,} tokens\n")

        if total_input_tokens > 0:
            overall_rate = (total_cached_tokens / total_input_tokens) * 100
            cost_savings = (total_cached_tokens * 0.075) / 1_000_000

            logger.info("-"*80)
            logger.info("📊 OVERALL CACHE PERFORMANCE:")
            logger.info(f"  Total Input Tokens:   {total_input_tokens:,}")
            logger.info(f"  Total Cached Tokens:  {total_cached_tokens:,} ({overall_rate:.1f}%)")
            logger.info(f"  💸 Estimated Savings: ${cost_savings:.6f}")
            logger.info("="*80 + "\n")
        else:
            logger.info("⚠️  No cache data available")
            logger.info("="*80 + "\n")
            
    # Additional helper method to debug pipeline structure
    def debug_pipeline_structure(self):
        """Debug method to print pipeline structure"""
        if hasattr(self.query_pipeline, '_modules'):
            print("Pipeline modules:")
            for name, module in self.query_pipeline._modules.items():
                print(f"  - {name}: {type(module)}")
        
        if hasattr(self.query_pipeline, '_links'):
            print("\nPipeline connections:")
            for link in self.query_pipeline._links:
                print(f"  {link}")