import os
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()

class Config:
    """Configuration class"""
    
    # ─── Database Configuration ────────────────────────────────────────────────
    DB_HOST: str = os.getenv("DB_HOST", "192.168.10.220")
    DB_PORT: str = os.getenv("DB_PORT", "1522")
    DB_NAME: str = os.getenv("DB_NAME", "ORCL")
    DB_USER: str = os.getenv("DB_USER", "dbm")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD","testdbm2025")

    @property 
    def DATABASE_URL(self) -> str:
        return (
            f"oracle+oracledb://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/?service_name={self.DB_NAME}"
        )
    
    
    # ─── LLM BACKEND SELECTION ────────────────────────────────────────────────    
    # "ollama" or "openai"
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama").lower() 
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "ollama").lower()

    AUX_LLM_BACKEND: str = os.getenv("AUX_LLM_BACKEND", "openai").lower()
    OLLAMA_AUX_LLM_MODEL: str = os.getenv("OLLAMA_AUX_LLM_MODEL", "qwen3:8b-custom")
    # ─── Ollama Configuration ───────────────────────────────────────────────
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "qwen3:8b-custom")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")
    OLLAMA_REQUEST_TIMEOUT: float = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600.0"))
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # ─── OpenAI Configuration ────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_COMPLETION_MODEL: str = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4.1")
    #OPENAI_COMPLETION_MODEL: str = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4.1")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    OPENAI_REQUEST_TIMEOUT: float = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "300.0"))
    OPENAI_AUX_MODEL: str = os.getenv("OPENAI_AUX_MODEL", "gpt-4.1-mini")
    # ------------ Gemini Configuration ------------------------
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL")
    GEMINI_REQUEST_TIMEOUT: float = float(os.getenv("GEMINI_REQUEST_TIMEOUT", "300.0"))
    # ─── Application Configuration ───────────────────────────────────────────
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO") 
    
    # ─── Storage Configuration ───────────────────────────────────────────────
    TABLE_INFO_DIR: str = os.getenv("TABLE_INFO_DIR", "PostgreSQL_TableInfo")
    TABLE_INDEX_DIR: str = os.getenv("TABLE_INDEX_DIR", "table_index_dir")
    
    # ─── API (FastAPI) Configuration ────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # ─── Query Pipeline Limits ──────────────────────────────────────────────
    MAX_TABLE_RETRIEVAL: int = int(os.getenv("MAX_TABLE_RETRIEVAL", "3"))
    MAX_ROW_RETRIEVAL: int = int(os.getenv("MAX_ROW_RETRIEVAL", "10"))
    MAX_ROWS_PER_TABLE: int = int(os.getenv("MAX_ROWS_PER_TABLE", "500"))
    TWO_STAGE_RETRIEVAL: bool = os.getenv("TWO_STAGE_RETRIEVAL", "False").lower() in ("true", "1", "yes")
    
    # ─── PII Masking Configuration (SIMPLIFIED) ────────────────────────────
    # Enable/disable PII masking - just masks columns with 'name' or 'нэр'
    ENABLE_PII_MASKING: bool = os.getenv("ENABLE_PII_MASKING", "True").lower() in ("true", "1", "yes")

    # ------------------ Testing Configuration ------------------------
    # Set to True to limit tables for testing with real DB
    TEST_MODE: bool = os.getenv("TEST_MODE", "True").lower() in ("true", "1", "yes")
    
    # Tables to use in test mode (comma separated)
    TEST_TABLES: List[str] = os.getenv("TEST_TABLES", "DBM.LOAN_BALANCE_DETAIL,DBM.BCOM_NRS_DETAIL,DBM.LOAN_TXN,DBM.SECURITY_BALANCE_DETAIL,DBM.PLACEMENT_BALANCE_DETAIL,DBM.BORROWING_BALANCE_DETAIL").split(",")
    
    # Row limit for test mode
    TEST_ROW_LIMIT: int = int(os.getenv("TEST_ROW_LIMIT", "15"))
    
    ENABLE_UNIQUE_FILTERING: bool = os.getenv("ENABLE_UNIQUE_FILTERING", "True").lower() in ("true", "1", "yes")
    
    # Table-specific unique filtering rules
    # Format: "table_name:column1,column2;other_table:column3"
    UNIQUE_FILTER_RULES: str = os.getenv(
        "UNIQUE_FILTER_RULES",
        "DBM.LOAN_BALANCE_DETAIL:acnt_code; " \
        "DBM.SECURITY_BALANCE_DETAIL:security_code; " \
        "DBM.PLACEMENT_BALANCE_DETAIL:acnt_code; " \
        "DBM.BORROWING_BALANCE_DETAIL:acnt_code; " \
        "DBM.BAC_BALANCE_DETAIL:acnt_code; " \
        "DBM.CASA_BALANCE:acnt_code; " \
        "DBM.FX_DEAL_DETAIL:deal_code; " \
        "DBM.FX_SWAP_BALANCE_DETAIL:deal_code; " \
        "DBM.COLL_BALANCE_DETAIL:acnt_code; " \
    )

    # Default unique columns to try if no specific rule is defined
    DEFAULT_UNIQUE_COLUMNS: List[str] = os.getenv(
        "DEFAULT_UNIQUE_COLUMNS", 
        "acnt_code,account_code,customer_id,id,code"
    ).split(",")
    
    ENABLE_NAME_INDEX: bool = os.getenv("ENABLE_NAME_INDEX", "True").lower() in ("true", "1", "yes")
    NAME_COLUMNS: List[str] = ["acnt_name", "customer_name", "name", "project_name", "cur_code"]
    NAME_INDEX_DIR: str = os.getenv("NAME_INDEX_DIR", "name_index_dir")
    # Increased from 500 to 5000 to ensure all unique customer names are indexed
    MAX_NAMES_PER_TABLE: int = int(os.getenv("MAX_NAMES_PER_TABLE", "5000"))
    # Maximum number of unique rows to index per table (prevents excessive memory usage)
    MAX_UNIQUE_ROWS_PER_TABLE: int = int(os.getenv("MAX_UNIQUE_ROWS_PER_TABLE", "1000"))

    # In class Config:
    # Tables to index (comma-separated, e.g., "DBM.LOAN_BALANCE,DBM.TRANSACTIONS"). Empty = all tables.
    INDEX_TABLES: List[str] = [t.strip() for t in os.getenv("INDEX_TABLES", "").split(",") if t.strip()]


    SEND_SAMPLE_ROWS: bool = os.getenv("SEND_SAMPLE_ROWS", "True").lower() in ("true", "1", "yes")
    MASK_SAMPLE_ROWS: bool = os.getenv("MASK_SAMPLE_ROWS", "True").lower() in ("true", "1", "yes")
    MASK_USER_INPUT: bool = os.getenv("MASK_USER_INPUT", "True").lower() in ("true", "1", "yes")
    
    SELF_ERROR_CORRECTION_ENABLED: bool = os.getenv("SELF_ERROR_CORRECTION_ENABLED", "True").lower() in ("true", "1", "yes")
    QUERY_ANALYZER_ENABLED: bool = os.getenv("QUERY_ANALYZER_ENABLED", "True").lower() in ("true", "1", "yes")
    COMPLEX_QUERY_SELF_CORRECTION: bool = os.getenv("COMPLEX_QUERY_SELF_CORRECTION", "True").lower() in ("true", "1", "yes")
    ALWAYS_REVIEW_SQL: bool = os.getenv("ALWAYS_REVIEW_SQL", "True").lower() in ("true", "1", "yes")
    
    # ─── NEW ARCHITECTURE: Triage & Retry Configuration ─────────────────────
    TRIAGE_ENABLED: bool = os.getenv("TRIAGE_ENABLED", "True").lower() in ("true", "1", "yes")
    MAX_SQL_RETRY_ATTEMPTS: int = int(os.getenv("MAX_SQL_RETRY_ATTEMPTS", "2"))
    POST_EXECUTION_VALIDATION: bool = os.getenv("POST_EXECUTION_VALIDATION", "True").lower() in ("true", "1", "yes")
    HISTORY_SUMMARIZATION_ENABLED: bool = os.getenv("HISTORY_SUMMARIZATION_ENABLED", "True").lower() in ("true", "1", "yes")
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "3"))
    # In __init__:
    # Already cleaning lists—good.
    def __init__(self):
        # ensure these directories exist
        Path(self.TABLE_INFO_DIR).mkdir(exist_ok=True)
        Path(self.TABLE_INDEX_DIR).mkdir(exist_ok=True)
        Path(self.NAME_INDEX_DIR).mkdir(exist_ok=True)
        # Parse unique filtering rules
        self.parsed_unique_rules = self._parse_unique_filter_rules()
        # Clean up default unique columns
        self.DEFAULT_UNIQUE_COLUMNS = [col.strip() for col in self.DEFAULT_UNIQUE_COLUMNS if col.strip()]

    def _parse_unique_filter_rules(self) -> Dict[str, List[str]]:
        """Parse the UNIQUE_FILTER_RULES string into a dictionary"""
        rules = {}
        if not self.UNIQUE_FILTER_RULES:
            return rules
        try:
            # Split by semicolon for different tables
            table_rules = self.UNIQUE_FILTER_RULES.split(';')
            for rule in table_rules:
                rule = rule.strip()
                if ':' in rule:
                    table_name, columns = rule.split(':', 1)
                    table_name = table_name.strip()
                    columns = [col.strip() for col in columns.split(',') if col.strip()]
                    if table_name and columns:
                        rules[table_name] = columns
        except Exception as e:
            logger.error(f"Error parsing UNIQUE_FILTER_RULES: {e}")
        
        return rules
    
    def get_unique_columns_for_table(self, table_name: str) -> Optional[List[str]]:
        """Get unique filtering columns for a specific table"""
        # Check specific rules first
        if table_name in self.parsed_unique_rules:
            return self.parsed_unique_rules[table_name]
        
        # If no specific rule and unique filtering is enabled, try default columns
        if self.ENABLE_UNIQUE_FILTERING:
            return self.DEFAULT_UNIQUE_COLUMNS
        
        return None
    
    
    def validate(self) -> bool:
        """Validate critical configuration values"""
        if not self.DB_PASSWORD:
            raise ValueError("DB_PASSWORD must be set in environment variables")
        if self.LLM_BACKEND not in ("ollama", "openai"):
            raise ValueError("LLM_BACKEND must be either 'ollama' or 'openai'")
        if self.LLM_BACKEND == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when LLM_BACKEND=openai")
        if self.EMBEDDING_BACKEND not in ("ollama", "openai"):
            raise ValueError("EMBEDDING_BACKEND must be either 'ollama' or 'openai'")
        if self.EMBEDDING_BACKEND == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when EMBEDDING_BACKEND=openai")
        return True

config = Config()
