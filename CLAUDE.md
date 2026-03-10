# Chatbot with Masking - Text-to-SQL for Development Bank of Mongolia

## Project Overview

A text-to-SQL chatbot for the Development Bank of Mongolia (DBM). Users ask questions in Mongolian about banking data, and the system generates Oracle SQL, executes it, and returns natural language answers with PII masking.

## Tech Stack

- **Web Framework:** Flask + Waitress (WSGI)
- **Database:** Oracle (via oracledb + SQLAlchemy)
- **LLM:** OpenAI GPT-4.1 (primary), Ollama qwen3:8b-custom (local), GPT-4.1-mini (auxiliary/analyzer)
- **Embeddings:** Ollama mxbai-embed-large (table indexing), nomic-embed-text-v1.5 (context builder)
- **Reranker:** BAAI/bge-reranker-v2-m3 (cross-encoder)
- **Language:** Python 3.13

## Quick Start

```bash
# Activate venv
cd C:\Eroolt\Chatbot\chatbot_with_masking_updated
.\venv_new\Scripts\activate

# Run the app
python run_web.py
```

App runs on `http://0.0.0.0:5000`

## Project Structure

```
app/
  pipeline.py        # Main pipeline: query → SQL → execute → response (core logic)
  prompts.py         # All prompt templates: modular domain blocks, analyzer, triage
  config.py          # Configuration (reads from .env)
  db.py              # Oracle DB connection, queries, name extraction
  llm.py             # LLM provider management (OpenAI/Ollama/Gemini)
  web_app.py         # Flask routes and API endpoints
  context_builder.py # Reranker + column selection (FastContextBuilder)
  sandbox.py         # SyntheticDataGenerator for new table summaries
  user_input_processor.py  # PII masking for user input
  example_retriever.py     # Retrieves similar SQL examples
  query_logger.py    # Query logging to files
  sql_examples.json  # 200+ verified Q&A pairs
  templates/         # Flask HTML templates
  static/            # CSS/JS assets

PostgreSQL_TableInfo/    # Table summary + column description JSONs (one per table)
table_index_dir/         # VectorStoreIndex per table (sample rows)
name_index_dir/          # VectorStoreIndex per table (unique entity names)
logs/queries/            # Session query logs
```

## Pipeline Flow

1. **Triage** → Classify query (DATA_QUESTION / GENERAL / OUT_OF_SCOPE)
2. **Input Masking** → Mask customer names in user query
3. **Example Retrieval** → Semantic search in sql_examples.json
4. **Table Context** → Embedding retrieval → Reranker (top 5 tables)
5. **Query Analyzer** → LLM selects final tables + columns + plan
6. **Minimal Schema** → Build pruned schema for selected tables
7. **SQL Generation** → Modular prompt composition → LLM generates SQL
8. **Lint + Execute** → SQLFluff check → Oracle execution
9. **Result Masking** → Mask sensitive data in results
10. **Response Synthesis** → LLM generates Mongolian answer
11. **Unmask** → Replace placeholders with real values

## Key Architecture Concepts

- **Modular Prompts:** Each table has its own domain rules block in `prompts.py` (e.g., `LOAN_BALANCE_BLOCK`). Only relevant blocks are included based on analyzer output.
- **NES Module Architecture:** Tables are grouped by NES banking modules (LOAN, FX, COLL, TREASURY, etc.) defined in `pipeline.py:NES_MODULE_TABLES`.
- **Two-phase table selection:** Reranker (fast, embedding-based) → Analyzer (LLM, picks final tables from candidates).
- **Name Index:** Unique entity names indexed per table for exact-match retrieval. Config: `SEND_SAMPLE_ROWS=False` enables name-only mode.
- **Daily Snapshots:** Most tables have one row per entity per day. Always use `txn_date = (SELECT MAX(txn_date) FROM table)` for latest state.

## Database Tables (12 tables, all in DBM schema)

| Module | Table | Purpose |
|--------|-------|---------|
| LOAN | loan_balance_detail | Loan balances, customer info, risk |
| LOAN | bcom_nrs_detail | Payment schedules |
| LOAN | loan_txn | Transaction history |
| FX | fx_deal_detail | FX trades (SPOT/FORWARD/SWAP) |
| FX | fx_swap_balance_detail | SWAP daily valuations |
| COLL | coll_balance_detail | Collateral balances |
| TREASURY | placement_balance_detail | Money placed out |
| TREASURY | borrowing_balance_detail | Money borrowed in |
| SECURITIES | security_balance_detail | Bonds/debt instruments |
| ACCOUNTS | bac_balance_detail | Internal accounts |
| ACCOUNTS | casa_balance | Customer accounts |
| GL | gl_balance_detail | General ledger |

## Configuration

All config is in `.env` file, read by `app/config.py`. Key variables:

- `LLM_BACKEND` — "ollama" or "openai"
- `OPENAI_API_KEY` — Required for OpenAI backend
- `SEND_SAMPLE_ROWS` — True=full rows in context, False=name-only mode
- `MAX_TABLE_RETRIEVAL` — Number of tables reranker returns (default: 5)
- `TEST_MODE` — True=use only configured test tables
- `ENABLE_PII_MASKING` — True=mask sensitive data

## Common Tasks

**Re-index name indices** (after config changes):
```bash
rm -rf name_index_dir
python run_web.py  # Rebuilds on startup
```

**Add a new table:**
1. Add table name to `TEST_TABLES` in `.env`
2. Delete its entry from `PostgreSQL_TableInfo/` if stale
3. Add domain rules block in `prompts.py` and register in `DOMAIN_BLOCKS` dict
4. Add to `NES_MODULE_TABLES` in `pipeline.py`
5. Add SQL examples to `sql_examples.json`
6. Restart app — summary + indices auto-generate

**Add SQL examples:**
Edit `app/sql_examples.json` — format: `{"user_question": "...", "sql_query": "...", "notes": "..."}`

## Testing

```bash
python test_name_retrieval.py
```

For manual testing, use the web UI at `http://localhost:5000` or the API endpoint.
