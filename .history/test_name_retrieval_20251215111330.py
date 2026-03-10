"""
Test script to check name index and retrieval
Run this to diagnose why certain names are not being retrieved.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from app.config import config
from app.db import db_manager
from app.llm import llm_manager
from llama_index.core import VectorStoreIndex, load_index_from_storage, Document
from llama_index.core.storage import StorageContext

def test_database_name_exists():
    """Check if the name exists in the database at all"""
    print("\n" + "="*80)
    print("TEST 1: Check if 'ЭРЧИМ ХҮЧНИЙ ЯАМ' exists in database")
    print("="*80)
    
    from sqlalchemy import text
    
    queries = [
        """
        SELECT DISTINCT customer_name 
        FROM dbm.loan_balance_detail 
        WHERE UPPER(customer_name) LIKE '%ЭРЧИМ%ЯАМ%'
        """,
        """
        SELECT DISTINCT customer_name 
        FROM dbm.loan_balance_detail 
        WHERE UPPER(customer_name) LIKE '%ЭРЧИМ ХҮЧНИЙ ЯАМ%'
        """,
        """
        SELECT COUNT(DISTINCT customer_name) as total_unique_customers
        FROM dbm.loan_balance_detail 
        WHERE customer_name IS NOT NULL
        """
    ]
    
    with db_manager.get_connection() as conn:
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}:")
            print(query.strip())
            result = conn.execute(text(query))
            rows = result.fetchall()
            print(f"Results: {len(rows)} rows")
            for row in rows[:20]:  # Show first 20
                print(f"  → {row}")


def test_indexed_names_count():
    """Check how many names are currently indexed"""
    print("\n" + "="*80)
    print("TEST 2: Check indexed names count in name_index_dir")
    print("="*80)
    
    name_index_dir = Path(config.NAME_INDEX_DIR)
    
    for table_dir in name_index_dir.iterdir():
        if table_dir.is_dir():
            docstore_path = table_dir / "docstore.json"
            if docstore_path.exists():
                import json
                with open(docstore_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc_count = len(data.get('docstore', {}).get('docs', {}))
                    print(f"📁 {table_dir.name}: {doc_count} documents indexed")
                    
                    # Show some sample documents
                    docs = data.get('docstore', {}).get('docs', {})
                    print("   Sample names:")
                    for i, (doc_id, doc_data) in enumerate(list(docs.items())[:5]):
                        text = doc_data.get('__data__', {}).get('text', 'N/A')
                        print(f"     {i+1}. {text[:100]}...")


def test_check_if_name_is_indexed():
    """Check if specific name is in the index"""
    print("\n" + "="*80)
    print("TEST 3: Search for 'ЭРЧИМ ХҮЧНИЙ ЯАМ' in index files")
    print("="*80)
    
    import json
    name_index_dir = Path(config.NAME_INDEX_DIR)
    search_term = "ЭРЧИМ ХҮЧНИЙ ЯАМ"
    
    for table_dir in name_index_dir.iterdir():
        if table_dir.is_dir():
            docstore_path = table_dir / "docstore.json"
            if docstore_path.exists():
                with open(docstore_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    docs = data.get('docstore', {}).get('docs', {})
                    
                    found = False
                    for doc_id, doc_data in docs.items():
                        text = doc_data.get('__data__', {}).get('text', '')
                        if search_term.upper() in text.upper():
                            print(f"✅ FOUND in {table_dir.name}:")
                            print(f"   {text}")
                            found = True
                    
                    if not found:
                        print(f"❌ NOT FOUND in {table_dir.name}")
                        
                        # Check for partial matches
                        partial_matches = []
                        for doc_id, doc_data in docs.items():
                            text = doc_data.get('__data__', {}).get('text', '')
                            if "ЭРЧИМ" in text.upper():
                                partial_matches.append(text)
                        
                        if partial_matches:
                            print(f"   Partial matches with 'ЭРЧИМ':")
                            for pm in partial_matches[:5]:
                                print(f"     → {pm}")


def test_embedding_retrieval():
    """Test actual embedding retrieval"""
    print("\n" + "="*80)
    print("TEST 4: Test embedding retrieval for 'Эрчим хүчний яам'")
    print("="*80)
    
    name_index_dir = Path(config.NAME_INDEX_DIR)
    table_name = "dbm.loan_balance_detail"
    safe_name = table_name.replace('.', '_')
    idx_path = name_index_dir / safe_name
    
    if not idx_path.exists():
        print(f"❌ Index path does not exist: {idx_path}")
        return
    
    try:
        # Load index
        ctx = StorageContext.from_defaults(persist_dir=str(idx_path))
        idx = load_index_from_storage(ctx, index_id="name_index")
        
        # Get embed model
        embed_model = llm_manager.get_embed_model()
        
        # Create retriever with more results
        retriever = idx.as_retriever(
            similarity_top_k=20,
            embed_model=embed_model
        )
        
        # Test queries
        test_queries = [
            "Эрчим хүчний яам",
            "ЭРЧИМ ХҮЧНИЙ ЯАМ",
            "эрчим хүч яам",
            "ЭРЧИМ",
            "яам"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            nodes = retriever.retrieve(query)
            print(f"   Retrieved {len(nodes)} nodes:")
            for i, node in enumerate(nodes[:5], 1):
                print(f"   {i}. (score: {node.score:.4f}) {node.text[:100]}...")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_get_unique_names_query():
    """Test the SQL query used to get unique names"""
    print("\n" + "="*80)
    print("TEST 5: Check unique names query (what's being indexed)")
    print("="*80)
    
    table_name = "dbm.loan_balance_detail"
    name_columns = config.NAME_COLUMNS
    limit = config.MAX_NAMES_PER_TABLE
    
    print(f"Config MAX_NAMES_PER_TABLE: {limit}")
    print(f"Config NAME_COLUMNS: {name_columns}")
    
    from sqlalchemy import inspect, text
    
    # Get valid columns
    if '.' in table_name:
        schema_name, table_only = table_name.split('.', 1)
    else:
        schema_name, table_only = 'DBM', table_name
    
    inspector = inspect(db_manager.engine)
    existing_columns = [col['name'] for col in inspector.get_columns(table_only, schema=schema_name)]
    valid_name_columns = [col for col in name_columns if col in existing_columns]
    
    print(f"Valid name columns for {table_name}: {valid_name_columns}")
    
    # Show the actual query
    columns_str = ", ".join(valid_name_columns)
    where_clause = " OR ".join([f"{col} IS NOT NULL" for col in valid_name_columns])
    
    query = f"""
    SELECT DISTINCT {columns_str}
    FROM {table_name}
    WHERE {where_clause}
    ORDER BY {valid_name_columns[0]}
    FETCH FIRST {limit} ROWS ONLY
    """
    print(f"\nActual query used for indexing:")
    print(query)
    
    # Check where ЭРЧИМ ХҮЧНИЙ ЯАМ falls in the ordering
    print("\n" + "-"*40)
    print("Checking alphabetical position of 'ЭРЧИМ ХҮЧНИЙ ЯАМ'...")
    
    position_query = f"""
    SELECT COUNT(*) as position
    FROM (
        SELECT DISTINCT customer_name
        FROM {table_name}
        WHERE customer_name IS NOT NULL
    )
    WHERE customer_name < 'ЭРЧИМ ХҮЧНИЙ ЯАМ'
    """
    
    with db_manager.get_connection() as conn:
        result = conn.execute(text(position_query))
        row = result.fetchone()
        print(f"'ЭРЧИМ ХҮЧНИЙ ЯАМ' is at position: {row[0]} (alphabetically by customer_name)")
        
        if row[0] > limit:
            print(f"⚠️  WARNING: Position {row[0]} > limit {limit} - NAME IS NOT INDEXED!")
        else:
            print(f"✅ Name should be within index limit")


def test_total_unique_names():
    """Count total unique names in database"""
    print("\n" + "="*80)
    print("TEST 6: Total unique names in database")
    print("="*80)
    
    from sqlalchemy import text
    
    queries = [
        ("Unique customer_names", "SELECT COUNT(DISTINCT customer_name) FROM dbm.loan_balance_detail WHERE customer_name IS NOT NULL"),
        ("Unique acnt_names", "SELECT COUNT(DISTINCT acnt_name) FROM dbm.loan_balance_detail WHERE acnt_name IS NOT NULL"),
        ("Unique combinations", """
            SELECT COUNT(*) FROM (
                SELECT DISTINCT customer_name, acnt_name, cur_code
                FROM dbm.loan_balance_detail
                WHERE customer_name IS NOT NULL OR acnt_name IS NOT NULL
            )
        """),
    ]
    
    with db_manager.get_connection() as conn:
        for name, query in queries:
            result = conn.execute(text(query))
            count = result.fetchone()[0]
            print(f"{name}: {count}")
            
            if count > config.MAX_NAMES_PER_TABLE:
                print(f"   ⚠️  WARNING: {count} > {config.MAX_NAMES_PER_TABLE} (MAX_NAMES_PER_TABLE)")
                print(f"   → Some names are NOT being indexed!")


def suggest_fix():
    """Suggest how to fix the issue"""
    print("\n" + "="*80)
    print("SUGGESTED FIXES")
    print("="*80)
    
    print("""
1. INCREASE MAX_NAMES_PER_TABLE in config.py:
   Current: MAX_NAMES_PER_TABLE = 500
   Suggested: MAX_NAMES_PER_TABLE = 5000 (or higher)

2. REBUILD the name index after changing config:
   - Delete the name_index_dir folder
   - Restart the application
   
3. ALTERNATIVE: Use customer_name only (not combinations):
   - Modify get_unique_names() to index customer_names separately
   - This reduces the number of documents needed

4. OPTIONAL: Index by popularity (loan count) instead of alphabetically:
   - Change ORDER BY to: ORDER BY loan_count DESC
   - This ensures most common customers are indexed first
""")


if __name__ == "__main__":
    print("="*80)
    print("NAME RETRIEVAL DIAGNOSTIC TEST")
    print("="*80)
    
    # Run all tests
    test_database_name_exists()
    test_total_unique_names()
    test_get_unique_names_query()
    test_indexed_names_count()
    test_check_if_name_is_indexed()
    test_embedding_retrieval()
    suggest_fix()
    
    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
