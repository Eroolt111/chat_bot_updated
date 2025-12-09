"""
OPTIMIZED PROMPTS - Streamlined for Performance
================================================
Changes from original:
1. Removed duplicate rules (identity rule, table selection, deduplication) 
2. Consolidated Text2SQL prompt from ~400 lines to ~150 lines
3. Simplified Analyzer prompt from ~150 lines to ~80 lines
4. Removed unused SCHEMA_ANALYSIS_PROMPT (duplicated analyzer)
5. Made SQL Review a lightweight post-execution check
"""

from venv import logger
from .config import config
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT

# =============================================================================
# TABLE INFO GENERATION (unchanged - needed for setup)
# =============================================================================
TABLE_INFO_PROMPT = """
Give me a summary of the table with the following JSON format:

{ 
  "table_name": "...",
  "table_summary": "...",
  "column_descriptions": {
    "column1": "description...",
    "column2": "description...",
    ... 
  }
}  

Instructions:
- The table name must be unique and describe its primary content.
- For each column, describe the type of data and its business meaning.
- For monetary columns, note the relationship with `CUR_CODE` (currency column).
- Do NOT make the table name one of the following: {exclude_table_name_list}

Table Name: {table_name}
Table Structure: {table_structure}
Sample Data:
{table_data}

Summary: """

# =============================================================================
# RESPONSE SYNTHESIS (API version - conversational)
# =============================================================================
API_RESPONSE_SYNTHESIS_PROMPT = """
You are a professional Mongolian banking assistant for **Development Bank of Mongolia (DBM)**.
Present the SQL results clearly in Mongolian.

**RULES:**
1. Never mention placeholders, masking, or technical details
2. Use actual database values (customer_name, acnt_name) exactly as returned
3. Format numbers with thousand separators (1,234,567)
4. Map currencies: 'MNT'→'төгрөг', 'USD'→'ам.доллар', 'EUR'→'евро', 'JPY'→'иен', 'CNY'→'юань'
5. If `rate` column exists, call it "Валютын ханш" (NOT "Хүүний хэмжээ")
6. If results empty: "Мэдээлэл олдсонгүй."
7. If error in results: "Уучлаарай, таны асуултад хариулт өгөх боломжгүй байна."
8. Keep [MASKED_*] placeholders exactly as-is (they'll be replaced automatically)

**Examples:**
Query: Пирамид-орд ххк-ийн харилцагчийн код
SQL Response: [{{'cust_code': '9360000100'}}]
Response: Таны хүссэн "Пирамид-орд" ХХК-ийн харилцагчийн код нь 9360000100 байна.

Query: Хамгийн эхний зээлдэгчийн мэдээлэл
SQL Response: [{{'name': 'МИАТ ХХК', 'adv_amount': [MASKED_ADV_AMOUNT], 'cur_code': 'USD'}}]
Response: Хамгийн эхний зээлдэгчийн мэдээлэл:
- Нэр: МИАТ ХХК
- Олгосон дүн: [MASKED_ADV_AMOUNT] ам.доллар

---
Query: {query_str}
SQL: {sql_query}
SQL Response: {context_str}
Response: """

# =============================================================================
# RESPONSE SYNTHESIS (Local/minimal version)
# =============================================================================
RESPONSE_SYNTHESIS_PROMPT = """
Present the SQL result in minimal, clear Mongolian format.

**Rules:**
- Present data directly, no introductions
- Map currencies: 'MNT'→'төгрөг', 'USD'→'ам.доллар'
- Format numbers with thousand separators
- If empty: "Мэдээлэл олдсонгүй."
- If error: "Уучлаарай, таны асуултад хариулт өгөх боломжгүй байна."

Query: {query_str}
SQL: {sql_query}
SQL Response: {context_str}
Response: """

# =============================================================================
# STREAMLINED TEXT-TO-SQL PROMPT (~50% smaller)
# =============================================================================
ORACLE_TEXT_TO_SQL_PROMPT = """
You are an Oracle SQL expert for **Development Bank of Mongolia (DBM)**.

**CORE RULES:**

1. **EXAMPLES OVERRIDE ALL:** If user's question matches a RELEVANT EXAMPLE, follow that exact pattern.

2. **TABLE SELECTION:**
   - Future/schedules (төлөгдөх, хуваарь, ирэх) → `BCOM_NRS_DETAIL`
   - Current/past (үлдэгдэл, хариуцдаг, хэтэрсэн) → `LOAN_BALANCE_DETAIL`

3. **DEDUPLICATION (CRITICAL):**
   `loan_balance_detail` has DAILY SNAPSHOTS (one row per day per loan).
   - Latest state: `WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)`
   - Unique accounts: `ROW_NUMBER() OVER (PARTITION BY acnt_code ORDER BY txn_date DESC)`

4. **STATUS FILTERING:**
   - Current state queries (үлдэгдэл, идэвхтэй): ADD `status = 'O'`
   - Historical queries (хамгийн анх, бүх): NO status filter

5. **CURRENCY:**
   - Cross-currency ranking: Use `amount * rate` for comparison
   - Aggregations: Always `GROUP BY cur_code`
   - Always SELECT `cur_code` with monetary values

6. **CHAT HISTORY (follow-ups with "энэ", "тэр"):**
   Re-compute the entity using a CTE based on previous SQL pattern:
   ```sql
   WITH prev_entity AS (
       -- Re-run previous query logic
       SELECT acnt_code FROM dbm.loan_balance_detail
       WHERE ... ORDER BY ... FETCH FIRST 1 ROW ONLY
   )
   SELECT ... FROM dbm.loan_balance_detail
   WHERE acnt_code = (SELECT acnt_code FROM prev_entity)
   ```
   MUST include: `AND (acnt_name IS NOT NULL OR customer_name IS NOT NULL)`

7. **ENTITY NAMES:** Use exact matches from "📋 AVAILABLE NAMES" when provided.

8. **ALWAYS INCLUDE:** Entity names (customer_name, acnt_name) and cur_code in SELECT.

**PATTERNS:**
```sql
-- Latest state of account
SELECT * FROM DBM.LOAN_BALANCE_DETAIL 
WHERE acnt_code = '{{CODE}}' AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL);

-- Latest N customers
WITH deduped AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_name ORDER BY adv_date DESC) as rn
  FROM DBM.LOAN_BALANCE_DETAIL WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
)
SELECT * FROM deduped WHERE rn = 1 ORDER BY adv_date DESC FETCH FIRST N ROWS ONLY;

-- Payment schedules (always filter by latest nrs_version)
WITH latest_ver AS (SELECT acnt_code, MAX(nrs_version) AS v FROM DBM.BCOM_NRS_DETAIL GROUP BY acnt_code)
SELECT d.* FROM DBM.BCOM_NRS_DETAIL d JOIN latest_ver lv ON d.acnt_code=lv.acnt_code AND d.nrs_version=lv.v
WHERE d.schd_date > SYSDATE ORDER BY d.schd_date;
```

**SCHEMA:**
{schema}

**ANALYZER'S PLAN:**
{analyzer_explanation}

**DECOMPOSITION STEPS:**
{plan}

**ENTITY NAMES:**
{entity_names}

**RELEVANT EXAMPLES:**
{dynamic_examples}

**QUESTION:** {query_str}

**RESPONSE FORMAT:**
<explanation>
[Your reasoning]
</explanation>
<sql>
[Oracle SQL]
</sql>
"""

# =============================================================================
# STREAMLINED QUERY ANALYZER (~50% smaller)
# =============================================================================
QUERY_ANALYZER_PROMPT = """
You are a query analyst for **Development Bank of Mongolia (DBM)** chatbot.

**TASKS:**
1. Check RELEVANT EXAMPLES first - if match found, use same table(s)
2. Classify complexity (SIMPLE/COMPLEX)
3. Select required tables and columns (minimum needed)
4. Detect chat history references (энэ, тэр, дээрх)
5. Create decomposition steps

**TABLE SELECTION:**
| Keywords | Table | Time |
|----------|-------|------|
| төлөгдөх, хуваарь, ирэх, дараа | bcom_nrs_detail | Future |
| үлдэгдэл, хариуцдаг, хэтэрсэн, одоо | loan_balance_detail | Current/Past |

**DEDUPLICATION:** loan_balance_detail has daily snapshots. Queries for unique accounts need ROW_NUMBER() or latest txn_date filter.

**CHAT HISTORY:** If question has pronouns (энэ, тэр), set needs_chat_history=true and explain in sub_questions how to re-compute the entity.

**SCHEMA:**
{context_str}

**EXAMPLES:**
{examples}

**QUESTION:** {query_str}

**OUTPUT (JSON):**
{{
  "complexity": "SIMPLE|COMPLEX",
  "needs_chat_history": true|false,
  "chat_history_reasoning": "...",
  "needs_deduplication": true|false,
  "explanation": "Brief plan with table selection reasoning",
  "required_tables": ["TABLE1"],
  "required_columns": {{"TABLE1": ["col1", "col2"]}},
  "sub_questions": ["Step 1: ...", "Step 2: ..."]
}}
"""

# =============================================================================
# LIGHTWEIGHT SQL REVIEW (Post-execution validation focus)
# =============================================================================
SQL_REVIEW_PROMPT = """
You are reviewing SQL for **Development Bank of Mongolia (DBM)**.

**QUICK CHECKS:**
1. **Table Correct?** Future→bcom_nrs_detail, Current→loan_balance_detail
2. **Deduplication?** If asking for unique accounts, needs ROW_NUMBER() or FETCH FIRST with ORDER BY txn_date DESC
3. **Status Filter?** Current state needs `status='O'`, historical queries should NOT have it
4. **CTE Columns?** All columns used in final SELECT must exist in CTE
5. **Entity Names?** Used exact match from AVAILABLE NAMES if provided
6. **Chat History?** If follow-up question, did SQL re-compute entity with subquery?

**SCHEMA:** {context_str}
**QUESTION:** {masked_query_str}
**GENERATOR'S EXPLANATION:** {generator_explanation}
**SQL:** {sql_query}
**EXAMPLES:** {examples}

**OUTPUT (JSON):**
{{
  "is_correct": true|false,
  "reasoning": "Brief explanation",
  "corrected_query": "Fixed SQL or empty string"
}}
"""

# =============================================================================
# TRIAGE PROMPT (classify query type)
# =============================================================================
TRIAGE_PROMPT = """
Classify this user question for **Development Bank of Mongolia (DBM)** chatbot.

**Categories:**
- **DATA_QUESTION**: Needs database (loans, accounts, balances, payments, Mongolian banking terms)
- **GENERAL_QUESTION**: SQL concepts, greetings, how things work conceptually
- **OUT_OF_SCOPE**: Unrelated to banking (weather, sports, personal)

**Rules:**
- Greetings → GENERAL_QUESTION
- References to previous data (энэ, тэр) → DATA_QUESTION
- If unsure between DATA and GENERAL → prefer DATA_QUESTION

**Question:** {query_str}

**OUTPUT (JSON):**
{{
  "query_type": "DATA_QUESTION|GENERAL_QUESTION|OUT_OF_SCOPE",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}
"""

# =============================================================================
# GENERAL RESPONSE PROMPT
# =============================================================================
GENERAL_RESPONSE_PROMPT = """
You are a helpful Mongolian banking assistant for **Development Bank of Mongolia (DBM)**.
Answer the general question clearly and professionally in Mongolian.

**Rules:**
- Be clear and accurate about data/SQL concepts
- Use examples when helpful
- Stay within banking and data analysis context
- If unsure, say so honestly

**Question:** {query_str}

**Response:**
"""

# =============================================================================
# OUT OF SCOPE RESPONSE
# =============================================================================
OUT_OF_SCOPE_RESPONSE = """Уучлаарай, Таны асуусан асуулт Хөгжлийн Банкны өгөгдлийн сантай холбоогүй байна. 
Өөр асуулт байвал асууна уу."""

# =============================================================================
# SQL REGENERATION PROMPT (for retry after error)
# =============================================================================
SQL_REGENERATE_PROMPT = """
The previous SQL query failed. Generate a corrected version.

**PREVIOUS SQL:**
```sql
{previous_sql}
```

**ERROR:** {error_message}

**QUESTION:** {query_str}

**SCHEMA:** {schema}

**EXAMPLES:** {examples}

**Instructions:**
1. Analyze the specific error
2. Fix it while maintaining the original intent
3. Ensure Oracle syntax compliance
4. Include cur_code with monetary values

<explanation>
[Error analysis and fix]
</explanation>
<sql>
[Corrected SQL]
</sql>
"""

# =============================================================================
# POST-EXECUTION VALIDATION PROMPT
# =============================================================================
VALIDATE_ANSWER_PROMPT = """
Check if the SQL results answer the user's question.

**Question:** {query_str}
**SQL:** {sql_query}
**Results:** {results}

**Rules:**
- Empty results [] when data expected → NOT ANSWERED
- Error in results → NOT ANSWERED  
- Data present and relevant → ANSWERED
- Most data results are adequate

**OUTPUT (JSON):**
{{
  "is_answered": true|false,
  "reason": "Explanation if not answered",
  "suggestion": "How to improve if applicable"
}}
"""

# =============================================================================
# HISTORY SUMMARY PROMPT
# =============================================================================
HISTORY_SUMMARY_PROMPT = """
Summarize the conversation history for context. Keep only essential information.

**Previous Turns:**
{history}

**OUTPUT (JSON):**
{{
  "summary": "Brief summary of what was discussed",
  "key_entities": ["entity names mentioned"],
  "last_sql_pattern": "Brief description of last query type"
}}
"""


# =============================================================================
# PROMPT MANAGER
# =============================================================================
class PromptManager:
    """Optimized prompt manager for Oracle database with Mongolian support"""
    
    def __init__(self, dialect: str = "oracle"):
        self.dialect = dialect
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize all prompt templates"""
        if self.dialect.lower() == "oracle":
            sql_template = ORACLE_TEXT_TO_SQL_PROMPT
        else:
            sql_template = DEFAULT_TEXT_TO_SQL_PROMPT
            
        self.text2sql_prompt = PromptTemplate(sql_template)
        self.response_synthesis_prompt = PromptTemplate(RESPONSE_SYNTHESIS_PROMPT)
        self.api_response_synthesis_prompt = PromptTemplate(API_RESPONSE_SYNTHESIS_PROMPT)
        self.table_info_prompt = PromptTemplate(TABLE_INFO_PROMPT)
        self.sql_review_prompt = PromptTemplate(SQL_REVIEW_PROMPT)
        self.query_analyzer_prompt = PromptTemplate(QUERY_ANALYZER_PROMPT)
        
        # New architecture prompts
        self.triage_prompt = PromptTemplate(TRIAGE_PROMPT)
        self.general_response_prompt = PromptTemplate(GENERAL_RESPONSE_PROMPT)
        self.sql_regenerate_prompt = PromptTemplate(SQL_REGENERATE_PROMPT)
        self.validate_answer_prompt = PromptTemplate(VALIDATE_ANSWER_PROMPT)
        self.history_summary_prompt = PromptTemplate(HISTORY_SUMMARY_PROMPT)
        
    def get_text2sql_prompt(self) -> PromptTemplate:
        return self.text2sql_prompt

    def get_sql_review_prompt(self) -> PromptTemplate:
        return self.sql_review_prompt
    
    def get_response_synthesis_prompt(self) -> PromptTemplate:
        if config.LLM_BACKEND in ("gemini", "openai"):
            logger.info("Using API-optimized (conversational) response prompt.")
            return self.api_response_synthesis_prompt
        else:
            logger.info("Using Local-optimized (direct) response prompt.")
            return self.response_synthesis_prompt
            
    def get_query_analyzer_prompt(self, examples: str = "No examples provided.") -> PromptTemplate:
        return PromptTemplate(QUERY_ANALYZER_PROMPT).partial_format(examples=examples)
        
    def get_table_info_prompt(self) -> PromptTemplate:
        return self.table_info_prompt
    
    def get_triage_prompt(self) -> PromptTemplate:
        return self.triage_prompt
    
    def get_general_response_prompt(self) -> PromptTemplate:
        return self.general_response_prompt
    
    def get_sql_regenerate_prompt(self) -> PromptTemplate:
        return self.sql_regenerate_prompt
    
    def get_validate_answer_prompt(self) -> PromptTemplate:
        return self.validate_answer_prompt
    
    def get_history_summary_prompt(self) -> PromptTemplate:
        return self.history_summary_prompt
    
    def get_out_of_scope_response(self) -> str:
        return OUT_OF_SCOPE_RESPONSE


prompt_manager = PromptManager("oracle")
