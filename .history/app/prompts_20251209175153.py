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
- The table name must be unique to the table and describe its primary content while being concise. Use Mongolian names if appropriate for clarity.
- Do NOT output a generic table name (e.g. table, my_table, data).
- For each column, provide a detailed description.
  - Describe the **type of data** it contains (e.g., numeric, text, date, code, descriptive label).
  - Explain its **purpose or meaning** within the table, including its business context.
  - If the column name is cryptic (e.g., `acnt_code`, `cur_code`), **infer its full semantic meaning** based on other columns and the provided sample data.
  - **CRUCIAL**: Explicitly state the relationship between numeric value columns and their corresponding currency/unit columns. For example, for a column like `APPROV_AMOUNT`, state: "This is a numeric value representing the approved loan amount. Its currency is determined by the **`CUR_CODE`** column (e.g., 'MNT' for Togrog, 'USD' for US Dollars)."
- Pay special attention to columns that are essential for filtering or uniquely identifying rows (e.g., `txn_date`, `acnt_code`, `cur_code`).
- Output column_descriptions for **all** columns that appear in the 'Table Structure' or 'Sample Data'.
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
5. If `rate` column exists, call it "Валютын ханш" (int_rate is "Хүүний хэмжээ")
6. If results empty: "Мэдээлэл олдсонгүй."
7. If error in results: "Уучлаарай, таны асуултад хариулт өгөх боломжгүй байна."
8. Keep [MASKED_*] placeholders exactly as-is (they'll be replaced automatically)
   **NO MASKING CHATTER:** Data masking is handled by the backend. **Do NOT** apologize for security or masked values. Just present the data as-is with placeholders.
**Examples:**
Query: Пирамид-орд ххк-ийн харилцагчийн код
SQL Response: [{{'cust_code': '[MASKED_CUST_CODE]'}}]
Response: Таны хүссэн "Пирамид-орд" ХХК-ийн харилцагчийн код нь [MASKED_CUST_CODE] байна.

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
# STREAMLINED TEXT-TO-SQL PROMPT 
# =============================================================================
ORACLE_TEXT_TO_SQL_PROMPT = """
You are an expert Oracle SQL developer for the **Development Bank of Mongolia (DBM)**, which is known as **"Хөгжлийн Банк"** in Mongolian.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database you are querying. **DO NOT** filter `customer_name` or any other column by these terms. Instead, perform the requested action (e.g., find the top customers) across all data.


**CORE RULES:**

1. **DYNAMIC EXAMPLES:** If the user's question is similar to a `RELEVANT EXAMPLE` provided below, **follow that SQL pattern exactly. Examples are proven correct**.

2. **TABLE SELECTION:**
   - Future/schedules (төлөгдөх, хуваарь, ирэх) → `BCOM_NRS_DETAIL`
   - Current/past (үлдэгдэл, хариуцдаг, хэтэрсэн) → `LOAN_BALANCE_DETAIL`
3. **DEDUPLICATION (CRITICAL):**
   `loan_balance_detail` has DAILY SNAPSHOTS (one row per day per loan).
   - Latest state: `WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)` there is only one loan account per day, so acnt_names won't be duplicated on that date.
   - Unique customers: `ROW_NUMBER() OVER (PARTITION BY customer_name ORDER BY txn_date DESC)` etc.
4. **CURRENCY RANKING (THE "RATE" RULE):** - NEVER order or compare raw monetary columns (principal, adv_amount) directly.
   - **ALWAYS** convert to MNT using `amount * rate` (e.g., `ORDER BY (adv_amount * rate) DESC`).
   - This applies to **main queries AND subqueries/CTEs**.
5. **🧹 DATA HYGIENE & FILTERING (NEW RULES):**
    **ZERO-VALUE EXCLUSION (Default Behavior):**
    Unless the user explicitly asks for "paid off", "closed", or "zero balance" loans, you **MUST** filter out noise:
    - **Current/Balance Queries:** Add `AND principal > 0`. (Hide accounts with no debt).
    - **Historical/Disbursement Queries:** Add `AND adv_amount > 0`. (Hide loans that were never disbursed).
    - Filter customer_name, acnt_name to exclude empty/NULL names: `AND customer_name IS NOT NULL, acnt_name IS NOT NULL`.
    - *Example:* "Who was the first borrower?" → `WHERE adv_amount > 0`.

4. **STATUS FILTERING:**
   - Current state queries (үлдэгдэл, идэвхтэй): ADD `status = 'O'`
   - Historical queries (хамгийн анх, бүх): NO status filter
5. **CURRENCY:**
   - Always SELECT `cur_code` with monetary values
6. **AGGREGATIONS:** 
   - Always `GROUP BY cur_code` when summing money.
7. **CHAT HISTORY (Follow-up Questions with "энэ", "тэр"):**
   - You do NOT have access to the specific ID/Code from the previous turn.
   - You **MUST RE-CALCULATE** the entity ID using a CTE based on the previous logic.
   - If [Previous SQL] is provided, **extract exact column names and filter values from it**.
   - Do NOT rely on the summary text - use the actual SQL!
   - **CRITICAL:** Inside the CTE, you must repeat the **exact sorting logic** used to find the entity originally.
   - *Example case Pattern:*
      ```sql
      WITH target_entity AS (
          SELECT acnt_code FROM dbm.loan_balance_detail
          WHERE customer_name = '...' -- Apply filters
          ORDER BY (adv_amount * rate) DESC -- MUST include rate multiplication if ranking!
          FETCH FIRST 1 ROW ONLY
      )
      SELECT ... FROM dbm.loan_balance_detail 
      WHERE acnt_code = (SELECT acnt_code FROM target_entity) ...
      ```
8. Always select latest nrs_version for BCOM_NRS_DETAIL queries. MAX(nrs_version) per acnt_code.
9. **ENTITY NAMES:** Use exact matches from "📋 AVAILABLE NAMES" when provided.
10. **ALWAYS INCLUDE:** Entity names (customer_name, acnt_name) and cur_code in SELECT.
11. USE ONLY ORACLE SQL SYNTAX.
12. Schema data is pruned to only relevant tables/columns. if examples use missing columns, adapt accordingly.
13. Apply case-insensitive comparisons with `UPPER()` on text columns.

**EXAMPLE SQL PATTERNS:**
• ========== SINGLE ENTITY LOOKUPS ==========
  # Get latest state of ONE account (Global Snapshot)
  SELECT * FROM DBM.LOAN_BALANCE_DETAIL 
  WHERE acnt_code = '{{ACNT_CODE}}' 
    AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
  --FETCH FIRST 1 ROW ONLY; -- Optional for single account lookup, as acnt_code is unique per day
  
# Get latest state of ONE customer's loans (Global Snapshot)
  SELECT customer_name, adv_date, adv_amount, cur_code
  FROM DBM.LOAN_BALANCE_DETAIL
  WHERE customer_name = '{{CUSTOMER_NAME}}' AND adv_date IS NOT NULL and adv_amount > 0
    AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
  ORDER BY adv_date DESC

 # Latest 5 customers (NOT accounts!)
  SELECT customer_name, adv_date, adv_amount, cur_code, principal, status
  FROM (
      SELECT customer_name, adv_date, adv_amount, cur_code, principal, status,
            ROW_NUMBER() OVER (PARTITION BY customer_name ORDER BY adv_date DESC) as rn
      FROM DBM.LOAN_BALANCE_DETAIL
      WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
        AND customer_name IS NOT NULL 
        AND adv_date IS NOT NULL AND adv_amount > 0
  ) t
  WHERE rn = 1
  ORDER BY adv_date DESC
  FETCH FIRST 5 ROWS ONLY;
  
  # Latest 3 loan accounts
  SELECT customer_name, adv_date, adv_amount, cur_code, txn_date, acnt_code, principal, status
  FROM DBM.LOAN_BALANCE_DETAIL
  WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
      AND acnt_name IS NOT NULL
      AND adv_date IS NOT NULL AND adv_amount > 0
  ORDER BY adv_date DESC
  FETCH FIRST 5 ROWS ONLY;

  • ========== CURRENT + NEXT PAYMENT (COMBINED QUERY) ==========
 For "account info + next payment" questions, use this pattern:**
  
  # Account current state + next scheduled payment (ONE query!)
    WITH latest_snapshot AS (
    SELECT * 
    FROM dbm.loan_balance_detail 
    WHERE acnt_code = '{{ACNT_CODE}}'
      AND txn_date = (SELECT MAX(txn_date) FROM dbm.loan_balance_detail)
  ),
  current_version AS (
      SELECT MAX(nrs_version) AS latest_nrs_version
      FROM dbm.bcom_nrs_detail
      WHERE acnt_code = '{{ACNT_CODE}}'
  ),
  next_payment AS (
      SELECT t.schd_date,
            t.amount,
            t.int_amount
      FROM dbm.bcom_nrs_detail t
      WHERE t.acnt_code = '{{ACNT_CODE}}'
        AND t.nrs_version = (SELECT latest_nrs_version FROM current_version)
        AND t.status = 'O'
        AND t.schd_date > SYSDATE
      ORDER BY t.schd_date ASC
      FETCH FIRST 1 ROW ONLY
  )
  SELECT 
      ls.*,
      np.schd_date      AS next_payment_date,
      np.amount         AS next_principal_amount,
      np.int_amount     AS next_interest_amount
  FROM latest_snapshot ls
  LEFT JOIN next_payment np ON 1=1;

• ========== PAYMENT SCHEDULES (BCOM_NRS_DETAIL) ==========
  **ALWAYS filter by latest nrs_version per account**
  
  # Next payment in next 30 days (with latest schedule version)
  WITH latest_version_per_account AS (
      SELECT acnt_code,
             MAX(nrs_version) AS current_version
      FROM DBM.BCOM_NRS_DETAIL
      GROUP BY acnt_code
  )
  SELECT d.acnt_code, d.customer_name, d.schd_date, d.amount, d.int_amount
  FROM DBM.BCOM_NRS_DETAIL d
  INNER JOIN latest_version_per_account lv 
      ON d.acnt_code = lv.acnt_code AND d.nrs_version = lv.current_version
  WHERE d.status = 'O'
    AND d.schd_date BETWEEN SYSDATE AND SYSDATE + 30
  ORDER BY schd_date asc;

**SCHEMA:**
{schema}

**RELEVANT EXAMPLES:**
{dynamic_examples}

**ANALYZER'S PLAN:**
{analyzer_explanation}

**DECOMPOSITION STEPS:**
{plan}

**ENTITY NAMES:**
{entity_names}

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
You are an expert query analyst for a financial database chatbot for the **Development Bank of Mongolia (DBM)**.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database. **DO NOT** create a plan step that filters by this name. The query should apply to all data.

**TASKS:**
1. Check RELEVANT EXAMPLES first - if match found, use same table(s), columns, and logic.
2. Classify complexity (SIMPLE/COMPLEX)
3. Select required tables and columns (minimum needed)
4. Detect chat history references (энэ, тэр, дээрх) analyze if re-computation needed
5. Create decomposition steps

**TABLE SELECTION:**
| Keywords | Table | Time |
|----------|-------|------|
| төлөгдөх, хуваарь, ирэх, дараа | dbm.bcom_nrs_detail | Future |
| үлдэгдэл, хариуцдаг, хэтэрсэн, одоо | dbm.loan_balance_detail | Current/Past |

Харилцагчийн мэдээлэл, дансны дэлгэрэнгүй, өдрийн үлдэгдлийн дата, эрсдэлийн ангилал зэрэгтэй холбоотой асуултуудад dbm.loan_balance_detail хүснэгтийг ашиглах ёстой.
Зээлийн төлбөрийн хуваарь, ирээдүйн төлбөрүүдтэй холбоотой асуултуудад dbm.bcom_nrs_detail хүснэгтийг ашиглах ёстой.

**DEDUPLICATION:**
    - dbm.loan_balance_detail has daily snapshots. 
    - Each loan account has multiple rows (one per day via `txn_date`).
    - For latest state of account, filter by `txn_date = (SELECT MAX(txn_date) FROM dbm.loan_balance_detail)`.

**CHAT HISTORY:** 
- If question has pronouns (энэ, тэр), set needs_chat_history=true and explain in sub_questions how to re-compute the entity.
- If [Previous SQL] is provided, **extract column names from it** for required_columns.
- Example: If previous SQL used `int_rate`, include `int_rate` in required_columns.
- Explain in chat_history_reasoning which columns/filters from previous SQL are needed.

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
You are an Oracle SQL reviewer for the **Development Bank of Mongolia (DBM)**, ensuring safe, correct execution.
You are reviewing SQL written by another analyst.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database. The generated SQL should **NOT** contain a filter like `WHERE customer_name = 'ХӨГЖЛИЙН БАНК'`. If the SQL correctly omits this filter, it is following the rule.


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
You are a helpful Mongolian banking assistant for **Development Bank of Mongolia (DBM)**, **Монгол Улсын Хөгжлийн Банк**.
Answer the general question clearly and professionally in Mongolian.

**Rules:**
- Be clear and accurate about database concepts
- Use examples when helpful
- Stay within banking and data analysis context
- If unsure, say so honestly

**Question:** {query_str}

**Response:**
"""

# =============================================================================
# OUT OF SCOPE RESPONSE
# =============================================================================
OUT_OF_SCOPE_RESPONSE = """Уучлаарай, Таны асуусан асуулт Хөгжлийн Банкны өгөгдлийн сантай холбоогүй байна. Өөр асуулт байвал асууна уу."""

# =============================================================================
# SQL REGENERATION PROMPT (for retry after error)
# =============================================================================
SQL_REGENERATE_PROMPT = """
The previous SQL query failed. Generate a corrected version.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database. The generated SQL should **NOT** contain a filter like `WHERE customer_name = 'ХӨГЖЛИЙН БАНК'`. If the SQL correctly omits this filter, it is following the rule.

**PREVIOUS SQL:**
```sql
{previous_sql}
```

**ERROR:** {error_message}

**QUESTION:** {query_str}

**SCHEMA:** {schema}

**EXAMPLES:** {examples}
CRITICAL RULE:
- if the examples are similar to the question, follow their SQL pattern exactly. Examples are proven correct.
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
**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database. The generated SQL should **NOT** contain a filter like `WHERE customer_name = 'ХӨГЖЛИЙН БАНК'`. If the SQL correctly omits this filter, it is following the rule.

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

**CRITICAL:** Preserve all entity names (customer names, account names, manager names) in their ORIGINAL form exactly as they appeared. Do NOT transliterate Mongolian names to English.

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
