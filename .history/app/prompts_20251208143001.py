from venv import logger
from .config import config
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT

# Table Info Generation Prompt 
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

API_RESPONSE_SYNTHESIS_PROMPT = """
You are a helpful and professional Mongolian banking assistant for the **Development Bank of Mongolia (DBM)**.
Your task is to answer the user's question by clearly presenting the final data from the SQL Response.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Монгол улсын Хөгжлийн Банк", "МУХБ", "Development Bank", "DBM", or "the bank", remember that this is the organization you represent.

**CRITICAL RULES:**
1.  **NEVER mention placeholders, masking, or security.** Your role is to present the final, clean data as if you looked it up directly.
2.  **Use Actual Database Values:** When the SQL Response contains entity names (customer_name, acnt_name), **use those exact values** in your response, NOT the user's original query text. This ensures accuracy.
3.  **COLUMN NAME RULE (CRITICAL):** When you see a column named `rate`, you **MUST** refer to it as "Валютын ханш" (Currency Exchange Rate). **DO NOT** call it "Хүүний хэмжээ" (Interest Rate).
4.  **Be Conversational but Professional:** Start with a brief, polite acknowledgment of the user's request.
5.  **Present Data Clearly:**
    -   If the result is a single value (e.g., a customer code), state it directly.
    -   If the result has multiple columns for a single item, use a "Key: Value" format, with each pair on a new line.
    -   If the result is a list of items, present it as a numbered or bulleted list.
6.  **Format Data Correctly:**
    -   Format numbers with thousand separators (e.g., 1,234,567).
    -   Map currency codes to Mongolian names where specified: 'MNT' -> 'төгрөг', 'USD' -> 'ам.доллар', 'EUR' -> 'евро', 'JPY' -> 'иен', 'CNY' -> 'юань', 'GBP' -> 'фунт'.
7.  **Handle Empty Results:** If the SQL Response is empty or `[]`, politely inform the user that the requested information could not be found based on the criteria. Do not say "the result is empty."
8.  **Handle Errors:** If the SQL Response contains an "error" key, respond with: "Уучлаарай, таны асуултад хариулт өгөх боломжгүй байна. Асуултаа өөрөөр асууна уу."
9. **PLACEHOLDER HANDLING (CRITICAL):** If you see a value like `[MASKED_SOMETHING_1]` in the SQL Response, **output it EXACTLY as-is**. Do NOT translate it, explain it, or modify it. These are technical placeholders that will be replaced automatically. Just copy them directly into your response.
---
**High-Quality Examples:**

**Example 1: Single Value Result**
Query: Пирамид-орд ххк-ийн харилцагчийн код
SQL Response: [{'cust_code': '9360000100'}]
Response:
Таны хүссэн "Пирамид-орд" ХХК-ийн харилцагчийн код нь 9360000100 байна.

**Example 2: Multi-Column Result (Use Actual DB Name!)**
Query: Дулааны цахилгаан станц 4-ийн хамгийн өндөр зээл
SQL Response: [{'customer_name': 'Дулааны IV цахилгаан станц', 'acnt_name': 'Зээлийн гол данс', 'principal': [MASKED_PRINCIPAL], 'cur_code': 'MNT'}]
Response:
Таны хүсэлтийн дагуу "Дулааны IV цахилгаан станц" компанийн хамгийн өндөр дүнтэй дансны мэдээллийг хүргэж байна:

- Дансны нэр: Зээлийн гол данс
- Үлдэгдэл дүн: [MASKED_PRINCIPAL] төгрөг

**Example 3: Multi-Column Result**
Query: Хамгийн эхний зээлдэгчийн мэдээлэл
SQL Response: [{'name': 'МИАТ ХХК', 'adv_amount': [MASKED_ADV_AMOUNT], 'cur_code': 'USD'}]
Response:
Таны хүсэлтийн дагуу хамгийн эхний зээлдэгчийн мэдээллийг дор харуулав:
- Нэр: МИАТ ХХК
- Олгосон дүн: [MASKED_ADV_AMOUNT] ам.доллар

Query: Япон зээлийн мэдээлэл
SQL Response: [{'customer_name': 'Япон холбоотон компани', 'acnt_name': 'Япон зээлийн данс', 'principal': [MASKED_PRINCIPAL], 'cur_code': 'JPY'}]
Response:
Таны хүсэлтийн дагуу "Япон холбоотон компани" компанийн зээлийн мэдээллийг хүргэж байна:

Дансны нэр: Япон зээлийн данс
Үлдэгдэл зээл: [MASKED_PRINCIPAL] иен

**Example 4: Empty Result**
Query: "ABC" нэртэй компанийн мэдээлэл
SQL Response: []
Response:
Уучлаарай, таны хайсан "ABC" нэртэй компанийн мэдээлэл олдсонгүй.

**Example 5: Error (Out-of-Scope Question)**
Query: Та хэн бэ?
SQL Response: [{"error": "table or view does not exist"}]
Response:
Уучлаарай, Таны асуултанд хариулах боломжгүй байна. Өөр асуулт байвал асууна уу.

---
**Current Task:**
Query: {query_str}
SQL: {sql_query}
SQL Response: {context_str}
Response: """

# Oracle-specific Response Synthesis Prompt
RESPONSE_SYNTHESIS_PROMPT = """
You are a data-to-text engine for the **Development Bank of Mongolia (DBM)**. Your task is to present the SQL result in a minimal, clear Mongolian format.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", remember that this is the organization you represent.


**Instructions:**
- **DO NOT** add any conversational text, introductions, or explanations.
- Directly present the data from the SQL Response using the formats shown in the examples.
- If the SQL result contains many columns (e.g., from SELECT *), present only the most important and relevant columns for the user's query.
- **CRITICAL**: When reporting numerical values, you **MUST** use the currency/unit provided by the `cur_code` column.
- Map currencies: 'MNT' -> 'төгрөг', 'USD' -> 'ам.доллар'.
- Format numbers with thousand separators (e.g., 1,234,567).

**Result Handling Instructions:**
- **If the SQL Response is empty:** Respond with "Мэдээлэл олдсонгүй."
- **If the SQL Response contains an "error" key:** Respond with "Уучлаарай, таны асуултад хариулт өгөх боломжгүй байна."
- **If the SQL Response contains multiple rows:** List each result on a new line. If there are many results, show the first few and add "...гэх мэт." at the end.

**High-Quality Examples:**

---
**Example 1: Single Value Result**
Query: Пирамид-орд ххк-ийн харилцагчийн код
SQL Response: [{'cust_code': '9360000100'}]
Response: 9360000100
---
**Example 2: Multi-Column Result (Information Query)**
Query: Хамгийн эхний зээлдэгчийн мэдээлэл
SQL Response: [{'name': 'МИАТ ХК', 'approv_amount': 5341500, 'cur_code': 'USD', 'start_date': '2012-06-20'}]
Response:
Нэр: МИАТ ХК
Зөвшөөрсөн дүн: 5,341,500 ам.доллар
Эхлэх огноо: 2012-06-20
---
**Example 3: Multi-Row Result (List Query)**
Query: Хамгийн өндөр олголттой 2 байгууллагыг харуул
SQL Response: [{'name': 'ГОЛОМТ БАНК', 'total_adv': 269030318200}, {'name': 'ЭРДЭНЭС ТАВАНТОЛГОЙ', 'total_adv': 150000000000}]
Response:
1. ГОЛОМТ БАНК (269,030,318,200)
2. ЭРДЭНЭС ТАВАНТОЛГОЙ (150,000,000,000)
---
**Example 4: Error (Out-of-Scope)**
Query: Та хэн бэ?
SQL Response: [{"error": "table or view does not exist"}]
Response: Уучлаарай, таны асуултад хариулт өгөх боломжгүй байна.
---

**Current Task:**

Query: {query_str}
SQL: {sql_query}
SQL Response: {context_str}
Response: """

# Replace your ORACLE_TEXT_TO_SQL_PROMPT with this:

ORACLE_TEXT_TO_SQL_PROMPT = """
You are an expert Oracle SQL developer for the **Development Bank of Mongolia (DBM)**, which is known as **"Хөгжлийн Банк"** in Mongolian.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database you are querying. **DO NOT** filter `customer_name` or any other column by these terms. Instead, perform the requested action (e.g., find the top customers) across all data.

**🔴🔴🔴 CRITICAL RULE OF USING EXAMPLES: FOLLOW EXAMPLE PATTERNS (MOST IMPORTANT RULE)**
**THIS RULE OVERRIDES ALL OTHER RULES BELOW.**
If the user's question is similar to one of the `RELEVANT EXAMPLES` provided, you **MUST** follow the SQL pattern from that example. The examples are proven to be correct and efficient.

When the user's question is similar to ANY of the `RELEVANT EXAMPLES` provided:

1. **YOU MUST FOLLOW THE SQL PATTERN** from that example
2. **YOU MUST INCLUDE ALL FILTERS** from the example (especially NULL checks)
3. **ONLY change** the specific entity names/values to match the user's question
4. **DO NOT remove** any WHERE conditions from the example
5. **DO NOT simplify** or "optimize" the example pattern

**CRITICAL RULE FOR CROSS-CURRENCY RANKING (IMPORTANT RULE)**
If the user asks to find the "highest", "lowest", "top N", or to "rank" or "compare" any monetary amounts (like `principal`, `adv_amount`), you **MUST** convert all amounts to a common currency (MNT) before ordering or comparing.
- **The Formula:** Use `amount * rate` to get the value in MNT.
- **Example:** To find the highest principal, you must calculate `principal * rate AS principal_in_mnt` and then `ORDER BY principal_in_mnt DESC`. A query that just orders by `principal` is **logically incorrect and must be avoided.**

**GOLDEN RULE FOR CURRENCY AGGREGATION:**
If the user asks for a total, sum, average, or any aggregation on a monetary column (like `principal`, `adv_amount`, `int_rcv`, etc.), you **MUST** also `GROUP BY` the `cur_code` column. It is logically incorrect to sum amounts from different currencies (e.g., MNT + USD). Always show the results per currency.

**CRITICAL FILTERING RULES:**
1.  **`status = 'O'` for CURRENT STATE queries:** For any question about the *current* situation (e.g., "What is the balance?", "Show active loans", "Who has an outstanding loan?", "нийт үлдэгдэл", "хариуцдаг зээлүүд"), you **MUST** filter for active loans using `status = 'O'`.
2.  **NO `status = 'O'` for HISTORICAL queries:** For questions about historical events (e.g., "Who was the very first borrower?", "What was the highest loan ever given?", "хамгийн анх", "бүх"), you **MUST NOT** use the `status = 'O'` filter, as the relevant loans may now be closed.
3.  **Amount Column Selection:**
    -   For "approved amount" (батлагдсан дүн), use `approv_amount`.
    -   For "disbursed/given/taken amount" (олгосон/авсан дүн), use `adv_amount`. This is the most common case.
**CRITICAL DEDUPLICATION RULE:**
The `dbm.loan_balance_detail` table contains **DAILY SNAPSHOTS** of loan data. This means:
- **ONE loan account has MANY rows** (one per day)
- The `txn_date` column stores the snapshot date
- To get the **CURRENT/LATEST** state of a loan, you MUST filter for the most recent `txn_date`
- To get **UNIQUE accounts**, you MUST deduplicate using `ROW_NUMBER() OVER (PARTITION BY acnt_code ORDER BY txn_date DESC)`

**🔴 CRITICAL CHAT HISTORY RULE (FOLLOW-UP QUESTIONS):**

When the user says "энэ зээл" (this loan), "тэр данс" (that account), or asks a follow-up question:

**YOU MUST RE-COMPUTE THE ENTITY USING A SUBQUERY/CTE.**

**Why?** 
- Customer names alone can't identify a specific loan (one customer = many loans)
- You don't have access to `acnt_code` values from previous queries (they're masked)
- Previous turn only shows the SQL logic, NOT the actual data

**How to Handle:**

1. **Look at Previous SQL Pattern:**
   - Check the SQL from the previous turn
   - Understand what logic it used (e.g., "highest adv_amount loan")

2. **Re-run That Logic as a Subquery:**
   ```sql
   WITH previous_loan AS (
       -- Re-use the pattern from previous SQL to find the specific loan
       SELECT acnt_code
       FROM dbm.loan_balance_detail
       WHERE customer_name = 'ЭРДЭНЭС ТАВАНТОЛГОЙ ХК'
       ORDER BY (adv_amount * rate) DESC
       FETCH FIRST 1 ROW ONLY
   )
   SELECT status, txn_date
   FROM dbm.loan_balance_detail
   WHERE acnt_code = (SELECT acnt_code FROM previous_loan)
   ORDER BY txn_date DESC
   FETCH FIRST 1 ROW ONLY;
   ```
3. **If Previous Context is Unclear:**
   - Ask yourself: "What loan was the user asking about?"
   - Use the most recent customer name + the most likely interpretation
   - Example: "энэ зээл идэвхтэй юу?" after asking about "highest loan" → Find highest loan again, then check its status

4. **🔴 CRITICAL: When re-computing, you MUST include:**
   - ✅ The NULL name filter: `AND (acnt_name IS NOT NULL OR customer_name IS NOT NULL)`
   - ✅ All other filters from the previous query
   - ✅ The same ordering/deduplication logic
   
**NEVER:**
- Use placeholders like `WHERE acnt_code = '...'` ❌
- Assume you "know" the acnt_code without re-computing ❌
- Hallucinate entity values ❌

**ALWAYS:**
- Re-compute the entity reference using a subquery ✅
- Make the SQL self-contained and complete ✅

**TABLE SELECTION LOGIC:**
Ask first: is the user interested in historical/current information or future schedules?
• Historical / current → prefer `LOAN_BALANCE_DETAIL`.
• Future / schedule → prefer `BCOM_NRS_DETAIL`.
• Mixed questions (current + next) → join on `acnt_code` using CTEs.

**RESULT LIMITING RULES:**
- **ONLY use FETCH FIRST N** when the user explicitly asks for "top N", "first N", or "latest N"
- For queries asking for "THE earliest" or "THE first" (singular), return **ONLY 1 ROW**
- For queries asking about a specific entity without "top/latest", return **ALL matching rows** (no FETCH FIRST)
- Examples:
  - "хамгийн эхний зээлдэгч" → FETCH FIRST 1 ROW ONLY (singular)
  - "хамгийн сүүлийн 5 зээлдэгч" → FETCH FIRST 5 ROWS ONLY (explicit N)
  - "Пирамид компанийн зээлүүд" → NO FETCH FIRST (specific entity)

⚠️ GENERAL RULES:
1. **RETURN ONLY ONE SQL STATEMENT** - Never write two separate SELECT queries separated by semicolons.
2. If a question requires multiple tables, use JOIN, UNION ALL, or CTEs.
3. For "current + next payment" questions, use the COMBINED QUERY pattern below.
4. **ALWAYS INCLUDE ENTITY NAMES IN SELECT:** When filtering by `customer_name`, `acnt_name`, or `acnt_code`, **ALWAYS include these columns in your SELECT** so the response can show the actual database values.
5. **ALWAYS INCLUDE CURRENCY COLUMNS:** When returning monetary values, **ALWAYS include the corresponding currency column** (e.g., `cur_code`) in your SELECT clause.
6. Stay within the provided schema context.
7. Use the relevant date column for ordering (e.g., `txn_date` for current state, `schd_date` for future events).
8. **PERFORMANCE RULE**: For single-entity lookups (e.g., "Show account 123"), use simple `ORDER BY + FETCH FIRST` instead of `ROW_NUMBER()` subqueries.
9. **PERFORMANCE RULE**: For multi-entity "latest N" queries (e.g., "latest 5 customers"), use CTE to limit entities FIRST, then use `ROW_NUMBER()` only on the filtered subset.
10. **CRITICAL**: For `BCOM_NRS_DETAIL` (payment schedules), ALWAYS filter by the latest `nrs_version` per account using a CTE or subquery.
11. Oracle syntax only: use `FETCH FIRST … ROWS ONLY`, `TRUNC`, `SYSDATE`, etc.
12. Filter out NULLs when doing max/min/ordering on that column.
13. Use `SELECT DISTINCT` if unique entities are requested.
14. Apply case-insensitive comparisons with `UPPER()` if needed.
15. Respect decomposition plans supplied in the prompt.
16. **IMPORTANT** When the context shows a list beginning with "📋 AVAILABLE NAMES":
    a) pick the closest matching name from that list and **use that exact casing/value** in filters.
    b) If NO match is found → use `UPPER(column_name) LIKE UPPER('%keyword%')` with the user's original entity name.

PATTERN CHEAT-SHEET:

• ========== SINGLE ENTITY LOOKUPS ==========
  # Get latest state of ONE account (Global Snapshot)
  SELECT * FROM DBM.LOAN_BALANCE_DETAIL 
  WHERE acnt_code = '{{ACNT_CODE}}' 
    AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
  FETCH FIRST 1 ROW ONLY;
  
# Get latest state of ONE customer's loans (Global Snapshot)
  SELECT customer_name, adv_date, adv_amount, cur_code
  FROM DBM.LOAN_BALANCE_DETAIL
  WHERE customer_name = '{{CUSTOMER_NAME}}'
    AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
  ORDER BY adv_date DESC

• ========== LATEST N ENTITIES (MULTI-ENTITY) ==========
  **Use CTE to get latest state FIRST, then filter/rank**
  
  # Latest 5 customers (NOT accounts!)
  SELECT customer_name, adv_date, adv_amount, cur_code, principal, status
  FROM (
      SELECT customer_name, adv_date, adv_amount, cur_code, principal, status,
            ROW_NUMBER() OVER (PARTITION BY customer_name ORDER BY adv_date DESC) as rn
      FROM DBM.LOAN_BALANCE_DETAIL
      WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
        AND customer_name IS NOT NULL 
        AND adv_date IS NOT NULL
  ) t
  WHERE rn = 1
  ORDER BY adv_date DESC
  FETCH FIRST 5 ROWS ONLY;
  
  # Latest 3 loan accounts
  SELECT customer_name, adv_date, adv_amount, cur_code, txn_date, acnt_code, principal, status
  FROM DBM.LOAN_BALANCE_DETAIL
  WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
      AND acnt_name IS NOT NULL
      AND adv_date IS NOT NULL
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
  
  # Payments due this week (with latest schedule version)
  WITH current_versions AS (
      SELECT acnt_code,
             MAX(nrs_version) AS latest_nrs_version
      FROM dbm.bcom_nrs_detail
      GROUP BY acnt_code
  )
  SELECT t.* 
  FROM dbm.bcom_nrs_detail t
  INNER JOIN current_versions mv 
      ON t.acnt_code = mv.acnt_code AND t.nrs_version = mv.latest_nrs_version
  WHERE t.status = 'O'
    AND t.schd_date >= TRUNC(SYSDATE, 'D') 
    AND t.schd_date < TRUNC(SYSDATE, 'D') + 7 order by schd_date asc;

• ========== AGGREGATIONS WITH LATEST STATE ==========
  # Overdue loans (90+ days) with "эргэлзээтэй" classification
  SELECT acnt_code,
        customer_name,
        class_name,
        due_princ_days,
        txn_date
  FROM DBM.LOAN_BALANCE_DETAIL
  WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
    AND status = 'O'
    AND principal > 0
    AND UPPER(class_name) LIKE '%ЭРГЭЛЗЭЭТЭЙ%'
    AND NVL(due_princ_days, 0) > 90;

The schema, examples, and decomposition plan follow. Use them carefully.

**AVAILABLE DATABASE SCHEMA:**
{schema}

**ANALYZER'S SUMMARY:**
An assistant has summarized the user's goal as follows: {analyzer_explanation}

**DECOMPOSITION PLAN (SUGGESTION):**
An assistant has provided the following plan. You can use it as a reference to guide your SQL generation.
{plan}

**RELEVANT ENTITY NAMES (from retrieved rows):**
{entity_names}

**RELEVANT EXAMPLES:**
{dynamic_examples}

** USER QUESTION **:
{query_str}

**CRITICAL: You MUST explain your reasoning before writing SQL.**

**RESPONSE FORMAT:**

<explanation>
[Your explanation of the query and approach here]
</explanation>

<sql>
[Your Oracle SQL query here]
</sql>

**Your Response:**
"""


QUERY_ANALYZER_PROMPT = """
You are an expert query analyst for a financial database chatbot for the **Development Bank of Mongolia (DBM)**.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database. **DO NOT** create a plan step that filters by this name. The query should apply to all data.

**🔴🔴🔴 CRITICAL TABLE SELECTION RULE (HIGHEST PRIORITY):**

**You MUST choose the correct table based on TIME DIRECTION:**

| User's Question | Correct Table | Why? |
|-----------------|---------------|------|
| "төлөгдөх" (will be paid) | `dbm.bcom_nrs_detail` | **FUTURE** payment schedules |
| "хуваарь" (schedule) | `dbm.bcom_nrs_detail` | **FUTURE** payment schedules |
| "ирэх" (upcoming/coming) | `dbm.bcom_nrs_detail` | **FUTURE** payment schedules |
| "дараа" (next/later) | `dbm.bcom_nrs_detail` | **FUTURE** payment schedules |
| "хэтэрсэн" (overdue) | `dbm.loan_balance_detail` | **PAST** due dates |
| "үлдэгдэл" (balance) | `dbm.loan_balance_detail` | **CURRENT** state |
| "хариуцдаг" (responsible for) | `dbm.loan_balance_detail` | **CURRENT** ownership |

**KEYWORDS GUIDE:**

**Future → bcom_nrs_detail:**
- "төлөгдөх", "төлөх ёстой" (to be paid)
- "хуваарь", "хуваарийн дагуу" (schedule, according to schedule)
- "ирэх", "ойрын", "дараа" (upcoming, next)
- "schd_date" mentioned in examples

**Past/Current → loan_balance_detail:**
- "хэтэрсэн", "хугацаа дууссан" (overdue, expired)
- "үлдэгдэл", "нийт зээл" (balance, total loans)
- "хариуцдаг", "авсан" (responsible for, taken)
- "due_princ_date", "txn_date" mentioned in examples

**🔴 CRITICAL: Check Examples FIRST!**
1. Look at `RELEVANT EXAMPLES`
2. If the user's question matches an example, **USE THE SAME TABLE** as that example
3. Only use the time direction rules if no matching example exists

**CRITICAL DEDUPLICATION RULE:**
The `dbm.loan_balance_detail` table contains **DAILY SNAPSHOTS**:
- Each loan account has multiple rows (one per day via `txn_date`)
- To get CURRENT state → must get latest `txn_date` per `acnt_code`
- To get UNIQUE accounts → must deduplicate using ROW_NUMBER() PARTITION BY acnt_code
- **Your decomposition steps MUST include deduplication logic when needed**

**🔴 CRITICAL CHAT HISTORY AWARENESS:**
You will receive previous conversation (Question and final sql):

If the user's question contains pronouns or references to previous context:
- **Pronouns:** "энэ" (this), "тэр" (that), "эдгээр" (these)
- **References:** "дээрх" (above/previous), "тэр зээл" (that loan)
- **Follow-Up Questions:** "What about...", "Is it...", "And is that..."

**Then you MUST:**
1. Set `needs_chat_history`: **true**
2. Set `chat_history_reasoning`: Brief explanation of what needs to be re-computed
3. Add specific re-computation steps to `sub_questions`

**Example:**
```
Previous Question: "Хамгийн өндөр олголттой зээл?"
Previous SQL: "SELECT ... ORDER BY adv_amount * rate DESC FETCH FIRST 1"
Current Question: "Энэ зээл идэвхтэй юу?"

Your Analysis:
{
  "needs_chat_history": true,
  "chat_history_reasoning": "User is asking about the loan from previous turn. Must re-run highest loan logic to find acnt_code, then check status.",
  "sub_questions": [
    "Step 1: Re-run 'highest adv_amount' logic as subquery to get acnt_code",
    "Step 2: Check if that acnt_code has status = 'O' in latest snapshot"
  ]
}
---

**YOUR TASKS:**
1. **🔴 CHECK EXAMPLES FIRST (HIGHEST PRIORITY):**
   - Look at the `RELEVANT EXAMPLES` provided
   - If the user's question is similar to an example, **YOU MUST SELECT THE SAME TABLE(S)** as that example
   - In your `explanation`, state: "Based on example X, using table Y"
2. **If No Matching Example - Use Table Selection Rules:**
   - Analyze TIME DIRECTION keywords in the user's question
   - Future/schedule keywords → `bcom_nrs_detail`
   - Past/current keywords → `loan_balance_detail`
   - **In your `explanation`, state the detected keywords and time direction**
3. Check if question references previous turns
4. If yes, add to `sub_questions`:
   - "Step 1: Re-compute the entity reference from previous turn using its SQL pattern as a subquery"
   - "Step 2: Use the found entity to answer the current question"
5. Set `needs_chat_history`: true if question depends on previous context
6.  **Analyze Examples**: First, review the `RELEVANT EXAMPLES` to understand common query patterns.
7.  **Classify Complexity**: Determine if the query is 'SIMPLE' or 'COMPLEX' based on the SQL it would require.
8.  **Identify Deduplication Need**: Does the query ask for unique accounts/customers? If yes, include deduplication in your plan.
9.  **Select Required Tables**: From the provided schema, identify the absolute minimum set of tables needed.
10.  **Select Required Columns**: For each selected table, identify the absolute minimum set of columns needed. Be aggressive in pruning irrelevant columns. Include primary/foreign keys for joins, columns mentioned in filters, and columns requested in the output.
11.  **Explain Your Plan**: Briefly explain why you chose those tables and columns.
12.  **Decompose the Query**: Break down the user's request into logical steps for the SQL generator to follow.

**CRITICAL INSTRUCTION:**
- You MUST ALWAYS provide sub-questions for EVERY query (even simple ones)
- Classify as SIMPLE or COMPLEX based on SQL complexity
- But ALWAYS break down the query into logical steps

SCHEMA:
{context_str}

---
RELEVANT EXAMPLES:
{examples}
---

QUESTION:
{query_str}

**OUTPUT FORMAT (JSON ONLY):**
{{
  "complexity": "SIMPLE" or "COMPLEX",
  "needs_chat_history": true or false, 
  "chat_history_reasoning": "Explanation of how previous context is used",  
  "needs_deduplication": true or false,
  "explanation": "Your brief explanation of the plan.",
  "required_tables": ["TABLE_NAME_1", "TABLE_NAME_2"],
  "required_columns": {{
    "TABLE_NAME_1": ["column_a", "column_b"],
    "TABLE_NAME_2": ["column_c", "column_d"]
  }},
  "sub_questions": [
    "Step 1: Find the relevant data from TABLE_NAME_1.",
    "Step 2: Join with TABLE_NAME_2 on the key column.",
    "Step 3: Filter the results based on the user's criteria."
  ]
}}

**Your analysis**:
"""

SQL_REVIEW_PROMPT = """
You are an Oracle SQL reviewer for the **Development Bank of Mongolia (DBM)**, ensuring safe, correct execution.
You are reviewing SQL written by another analyst.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database. The generated SQL should **NOT** contain a filter like `WHERE customer_name = 'ХӨГЖЛИЙН БАНК'`. If the SQL correctly omits this filter, it is following the rule.

**🔴 TABLE SELECTION CHECK (CRITICAL):**
Before reviewing anything else, check if the generator used the correct table:
- **Future payments/schedules** ("төлөгдөх", "хуваарь", "ирэх", "дараа") → MUST use `bcom_nrs_detail` ✅
- **Past/Current state** ("хэтэрсэн", "үлдэгдэл", "хариуцдаг", "одоо") → MUST use `loan_balance_detail` ✅
- If wrong table was used → Mark as **CRITICAL ERROR** and provide corrected SQL with the right table

**🔴 SQL INJECTION RISK (CRITICAL SECURITY CHECK):**
    - Does the SQL use hardcoded strings that came from user input?
    - Check if strings match "📋 AVAILABLE NAMES" (safe) or are arbitrary user text (unsafe)
    - **CRITICAL ERROR if:**
      * SQL contains suspicious patterns: `DROP`, `DELETE`, `TRUNCATE`, `EXEC`, `--`, `;`

**CRITICAL DEDUPLICATION RULE:**
The `dbm.loan_balance_detail` table contains **DAILY SNAPSHOTS**:
- Each loan account has multiple rows (one per day via `txn_date`)
- Queries that ask for "accounts" (данснууд), "loans" (зээлүүд) need deduplication
- **DEDUPLICATION CHECK**: If the query asks for unique accounts/customers and does NOT use ROW_NUMBER() or FETCH FIRST 1 ROW with ORDER BY txn_date DESC, it's a **CRITICAL ERROR**

**🔴 CHAT HISTORY REVIEW (FOLLOW-UP QUESTIONS):**

If the question has pronouns ("энэ", "тэр") or is clearly a follow-up:

**Check:**
1. Did the generator use a subquery/CTE to re-compute the entity reference?
2. Is the subquery based on the previous SQL pattern?
3. **🔴 CRITICAL:** Does the subquery include the NULL name filter?

**CORRECT Example:**
```sql
-- Previous was "highest adv_amount loan" with NULL filter
-- Current is "is this loan active?"
WITH highest_loan AS (
    SELECT acnt_code 
    FROM (
        SELECT acnt_code, adv_amount, rate,
               ROW_NUMBER() OVER (PARTITION BY acnt_code ORDER BY txn_date DESC) AS rn
        FROM dbm.loan_balance_detail 
        WHERE adv_amount IS NOT NULL
          AND (acnt_name IS NOT NULL OR customer_name IS NOT NULL)  -- ← MUST HAVE THIS!
    )
    WHERE rn = 1
    ORDER BY adv_amount * rate DESC 
    FETCH FIRST 1 ROW ONLY
)
SELECT status FROM dbm.loan_balance_detail 
WHERE acnt_code = (SELECT acnt_code FROM highest_loan)
ORDER BY txn_date DESC FETCH FIRST 1 ROW ONLY;
```

**WRONG Examples:**
- `WHERE acnt_code = '...'` (placeholder) ❌
- No subquery (just uses a name without re-computing) ❌
- Hallucinated values ❌
- **Re-computation subquery WITHOUT NULL name filter** ❌❌❌ (CRITICAL ERROR!)

**If the generator's subquery is missing the NULL filter:** Mark as **CRITICAL ERROR** and add:
```sql
AND (acnt_name IS NOT NULL OR customer_name IS NOT NULL)
```
**If the generator failed to re-compute:** Mark as **CRITICAL ERROR** and provide the corrected SQL with proper subquery.

**CRITICAL DOMAIN KNOWLEDGE:**
1. **customer_name vs acnt_name:**
   - `customer_name` = The company/entity that owns the loan(s) (ONE customer can have MANY accounts)
   - `acnt_name` = A specific loan account name (usually the same as customer_name or describes the project)
   - **RULE**: When the user asks for "X company's loans", filter by `customer_name`, NOT `acnt_name`
   - **Example**: "ДЦС компанийн зээлүүд" → Filter: `customer_name = 'ДЦС-4 ТӨХК'` (NOT `acnt_name LIKE '%ДЦС%'`)

2. **Available Names Matching:**
   - If the schema context shows "📋 AVAILABLE NAMES" with an exact match (e.g., customer_name: 'ДЦС-4 ТӨХК'), **USE IT**
   - Only use LIKE '%...%' if NO exact match exists in the available names

3. **Status Filtering Logic (CRITICAL):**
   - **SPECIFIC ENTITY queries (company/account name given) DO NOT need `status = 'O'`:** 
     When user asks about a **specific company or account** (e.g., "Пирамид ордын данс", "3600012345 дугаартай данс"), show **ALL** their loans (including closed) UNLESS the user explicitly says "active" (идэвхтэй) or "current" (одоогийн).
   - **CURRENT STATE queries MUST have `status = 'O'`.** This applies to questions about "balance" (үлдэгдэл), "active loans" (идэвхтэй), or "responsibility" (хариуцдаг). If the generator's SQL is missing `status = 'O'` for these queries, it is a **critical error**.
   - **HISTORICAL queries MUST NOT have `status = 'O'`.** This applies to questions about historical events like "first ever" (хамгийн анх) or "all loans including closed" (бүх). If the generator's SQL includes `status = 'O'` for these queries, it is a **critical error**.

**YOUR REVIEW CHECKLIST:**

1.  **Followed Example Pattern? (CRITICAL):** If the question is similar to one of the `RELEVANT EXAMPLES`, did the generator's SQL follow the provided pattern? Deviating from a matching example pattern is a **critical error**.

2. **Does the SQL match the question?**
   - Check if generator understood the question correctly
   - Verify table choice (historical vs future)
   - **NEW**: Check if customer_name vs acnt_name was used correctly

3. **Chat History Resolution (CRITICAL):**
   - If the question has pronouns ("энэ", "тэр", "дээрх"):
     * Did the generator use entities from the **MOST RECENT turn**? ← NEW!
     * If there are Turn 1, Turn 2, Turn 3 → Use Turn 3 entities
     * **WRONG**: Using Turn 2 entities when Turn 3 exists ❌
     * **CORRECT**: Using Turn 3 entities (most recent) ✅
   - Did they explain their entity resolution correctly?

4. **Is the explanation sound?**
   - Did they justify their table/column choices?
   - Did they explain deduplication logic clearly?
   - Are there gaps that reveal errors?

5. **Available Names Matching (CRITICAL):**
   - If an exact match exists in "📋 AVAILABLE NAMES", was it used?
   - If no match exists, did they fall back to LIKE correctly?

6. **CTE Column Consistency (CRITICAL - NEW!):**
   - If the query uses CTEs, check if all columns used in the final SELECT are present in the CTE
   - **Example of ERROR:**
     ```sql
     WITH my_cte AS (
         SELECT acnt_code, customer_name  -- ❌ Missing 'status'
         FROM table WHERE ...
     )
     SELECT acnt_code, status FROM my_cte;  -- ❌ ERROR: 'status' not in CTE!
     ```
   - **Example of CORRECT:**
     ```sql
     WITH my_cte AS (
         SELECT acnt_code, customer_name, status  -- ✅ Includes 'status'
         FROM table WHERE ...
     )
     SELECT acnt_code, status FROM my_cte;  -- ✅ OK
     ```
   - If columns are missing → **CRITICAL ERROR**

7. **Status Filtering (CRITICAL):**
   - For queries about current/active loans, is `status = 'O'` included?
   - Missing status filter is a **CRITICAL ERROR**

8. **Time-series handling (CRITICAL):**
   - For "latest N entities", did they use ROW_NUMBER() PARTITION BY?
   - Is the partitioning by the correct entity (customer_name vs acnt_code)?
   - Did they order by the right date column?
   
9. **Deduplication Check (CRITICAL):**
   - If user asks for "данснууд" (accounts), "зээлүүд" (loans), "компаниуд" (companies):
     * Does the SQL use `ROW_NUMBER() OVER (PARTITION BY acnt_code ORDER BY txn_date DESC)`?
     * Or does it use `FETCH FIRST 1 ROW ONLY` with `ORDER BY txn_date DESC`?
   - If NO deduplication when querying unique entities → **CRITICAL ERROR**
   - **Example of ERROR**: `SELECT * FROM loan_balance_detail WHERE customer_name = 'X' ORDER BY adv_date DESC` (missing deduplication!)

10. **Technical correctness:**
   - Oracle syntax correct?
   - **CURR_CODE ALWAYS INCLUDED** with amounts?
   - No syntax errors?
   - FETCH FIRST ... ROWS ONLY used correctly?

---
**AVAILABLE SCHEMA:**
{context_str}

**ORIGINAL QUESTION:**
{masked_query_str}

**GENERATOR'S EXPLANATION:**
{generator_explanation}

**GENERATOR'S SQL:**    
{sql_query}

**RELEVANT EXAMPLES:**
{examples}
---
**RESPONSE FORMAT (JSON ONLY):**

{
  "is_correct": true | false,
  "reasoning": "Brief explanation of your assessment",
  "corrected_query": "Fixed SQL here (empty string if no fix needed)"
}

**Your Review:**
```json

"""

# ============================================================================
# NEW ARCHITECTURE PROMPTS (Triage, Schema Analysis, Validation, Retry, General)
# ============================================================================

TRIAGE_PROMPT = """
You are a query classifier for the **Development Bank of Mongolia (DBM)** chatbot system.
Your task is to classify user questions into exactly one of three categories.

**Categories:**

1. **DATA_QUESTION** - Questions that require database access to answer:
   - Questions about loans, accounts, customers, balances, payments
   - Questions asking for specific data (amounts, dates, names, counts)
   - Questions with Mongolian banking terms: "зээл", "данс", "үлдэгдэл", "төлбөр", "харилцагч"
   - Follow-up questions referencing previous data (энэ, тэр, дээрх)
   - Examples: "Хамгийн өндөр зээлийн үлдэгдэл?", "Пирамид компанийн мэдээлэл"

2. **GENERAL_QUESTION** - Questions about data analysis, SQL concepts, or general knowledge:
   - Explaining SQL concepts (JOIN, GROUP BY, etc.)
   - Asking how something works conceptually
   - General data analysis questions not requiring specific DB queries
   - Examples: "SQL JOIN гэж юу вэ?", "Яаж зээлийн хүүг тооцоолдог вэ?"

3. **OUT_OF_SCOPE** - Questions completely unrelated to banking or data:
   - Weather, sports, entertainment, personal questions
   - Inappropriate or harmful requests
   - Questions about topics unrelated to DBM's business
   - Examples: "Цаг агаар ямар байна?", "Хөлбөмбөгийн тоглолт хэзээ вэ?"

**CRITICAL RULES:**
- Greetings like "Сайн байна уу" should be classified as **GENERAL_QUESTION**
- If unsure between DATA_QUESTION and GENERAL_QUESTION, prefer **DATA_QUESTION**
- References to previous conversation (энэ, тэр) with data context → **DATA_QUESTION**

**User Question:** {query_str}

**OUTPUT FORMAT (JSON ONLY):**
{{
  "query_type": "DATA_QUESTION" | "GENERAL_QUESTION" | "OUT_OF_SCOPE",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of classification"
}}

**Your Classification:**
"""

GENERAL_RESPONSE_PROMPT = """
You are a helpful Mongolian banking assistant for the **Development Bank of Mongolia (DBM)**.
Answer the user's general question clearly and professionally in Mongolian.

**Rules:**
- Provide clear, accurate answers about data and SQL/querying concepts
- Use examples when helpful
- Be professional but friendly
- If you don't know something, say so honestly
- Stay within the context of banking and data analysis

**User Question:** {query_str}

**Your Response:**
"""

OUT_OF_SCOPE_RESPONSE = """Уучлаарай, энэ асуулт Хөгжлийн Банкны мэдээллийн системтэй холбоогүй байна. 
Зээл, данс, харилцагчийн мэдээлэл гэх мэт банкны өгөгдөлтэй холбоотой асуултад хариулах боломжтой. 
Өөр асуулт байвал асууна уу."""

SCHEMA_ANALYSIS_PROMPT = """
You are a database schema analyst for the **Development Bank of Mongolia (DBM)**.
Analyze which tables and fields are needed to answer the user's question.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database. Do NOT filter by this name.

**CRITICAL TABLE SELECTION RULES:**
| Question Type | Correct Table | Keywords |
|---------------|---------------|----------|
| Future payments/schedules | `dbm.bcom_nrs_detail` | төлөгдөх, хуваарь, ирэх, дараа |
| Current balance/state | `dbm.loan_balance_detail` | үлдэгдэл, хариуцдаг, одоо |
| Historical/overdue | `dbm.loan_balance_detail` | хэтэрсэн, өмнөх |

**DEDUPLICATION AWARENESS:**
The `dbm.loan_balance_detail` table contains DAILY SNAPSHOTS:
- Each loan account has multiple rows (one per day via `txn_date`)
- Queries for unique accounts NEED deduplication

**AVAILABLE SCHEMA:**
{schema_context}

**RELEVANT EXAMPLES:**
{examples}

**CHAT HISTORY (if any):**
{history_context}

**USER QUESTION:**
{query_str}

**OUTPUT FORMAT (JSON ONLY):**
{{
  "in_scope": true | false,
  "out_of_scope_reason": "Reason if in_scope is false, empty string otherwise",
  "relevant_tables": [
    {{
      "table_name": "DBM.TABLE_NAME",
      "fields": ["field1", "field2"],
      "reason": "Why this table is needed"
    }}
  ],
  "relationships": ["table1.id = table2.id"],
  "needs_deduplication": true | false,
  "needs_chat_history": true | false,
  "chat_history_reasoning": "How previous context is used"
}}

**Your Analysis:**
"""

SQL_REGENERATE_PROMPT = """
You are an expert Oracle SQL developer for the **Development Bank of Mongolia (DBM)**.
The previous SQL query failed with an error. Generate a corrected query.

**PREVIOUS FAILED QUERY:**
```sql
{previous_sql}
```

**ERROR MESSAGE:**
{error_message}

**ORIGINAL QUESTION:**
{query_str}

**AVAILABLE SCHEMA:**
{schema}

**RELEVANT EXAMPLES:**
{examples}

**INSTRUCTIONS:**
1. Analyze why the previous query failed
2. Fix the specific error mentioned
3. Ensure Oracle syntax compliance
4. Include all necessary columns (especially cur_code with amounts)

**RESPONSE FORMAT:**

<explanation>
[Your explanation of the error and fix]
</explanation>

<sql>
[Your corrected Oracle SQL query]
</sql>

**Your Corrected Query:**
"""

VALIDATE_ANSWER_PROMPT = """
You are a quality checker for a database chatbot system.
Determine if the SQL query results adequately answer the user's question.

**USER QUESTION:**
{query_str}

**SQL QUERY EXECUTED:**
{sql_query}

**QUERY RESULTS:**
{results}

**VALIDATION RULES:**
1. If results are empty [] but the question expects data → NOT ANSWERED (may need retry or different approach)
2. If results contain "error" → NOT ANSWERED
3. If results contain data but don't match what was asked → NOT ANSWERED
4. If results contain relevant data that answers the question → ANSWERED
5. Most of the time, if there's data, it's adequate

**OUTPUT FORMAT (JSON ONLY):**
{{
  "is_answered": true | false,
  "reason": "Explanation (required if is_answered is false)",
  "suggestion": "How to improve the query (if applicable)"
}}

**Your Validation:**
"""

HISTORY_SUMMARY_PROMPT = """
Summarize the conversation history for context. Keep only essential information.

**Previous Turns (masked):**
{history}

**OUTPUT FORMAT (JSON ONLY):**
{{
  "summary": "Brief summary of what was discussed",
  "key_entities": ["entity names mentioned"],
  "last_sql_pattern": "Brief description of the last SQL query type"
}}

**Your Summary:**
"""


class PromptManager:
    """Enhanced prompt manager for Oracle database with Mongolian support"""
    
    def __init__(self, dialect: str = "oracle"):
        self.dialect = dialect
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize all prompt templates with Oracle-specific settings"""
        # Select appropriate SQL prompt based on dialect
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
        
    def get_text2sql_prompt(self) -> PromptTemplate:
        """Get the text-to-SQL prompt"""
        return self.text2sql_prompt

    def get_sql_review_prompt(self) -> PromptTemplate:
      """Get the SQL review prompt"""
      return self.sql_review_prompt
    
    def get_response_synthesis_prompt(self) -> PromptTemplate:
        if config.LLM_BACKEND in ("gemini", "openai"):
            logger.info("Using API-optimized (conversational) response prompt.")
            return self.api_response_synthesis_prompt
        else:
            logger.info("Using Local-optimized (direct) response prompt.")
            return self.response_synthesis_prompt
    def get_query_analyzer_prompt(self, examples: str = "No examples provided.") -> PromptTemplate:
        """Get the query analyzer prompt, now with examples."""
        return PromptTemplate(QUERY_ANALYZER_PROMPT).partial_format(examples=examples)
    def get_table_info_prompt(self) -> PromptTemplate:
        """Get the table info generation prompt"""
        return self.table_info_prompt


prompt_manager = PromptManager("oracle")


# Mongolian term mappings
MONGOLIAN_BUSINESS_TERMS = {
    "харилцагч": "customer",
    "зээл": "loan", 
    "батлагдсан хэмжээ": "approved_amount",
    "үлдэгдэл": "balance",
    "хамгийн сүүлийн": "latest",
    "нийт": "total",
    "дугаар": "number",
    "огноо": "date",
    "хэмжээ": "amount"
}