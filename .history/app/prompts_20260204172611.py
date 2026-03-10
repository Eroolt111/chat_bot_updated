from venv import logger
from typing import List, Dict, Set
from .config import config
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT

# =============================================================================
# MODULAR PROMPT BLOCKS - COMPOSABLE DOMAIN-SPECIFIC RULES
# =============================================================================
# Each block contains RULES + PATTERNS for a specific table/view
# The PromptComposer assembles only relevant blocks based on QueryAnalyzer output
# =============================================================================

# -----------------------------------------------------------------------------
# CORE RULES BLOCK - Always included in every SQL generation prompt
# -----------------------------------------------------------------------------
CORE_RULES_BLOCK = """
**CORE ORACLE SQL RULES (ALWAYS APPLY):**

1. **DYNAMIC EXAMPLES FIRST:** If the user's question is similar to a `RELEVANT EXAMPLE` provided below, **follow that SQL pattern exactly. Examples are proven correct**.

2. **CURRENCY HANDLING:**
   - Always SELECT `cur_code` with monetary values
   - Always `GROUP BY cur_code` when summing money
   - For ranking/ordering by amounts: `ORDER BY (amount * rate) DESC` (convert to MNT)

3. **TEXT MATCHING (CRITICAL):**
   - ALWAYS use `UPPER()` for text comparisons: `UPPER(customer_name) = UPPER('...')`
   - Exception: If ENTITY NAMES section provides exact uppercase names from database, use those directly

4. **DYNAMIC DATES:**
   - "энэ он" (this year) → `txn_date >= TRUNC(SYSDATE, 'YYYY')`
   - "энэ сар" (this month) → `txn_date >= TRUNC(SYSDATE, 'MM')`
   - "өнөөдөр" (today) → `txn_date = TRUNC(SYSDATE)`
   - Only use `TO_DATE` if user specifies exact year

5. **CHAT HISTORY (Follow-up with "энэ", "тэр"):**
   - You do NOT have access to specific ID/Code from previous turn
   - MUST re-calculate entity using CTE with previous logic
   - Extract exact columns/filters from [Previous SQL]

6. **IDENTITY RULE:**
   - If user mentions "Хөгжлийн Банк", "Development Bank", "DBM" → refers to entire database
   - DO NOT filter by these terms, query ALL data

7. **ALWAYS INCLUDE:** Entity names (customer_name, acnt_name) and cur_code in SELECT

8. USE ONLY ORACLE SQL SYNTAX. Schema is pruned to relevant tables/columns only.
"""

# -----------------------------------------------------------------------------
# LOAN_BALANCE_DETAIL BLOCK - Current State / Historical Data
# -----------------------------------------------------------------------------
LOAN_BALANCE_BLOCK = """
**━━━ DBM.LOAN_BALANCE_DETAIL RULES (Current State / Historical) ━━━**

 **PURPOSE:** Daily snapshots of loan states - customer info, balances, risk classification
   Use for: үлдэгдэл (balance), хариуцдаг (manages), хэтэрсэн (overdue), одоо (current), харилцагч (customer)

 **KEY COLUMNS:**
   - `txn_date`: Snapshot date (CRITICAL for time-based queries)
   - `acnt_code`: Unique loan account identifier
   - `customer_name`, `acnt_name`: Entity names
   - `principal`: Remaining principal balance
   - `adv_amount`: Disbursed amount | `adv_date`: Disbursement date
   - `cur_code`, `rate`: Currency and exchange rate
   - `status`: O=Open, C=Closed | `class_name`: Risk classification

 **DEDUPLICATION (CRITICAL):**
   This table has DAILY SNAPSHOTS - one row per loan per day!
   
   • **Latest State (Global Snapshot):**
     ```sql
     WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
     ```
   
   • **Unique Customers (not accounts):**
     ```sql
     ROW_NUMBER() OVER (PARTITION BY customer_name ORDER BY txn_date DESC) = 1
     ```

 **VALID LOAN FILTERING:**
   - **Current Balance:** `AND status = 'O' AND principal > 0`
   - **Historical/Disbursement:** `AND adv_amount > 0`
   - **Skip filter if:** User asks for "closed", "zero balance", or specific account by code

 **PATTERNS:**

   -- Single account lookup (Global Snapshot)
   SELECT * FROM DBM.LOAN_BALANCE_DETAIL 
   WHERE acnt_code = '{{ACNT_CODE}}' 
     AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)

   -- Customer's loans (case-insensitive match)
   SELECT customer_name, adv_date, adv_amount, cur_code
   FROM DBM.LOAN_BALANCE_DETAIL
   WHERE UPPER(customer_name) = UPPER('{{CUSTOMER_NAME}}')
     AND adv_date IS NOT NULL AND adv_amount > 0
     AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
   ORDER BY adv_date DESC

   -- Latest N unique customers with active loans
   SELECT customer_name, adv_date, adv_amount, cur_code, principal, status
   FROM (
       SELECT customer_name, adv_date, adv_amount, cur_code, principal, status,
              ROW_NUMBER() OVER (PARTITION BY customer_name ORDER BY adv_date DESC) as rn
       FROM DBM.LOAN_BALANCE_DETAIL
       WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
         AND customer_name IS NOT NULL 
         AND adv_date IS NOT NULL AND adv_amount > 0
         AND principal > 0
   ) t
   WHERE rn = 1
   ORDER BY adv_date DESC
   FETCH FIRST {{N}} ROWS ONLY
"""

# -----------------------------------------------------------------------------
# BCOM_NRS_DETAIL BLOCK - Future Payment Schedules
# -----------------------------------------------------------------------------
BCOM_NRS_BLOCK = """
**━━━ DBM.BCOM_NRS_DETAIL RULES (Future Payment Schedules) ━━━**

 **PURPOSE:** Future payment schedules for loans - scheduled dates and amounts
   Use for: төлөгдөх (to be paid), хуваарь (schedule), ирэх (upcoming), дараа (next)

 **KEY COLUMNS:**
   - `acnt_code`: Loan account identifier (join key with LOAN_BALANCE_DETAIL)
   - `nrs_version`: Schedule version (CRITICAL - must filter for MAX per account!)
   - `schd_date`: Scheduled payment date
   - `amount`: Scheduled principal payment
   - `int_amount`: Scheduled interest payment
   - `theor_bal`: Theoretical balance after payment
   - `cur_code`: Currency | `status`: O=Open

 **NRS_VERSION FILTERING (CRITICAL):**
   A loan can have multiple schedule versions! ALWAYS filter by latest version:
   
   ```sql
   WITH latest_ver AS (
       SELECT acnt_code, MAX(nrs_version) AS max_ver 
       FROM DBM.BCOM_NRS_DETAIL 
       GROUP BY acnt_code
   )
   SELECT t.* FROM DBM.BCOM_NRS_DETAIL t
   JOIN latest_ver lv ON t.acnt_code = lv.acnt_code AND t.nrs_version = lv.max_ver
   ```

 **PATTERNS:**

   -- Next payment in next 30 days (with latest schedule version)
   WITH latest_version_per_account AS (
       SELECT acnt_code, MAX(nrs_version) AS current_version
       FROM DBM.BCOM_NRS_DETAIL
       GROUP BY acnt_code
   )
   SELECT d.acnt_code, d.customer_name, d.schd_date, d.amount, d.int_amount
   FROM DBM.BCOM_NRS_DETAIL d
   INNER JOIN latest_version_per_account lv 
       ON d.acnt_code = lv.acnt_code AND d.nrs_version = lv.current_version
   WHERE d.status = 'O'
     AND d.schd_date BETWEEN SYSDATE AND SYSDATE + 30
   ORDER BY d.schd_date ASC

   -- Total scheduled payments for this year
   WITH latest_ver AS (
       SELECT acnt_code, MAX(nrs_version) AS max_ver 
       FROM DBM.BCOM_NRS_DETAIL GROUP BY acnt_code
   )
   SELECT SUM(t.amount) AS total_principal, SUM(t.int_amount) AS total_interest, t.cur_code
   FROM DBM.BCOM_NRS_DETAIL t
   JOIN latest_ver lv ON t.acnt_code = lv.acnt_code AND t.nrs_version = lv.max_ver
   WHERE t.status = 'O'
     AND t.schd_date >= TRUNC(SYSDATE, 'YYYY')
     AND t.schd_date < ADD_MONTHS(TRUNC(SYSDATE, 'YYYY'), 12)
   GROUP BY t.cur_code
"""

# -----------------------------------------------------------------------------
# LOAN_TXN BLOCK - Transaction History (Actual Payments Made)
# -----------------------------------------------------------------------------
LOAN_TXN_BLOCK = """
**━━━ DBM.LOAN_TXN RULES (Transaction History / Actual Payments) ━━━**

 **PURPOSE:** Actual transaction records - disbursements, repayments, fees
   Use for: гүйлгээ (transaction), төлсөн (paid), үндсэн төлбөр (principal payment), 
            хүүний төлбөр (interest payment), торгууль (penalty)

 **KEY COLUMNS:**
   - `txn_jrno`: Transaction journal number (unique per transaction batch)
   - `txn_date`: Transaction date (when payment actually occurred)
   - `acnt_code`: Loan account identifier (join key)
   - `txn_amount`: Transaction amount
   - `cur_code`, `currate`: Currency and exchange rate
   - `txn_code`: Transaction type code

 **TRANSACTION CODES (txn_code):**
   - `ADV` or similar: Loan disbursement/advancement
   - `PAY_PRINC`: Principal payment
   - `PAY_INT`: Interest payment
   - `PAY_FINE` or `PENALTY`: Penalty/fine payment

 **PATTERNS:**

   -- Principal payments this year
   SELECT txn_jrno, txn_date, acnt_code, txn_amount, cur_code, currate
   FROM DBM.LOAN_TXN
   WHERE txn_code = 'PAY_PRINC'
     AND txn_date >= TRUNC(SYSDATE, 'YYYY')
   ORDER BY txn_date DESC

   -- Total paid by customer this year
   SELECT SUM(txn_amount) AS total_paid, cur_code
   FROM DBM.LOAN_TXN
   WHERE acnt_code = '{{ACNT_CODE}}'
     AND txn_code IN ('PAY_PRINC', 'PAY_INT')
     AND txn_date >= TRUNC(SYSDATE, 'YYYY')
   GROUP BY cur_code

   -- All transactions for an account
   SELECT txn_jrno, txn_date, txn_code, txn_amount, cur_code
   FROM DBM.LOAN_TXN
   WHERE acnt_code = '{{ACNT_CODE}}'
   ORDER BY txn_date DESC
"""

# -----------------------------------------------------------------------------
# SECURITY_BALANCE_DETAIL BLOCK - Securities/Bonds Transactions
# -----------------------------------------------------------------------------
SECURITY_BALANCE_BLOCK = """
**━━━ DBM.SECURITY_BALANCE_DETAIL RULES (Securities / Bonds / Investments) ━━━**

  **PURPOSE:** Securities transactions, bonds, debt instruments - bought or issued by the bank
   Use for: үнэт цаас (securities), бонд (bond), өрийн бичиг (debt instrument), 
            хөрөнгө оруулалт (investment), арилжаа (trading), хэлцэл (deal)

  **KEY COLUMNS:**
   - `acnt_code`: Security account identifier
   - `security_code`: Unique security/bond identifier (e.g., 'W190417003', 'АСЕМ 1')
   - `txn_date`: Transaction/snapshot date (use for latest state queries)
   - `deal_date`: When the deal was executed
   - `start_date`, `end_date`: Security validity period
   - `cur_code`: Currency | `rate`: Exchange rate
   - `customer_name`: Counterparty/issuer name

  **FINANCIAL COLUMNS:**
   - `book_value`: Book value of the security
   - `face_value`: Nominal/face value
   - `net_amount`: Net transaction amount
   - `deal_quantity`: Deal amount or quantity
   - `int_rate`: Interest rate (%)
   - `int_inc`, `int_exp`: Interest income/expense
   - `trade_profit`, `trade_loss`: Trading profit/loss

  **KEY STATUS CODES:**

   **buy_sell (Transaction Direction):**
   - `B` = Buy (Худалдан авсан) - Bank purchased securities
   - `S` = Sell (Зарсан/Гаргасан) - Bank sold or issued securities

   **deal_status (Deal State):**
   - `N` = New (Шинэ)
   - `S` = Settled (Гүйцэтгэгдсэн)
   - `C` = Closed (Хаагдсан)
   - `D` = Cancelled (Цуцлагдсан)

   **status (Security State):**
   - `O` = Open (Нээлттэй/Идэвхтэй)
   - `N` = New (Шинэ)
   - `C` = Closed (Хаагдсан)
   - `D` = Cancelled (Цуцлагдсан)

   **invest_type (Investment Classification):**
   - `A` = Available-for-sale (Худалдаанд бэлэн)
   - `T` = Trading (Арилжааны)
   - `H` = Held-to-maturity (Хугацаа дуустал эзэмших)
   - `L` = Loan and receivable (Зээл ба авлага)

   **prod_type (Portfolio Type):**
   - `BANK_PORTFOLIO` = Securities bought by bank (Банкны худалдан авсан)
   - `ISSUER_PORTFOLIO` = Securities issued by bank (Банкны гаргасан)

   **type (Account Type):**
   - `DEAL` = Deal account (Хэлцлийн данс)
   - `PROGRAM` = Program account (Программын данс)
   - `SERIES` = Series account (Серийн данс)

  **DEDUPLICATION:**
   This table may have daily snapshots. For latest state:
   ```sql
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.SECURITY_BALANCE_DETAIL)
   ```

  **PATTERNS:**

   -- All active/open securities (latest state)
   SELECT acnt_code, acnt_name, security_code, customer_name, 
          book_value, face_value, cur_code, status, invest_type
   FROM DBM.SECURITY_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.SECURITY_BALANCE_DETAIL)
     AND status = 'O'
   ORDER BY book_value DESC

   -- Securities bought by bank (BANK_PORTFOLIO)
   SELECT acnt_code, security_code, customer_name, deal_date, 
          net_amount, cur_code, int_rate, end_date
   FROM DBM.SECURITY_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.SECURITY_BALANCE_DETAIL)
     AND prod_type = 'BANK_PORTFOLIO'
     AND buy_sell = 'B'
   ORDER BY deal_date DESC

   -- Securities/bonds issued by bank (ISSUER_PORTFOLIO)
   SELECT acnt_code, acnt_name, security_code, customer_name, 
          net_amount, cur_code, int_rate, start_date, end_date
   FROM DBM.SECURITY_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.SECURITY_BALANCE_DETAIL)
     AND prod_type = 'ISSUER_PORTFOLIO'
     AND buy_sell = 'S'
   ORDER BY start_date DESC

   -- Total securities by currency (active only)
   SELECT cur_code, 
          SUM(book_value) AS total_book_value,
          SUM(net_amount) AS total_net_amount,
          COUNT(*) AS security_count
   FROM DBM.SECURITY_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.SECURITY_BALANCE_DETAIL)
     AND status = 'O'
   GROUP BY cur_code
   ORDER BY SUM(book_value * rate) DESC

   -- Securities maturing this year
   SELECT acnt_code, security_code, customer_name, 
          end_date, net_amount, cur_code, int_rate
   FROM DBM.SECURITY_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.SECURITY_BALANCE_DETAIL)
     AND status = 'O'
     AND end_date >= TRUNC(SYSDATE, 'YYYY')
     AND end_date < ADD_MONTHS(TRUNC(SYSDATE, 'YYYY'), 12)
   ORDER BY end_date ASC

   -- Search by counterparty name
   SELECT acnt_code, security_code, acnt_name, customer_name,
          net_amount, cur_code, deal_date, status
   FROM DBM.SECURITY_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.SECURITY_BALANCE_DETAIL)
     AND UPPER(customer_name) LIKE UPPER('%{{CUSTOMER_NAME}}%')
   ORDER BY deal_date DESC

   -- Interest income from securities this year
   SELECT security_code, customer_name, int_inc, cur_code
   FROM DBM.SECURITY_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.SECURITY_BALANCE_DETAIL)
     AND int_inc > 0
   ORDER BY (int_inc * rate) DESC
"""

# -----------------------------------------------------------------------------
# PLACEMENT_BALANCE_DETAIL BLOCK - Bank's Placements (байршуулсан эх үүсвэр)
# -----------------------------------------------------------------------------
PLACEMENT_BALANCE_BLOCK = """
**━━━ DBM.PLACEMENT_BALANCE_DETAIL RULES (Байршуулсан эх үүсвэр - Money Placed/Lent Out) ━━━**

 **PURPOSE:** Daily snapshots of placements made BY the bank TO other institutions.
   This is money the bank has PLACED or LENT OUT to other banks/institutions.
   Bank EARNS interest (хүүний орлого) from placements.

   **KEYWORDS → Use this table when user says:**
   - байршуулсан эх үүсвэр, байршуулсан зээл, байршуулсан хадгаламж
   - байршуулалт, зээлүүлсэн (to other institutions, NOT customers)

 **⚠️ DISTINGUISH FROM OTHER TABLES:**
   - PLACEMENT (this) = Bank PLACES money TO other institutions (bank EARNS interest)
   - BORROWING = Bank BORROWS money FROM other institutions (bank PAYS interest)
   - LOAN (loan_balance_detail) = Bank lends to CUSTOMERS (completely different!)

 **MODULE FILTERING (CRITICAL):**
   - `module = 'LD'` → Зээл (Long-term loan placements)
   - `module = 'MM'` → Хадгаламж (Money Market / short-term deposits)

   **FILTERING RULES:**
   - "байршуулсан зээл" → `AND module = 'LD'`
   - "байршуулсан хадгаламж" → `AND module = 'MM'`
   - "байршуулсан эх үүсвэр" → NO module filter (includes BOTH)

 **KEY COLUMNS:**
   - `module`: LD=Зээл (loan), MM=Хадгаламж (deposit) **FILTER BY THIS!**
   - `txn_date`: Snapshot date (for latest state: MAX(txn_date))
   - `acnt_code`: Unique placement account identifier
   - `acnt_name`, `cust_name`: Counterparty institution names
   - `principal`: Remaining principal balance (money placed)
   - `cur_code`, `rate`: Currency and exchange rate
   - `int_rate`: Interest rate (%) | `int_rcv`: Accrued interest receivable (INCOME)
   - `maturity_date`: When placement matures
   - `status`: O=Open, C=Closed, N=New, D=Deleted

 **DEDUPLICATION:** Daily snapshots - use global snapshot pattern:
   ```sql
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.PLACEMENT_BALANCE_DETAIL)
   ```

 **PATTERNS:**

   -- Байршуулсан эх үүсвэр (ALL - no module filter)
   SELECT acnt_code, acnt_name, cust_name, principal, cur_code, int_rate, maturity_date, module
   FROM DBM.PLACEMENT_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.PLACEMENT_BALANCE_DETAIL)
     AND status = 'O' AND principal > 0
   ORDER BY (principal * rate) DESC

   -- Байршуулсан зээл (LD module only)
   SELECT acnt_code, acnt_name, cust_name, principal, cur_code, int_rate, maturity_date
   FROM DBM.PLACEMENT_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.PLACEMENT_BALANCE_DETAIL)
     AND module = 'LD'
     AND status = 'O' AND principal > 0
   ORDER BY (principal * rate) DESC

   -- Байршуулсан хадгаламж (MM module only)
   SELECT acnt_code, acnt_name, cust_name, principal, cur_code, int_rate, maturity_date
   FROM DBM.PLACEMENT_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.PLACEMENT_BALANCE_DETAIL)
     AND module = 'MM'
     AND status = 'O' AND principal > 0
   ORDER BY (principal * rate) DESC

   -- Total by module
   SELECT module, cur_code, SUM(principal) AS total_principal, COUNT(*) AS cnt
   FROM DBM.PLACEMENT_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.PLACEMENT_BALANCE_DETAIL)
     AND status = 'O' AND principal > 0
   GROUP BY module, cur_code
   ORDER BY module, SUM(principal * rate) DESC
"""

# -----------------------------------------------------------------------------
# BORROWING_BALANCE_DETAIL BLOCK - Bank's Borrowings (татсан эх үүсвэр)
# -----------------------------------------------------------------------------
BORROWING_BALANCE_BLOCK = """
**━━━ DBM.BORROWING_BALANCE_DETAIL RULES (Татсан эх үүсвэр - Money Borrowed/Received) ━━━**

 **PURPOSE:** Daily snapshots of borrowings received BY the bank FROM other institutions.
   This is money the bank has BORROWED or received as funding from others.
   Bank PAYS interest (хүүний зардал) on borrowings.

   **KEYWORDS → Use this table when user says:**
   - татсан эх үүсвэр, татсан зээл, татсан хадгаламж
   - татан төвлөрүүлсэн, зээлж авсан, өр

 **⚠️ DISTINGUISH FROM OTHER TABLES:**
   - BORROWING (this) = Bank BORROWS money FROM other institutions (bank PAYS interest)
   - Татсан зээл = BORROWING, not LOAN or PLACEMENT, LOAN = Normally customer loans
   - PLACEMENT = Bank PLACES money TO other institutions (bank EARNS interest)
   - LOAN (loan_balance_detail) = Bank lends to CUSTOMERS (completely different!)

 **MODULE FILTERING (CRITICAL):**
   - `module = 'LD'` → Зээл (Long-term borrowings: АХБ, JICA, World Bank, etc.)
   - `module = 'MM'` → Хадгаламж (Money Market / short-term borrowings)

   **FILTERING RULES:**
   - "татсан зээл" → `AND module = 'LD'`
   - "татсан хадгаламж" → `AND module = 'MM'`
   - "татсан эх үүсвэр" → NO module filter (includes BOTH)

 **KEY COLUMNS:**
   - `module`: LD=Зээл (loan), MM=Хадгаламж (deposit) **FILTER BY THIS!**
   - `txn_date`: Snapshot date (for latest state: MAX(txn_date))
   - `acnt_code`: Unique borrowing account identifier
   - `acnt_name`, `cust_name`: Lender institution names
   - `principal`: Remaining principal balance (outstanding debt)
   - `cur_code`, `rate`: Currency and exchange rate
   - `int_rate`: Interest rate (%) | `int_exp`: Interest expense (bank PAYS this)
   - `int_pay`: Accrued interest payable
   - `maturity_date`: Repayment due date
   - `status`: O=Open, C=Closed, N=New, D=Deleted

 **DEDUPLICATION:** Daily snapshots - use global snapshot pattern:
   ```sql
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.BORROWING_BALANCE_DETAIL)
   ```

 **PATTERNS:**

   -- Татсан эх үүсвэр (ALL - no module filter)
   SELECT acnt_code, acnt_name, cust_name, principal, cur_code, int_rate, maturity_date, module
   FROM DBM.BORROWING_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.BORROWING_BALANCE_DETAIL)
     AND status = 'O' AND principal > 0
   ORDER BY (principal * rate) DESC

   -- Татсан зээл (LD module only - АХБ, JICA, World Bank, etc.)
   SELECT acnt_code, acnt_name, cust_name, principal, cur_code, int_rate, maturity_date
   FROM DBM.BORROWING_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.BORROWING_BALANCE_DETAIL)
     AND module = 'LD'
     AND status = 'O' AND principal > 0
   ORDER BY (principal * rate) DESC

   -- Татсан хадгаламж (MM module only)
   SELECT acnt_code, acnt_name, cust_name, principal, cur_code, int_rate, maturity_date
   FROM DBM.BORROWING_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.BORROWING_BALANCE_DETAIL)
     AND module = 'MM'
     AND status = 'O' AND principal > 0
   ORDER BY (principal * rate) DESC

   -- Total by module
   SELECT module, cur_code, SUM(principal) AS total_principal, COUNT(*) AS cnt
   FROM DBM.BORROWING_BALANCE_DETAIL
   WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.BORROWING_BALANCE_DETAIL)
     AND status = 'O' AND principal > 0
   GROUP BY module, cur_code
   ORDER BY module, SUM(principal * rate) DESC
"""

# -----------------------------------------------------------------------------
# MULTI-TABLE PATTERNS - When queries require joining multiple tables
# -----------------------------------------------------------------------------
MULTI_TABLE_PATTERNS_BLOCK = """
**━━━ MULTI-TABLE QUERY PATTERNS ━━━**

 **JOIN KEY:** All tables join on `acnt_code`

**PATTERN 1: Current Balance + Next Payment (LOAN_BALANCE + BCOM_NRS)**
```sql
WITH latest_snapshot AS (
    SELECT * FROM DBM.LOAN_BALANCE_DETAIL 
    WHERE acnt_code = '{{ACNT_CODE}}'
      AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
),
current_version AS (
    SELECT MAX(nrs_version) AS latest_nrs_version
    FROM DBM.BCOM_NRS_DETAIL WHERE acnt_code = '{{ACNT_CODE}}'
),
next_payment AS (
    SELECT schd_date, amount, int_amount
    FROM DBM.BCOM_NRS_DETAIL t
    WHERE t.acnt_code = '{{ACNT_CODE}}'
      AND t.nrs_version = (SELECT latest_nrs_version FROM current_version)
      AND t.status = 'O' AND t.schd_date > SYSDATE
    ORDER BY t.schd_date ASC FETCH FIRST 1 ROW ONLY
)
SELECT ls.*, np.schd_date AS next_payment_date, np.amount AS next_principal, np.int_amount AS next_interest
FROM latest_snapshot ls LEFT JOIN next_payment np ON 1=1
```

**PATTERN 2: Remaining Payments (Scheduled - Paid) (BCOM_NRS + LOAN_TXN)**
Use when user asks: "үлдсэн төлбөр" (remaining payment), "төлөгдөх ёстой" (due to be paid)
Formula: SCHEDULED (from BCOM_NRS) - ALREADY_PAID (from LOAN_TXN) = REMAINING

```sql
WITH scheduled AS (
    -- Get scheduled payments from BCOM_NRS_DETAIL (future amounts)
    SELECT t.acnt_code, t.cur_code, 
           SUM(t.amount) AS scheduled_principal,
           SUM(t.int_amount) AS scheduled_interest
    FROM DBM.BCOM_NRS_DETAIL t
    INNER JOIN (
        SELECT acnt_code, MAX(nrs_version) AS max_ver 
        FROM DBM.BCOM_NRS_DETAIL GROUP BY acnt_code
    ) lv ON t.acnt_code = lv.acnt_code AND t.nrs_version = lv.max_ver
    WHERE t.status = 'O'
      AND t.schd_date >= TRUNC(SYSDATE, 'YYYY')  -- This year
      AND t.schd_date < ADD_MONTHS(TRUNC(SYSDATE, 'YYYY'), 12)
    GROUP BY t.acnt_code, t.cur_code
),
paid AS (
    -- Get actual payments from LOAN_TXN (what was already paid)
    SELECT acnt_code, cur_code,
           SUM(CASE WHEN txn_code = 'PAY_PRINC' THEN txn_amount ELSE 0 END) AS paid_principal,
           SUM(CASE WHEN txn_code = 'PAY_INT' THEN txn_amount ELSE 0 END) AS paid_interest
    FROM DBM.LOAN_TXN
    WHERE txn_date >= TRUNC(SYSDATE, 'YYYY')  -- This year
    GROUP BY acnt_code, cur_code
)
SELECT s.acnt_code, s.cur_code,
       s.scheduled_principal,
       NVL(p.paid_principal, 0) AS paid_principal,
       (s.scheduled_principal - NVL(p.paid_principal, 0)) AS remaining_principal,
       s.scheduled_interest,
       NVL(p.paid_interest, 0) AS paid_interest,
       (s.scheduled_interest - NVL(p.paid_interest, 0)) AS remaining_interest
FROM scheduled s
LEFT JOIN paid p ON s.acnt_code = p.acnt_code AND s.cur_code = p.cur_code
WHERE (s.scheduled_principal - NVL(p.paid_principal, 0)) > 0  -- Only show with remaining balance
ORDER BY (s.scheduled_principal - NVL(p.paid_principal, 0)) DESC
```

**PATTERN 3: Account Summary (All Three Tables)**
```sql
WITH account_info AS (
    SELECT acnt_code, customer_name, principal, adv_amount, adv_date, cur_code, status
    FROM DBM.LOAN_BALANCE_DETAIL
    WHERE acnt_code = '{{ACNT_CODE}}'
      AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
),
total_paid AS (
    SELECT acnt_code, SUM(txn_amount) AS total_paid
    FROM DBM.LOAN_TXN
    WHERE acnt_code = '{{ACNT_CODE}}' AND txn_code IN ('PAY_PRINC', 'PAY_INT')
    GROUP BY acnt_code
),
next_sched AS (
    SELECT acnt_code, schd_date AS next_payment_date, amount + int_amount AS next_amount
    FROM DBM.BCOM_NRS_DETAIL
    WHERE acnt_code = '{{ACNT_CODE}}'
      AND nrs_version = (SELECT MAX(nrs_version) FROM DBM.BCOM_NRS_DETAIL WHERE acnt_code = '{{ACNT_CODE}}')
      AND schd_date > SYSDATE AND status = 'O'
    ORDER BY schd_date FETCH FIRST 1 ROW ONLY
)
SELECT a.*, tp.total_paid, ns.next_payment_date, ns.next_amount
FROM account_info a
LEFT JOIN total_paid tp ON a.acnt_code = tp.acnt_code
LEFT JOIN next_sched ns ON a.acnt_code = ns.acnt_code
```

**PATTERN 4: Эх үүсвэр Overview (PLACEMENT + BORROWING)**
Use when user asks: "эх үүсвэр" (source of funds), "татсан болон байршуулсан" (borrowed and placed)
Shows BOTH sides: money placed out (байршуулсан) and money borrowed in (татсан)

```sql
-- Байршуулсан эх үүсвэр (bank PLACED to others - earns interest)
SELECT 'БАЙРШУУЛСАН' AS type, acnt_code, acnt_name, cust_name, principal, cur_code, int_rate, maturity_date, module
FROM DBM.PLACEMENT_BALANCE_DETAIL
WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.PLACEMENT_BALANCE_DETAIL)
  AND status = 'O' AND principal > 0

UNION ALL

-- Татсан эх үүсвэр (bank BORROWED from others - pays interest)
SELECT 'ТАТСАН' AS type, acnt_code, acnt_name, cust_name, principal, cur_code, int_rate, maturity_date, module
FROM DBM.BORROWING_BALANCE_DETAIL
WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.BORROWING_BALANCE_DETAIL)
  AND status = 'O' AND principal > 0

ORDER BY type, 5 DESC
```

**PATTERN 5: Хадгаламж (MM module) - Both directions**
Use when user asks about "хадгаламж" (deposits/money market)

```sql
-- Байршуулсан хадгаламж (MM module)
SELECT 'БАЙРШУУЛСАН ХАДГАЛАМЖ' AS type, acnt_code, acnt_name, cust_name, principal, cur_code, int_rate
FROM DBM.PLACEMENT_BALANCE_DETAIL
WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.PLACEMENT_BALANCE_DETAIL)
  AND module = 'MM' AND status = 'O' AND principal > 0

UNION ALL

-- Татсан хадгаламж (MM module)
SELECT 'ТАТСАН ХАДГАЛАМЖ' AS type, acnt_code, acnt_name, cust_name, principal, cur_code, int_rate
FROM DBM.BORROWING_BALANCE_DETAIL
WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.BORROWING_BALANCE_DETAIL)
  AND module = 'MM' AND status = 'O' AND principal > 0

ORDER BY type, 5 DESC
```

**PATTERN 6: Зээл (LD module) - Both directions**
Use when user asks about long-term loans (зээл) in both directions

```sql
-- Байршуулсан зээл (LD module)
SELECT 'БАЙРШУУЛСАН ЗЭЭЛ' AS type, acnt_code, acnt_name, cust_name, principal, cur_code, int_rate
FROM DBM.PLACEMENT_BALANCE_DETAIL
WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.PLACEMENT_BALANCE_DETAIL)
  AND module = 'LD' AND status = 'O' AND principal > 0

UNION ALL

-- Татсан зээл (LD module - АХБ, JICA, etc.)
SELECT 'ТАТСАН ЗЭЭЛ' AS type, acnt_code, acnt_name, cust_name, principal, cur_code, int_rate
FROM DBM.BORROWING_BALANCE_DETAIL
WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.BORROWING_BALANCE_DETAIL)
  AND module = 'LD' AND status = 'O' AND principal > 0

ORDER BY type, 5 DESC
```
"""

# -----------------------------------------------------------------------------
# TABLE MAPPING - Maps table names to their blocks
# -----------------------------------------------------------------------------
DOMAIN_BLOCKS = {
    "dbm.loan_balance_detail": LOAN_BALANCE_BLOCK,
    "dbm.bcom_nrs_detail": BCOM_NRS_BLOCK,
    "dbm.loan_txn": LOAN_TXN_BLOCK,
    "dbm.security_balance_detail": SECURITY_BALANCE_BLOCK,
    "dbm.placement_balance_detail": PLACEMENT_BALANCE_BLOCK,
    "dbm.borrowing_balance_detail": BORROWING_BALANCE_BLOCK,
}


# =============================================================================
# PROMPT COMPOSER - Dynamically assembles prompts based on required tables
# =============================================================================
class PromptComposer:
    """
    Composes domain-specific prompts by selecting only relevant table blocks
    based on QueryAnalyzer's required_tables output.
    """
    
    def __init__(self):
        self.domain_blocks = DOMAIN_BLOCKS
        self.core_rules = CORE_RULES_BLOCK
        self.multi_table_patterns = MULTI_TABLE_PATTERNS_BLOCK
    
    def compose_domain_rules(self, required_tables: List[str]) -> str:
        """
        Compose domain-specific rules block based on required tables.
        
        Args:
            required_tables: List of table names from QueryAnalyzer 
                            (e.g., ["dbm.loan_balance_detail", "dbm.bcom_nrs_detail"])
        
        Returns:
            Composed rules string with only relevant table blocks
        """
        if not required_tables:
            # If no tables specified, include all blocks as fallback
            logger.warning("No required_tables specified. Including all domain blocks.")
            required_tables = list(self.domain_blocks.keys())
        
        # Normalize table names to lowercase for matching
        normalized_tables = [t.lower().strip() for t in required_tables]
        
        # Collect relevant blocks
        selected_blocks = []
        for table_name in normalized_tables:
            if table_name in self.domain_blocks:
                selected_blocks.append(self.domain_blocks[table_name])
            else:
                logger.warning(f"Unknown table '{table_name}' - no domain block available")
        
        # Build the composed rules
        composed_rules = self.core_rules + "\n"
        
        # Add table-specific blocks
        if selected_blocks:
            composed_rules += "\n".join(selected_blocks)
        
        # Add multi-table patterns if multiple tables are selected
        if len(normalized_tables) > 1:
            composed_rules += "\n" + self.multi_table_patterns
        
        return composed_rules
    
    def get_table_selection_hint(self, required_tables: List[str]) -> str:
        """
        Generate a hint about which tables were selected and why they should be used.
        """
        if not required_tables:
            return ""
        
        table_names = ", ".join(required_tables)
        return f"\n**📌 SELECTED TABLES:** {table_names}\nUse ONLY these tables. The analyzer has determined these are sufficient for this query.\n"


# Global instance
prompt_composer = PromptComposer()


# =============================================================================
# TABLE INFO GENERATION (unchanged - needed for setup)
# =============================================================================
TABLE_INFO_PROMPT = """
Give me a summary of the table with the following JSON format:

{
  "table_name": "...",
  "table_summary": "...",
  "column_descriptions": [
  {"column_name": "column1", "description": "description..."},
  {"column_name": "column2", "description": "description..."}
  ]
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
# MODULAR TEXT-TO-SQL PROMPT TEMPLATE
# Uses dynamically composed domain rules from PromptComposer
# =============================================================================
MODULAR_TEXT_TO_SQL_TEMPLATE = """
You are an expert Oracle SQL developer for the **Development Bank of Mongolia (DBM)**, which is known as **"Хөгжлийн Банк"** in Mongolian.

{domain_rules}

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
[Your reasoning - mention which example/pattern you're following]
</explanation>
<sql>
[Oracle SQL]
</sql>
"""

# =============================================================================
# LEGACY TEXT-TO-SQL PROMPT (kept for backward compatibility)
# This is the full monolithic prompt - use MODULAR_TEXT_TO_SQL_TEMPLATE instead
# =============================================================================
ORACLE_TEXT_TO_SQL_PROMPT = """
You are an expert Oracle SQL developer for the **Development Bank of Mongolia (DBM)**, which is known as **"Хөгжлийн Банк"** in Mongolian.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database you are querying. **DO NOT** filter `customer_name` or any other column by these terms. Instead, perform the requested action (e.g., find the top customers) across all data.


**CORE RULES:**

1. (IMPORTANT RULE) **DYNAMIC EXAMPLES:** If the user's question is similar to a `RELEVANT EXAMPLE` provided below, **follow that SQL pattern exactly. Examples are proven correct**.

2. **TABLE SELECTION:**
   - Future/schedules (төлөгдөх, хуваарь, ирэх) → `BCOM_NRS_DETAIL`
   - Current/past (үлдэгдэл, хариуцдаг, хэтэрсэн) → `LOAN_BALANCE_DETAIL`
   - Event-based, Transaction history, actual payments (гүйлгээ, төлсөн, төлбөр хийсэн) → `LOAN_TXN`
3. **DEDUPLICATION (CRITICAL):**
   `loan_balance_detail` has DAILY SNAPSHOTS (one row per day per loan).
   - Latest state: `WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)` there is only one loan account per day, so acnt_names won't be duplicated on that date.
   - Unique customers: `ROW_NUMBER() OVER (PARTITION BY customer_name ORDER BY txn_date DESC)` etc.
4. **CURRENCY RANKING (THE "RATE" RULE):** - NEVER order or compare raw monetary columns (principal, adv_amount) directly.
   - **ALWAYS** convert to MNT using `amount * rate` (e.g., `ORDER BY (adv_amount * rate) DESC`).
   - This applies to **main queries AND subqueries/CTEs**.

5. **🧹 VALID LOAN FILTERING (CRITICAL - ALWAYS APPLY):**
   Many loan accounts exist with `principal = 0` (created but not disbursed, or fully paid off).
   **To get REAL/ACTIVE loans, you MUST filter:**
   - **Current Balance Queries:** `AND status = 'O' AND principal > 0` (exclude zero-balance accounts, closed loans)
   - You can filter those > 0 almost defaulty, except user specifically asks for closed/zero-balance loans.
   - **Historical/Disbursement Queries:** `AND adv_amount > 0` (exclude never-disbursed loans)
   - **Name Filters:** `AND customer_name IS NOT NULL AND acnt_name IS NOT NULL`
   
   **WHEN TO SKIP THIS FILTER:**
   - User explicitly asks for "paid off", "closed", "zero balance", or "бүх" (all) loans
   - User asks about a specific account by code (e.g., `acnt_code = 'XXX'`)
   
   *Examples:*
   - "Хамгийн их зээлтэй харилцагч?" → `WHERE principal > 0` (active loans only)
   - "Хамгийн анхны зээлдэгч?" → `WHERE adv_amount > 0` (actually disbursed)
   - "Энэ дансны мэдээлэл" → No filter needed (specific account lookup)

6. **STATUS FILTERING:**
   - Current state queries (үлдэгдэл, идэвхтэй): ADD `status = 'O'`
   - Historical queries (хамгийн анх, бүх): NO status filter
7. **CURRENCY:**
   - Always SELECT `cur_code` with monetary values
8. **AGGREGATIONS:** 
   - Always `GROUP BY cur_code` when summing money.
9. **CHAT HISTORY (Follow-up Questions with "энэ", "тэр"):**
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
10. Always select latest nrs_version for BCOM_NRS_DETAIL queries. MAX(nrs_version) per acnt_code.
11. **CASE-INSENSITIVE TEXT MATCHING (CRITICAL!):**
    **ALWAYS use UPPER() for text comparisons.** Database values may be in different cases.
    - CORRECT: `UPPER(customer_name) = UPPER('Эрчим хүчний яам')`
    - CORRECT: `UPPER(customer_name) LIKE UPPER('%МИАТ%')`
    - WRONG: `customer_name = 'Эрчим хүчний яам'` (case mismatch = 0 results!)
    - **Exception:** Only skip UPPER() if ENTITY NAMES section provides the EXACT uppercase name from database.
12. **ENTITY NAMES:** If ENTITY NAMES section provides names, use them EXACTLY as shown (they're already correct case).
13. **ALWAYS INCLUDE:** Entity names (customer_name, acnt_name) and cur_code in SELECT.
14. USE ONLY ORACLE SQL SYNTAX.
15. Schema data is pruned to only relevant tables/columns. if examples use missing columns, adapt accordingly.
16. **DYNAMIC DATE HANDLING (CRITICAL):**
    - If the user says "энэ он" (this year), "энэ сар" (this month), or "өнөөдөр" (today):
      **DO NOT HARDCODE YEARS (e.g., '2024', '2025').**
    - **ALWAYS** use Oracle dynamic functions:
      - "This year" → `txn_date >= TRUNC(SYSDATE, 'YYYY')`
      - "This month" → `txn_date >= TRUNC(SYSDATE, 'MM')`
      - "Today" → `txn_date = TRUNC(SYSDATE)`
    - Only use `TO_DATE` if the user explicitly types a specific year (e.g., "2023 оны").

**EXAMPLE SQL PATTERNS:**
• ========== SINGLE ENTITY LOOKUPS ==========
  # Get latest state of ONE account (Global Snapshot)
  SELECT * FROM DBM.LOAN_BALANCE_DETAIL 
  WHERE acnt_code = '{{ACNT_CODE}}' 
    AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
  --FETCH FIRST 1 ROW ONLY; -- Optional for single account lookup, as acnt_code is unique per day
  
# Get latest state of ONE customer's loans (Global Snapshot)
  -- ALWAYS use UPPER() for text matching!
  SELECT customer_name, adv_date, adv_amount, cur_code
  FROM DBM.LOAN_BALANCE_DETAIL
  WHERE UPPER(customer_name) = UPPER('{{CUSTOMER_NAME}}')
    AND adv_date IS NOT NULL AND adv_amount > 0
    AND txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
  ORDER BY adv_date DESC

 # Latest 5 customers (NOT accounts!) with active loans
  SELECT customer_name, adv_date, adv_amount, cur_code, principal, status
  FROM (
      SELECT customer_name, adv_date, adv_amount, cur_code, principal, status,
            ROW_NUMBER() OVER (PARTITION BY customer_name ORDER BY adv_date DESC) as rn
      FROM DBM.LOAN_BALANCE_DETAIL
      WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
        AND customer_name IS NOT NULL 
        AND adv_date IS NOT NULL AND adv_amount > 0
        AND principal > 0  -- Only active loans with balance
  ) t
  WHERE rn = 1
  ORDER BY adv_date DESC
  FETCH FIRST 5 ROWS ONLY;
  
  # Latest 3 loan accounts with active balance
  SELECT customer_name, adv_date, adv_amount, cur_code, txn_date, acnt_code, principal, status
  FROM DBM.LOAN_BALANCE_DETAIL
  WHERE txn_date = (SELECT MAX(txn_date) FROM DBM.LOAN_BALANCE_DETAIL)
      AND acnt_name IS NOT NULL
      AND adv_date IS NOT NULL AND adv_amount > 0
      AND principal > 0  -- Only active loans with balance
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

• ========== PRINCIPAL PAYMENT (DBM.LOAN_TXN) ==========
  # "This year principal payments" (Энэ онд төлөгдсөн...) -> USE SYSDATE!
  SELECT txn_jrno, txn_date, acnt_code, txn_amount, cur_code, currate
  FROM DBM.LOAN_TXN
  WHERE txn_code = 'PAY_PRINC'
    AND txn_date >= TRUNC(SYSDATE, 'YYYY') -- Dynamic start of current year
  ORDER BY txn_date DESC;

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
# STREAMLINED QUERY ANALYZER - Example-Driven Table Selection
# =============================================================================
QUERY_ANALYZER_PROMPT = """
You are an expert query analyst for a financial database chatbot for the **Development Bank of Mongolia (DBM)**.

**CRITICAL IDENTITY RULE:**
If the user's question mentions "Хөгжлийн Банк", "Development Bank", "DBM", or "the bank", they are referring to the entire database. **DO NOT** create a plan step that filters by this name. The query should apply to all data.

**YOUR PRIMARY TASK: SELECT THE CORRECT TABLES**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE #1: EXAMPLES ARE YOUR BEST GUIDE 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**BEFORE selecting tables, SEARCH the EXAMPLES section below!**

1. Find examples with SIMILAR LOGIC to the user's question
2. Look at which tables those examples use in their SQL
3. **USE THE EXACT SAME TABLES** as the matching example

**Example Matching Process:**
- User asks: "Энэ онд төлөгдөх ёстой үлдсэн төлбөрийг харуул" (remaining payments due this year)
- Find example: "Энэ онд төлөгдөх хуваарьт төлбөр" uses BCOM_NRS_DETAIL + LOAN_TXN
- Reasoning: "remaining" = scheduled - paid, so need BOTH tables
- Output: required_tables = ["dbm.bcom_nrs_detail", "dbm.loan_txn"]

**CRITICAL:** If an example uses MULTIPLE tables, you MUST also select MULTIPLE tables!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**TABLE SELECTION GUIDE (if no matching example found):**

| User Intent | Keywords | Table(s) Required |
|-------------|----------|-------------------|
| Future payments/schedules | төлөгдөх, хуваарь, ирэх, дараагийн төлбөр | `dbm.bcom_nrs_detail` |
| Current balance/state | Зээлийн үлдэгдэл, хариуцдаг, хэтэрсэн, одоо, principal | `dbm.loan_balance_detail` |
| Actual payments made | гүйлгээ, төлсөн, төлбөр хийсэн | `dbm.loan_txn` |
| **Remaining to pay** | үлдсэн төлбөр, төлөгдөх ёстой | `dbm.bcom_nrs_detail` + `dbm.loan_txn` |
| Account + next payment | дансны мэдээлэл + дараагийн төлбөр | `dbm.loan_balance_detail` + `dbm.bcom_nrs_detail` |
| Payment history + balance | төлөлтийн түүх + үлдэгдэл | `dbm.loan_balance_detail` + `dbm.loan_txn` |
| **Securities/Bonds** | үнэт цаас, бонд, өрийн бичиг, хөрөнгө оруулалт, арилжаа | `dbm.security_balance_detail` |
| **Placements (байршуулсан)** | байршуулсан эх үүсвэр, байршуулсан зээл, байршуулсан хадгаламж | `dbm.placement_balance_detail` |
| **Borrowings (татсан)** | татсан эх үүсвэр, татсан зээл, татсан хадгаламж, татан төвлөрүүлсэн | `dbm.borrowing_balance_detail` |
| **Both Placement + Borrowing** | эх үүсвэр (without татсан/байршуулсан), татсан болон байршуулсан | `dbm.placement_balance_detail` + `dbm.borrowing_balance_detail` |

**⚠️ CRITICAL DISTINCTIONS:**
- **зээл олгосон (loans to CUSTOMERS)** → `dbm.loan_balance_detail` (bank's loan portfolio)
- **байршуулсан** → `dbm.placement_balance_detail` (bank PLACES money TO others, EARNS interest)
- **татсан** → `dbm.borrowing_balance_detail` (bank BORROWS money FROM others, PAYS interest)

**MODULE FILTERING (for placement & borrowing tables):**
- `module = 'LD'` → Зээл (long-term loans)
- `module = 'MM'` → Хадгаламж (money market/deposits)
- **эх үүсвэр** → NO module filter (includes both LD and MM)

**Examples:**
- "татсан зээл" → borrowing_balance_detail WHERE module = 'LD'
- "татсан хадгаламж" → borrowing_balance_detail WHERE module = 'MM'
- "татсан эх үүсвэр" → borrowing_balance_detail (no module filter)
- "байршуулсан зээл" → placement_balance_detail WHERE module = 'LD'
- "байршуулсан хадгаламж" → placement_balance_detail WHERE module = 'MM'
- "байршуулсан эх үүсвэр" → placement_balance_detail (no module filter)

**MULTI-TABLE HINTS:**
- "үлдсэн" (remaining) = scheduled - paid → needs 2 tables
- "төлөгдсөн + үлдэгдэл" (paid + balance) → needs 2 tables
- "дансны мэдээлэл" + "дараагийн төлбөр" → needs 2 tables

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**TABLE PURPOSES:**
- `dbm.loan_balance_detail`: LOANS to CUSTOMERS - bank's loan portfolio, customer info, risk classification
- `dbm.bcom_nrs_detail`: Future payment schedules (dates and amounts) for loans
- `dbm.loan_txn`: Transaction history (actual payments that occurred)
- `dbm.security_balance_detail`: Securities, bonds, debt instruments - bought or issued by bank
- `dbm.placement_balance_detail`: БАЙРШУУЛСАН - bank PLACES money TO other institutions (хүүний ОРЛОГО)
- `dbm.borrowing_balance_detail`: ТАТСАН - bank BORROWS money FROM other institutions (хүүний ЗАРДАЛ)

**MODULE INFO (for placement & borrowing tables):**
- `module = 'LD'` → Зээл (long-term loans)
- `module = 'MM'` → Хадгаламж (money market/deposits - short-term)
- If user says "эх үүсвэр" without зээл/хадгаламж → NO module filter needed

**DEDUPLICATION:**
- `dbm.loan_balance_detail` has daily snapshots (one row per loan per day)
- For latest state: `txn_date = (SELECT MAX(txn_date) FROM dbm.loan_balance_detail)`
- `dbm.security_balance_detail` also has daily snapshots - use same pattern
- `dbm.placement_balance_detail` has daily snapshots - use same pattern
- `dbm.borrowing_balance_detail` has daily snapshots - use same pattern

**CHAT HISTORY (CRITICAL FOR FOLLOW-UP QUESTIONS):**
If question has pronouns (энэ, тэр, дээрх) or refers to previous result:
1. Set `needs_chat_history=true`
2. Include ALL columns from [Previous SQL] in `required_columns` for CTE recreation

**SCHEMA:**
{context_str}

**EXAMPLES:**
{examples}

**QUESTION:** {query_str}

**OUTPUT (JSON):**
{{
  "complexity": "SIMPLE|COMPLEX",
  "needs_chat_history": true|false,
  "chat_history_reasoning": "Explanation if applicable, empty string otherwise",
  "needs_deduplication": true|false,
  "explanation": "Brief plan. MUST include: 1) Which example matched (if any), 2) Why these tables were selected",
  "required_tables": ["dbm.table1", "dbm.table2"],
  "required_columns": {{"dbm.table1": ["col1", "col2"]}},
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
Summarize the conversation history for context in Mongolian. Keep only essential information.

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
    
    # =========================================================================
    # MODULAR PROMPT COMPOSITION METHODS
    # =========================================================================
    
    def get_modular_text2sql_prompt(self, required_tables: List[str]) -> PromptTemplate:
        """
        Get a dynamically composed text-to-SQL prompt based on required tables.
        
        Args:
            required_tables: List of table names from QueryAnalyzer
                            (e.g., ["dbm.loan_balance_detail", "dbm.bcom_nrs_detail"])
        
        Returns:
            PromptTemplate with domain-specific rules for only the required tables
        """
        # Compose domain-specific rules using PromptComposer
        domain_rules = prompt_composer.compose_domain_rules(required_tables)
        
        # Create the modular prompt with composed rules
        prompt_with_rules = MODULAR_TEXT_TO_SQL_TEMPLATE.replace("{domain_rules}", domain_rules)
        
        return PromptTemplate(prompt_with_rules)
    
    def format_modular_prompt(
        self,
        required_tables: List[str],
        query_str: str,
        schema: str,
        dynamic_examples: str,
        analyzer_explanation: str,
        plan: str,
        entity_names: str
    ) -> str:
        """
        Format the modular text-to-SQL prompt with all variables filled in.
        
        Args:
            required_tables: List of table names to compose domain rules for
            query_str: User's question
            schema: Database schema (pruned to relevant tables/columns)
            dynamic_examples: Retrieved similar SQL examples
            analyzer_explanation: QueryAnalyzer's explanation
            plan: Decomposition steps from QueryAnalyzer
            entity_names: Extracted entity names from database
        
        Returns:
            Fully formatted prompt string ready to send to LLM
        """
        # Get the modular prompt template
        prompt_template = self.get_modular_text2sql_prompt(required_tables)
        
        # Add table selection hint
        table_hint = prompt_composer.get_table_selection_hint(required_tables)
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            query_str=query_str,
            schema=schema,
            dynamic_examples=dynamic_examples,
            analyzer_explanation=analyzer_explanation + table_hint,
            plan=plan,
            entity_names=entity_names
        )
        
        return formatted_prompt
    
    def get_available_tables(self) -> List[str]:
        """Get list of all available tables that have domain blocks defined."""
        return list(DOMAIN_BLOCKS.keys())


prompt_manager = PromptManager("oracle")
