"""
Northbridge Bank — Credit Risk Query Engine (Streamlit app)
----------------------------------------------------------
Business users ask commercial-lending portfolio questions in plain English.
The engine routes each question to a pre-approved SQL template or generates
new read-only SQL, validates it through five checks, retries once on
failure, escalates to a human analyst if it still fails, and shows the SQL,
raw data, confidence score, and an audit trail.

Run with:
    streamlit run app.py

Files needed in the same folder as this app:
    - credit_risk_portfolio.db

Secrets (Streamlit Cloud -> App settings -> Secrets, or .streamlit/secrets.toml
locally), or equivalent environment variables / a local config.json:
    OPENAI_API_KEY = "your-api-key"
    OPENAI_API_BASE = "https://aibe.mygreatlearning.com/openai/v1"
"""

import os
import re
import json
import sqlite3
import warnings
from datetime import datetime

import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(page_title="Credit Risk Query Engine", page_icon="🏦", layout="wide")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_risk_portfolio.db")


# ============================================================
# Credential Loading
# ============================================================
# Resolution order:
#   1. st.secrets (Streamlit Community Cloud / secrets.toml)
#   2. environment variables (already set, e.g. by the host)
#   3. config.json in the working directory (notebook-style local dev)
#
# The notebook's default OPENAI_API_BASE (the course's OpenAI-compatible
# proxy) is kept as a fallback default so the app works out of the box if
# only an API key is supplied.

DEFAULT_API_BASE = "https://aibe.mygreatlearning.com/openai/v1"


def get_secret(name, default=None):
    """Read a value from Streamlit secrets, environment variables, or config.json."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    value = os.environ.get(name)
    if value:
        return value

    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as file:
                config = json.load(file)
                if config.get(name):
                    return config.get(name)
        except Exception:
            pass

    return default


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_API_BASE = get_secret("OPENAI_API_BASE", DEFAULT_API_BASE)

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. Add it to the app's Secrets, as an "
        "environment variable, or in a local config.json before running."
    )
    st.stop()

if not os.path.exists(DB_PATH):
    st.error(f"Database file not found: {DB_PATH}. Upload credit_risk_portfolio.db next to app.py.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE


# ============================================================
# LLM and Database Setup
# ============================================================

@st.cache_resource
def get_llms():
    """
    llm: generates routing decisions, SQL, and narratives.
    evaluator_llm: stronger model used as an independent reviewer in the
    validation gate.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, base_url=OPENAI_API_BASE)
    evaluator_llm = ChatOpenAI(model_name="gpt-4o", temperature=0, base_url=OPENAI_API_BASE)
    return llm, evaluator_llm


@st.cache_resource
def get_connection():
    """Read-only SQLite connection shared across Streamlit reruns."""
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)


llm, evaluator_llm = get_llms()
conn = get_connection()


# ============================================================
# Database Schema (passed to the LLM for SQL generation)
# ============================================================

database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ============================================================
# Verified Query Template Library
# ============================================================

sql_1 = """
SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
    ROUND(SUM(CASE
                WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                THEN l.total_outstanding
                ELSE 0
              END) / 1000000.0, 2) AS npa_exposure_mn
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_mn DESC;
"""

sql_2 = """
SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC;
"""

sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_mn,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage;
"""

sql_4 = """
SELECT
    s.sector_name,
    COUNT(*) AS loan_count,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_coverage_ratio
FROM provisioning AS p
JOIN loan_master AS l
    ON p.loan_account_number = l.loan_account_number
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_coverage_ratio DESC;
"""

sql_5 = """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS outstanding_mn,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10;
"""

sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL
  AND TRIM(group_name) <> ''
GROUP BY group_name
ORDER BY SUM(total_outstanding) DESC
LIMIT 5;
"""

sql_7 = """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS outstanding_mn,
    l.days_past_due,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC, l.total_outstanding DESC;
"""

sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due);
"""

sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;
"""

sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date;
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },

    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ============================================================
# Tool 1: Intent Classification
# ============================================================

def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
You are a routing assistant for a commercial bank's credit risk query engine.

Your job is to decide whether the user's question can be answered by one of the
pre-approved verified query templates listed below, or whether new SQL must be generated.

User question:
{user_question}

Verified query templates:
{library_descriptions}

Rules:
1. Choose "verified" if a template's output already contains the data needed to answer
   the question. A template may return more rows than the user needs (for example, all
   sectors when the user asks about one sector). That is acceptable, because a later
   step picks out the relevant rows from the full result.
2. Choose "generated" if the question needs a metric, table, filter, or calculation
   that none of the templates provides.
3. Match on meaning, not on exact wording.
4. If more than one template could apply, pick the one whose metric matches the
   question most closely.
5. If you choose "generated", set query_id to null.

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            classification = json.loads(json_match.group())
            classification['route'] = str(classification.get('route', 'generated')).strip().lower()
            if classification['route'] != 'verified':
                classification['query_id'] = None
            return classification
        except json.JSONDecodeError:
            pass
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ============================================================
# Tool 2: Query Generation
# ============================================================

def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
You are a senior SQL developer writing queries for a SQLite credit risk database
at a commercial bank.

Write one SQLite query that answers the user's question.

User question:
{user_question}

Database schema:
{schema_context}

Rules:
1. Return only the SQL query: no explanation, comments, or markdown code fences.
2. Write a single read-only SELECT (or WITH ... SELECT) statement. Never use INSERT,
   UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, or ATTACH.
3. Use only the tables and columns listed in the schema. Do not invent names.
4. Use JOINs when the question needs data from more than one table (for example,
   join sector_master on sector_code to show sector names).
5. For provisioning or rating questions, use the latest date (2025-09-30) unless the
   user asks for a different date or for a trend across dates.
6. NPA means asset_classification IN ('Substandard', 'Doubtful', 'Loss').
7. Show monetary amounts in millions, rounded to 2 decimals, with column names
   ending in _mn.
8. Give every calculated column a clear alias.
9. Sort the results in the order that best answers the question.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ============================================================
# Tool 3: Query Validation
# ============================================================

def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (float, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip().rstrip(';'):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    # Collects identifiers that are not known tables, columns, or keywords.
    # These are reported for audit purposes only, because column aliases
    # (e.g. total_outstanding_mn) and SQL functions would otherwise be flagged.
    # Unknown tables/columns are caught definitively by Check 3 (EXPLAIN).
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
You are a senior credit risk analyst reviewing SQL before it runs against the bank's
commercial lending portfolio database.

Decide whether this SQL correctly answers the user's question.

Context:
{track_context}

User question:
{user_question}

SQL to review:
{candidate_sql}

Check:
1. Does it use the right tables and columns for what was asked?
2. Does it calculate the right metric with the right aggregation (SUM, COUNT, AVG)
   and grouping?
3. If the question involves NPAs, does it use
   asset_classification IN ('Substandard', 'Doubtful', 'Loss')?
4. If the question involves provisioning or ratings, does it use the right date
   (2025-09-30 for "latest" or "current"; all dates for a trend)?

Answer "no" only if the SQL would give a wrong or misleading answer.
Set confidence to how sure you are that the SQL correctly answers the question
(1.0 = certainly correct, 0.0 = certainly wrong).

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if not json_match:
        result['failed_check'] = 'llm_relevance'
        result['details'] = 'Relevance check failed: evaluator did not return a valid verdict'
        return result
    try:
        relevance_json = json.loads(json_match.group())
    except json.JSONDecodeError:
        result['failed_check'] = 'llm_relevance'
        result['details'] = 'Relevance check failed: evaluator response could not be parsed'
        return result
    result['relevance_confidence'] = float(relevance_json.get('confidence', 0.0))
    if str(relevance_json.get('verdict', '')).lower() == 'no' or result['relevance_confidence'] < 0.6:
        result['failed_check'] = 'llm_relevance'
        result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
        return result

    # Check 5: Verified template integrity check (verified track only)
    # Each query is wrapped as a subquery so LIMIT 0 works even when the query
    # ends with a semicolon or already has its own LIMIT clause.
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql'].strip().rstrip(';')
        clean_candidate = candidate_sql.strip().rstrip(';')
        try:
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({clean_candidate}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ============================================================
# Tool 4: Retry Generation
# ============================================================

def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
You are a senior SQL developer. A SQLite query you wrote for the bank's credit risk
database failed validation. Write a corrected version.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

Rules:
1. Fix the specific problem described in the validation error.
2. Keep answering the original user question.
3. Write a single read-only SELECT (or WITH ... SELECT) statement using only tables
   and columns from the schema.
4. NPA means asset_classification IN ('Substandard', 'Doubtful', 'Loss').
5. Return only the corrected SQL: no explanation, comments, or markdown code fences.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ============================================================
# Tool 5: Query Execution
# ============================================================

def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ============================================================
# Tool 6: Response Generation
# ============================================================

def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
You are a credit risk analyst writing a short answer for bank executives and the
Board Risk Management Committee.

Answer the user's question using only the query results below.

User question:
{user_question}

Query results:
{dataframe.to_string(index=False)}

Rules:
1. Answer in 2-4 sentences of plain English.
2. Focus on what the user asked. If the results contain more rows than needed (for
   example, all sectors when the user asked about one), mention only the relevant rows.
3. Quote exact figures from the results. Amounts in columns ending in _mn are in USD
   millions; write them as, for example, "$245.30 million".
4. Do not invent numbers or draw conclusions the data does not support.
5. If the results are empty, say that no matching records were found.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ============================================================
# Pipeline Orchestration
# ============================================================

def run_pipeline(user_question, db_connection, query_library, schema_context, status=None):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - status: optional Streamlit status/container object with a .write()
      method, used in place of the notebook's verbose print statements.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    def report(message):
        if status is not None:
            status.write(message)

    log = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'warnings': [],
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    report(f"**[1] Intent Classification:** route=`{log['route']}`, query_id=`{log['query_id']}`  \n{log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
        report("**[2] Query Construction:** loaded from verified library")
    else:
        candidate_sql = generate_query(user_question, schema_context)
        report("**[2] Query Construction:** generated fresh SQL")
    log['candidate_sql'] = candidate_sql

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    report(f"**[3] Validation Gate:** passed=`{gate['passed']}`, relevance_confidence=`{gate.get('relevance_confidence')}`")
    if not gate['passed']:
        report(f"Failed check: `{gate.get('failed_check')}`  \nDetails: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        report(f"Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        report(f"**Retry Validation Gate:** passed=`{gate['passed']}`, relevance_confidence=`{gate.get('relevance_confidence')}`")
        if not gate['passed']:
            report(f"Retry failed check: `{gate.get('failed_check')}`  \nRetry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        report(f"**[!] Escalated to human:** {gate['details']}")
        return {'log': log, 'dataframe': None}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    report(f"**[4] Execute:** {len(df)} rows returned")
    if exec_result['warnings']:
        report(f"Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    log['narrative'] = generate_response(user_question, df, log['route'], log['query_id'])

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    report(f"**[6] Response Generation:** confidence=`{log['confidence']}`")

    return {'log': log, 'dataframe': df}


# ============================================================
# Streamlit User Interface
# ============================================================

if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []
if "question" not in st.session_state:
    st.session_state.question = ""

EXAMPLE_QUESTIONS = [
    "What is the NPA exposure in the Real Estate sector?",
    "Show the aging profile of the portfolio by days past due.",
    "What is the total outstanding of restructured loans by sector?",
    "What is the average interest rate by sector?",
    "How has Stage 3 ECL changed across reporting quarters?",
]

with st.sidebar:
    st.header("About")
    st.write(
        "This application routes portfolio questions through a verified SQL "
        "template library when possible, or generates fresh SQL for novel "
        "questions. Every query passes through a five-check validation gate "
        "before it touches the read-only database, and failed generated "
        "queries get one automatic retry before escalating to a human analyst."
    )

    st.header("Example questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.question = q

    st.divider()
    st.header("Verified query library")
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}"):
            st.caption(entry['description'])
            st.code(entry['sql'].strip(), language="sql")

st.title("🏦 Credit Risk Query Engine")
st.caption(
    "Ask a question about the commercial lending portfolio in plain English. "
    "Every answer shows the SQL used, the raw data, and a confidence score. "
    "Questions the engine cannot answer reliably are escalated to a human analyst."
)

question = st.text_area(
    "Your question",
    key="question",
    height=90,
    placeholder="e.g. Which sectors have the highest NPA exposure?"
)

show_trace = st.checkbox("Show pipeline trace", value=True)
run = st.button("Get answer", type="primary")

if run:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        trace_container = st.status("Running pipeline...", expanded=show_trace) if show_trace else None

        try:
            output = run_pipeline(
                question.strip(), conn, verified_query_library, database_schema, status=trace_container
            )
        except Exception as e:
            if trace_container is not None:
                trace_container.update(label="Pipeline error", state="error")
            st.error(f"The pipeline hit an unexpected error: {e}")
            output = None

        if output:
            log = output['log']
            df = output['dataframe']
            st.session_state.audit_trail.append(log)

            if trace_container is not None:
                trace_container.update(
                    label="Pipeline complete" if not log['escalated'] else "Escalated to human review",
                    state="complete" if not log['escalated'] else "error"
                )

            st.divider()

            if log['escalated']:
                st.error("⚠️ Escalated to a human analyst")
                st.write(log['narrative'])
                with st.expander("Validation details"):
                    st.json(log['gate_result'])
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Route", str(log['route']).title())
                c2.metric("Template", log['query_id'] or "Generated SQL")
                c3.metric("Confidence", f"{log['confidence']:.2f}" if isinstance(log['confidence'], (int, float)) else "n/a")
                c4.metric("Rows returned", log['row_count'])

                st.subheader("Answer")
                st.success(log['narrative'])

                if log['warnings']:
                    st.warning("Data checks: " + "; ".join(log['warnings']))

                st.subheader("Data returned")
                st.dataframe(df, use_container_width=True)

                st.subheader("SQL used")
                st.code(sqlparse.format(log['executed_sql'], reindent=True, keyword_case='upper'), language="sql")

            with st.expander("Pipeline trace (routing and validation details)"):
                st.write(f"**Routing reason:** {log['match_reason']}")
                st.write(f"**Retry used:** {log['retry_used']}")
                st.json(log['gate_result'])

st.divider()
st.subheader("Audit trail (this session)")
if st.session_state.audit_trail:
    audit_df = pd.DataFrame([
        {
            'Timestamp': e['timestamp'],
            'Question': e['user_question'],
            'Route': e['route'],
            'Query ID': e['query_id'],
            'Retry': e['retry_used'],
            'Escalated': e['escalated'],
            'Confidence': e['confidence'],
            'Rows': e['row_count'],
        }
        for e in st.session_state.audit_trail
    ])
    st.dataframe(audit_df, use_container_width=True)
    st.download_button(
        "Download full audit log (JSON)",
        data=json.dumps(st.session_state.audit_trail, indent=2, default=str),
        file_name="audit_log.json",
        mime="application/json",
    )
else:
    st.caption("No questions asked yet in this session.")
