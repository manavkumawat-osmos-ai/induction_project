# constants.py

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Constants
# ──────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 20
CACHE_TTL_SECONDS = "1800s"  

APP_CONFIG = {
    "test": {
        "APPLICATION": "irisTestApplication",
        "S3_APP_KEY": "os_PERFORMANCE_MASTER_KEY",
        "BIG_QUERY_APP_KEY": "DATA_VALIDATION_FRAMEWORK_BQ_REPORTING_CREDS",
    },

    "prod": {
        "APPLICATION": "irisTestApplication",
        "S3_APP_KEY": "os_PERFORMANCE_MASTER_KEY",
        "BIG_QUERY_APP_KEY": "DATA_VALIDATION_FRAMEWORK_BQ_REPORTING_CREDS",
    }
}

# S3 output path template — MARKETPLACE_CLIENT_ID is replaced at runtime
S3_OUTPUT_PATH_TEMPLATE = "s3://os-performance-dev-bucket/automated_mapping/{marketplace_id}/mapped_keywords.tsv"

# Prompt templates available
PROMPT_TEMPLATES = [
    "prompt_e-commerce.txt",
    "prompt_fashion.txt",
    "prompt_grocery.txt",
    "prompt_pharmacy.txt",
]
DEFAULT_PROMPT = "prompt_e-commerce.txt"

# LLM configuration
LLM_SCORE_THRESHOLD = 70

# SQL query templates — marketplace_id is injected at runtime
PRODUCT_QUERY_TEMPLATE = """
SELECT e_name, e_brand, e_product_type
FROM `reporting.oltp_merchandise_product_dimensions_{marketplace_id}`
WHERE e_product_type IS NOT NULL
"""

KEYWORDS_QUERY_TEMPLATE = """
SELECT
  os_product_ads_search_query_request_report.marketplace_client_id,
  lower(
    REPLACE( REPLACE( os_product_ads_search_query_request_report.search_query, '[', '' ), ']', '' )
  ) as keywords,
  replace(
    lower(
      REPLACE( REPLACE( os_product_ads_search_query_request_report.search_query, '[', '' ), ']', '' )    ),
    ' ',
    ''
  ) as trimmed_keywords,
  (
    length(os_product_ads_search_query_request_report.search_query) -
    length(REGEXP_REPLACE(os_product_ads_search_query_request_report.search_query, ' ', ''))
  )  + 1 as word_cnt,
  SUM(os_product_ads_search_query_request_report.request) as request_cnt,
FROM
  reporting.os_product_ads_search_query_request_report,
  reporting.clients
where
  clients.client_id = os_product_ads_search_query_request_report.marketplace_client_id
  AND os_product_ads_search_query_request_report.marketplace_client_id = "{marketplace_id}"
  and os_product_ads_search_query_request_report.date >= CURRENT_DATE() - 3
  AND os_product_ads_search_query_request_report.date_hour_utc >= CURRENT_DATE() - 4
  and os_product_ads_search_query_request_report.date <= CURRENT_DATE() - 1
  and os_product_ads_search_query_request_report.search_query is not NULL
  and os_product_ads_search_query_request_report.search_query <> ''
GROUP BY
  1,
  2,
  3,
  4
"""
