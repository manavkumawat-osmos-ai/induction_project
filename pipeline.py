# pipeline.py

import csv
import gzip
import subprocess
import sys
from pathlib import Path
import pandas as pd
from osClient4pyV2.big_query_client import BigQueryServiceClient
from osClient4pyV2.s3_client import S3Client
from osSvcClient4pyV2.hades_svc_client import HadesSvcClient
from osUtilsV2.log_utils import LogUtils


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE CONFIGURATION  –  edit these before each run
# ──────────────────────────────────────────────────────────────────────────────

RUN_KEYWORDS_QUERY = True         # Set to True to also run KEYWORDS_QUERY
marketplace_id = 290039             # Set to appropriate marketplace ID for queries and product feed table
UPLOAD_TO_S3      = False           # Set to True to upload output to S3
S3_OUTPUT_PATH    = ""            
prompt="prompt_e-commerce.txt"     # Prompt template for LLM ( needed by transform.py) list=["prompt_e-commerce.txt", "prompt_fashion.txt", "prompt_grocery.txt", "prompt_pharmacy.txt"]
LLM_score_threshold=70

OUTPUT_DIR        = Path(".")       # Root directory (data/ and output/ live here)
TRANSFORM_SCRIPT  = Path("transform.py")
TRANSFORM_OUTPUT  = Path("output/query_category_mapping.tsv")

# ──────────────────────────────────────────────────────────────────────────────


APP_CONFIG = {
    "prod": {
        "APPLICATION": "irisTestApplication",
        "S3_APP_KEY": "os_PERFORMANCE_MASTER_KEY",
        "BIG_QUERY_APP_KEY": "DATA_VALIDATION_FRAMEWORK_BQ_REPORTING_CREDS",
    }
}

PRODUCT_QUERY = f"""
SELECT e_name,e_brand, e_product_type
FROM `reporting.oltp_merchandise_product_dimensions_{marketplace_id}`
WHERE r_product_type IS NOT NULL
"""

KEYWORDS_QUERY = f"""
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

LOGGER = LogUtils.configure_console_logger()


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def get_bigquery_client() -> BigQueryServiceClient:
    config = APP_CONFIG["prod"]
    hades_client = HadesSvcClient(config["APPLICATION"], env_domain="prod")
    credentials = hades_client.get_app_context_by_app_key(config["BIG_QUERY_APP_KEY"])
    return BigQueryServiceClient(big_query_cred_context=credentials)


def extract_query_to_csv(bq_client, query: str, output_path: Path) -> Path:
    LOGGER.info("Running BigQuery query and writing to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = bq_client.fetch_query(query=query)
    if not df:
        LOGGER.warning("No rows returned for query %s", query)
    pd.DataFrame(df).to_csv(output_path, index=False)
    return output_path


def extract_and_append_unique(bq_client, query: str, output_path: Path, unique_col: str) -> Path:
    """Fetch query results and append only rows whose `unique_col` value is new."""
    LOGGER.info("Running BigQuery query for append-unique to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(bq_client.fetch_query(query=query))

    if new_df.empty:
        LOGGER.warning("No rows returned for query %s", query)
        return output_path

    if output_path.exists() and output_path.stat().st_size > 0:
        existing_df = pd.read_csv(output_path)
        existing_keys = set(existing_df[unique_col])
        unique_rows = new_df[~new_df[unique_col].isin(existing_keys)]
        LOGGER.info(
            "Existing rows: %d | Fetched: %d | New unique: %d",
            len(existing_df), len(new_df), len(unique_rows),
        )
        if not unique_rows.empty:
            combined_df = pd.concat([existing_df, unique_rows], ignore_index=True)
            combined_df.to_csv(output_path, index=False)
        else:
            LOGGER.info("No new unique rows to append")
    else:
        new_df.to_csv(output_path, index=False)

    return output_path


def run_transform_script(script_path: Path) -> None:
    LOGGER.info("Running transform script: %s", script_path)
    result = subprocess.run([sys.executable, str(script_path)], check=True)
    LOGGER.info("Transform script finished with return code %s", result.returncode)


def upload_to_s3(input_file: Path, s3_output_path: str) -> None:
    if not s3_output_path:
        raise ValueError("S3_OUTPUT_PATH must be set when UPLOAD_TO_S3 is True")

    config = APP_CONFIG["prod"]
    hades_client = HadesSvcClient(config["APPLICATION"], env_domain="prod")
    s3_context = hades_client.get_app_context_by_app_key(config["S3_APP_KEY"])

    access_key = s3_context.get("s3_access_key")
    secret_key = s3_context.get("s3_secret_key")
    if not access_key or not secret_key:
        raise RuntimeError("Missing S3 credentials")

    s3_client = S3Client(access_key, secret_key)

    compressed_file = Path("/tmp") / f"{input_file.stem}.tsv.gz"
    LOGGER.info("Compressing %s to %s", input_file, compressed_file)

    df = pd.read_csv(input_file, sep="\t")
    with gzip.open(compressed_file, "wt", newline="") as f:
        df.to_csv(f, header=False, index=False, sep="\t", quoting=csv.QUOTE_NONE)

    LOGGER.info("Uploading compressed file to S3: %s", s3_output_path)
    s3_client.upload_s3_file(str(compressed_file), s3_output_path)
    LOGGER.info("Upload completed")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Pipeline started  |  RUN_KEYWORDS_QUERY=%s  UPLOAD_TO_S3=%s", RUN_KEYWORDS_QUERY, UPLOAD_TO_S3)

    # ── Step 1: Extract data from BigQuery ───────────────────────────────
    bq_client = get_bigquery_client()
    extract_query_to_csv(bq_client, PRODUCT_QUERY, OUTPUT_DIR / "data/product_feed.csv")

    if RUN_KEYWORDS_QUERY:
        extract_and_append_unique(bq_client, KEYWORDS_QUERY, OUTPUT_DIR / "data/search_queries.csv", unique_col="keywords")
    else:
        LOGGER.info("Skipping KEYWORDS_QUERY (RUN_KEYWORDS_QUERY is False)")

    # ── Step 2: Run transform script ─────────────────────────────────────
    try:
        run_transform_script(TRANSFORM_SCRIPT)
    except subprocess.CalledProcessError:
        LOGGER.exception("Transform script failed")
        raise

    # ── Step 3: Upload to S3 (optional) ──────────────────────────────────
    if UPLOAD_TO_S3:
        upload_to_s3(OUTPUT_DIR / TRANSFORM_OUTPUT, S3_OUTPUT_PATH)
    else:
        LOGGER.info("Skipping S3 upload (UPLOAD_TO_S3 is False)")


def main() -> int:
    try:
        run_pipeline()
        LOGGER.info("Pipeline completed successfully")
        return 0
    except Exception:
        LOGGER.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())