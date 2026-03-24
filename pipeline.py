# pipeline.py

import argparse
import csv
import gzip
import io
from pathlib import Path
import pandas as pd
from osClient4pyV2.big_query_client import BigQueryServiceClient
from osClient4pyV2.s3_client import S3Client
from osSvcClient4pyV2.hades_svc_client import HadesSvcClient
from osUtilsV2.log_utils import LogUtils

from constants import (
    APP_CONFIG,
    DEFAULT_PROMPT,
    LLM_SCORE_THRESHOLD,
    PRODUCT_QUERY_TEMPLATE,
    KEYWORDS_QUERY_TEMPLATE,
    S3_OUTPUT_PATH_TEMPLATE,
)
from transform import run_mapping

LOGGER = LogUtils.configure_console_logger()

OUTPUT_DIR = Path(".")


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ETL pipeline for keyword-to-category mapping")
    parser.add_argument("--marketplace-id", type=int, required=True,
                        help="Marketplace client ID for queries and product feed table")
    parser.add_argument("--env", type=str, default="prod", choices=["test", "prod"],
                        help="Environment to use (default: prod)")
    parser.add_argument("--upload-to-s3", action="store_true", default=False,
                        help="Upload output to S3")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Prompt template filename (default: prompt_e-commerce.txt)")
    parser.add_argument("--llm-score-threshold", type=int, default=LLM_SCORE_THRESHOLD,
                        help="LLM score threshold (default: 70)")

    # Keywords source: BQ query OR manual file (mutually exclusive)
    kw_group = parser.add_mutually_exclusive_group(required=True)
    kw_group.add_argument("--run-keywords-query", action="store_true", default=False,
                          help="Fetch keywords from BigQuery")
    kw_group.add_argument("--keywords-file", type=str, default=None,
                          help="Path to a manual keywords CSV file (must have a 'keywords' column)")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def get_bigquery_client(env: str) -> BigQueryServiceClient:
    config = APP_CONFIG[env]
    hades_client = HadesSvcClient(config["APPLICATION"], env_domain=env)
    credentials = hades_client.get_app_context_by_app_key(config["BIG_QUERY_APP_KEY"])
    return BigQueryServiceClient(big_query_cred_context=credentials)


def get_s3_client(env: str) -> S3Client:
    config = APP_CONFIG[env]
    hades_client = HadesSvcClient(config["APPLICATION"], env_domain=env)
    s3_context = hades_client.get_app_context_by_app_key(config["S3_APP_KEY"])
    access_key = s3_context.get("s3_access_key")
    secret_key = s3_context.get("s3_secret_key")
    if not access_key or not secret_key:
        raise RuntimeError("Missing S3 credentials")
    return S3Client(access_key, secret_key)


def fetch_query_as_df(bq_client, query: str, label: str) -> pd.DataFrame:
    """Fetch a BigQuery query and return results as an in-memory DataFrame."""
    LOGGER.info("Running BigQuery query: %s", label)
    result = bq_client.fetch_query(query=query)
    df = pd.DataFrame(result)
    if df.empty:
        LOGGER.warning("No rows returned for %s", label)
    else:
        LOGGER.info("Fetched %d rows for %s", len(df), label)
    return df


def download_mapped_keywords_from_s3(s3_client: S3Client, s3_path: str) -> set:
    """Download existing mapped keywords from S3 and return the set of already-mapped keywords."""
    LOGGER.info("Downloading existing mapped keywords from S3: %s", s3_path)
    try:
        raw_data = s3_client.download_s3_file(s3_path)
        if raw_data is None:
            LOGGER.info("No existing mapped file found at S3 path")
            return set()

        with gzip.open(io.BytesIO(raw_data), "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            mapped_keywords = set()
            for row in reader:
                if row and row[0]:
                    mapped_keywords.add(row[0].strip().lower())

        LOGGER.info("Found %d already-mapped keywords in S3", len(mapped_keywords))
        return mapped_keywords
    except Exception as e:
        LOGGER.warning("Could not download existing mappings from S3: %s. Proceeding without filtering.", e)
        return set()


def download_existing_mappings_df(s3_client: S3Client, s3_path: str) -> pd.DataFrame:
    """Download existing mapped data from S3 as a DataFrame (columns: query, category_path, score)."""
    LOGGER.info("Downloading existing mappings from S3 for merge: %s", s3_path)
    try:
        raw_data = s3_client.download_s3_file(s3_path)
        if raw_data is None:
            LOGGER.info("No existing mapped file found at S3 path")
            return pd.DataFrame(columns=["query", "category_path", "score"])

        with gzip.open(io.BytesIO(raw_data), "rt") as f:
            existing_df = pd.read_csv(f, sep="\t", header=None,
                                      names=["query", "category_path", "score"])

        LOGGER.info("Downloaded %d existing mapping rows from S3", len(existing_df))
        return existing_df
    except Exception as e:
        LOGGER.warning("Could not download existing mappings from S3: %s. Will upload new data only.", e)
        return pd.DataFrame(columns=["query", "category_path", "score"])


def upload_to_s3(new_df: pd.DataFrame, s3_output_path: str, s3_client: S3Client) -> None:
    """Append new mappings to existing S3 data and upload the combined result."""
    if not s3_output_path:
        raise ValueError("S3 output path must be set when uploading to S3")

    # Download existing data and append new mappings
    existing_df = download_existing_mappings_df(s3_client, s3_output_path)

    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        LOGGER.info("Merged: %d existing + %d new = %d total rows",
                     len(existing_df), len(new_df), len(combined_df))
    else:
        combined_df = new_df
        LOGGER.info("No existing data. Uploading %d new rows.", len(combined_df))

    compressed_file = Path("/tmp") / "mapped_keywords.tsv.gz"
    LOGGER.info("Compressing combined data to %s", compressed_file)

    with gzip.open(compressed_file, "wt", newline="") as f:
        combined_df.to_csv(f, header=False, index=False, sep="\t", quoting=csv.QUOTE_NONE)

    LOGGER.info("Uploading compressed file to S3: %s", s3_output_path)
    s3_client.upload_s3_file(str(compressed_file), s3_output_path)
    LOGGER.info("Upload completed (%d total rows)", len(combined_df))


def get_keyword_column(df: pd.DataFrame) -> str:
    """Find the keyword column name in a DataFrame."""
    for col in ["keywords", "keyword", "search_query"]:
        if col in df.columns:
            return col
    raise ValueError(f"DataFrame must have one of: keywords, keyword, search_query. Found: {list(df.columns)}")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    marketplace_id = args.marketplace_id
    env = args.env
    s3_path = S3_OUTPUT_PATH_TEMPLATE.format(marketplace_client_id=marketplace_id)

    LOGGER.info("Pipeline started  |  marketplace_id=%s  env=%s  upload_to_s3=%s",
                marketplace_id, env, args.upload_to_s3)

    # ── Step 0: Initialize clients early ─────────────────────────────────
    bq_client = get_bigquery_client(env)
    s3_client = get_s3_client(env)

    # ── Step 1: Always run product query (in memory) ─────────────────────
    product_query = PRODUCT_QUERY_TEMPLATE.format(marketplace_id=marketplace_id)
    product_df = fetch_query_as_df(bq_client, product_query, "product_feed")

    # ── Step 2: Get keywords (from BQ query OR manual file, in memory) ───
    if args.run_keywords_query:
        LOGGER.info("Fetching keywords from BigQuery")
        keywords_query = KEYWORDS_QUERY_TEMPLATE.format(marketplace_id=marketplace_id)
        keywords_df = fetch_query_as_df(bq_client, keywords_query, "keywords")
    elif args.keywords_file:
        keywords_file = Path(args.keywords_file)
        if not keywords_file.exists():
            raise FileNotFoundError(f"Keywords file not found: {keywords_file}")
        LOGGER.info("Loading keywords from manual file: %s", keywords_file)
        keywords_df = pd.read_csv(keywords_file)
    else:
        raise ValueError("Either --run-keywords-query or --keywords-file must be provided")

    if keywords_df.empty:
        LOGGER.warning("No keywords found. Nothing to process.")
        return

    LOGGER.info("Total keywords loaded: %d", len(keywords_df))

    # ── Step 3: Download existing mappings from S3 and filter ────────────
    already_mapped = download_mapped_keywords_from_s3(s3_client, s3_path)

    if already_mapped:
        kw_col = get_keyword_column(keywords_df)
        before_count = len(keywords_df)
        keywords_df = keywords_df[~keywords_df[kw_col].str.strip().str.lower().isin(already_mapped)]
        filtered_count = before_count - len(keywords_df)
        LOGGER.info("Filtered out %d already-mapped keywords. Remaining: %d", filtered_count, len(keywords_df))

        if keywords_df.empty:
            LOGGER.info("All keywords already mapped. Nothing new to process.")
            return
    else:
        LOGGER.info("No existing mappings found in S3. All keywords will be processed.")

    # ── Step 4: Run LLM mapping (in memory) ──────────────────────────────
    result_df = run_mapping(
        product_df=product_df,
        keywords_df=keywords_df,
        prompt_template=args.prompt,
        score_threshold=args.llm_score_threshold,
    )

    LOGGER.info("Mapping complete. Result rows: %d", len(result_df))

    # ── Step 5: Upload to S3 (optional) ──────────────────────────────────
    if args.upload_to_s3:
        upload_to_s3(result_df, s3_path, s3_client)
    else:
        LOGGER.info("Skipping S3 upload (--upload-to-s3 not set)")


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(args)
        LOGGER.info("Pipeline completed successfully")
        return 0
    except Exception:
        LOGGER.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
