# ETL Pipeline — Keyword-to-Category Mapping

An ETL pipeline that maps search keywords to product categories using **Gemini 2.5 Flash Lite** with explicit context caching. It extracts product feed and keyword data from BigQuery, runs LLM-based category mapping, and optionally uploads results to S3.

## Features

- Extracts product feed and search keywords from BigQuery
- Supports manual keyword input via CSV file
- Maps keywords to product categories using Gemini 2.5 Flash Lite with context caching
- Filters out already-mapped keywords by checking existing S3 data
- Crash recovery via progress checkpointing
- Batch processing with configurable batch size and score threshold
- Token usage tracking and cost estimation
- Appends new mappings to existing S3 data on upload

## Prerequisites

- Python 3.8+
- Access to BigQuery and S3 via Osmos clients (`osClient4pyV2`, `osSvcClient4pyV2`, `osUtilsV2`)
- `GOOGLE_API_KEY` environment variable set for Gemini API access
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/manavkumawat-osmos-ai/induction_project.git
   cd induction_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export GOOGLE_API_KEY="your-gemini-api-key"
   ```

## Usage

The pipeline is driven entirely via CLI arguments:

```bash
# Fetch keywords from BigQuery
python pipeline.py --marketplace-id 100002 --run-keywords-query

# Use a manual keywords CSV file
python pipeline.py --marketplace-id 100002 --keywords-file data/keywords.csv

# Upload results to S3
python pipeline.py --marketplace-id 100002 --run-keywords-query --upload-to-s3

# Use a specific prompt template and custom score threshold
python pipeline.py --marketplace-id 100002 --run-keywords-query --prompt prompt_fashion.txt --llm-score-threshold 80
```

### CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--marketplace-id` | Yes | — | Marketplace client ID for BigQuery queries |
| `--env` | No | `prod` | Environment (`test` or `prod`) |
| `--run-keywords-query` | One of these | — | Fetch keywords from BigQuery |
| `--keywords-file` | is required | — | Path to a CSV with a `keywords` column |
| `--upload-to-s3` | No | `False` | Upload output to S3 |
| `--prompt` | No | `prompt_e-commerce.txt` | Prompt template filename |
| `--llm-score-threshold` | No | `70` | Minimum LLM confidence score (0–100) |

## How It Works

1. **Extract** — Fetches the product feed from BigQuery using the marketplace ID. Keywords are either queried from BigQuery or loaded from a CSV file. Existing mapped keywords are downloaded from S3 and filtered out to avoid redundant processing.

2. **Transform** — Unique category paths are extracted from the product feed and assigned IDs. A Gemini context cache is created with the prompt template and category list. Keywords are sent to the LLM in batches, and each keyword is mapped to 0–3 categories with confidence scores. Results below the score threshold are discarded.

3. **Load** — If `--upload-to-s3` is set, new mappings are merged with existing S3 data and uploaded as a gzip-compressed TSV file.

## Configuration

Pipeline constants are centralized in `constants.py`:

| Constant | Value | Description |
|---|---|---|
| `BATCH_SIZE` | `20` | Number of keywords per LLM batch |
| `CACHE_TTL_SECONDS` | `1800s` | Gemini context cache time-to-live |
| `LLM_SCORE_THRESHOLD` | `70` | Default minimum confidence score |
| `PROMPT_TEMPLATES` | 4 templates | Available prompt files (e-commerce, fashion, grocery, pharmacy) |
| `APP_CONFIG` | test/prod | Application, S3, and BigQuery credential keys per environment |

## Project Structure

```
.
├── pipeline.py          # Main pipeline — CLI parsing, BigQuery/S3 I/O, orchestration
├── transform.py         # LLM mapping — Gemini context caching, batching, scoring
├── constants.py         # Centralized configuration and SQL query templates
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (GOOGLE_API_KEY)
├── data/                # Prompt templates and input data
│   ├── prompt_e-commerce.txt
│   ├── prompt_fashion.txt
│   ├── prompt_grocery.txt
│   └── prompt_pharmacy.txt
└── output/              # Runtime output (git-ignored)
```

## Output Format

The pipeline produces a TSV with three columns:

| Column | Description |
|---|---|
| `query` | Original search keyword |
| `category_path` | Matched product category path |
| `score` | LLM confidence score (0–100) |

## Notes

- The pipeline operates entirely in memory — no intermediate CSV files are written to disk.
- Crash recovery is supported via a progress file (`output/.mapping_progress.json`) that is cleaned up on successful completion.
- The `data/` and `output/` directories are git-ignored to keep the repository lightweight.
- Ensure proper credentials are configured for BigQuery and S3 access via Osmos clients.
