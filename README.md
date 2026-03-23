# ETL Pipeline Project

This project implements an ETL (Extract, Transform, Load) pipeline for processing product feeds and search keywords using BigQuery, transforming data with an LLM, and optionally uploading results to S3.

## Features

- Extracts product feed data from BigQuery based on marketplace ID
- Fetches or uses provided search keywords
- Applies LLM-based transformation to generate category mappings
- Supports optional upload of results to Amazon S3

## Prerequisites

- Python 3.8+
- Access to BigQuery and S3 services via Osmos clients
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

3. Set up environment variables or configuration as needed for Osmos clients.

## Configuration

Edit the configuration variables in `pipeline.py`:

- `RUN_KEYWORDS_QUERY`: Set to `True` to fetch keywords from BigQuery, or `False` to provide `keywords.csv` manually in the `data/` directory.
- `marketplace_id`: Required marketplace ID for BigQuery queries (e.g., "100002").
- `UPLOAD_TO_S3`: Set to `True` to upload results to S3, or `False` to skip.
- `S3_OUTPUT_PATH`: S3 path for upload (e.g., "s3://bucket/path/to/output.tsv.gz") if `UPLOAD_TO_S3` is `True`.
- `prompt`: Select a prompt template from `["prompt_e-commerce.txt", "prompt_fashion.txt", "prompt_grocery.txt", "prompt_pharmacy.txt"]`.

## Usage

Run the pipeline:

```bash
python pipeline.py
```

## How It Works

1. **Extract**: Fetches product feed data from BigQuery using the specified `marketplace_id` and saves it to `data/product_feed.csv`. If `RUN_KEYWORDS_QUERY` is `True`, also fetches search keywords and appends unique entries to `data/keywords.csv`.

2. **Transform**: Executes `transform.py`, which uses the selected prompt template and LLM to process the data, generating category mappings saved as `output/query_category_mapping.tsv`.

3. **Load**: If `UPLOAD_TO_S3` is `True`, compresses the output TSV and uploads it to the specified S3 path. Otherwise, the pipeline completes after transformation.

## Project Structure

```
.
├── pipeline.py          # Main pipeline script
├── transform.py         # Data transformation logic
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
├── data/                # Input data directory
│   ├── prompt.txt       # Selected prompt template
│   └── ...              # Other data files (ignored by git)
└── output/              # Output directory (ignored by git)
    └── query_category_mapping.tsv
```

## Notes

- Large data files in `data/` and `output/` are ignored by Git to keep the repository lightweight.
- Ensure proper credentials are configured for BigQuery and S3 access via Osmos clients.
- The pipeline logs progress and errors to the console.