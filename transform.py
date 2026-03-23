#!/usr/bin/env python3
"""
Query-to-Category Mapper using Gemini 2.5 Flash Lite with Explicit Context Caching.

The LLM returns category path strings directly, which are validated against the known paths.

Usage:
    export GOOGLE_API_KEY="your-api-key"
    python3 query_category_mapper.py               # full run (all queries)
    python3 query_category_mapper.py --limit 20     # test run (first 20 queries)
    python3 query_category_mapper.py --batch-size 5 # custom batch size
"""

import argparse
import csv
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pipeline import prompt, LLM_score_threshold  # from pipeline.py configuration
# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "gemini-2.5-flash-lite"
DEFAULT_BATCH_SIZE = 20          # queries per API call
CACHE_TTL_SECONDS = "1800s"      # 3 minutes
MAX_RETRIES = 5
INITIAL_BACKOFF = 2              # seconds

INPUT_FEED = "data/product_feed.csv"
INPUT_QUERIES = "data/search_queries.csv"
PROMPT_FILE = f"data/{prompt}"
OUTPUT_FILE = "output/query_category_mapping.tsv"
PROGRESS_FILE = "output/.mapping_progress.json"  # tracks completed queries for resume
LLM_RESPONSE_FILE = "output/llm_responses.json"  # saves raw LLM JSON responses
MIN_SCORE_THRESHOLD = LLM_score_threshold              # drop categories scoring below 70 (0-100 scale)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# 0.  Token usage and cost calculation
# ---------------------------------------------------------------------------
def get_token_usage(response):
    """Extract token counts from a Gemini SDK response object."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {"total_input": 0, "output": 0, "cached": 0, "thoughts": 0}
    
    return {
        "total_input": getattr(usage, "prompt_token_count", 0) or 0,
        "output": getattr(usage, "candidates_token_count", 0) or 0,
        "cached": getattr(usage, "cached_content_token_count", 0) or 0,
        "thoughts": getattr(usage, "thoughts_token_count", 0) or 0,
    }


def calculate_cost(token_usage):
    """Calculate USD cost based on Gemini 2.5 Flash Lite pricing."""
    INPUT_PRICE_PER_M = 0.10    # $0.10 per 1M newly sent tokens
    OUTPUT_PRICE_PER_M = 0.40   # $0.40 per 1M output tokens
    CACHE_PRICE_PER_M = 0.025   # $0.10 * 0.25 (75% discount) for cached tokens
    THOUGHT_PRICE_PER_M = 0.40

    # total_input includes both cached and new tokens
    new_input_tokens = max(0, token_usage["total_input"] - token_usage["cached"])
    
    input_cost = (new_input_tokens / 1_000_000) * INPUT_PRICE_PER_M
    cache_cost = (token_usage["cached"] / 1_000_000) * CACHE_PRICE_PER_M
    output_cost = (token_usage["output"] / 1_000_000) * OUTPUT_PRICE_PER_M
    thought_cost = (token_usage["thoughts"] / 1_000_000) * THOUGHT_PRICE_PER_M

    total_cost = input_cost + cache_cost + output_cost + thought_cost 
    
    # Apply 50% Batch API discount if applicable
    BATCH_DISCOUNT = 0.5
    discounted_total = total_cost * BATCH_DISCOUNT

    return {
        "input_cost": input_cost,
        "cache_cost": cache_cost,
        "output_cost": output_cost,
        "total_cost": discounted_total,
        "raw_total_cost": total_cost,
        "batch_discount": BATCH_DISCOUNT
    }


# ---------------------------------------------------------------------------
# 1.  Extract unique category paths
# ---------------------------------------------------------------------------
def extract_category_paths(feed_path):
    # type: (str) -> List[str]
    """Read product_feed.csv and return sorted unique e_product_type paths."""
    paths = set()
    with open(feed_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("e_product_type", "").strip()
            if path:
                paths.add(path)
    return sorted(paths)


# ---------------------------------------------------------------------------
# 2.  Read queries
# ---------------------------------------------------------------------------
def read_queries(queries_path, limit=None):
    # type: (str, Optional[int]) -> List[str]
    """Read search queries from queries.csv."""
    queries = []  # type: List[str]
    with open(queries_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Fallback for checking "keyword", "keywords", or "search_query"
            q = row.get("keyword", row.get("keywords", row.get("search_query", ""))).strip()
            if q:
                queries.append(q)
            if limit and len(queries) >= limit:
                break
    return queries


# ---------------------------------------------------------------------------
# 3.  Progress / resume helpers
# ---------------------------------------------------------------------------
def load_existing_mappings(output_path):
    # type: (str) -> Dict
    """Read the existing output CSV and return a dict of query -> categories.
    This allows re-runs to skip queries that were already mapped."""
    mappings = {}  # type: Dict[str, List[Dict]]
    if not os.path.exists(output_path):
        return mappings
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get("query", "").strip()
            cat_path = row.get("category_path", "").strip()
            score = row.get("score", "").strip()
            if not query:
                continue
            if query not in mappings:
                mappings[query] = []
            if cat_path:  # only add non-empty category entries
                mappings[query].append({
                    "path": cat_path,
                    "score": int(score) if score else 0,
                })
    return mappings


def load_progress(progress_path):
    # type: (str) -> Dict
    """Load in-flight mapping progress from disk (for mid-run crash recovery)."""
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress_path, progress):
    # type: (str, dict) -> None
    """Persist mapping progress to disk."""
    with open(progress_path, "w") as f:
        json.dump(progress, f)


# ---------------------------------------------------------------------------
# 4.  Build the context cache
# ---------------------------------------------------------------------------
def load_prompt(prompt_path, id_to_path):
    # type: (str, Dict[str, str]) -> str
    """Read prompt.txt and append the category list with IDs to build the system instruction."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    
    # Format: ID: Category Path
    category_list_text = "\n".join("- {}: {}".format(cid, path) for cid, path in sorted(id_to_path.items()))
    return prompt_text + "\n\nAvailable Categories:\n" + category_list_text


USER_PROMPT_TEMPLATE = """\
Map each of the following search queries to 0-3 best matching category IDs from the list.
Return a JSON object with a "mappings" array. Each item must have:
  - "query": the original search query
  - "categories": array of objects, each with "category_id" (string) and "score" (integer 0-100)

Example: {{"mappings": [{{"query": "blue jeans", "categories": [{{"category_id": "CAT_001", "score": 95}}, {{"category_id": "CAT_002", "score": 80}}]}}]}}

Queries:
{queries}
"""


def create_cache(client, id_to_path, prompt_path):
    # type: (genai.Client, Dict[str, str], str) -> Optional[str]
    """Create an explicit context cache with category paths and IDs.
    Returns the cache name, or None if the content is too small for caching."""
    system_text = load_prompt(prompt_path, id_to_path)
    
    # Save the exact cache input to a file for review
    cache_input_file = Path(__file__).parent / "llm_cache_input.txt"
    with open(cache_input_file, "w", encoding="utf-8") as f:
        f.write(system_text)
    log.info("Saved cache input to %s", cache_input_file.name)

    log.info("Creating context cache with %d category paths …", len(id_to_path))

    try:
        cache = client.caches.create(
            model="models/{}".format(MODEL),
            config=types.CreateCachedContentConfig(
                display_name="product-category-paths",
                system_instruction=system_text,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text="Understood. I will only return category_id values from the available categories list.")],
                    ),
                    types.Content(
                        role="model",
                        parts=[types.Part(text="Ready. Send me the search queries and I will map each one to 0-3 category IDs from the list, returning structured JSON.")],
                    ),
                ],
                ttl=CACHE_TTL_SECONDS,
            ),
        )
        log.info("Cache created: %s", cache.name)
        return cache.name
    except Exception as e:
        if "too small" in str(e).lower() or "min_total_token_count" in str(e):
            log.warning("Content too small for caching (need ≥2048 tokens). "
                        "Falling back to non-cached mode (system instruction per request).")
            return None
        raise  # re-raise unexpected errors


# ---------------------------------------------------------------------------
# 5.  Call the LLM with retry — returns resolved category paths
# ---------------------------------------------------------------------------
def call_llm(client, cache_name, queries_batch, id_to_path, system_text=None, batch_num=0):
    # type: (genai.Client, Optional[str], List[str], Dict[str, str], Optional[str], int) -> List[Dict]
    """Send a batch of queries to Gemini and return parsed results with validated paths.
    If cache_name is None, uses system_text as a per-request system instruction."""
    queries_text = "\n".join("{}. {}".format(i + 1, q) for i, q in enumerate(queries_batch))
    prompt = USER_PROMPT_TEMPLATE.format(queries=queries_text)

    # Build config based on whether caching is available
    if cache_name:
        config = types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=8192,
        )
    else:
        config = types.GenerateContentConfig(
            system_instruction=system_text,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=8192,
        )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model="models/{}".format(MODEL),
                contents=prompt,
                config=config,
            )

            # Parse the JSON response
            text = response.text.strip()
            parsed = json.loads(text)

            # Save raw LLM response to file
            _save_llm_response(parsed, batch_num)

            # Handle both formats: {"mappings": [...]} or [...]
            if isinstance(parsed, dict) and "mappings" in parsed:
                results = parsed["mappings"]
            elif isinstance(parsed, list):
                results = parsed
            else:
                results = [parsed]

            # Validate category IDs and collect scores
            validated = []  # type: List[Dict]
            for item in results:
                raw_categories = item.get("categories", [])[:3]  # cap at 3
                resolved = []
                for cat in raw_categories:
                    if isinstance(cat, dict):
                        cat_id = cat.get("category_id", "").strip()
                        score = int(cat.get("score", 0))
                    else:
                        cat_id = str(cat).strip()
                        score = 100

                    if cat_id in id_to_path:
                        cat_path = id_to_path[cat_id]
                        if score >= MIN_SCORE_THRESHOLD:
                            resolved.append({"path": cat_path, "score": score})
                        else:
                            log.info("Score %d below threshold for ID '%s' ('%s') → query '%s' — skipping",
                                     score, cat_id, cat_path, item.get("query", ""))
                    elif cat_id and cat_id.lower() != "null":
                        log.warning("Unknown category ID '%s' for query '%s' — skipping",
                                    cat_id, item.get("query", ""))
                validated.append({
                    "query": item.get("query", ""),
                    "categories": resolved,
                })

            # Reconcile LLM-returned queries back to the original batch queries.
            # The LLM may change casing, spacing, or punctuation, so we match
            # by position first, then by normalized text as a fallback.
            reconciled = []  # type: List[Dict]
            if len(validated) == len(queries_batch):
                # Same count — assume same order, use original query text
                for orig_q, v_item in zip(queries_batch, validated):
                    reconciled.append({"query": orig_q, "categories": v_item["categories"]})
            else:
                # Count mismatch — match by normalized query text
                norm_map = {}  # type: Dict[str, List]
                for v_item in validated:
                    norm_key = v_item["query"].strip().lower()
                    norm_map[norm_key] = v_item["categories"]
                for orig_q in queries_batch:
                    norm_key = orig_q.strip().lower()
                    cats = norm_map.get(norm_key, [])
                    reconciled.append({"query": orig_q, "categories": cats})
                    if not norm_map.get(norm_key):
                        log.warning("No LLM result matched for query '%s'", orig_q)

            return reconciled, get_token_usage(response)

        except json.JSONDecodeError as e:
            log.warning("Attempt %d/%d: JSON parse error: %s", attempt, MAX_RETRIES, e)
            log.warning("Raw response text: %s", response.text[:500] if response.text else "EMPTY")
        except Exception as e:
            log.warning("Attempt %d/%d: API error: %s", attempt, MAX_RETRIES, e)

        if attempt < MAX_RETRIES:
            wait = INITIAL_BACKOFF * (2 ** (attempt - 1))
            log.info("Retrying in %d seconds …", wait)
            time.sleep(wait)

    # All retries exhausted — return empty mappings so the script continues
    log.error("All retries exhausted for batch. Returning empty mappings.")
    return [{"query": q, "categories": []} for q in queries_batch], {"total_input": 0, "output": 0, "cached": 0, "thoughts": 0}


def _save_llm_response(parsed, batch_num):
    # type: (dict, int) -> None
    """Append the raw LLM JSON response to the responses file."""
    response_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LLM_RESPONSE_FILE)
    try:
        if os.path.exists(response_path):
            with open(response_path, "r", encoding="utf-8") as f:
                all_responses = json.load(f)
        else:
            all_responses = []
        all_responses.append({
            "batch": batch_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "response": parsed,
        })
        with open(response_path, "w", encoding="utf-8") as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning("Could not save LLM response: %s", e)


# ---------------------------------------------------------------------------
# 6.  Write output CSV
# ---------------------------------------------------------------------------
def write_output(output_path, all_results):
    # type: (str, List[Dict]) -> None
    """Write all results to CSV — one row per query-category pair with score.
    Overwrites the file to keep it in sync with the full query list."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "category_path", "score"])
        for item in all_results:
            query = item.get("query", "")
            categories = item.get("categories", [])
            if categories:
                for cat in categories:
                    writer.writerow([query, cat["path"], cat["score"]])
            else:
                writer.writerow([query, "", ""])


def append_to_output(output_path, new_results):
    # type: (str, List[Dict]) -> None
    """Append newly mapped results to the output CSV (creates file + header if missing)."""
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["query", "category_path", "score"])
        for item in new_results:
            query = item.get("query", "")
            categories = item.get("categories", [])
            if categories:
                for cat in categories:
                    writer.writerow([query, cat["path"], cat["score"]])
            else:
                writer.writerow([query, "", ""])


# ---------------------------------------------------------------------------
# 7.  Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Map search queries to product categories using Gemini.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N queries (for testing)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Queries per API call (default: {})".format(DEFAULT_BATCH_SIZE))
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output CSV file (default: {})".format(OUTPUT_FILE))
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignoring previous progress and output")
    args = parser.parse_args()

    # Validate API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("GOOGLE_API_KEY environment variable is not set.")
        sys.exit(1)

    base_dir = Path(__file__).parent
    output_path = str(base_dir / args.output)
    progress_path = str(base_dir / PROGRESS_FILE)

    # Step 1: Extract category paths and create ID mapping
    log.info("Step 1/4: Extracting category paths and creating ID mapping from %s …", INPUT_FEED)
    category_paths = extract_category_paths(str(base_dir / INPUT_FEED))
    log.info("  Found %d unique category paths.", len(category_paths))
    
    # Create bi-directional mapping
    id_to_path = {"CAT_{:04d}".format(i+1): path for i, path in enumerate(category_paths)}
    # valid_paths = set(category_paths)  # No longer used, replaced by id_to_path check

    # Step 2: Read queries
    log.info("Step 2/4: Reading queries from %s …", INPUT_QUERIES)
    queries = read_queries(str(base_dir / INPUT_QUERIES), limit=args.limit)
    log.info("  Loaded %d queries.", len(queries))

    # Step 2b: Load already-completed mappings (from output CSV + in-flight progress)
    if args.no_resume:
        completed = {}  # type: Dict[str, List]
        progress = {}   # type: Dict[str, List]
        # Remove old output & progress files for a clean start
        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(progress_path):
            os.remove(progress_path)
        log.info("  Fresh start — cleared previous output and progress.")
    else:
        # 1) Load queries already written to the output CSV (from previous completed runs)
        completed = load_existing_mappings(output_path)
        # 2) Load queries from in-flight progress (from a crashed/interrupted run)
        progress = load_progress(progress_path)
        already_done = len(completed) + len(set(progress.keys()) - set(completed.keys()))
        if already_done:
            log.info("  Found %d queries already mapped (%d from output CSV, %d from in-flight progress).",
                     already_done, len(completed), len(progress))

    # Merge: anything in completed or progress is considered done
    all_done = set(completed.keys()) | set(progress.keys())
    pending_queries = [q for q in queries if q not in all_done]

    if not pending_queries:
        log.info("  All %d queries already mapped. Nothing to do!", len(queries))
        # Still write a unified output in case progress file had entries not in CSV
        if progress:
            _write_unified_output(output_path, queries, completed, progress)
        log.info("Done!")
        return

    # Step 3: Create cache & process only new queries
    log.info("Step 3/4: Creating Gemini context cache with IDs …")
    client = genai.Client(api_key=api_key)
    prompt_path = str(base_dir / PROMPT_FILE)
    cache_name = create_cache(client, id_to_path, prompt_path)

    # If caching failed (content too small), load system text for per-request use
    system_text = None
    if cache_name is None:
        system_text = load_prompt(prompt_path, id_to_path)
        log.info("  Using non-cached mode with system instruction per request.")

    total_batches = (len(pending_queries) + args.batch_size - 1) // args.batch_size
    log.info("  Processing %d NEW queries in %d batches (batch size = %d) — skipping %d already mapped …",
             len(pending_queries), total_batches, args.batch_size, len(all_done))

    newly_mapped = []  # type: List[Dict]
    total_usage = {"total_input": 0, "output": 0, "cached": 0, "thoughts": 0}
    for batch_idx in range(0, len(pending_queries), args.batch_size):
        batch = pending_queries[batch_idx : batch_idx + args.batch_size]
        batch_num = batch_idx // args.batch_size + 1

        log.info("  Batch %d/%d  (%d queries) …", batch_num, total_batches, len(batch))
        results, batch_usage = call_llm(client, cache_name, batch, id_to_path, system_text=system_text, batch_num=batch_num)

        # Accumulate token usage
        for key in total_usage:
            total_usage[key] += batch_usage.get(key, 0)
        batch_cost = calculate_cost(batch_usage)
        log.info("    Tokens — new input: %d, cached: %d, output: %d | Cost: $%.6f (Raw: $%.6f)",
                 max(0, batch_usage["total_input"] - batch_usage["cached"]), 
                 batch_usage["cached"], 
                 batch_usage["output"], 
                 batch_cost["total_cost"],
                 batch_cost["raw_total_cost"])

        # Store in progress dict (for crash recovery) and newly_mapped (for appending)
        for item in results:
            q = item["query"]
            progress[q] = item["categories"]
            newly_mapped.append(item)

        # Save progress after every batch (crash recovery)
        save_progress(progress_path, progress)

        # Append newly mapped results to CSV after every batch
        append_to_output(output_path, results)

    # Step 4: Write unified output (ensures order matches query list & includes all)
    log.info("Step 4/4: Writing unified results to %s …", args.output)
    _write_unified_output(output_path, queries, completed, progress)

    # Clean up cache to avoid unnecessary costs
    if cache_name:
        try:
            client.caches.delete(name=cache_name)
            log.info("Deleted context cache: %s", cache_name)
        except Exception as e:
            log.warning("Could not delete cache: %s", e)

    # Clean up progress file on successful completion
    if os.path.exists(progress_path):
        os.remove(progress_path)
        log.info("Cleaned up progress file.")

    # Log total token usage and cost summary
    total_cost = calculate_cost(total_usage)
    log.info("=" * 60)
    log.info("TOKEN USAGE SUMMARY")
    log.info("  New Input tokens: %d", max(0, total_usage["total_input"] - total_usage["cached"]))
    log.info("  Cached tokens:    %d", total_usage["cached"])
    log.info("  Output tokens:    %d", total_usage["output"])
    log.info("  Total Input (incl. cached): %d", total_usage["total_input"])
    log.info("COST SUMMARY (includes 50%% Batch API discount)")
    log.info("  New Input cost: $%.6f", total_cost["input_cost"])
    log.info("  Cached cost:    $%.6f", total_cost["cache_cost"])
    log.info("  Output cost:    $%.6f", total_cost["output_cost"])
    log.info("  Raw Total:      $%.6f", total_cost["raw_total_cost"])
    log.info("  Discounted:     $%.6f (Batch: -50%%)", total_cost["total_cost"])
    log.info("=" * 60)

    log.info("Done! %d total queries in output (%d newly mapped this run) → %s",
             len(queries), len(newly_mapped), args.output)


def _write_unified_output(output_path, queries, completed, progress):
    # type: (str, List[str], Dict, Dict) -> None
    """Write a clean, unified output CSV combining completed + progress data."""
    all_results = []
    for q in queries:
        # Progress (from this run) takes priority over completed (from previous runs)
        cats = progress.get(q, completed.get(q, []))
        all_results.append({"query": q, "categories": cats})
    write_output(output_path, all_results)


if __name__ == "__main__":
    main()
