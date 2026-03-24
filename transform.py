#!/usr/bin/env python3
"""
Query-to-Category Mapper using Gemini 2.5 Flash Lite with Explicit Context Caching.

Called from pipeline.py via run_mapping() with in-memory DataFrames.
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from constants import DEFAULT_PROMPT, LLM_SCORE_THRESHOLD, BATCH_SIZE, CACHE_TTL_SECONDS

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "gemini-2.5-flash-lite"
DEFAULT_BATCH_SIZE = BATCH_SIZE
CACHE_TTL_SECONDS = CACHE_TTL_SECONDS      # 30 minutes
MAX_RETRIES = 5
INITIAL_BACKOFF = 2              # seconds

PROGRESS_FILE = "output/.mapping_progress.json"
MIN_SCORE_THRESHOLD = LLM_SCORE_THRESHOLD

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
    INPUT_PRICE_PER_M = 0.10
    OUTPUT_PRICE_PER_M = 0.40
    CACHE_PRICE_PER_M = 0.025
    THOUGHT_PRICE_PER_M = 0.40
    BATCH_DISCOUNT = 0.5

    new_input_tokens = max(0, token_usage["total_input"] - token_usage["cached"])

    input_cost = (new_input_tokens / 1_000_000) * INPUT_PRICE_PER_M
    cache_cost = (token_usage["cached"] / 1_000_000) * CACHE_PRICE_PER_M
    output_cost = (token_usage["output"] / 1_000_000) * OUTPUT_PRICE_PER_M
    thought_cost = (token_usage["thoughts"] / 1_000_000) * THOUGHT_PRICE_PER_M

    raw_total = input_cost + cache_cost + output_cost + thought_cost

    return {
        "input_cost": input_cost,
        "cache_cost": cache_cost,
        "output_cost": output_cost,
        "total_cost": raw_total * BATCH_DISCOUNT,
        "raw_total_cost": raw_total,
    }


# ---------------------------------------------------------------------------
# 1.  Extract unique category paths from DataFrame
# ---------------------------------------------------------------------------
def extract_category_paths_from_df(product_df):
    # type: (pd.DataFrame) -> List[str]
    """Extract sorted unique e_product_type paths from an in-memory DataFrame."""
    if "e_product_type" not in product_df.columns:
        raise ValueError("product_df must have an 'e_product_type' column")
    paths = product_df["e_product_type"].dropna().str.strip()
    return sorted(paths[paths != ""].unique().tolist())


# ---------------------------------------------------------------------------
# 2.  Extract queries from DataFrame
# ---------------------------------------------------------------------------
def extract_queries_from_df(keywords_df, limit=None):
    # type: (pd.DataFrame, Optional[int]) -> List[str]
    """Extract query strings from an in-memory DataFrame."""
    kw_col = None
    for col in ["keyword", "keywords", "search_query"]:
        if col in keywords_df.columns:
            kw_col = col
            break
    if kw_col is None:
        raise ValueError("keywords_df must have one of: keyword, keywords, search_query. Found: {}".format(list(keywords_df.columns)))
    queries = keywords_df[kw_col].dropna().str.strip()
    queries = queries[queries != ""].tolist()
    if limit:
        queries = queries[:limit]
    return queries


# ---------------------------------------------------------------------------
# 3.  Progress / resume helpers (crash recovery)
# ---------------------------------------------------------------------------
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
    """Read prompt file and append the category list with IDs."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

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
                        "Falling back to non-cached mode.")
            return None
        raise


# ---------------------------------------------------------------------------
# 5.  Call the LLM with retry
# ---------------------------------------------------------------------------
def call_llm(client, cache_name, queries_batch, id_to_path, system_text=None):
    # type: (genai.Client, Optional[str], List[str], Dict[str, str], Optional[str]) -> tuple
    """Send a batch of queries to Gemini and return parsed results with validated paths."""
    queries_text = "\n".join("{}. {}".format(i + 1, q) for i, q in enumerate(queries_batch))
    prompt = USER_PROMPT_TEMPLATE.format(queries=queries_text)

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

            text = response.text.strip()
            parsed = json.loads(text)

            if isinstance(parsed, dict) and "mappings" in parsed:
                results = parsed["mappings"]
            elif isinstance(parsed, list):
                results = parsed
            else:
                results = [parsed]

            validated = []  # type: List[Dict]
            for item in results:
                raw_categories = item.get("categories", [])[:3]
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
                            log.info("Score %d below threshold for ID '%s' → query '%s' — skipping",
                                     score, cat_id, item.get("query", ""))
                    elif cat_id and cat_id.lower() != "null":
                        log.warning("Unknown category ID '%s' for query '%s' — skipping",
                                    cat_id, item.get("query", ""))
                validated.append({
                    "query": item.get("query", ""),
                    "categories": resolved,
                })

            # Reconcile LLM-returned queries back to the original batch
            reconciled = []  # type: List[Dict]
            if len(validated) == len(queries_batch):
                for orig_q, v_item in zip(queries_batch, validated):
                    reconciled.append({"query": orig_q, "categories": v_item["categories"]})
            else:
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

    log.error("All retries exhausted for batch. Returning empty mappings (marked as failed).")
    return [{"query": q, "categories": [], "failed": True} for q in queries_batch], {"total_input": 0, "output": 0, "cached": 0, "thoughts": 0}


# ---------------------------------------------------------------------------
# 6.  Main entry point (called from pipeline.py)
# ---------------------------------------------------------------------------
def run_mapping(product_df, keywords_df, prompt_template=DEFAULT_PROMPT,
                score_threshold=LLM_SCORE_THRESHOLD, batch_size=DEFAULT_BATCH_SIZE,
                limit=None, no_resume=False):
    # type: (pd.DataFrame, pd.DataFrame, str, int, int, Optional[int], bool) -> pd.DataFrame
    """Run the full mapping pipeline using in-memory DataFrames.

    Returns:
        DataFrame with columns [query, category_path, score]
    """
    global MIN_SCORE_THRESHOLD
    MIN_SCORE_THRESHOLD = score_threshold

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

    base_dir = Path(__file__).parent
    progress_path = str(base_dir / PROGRESS_FILE)
    prompt_path = str(base_dir / "data" / prompt_template)

    # Step 1: Extract category paths
    log.info("Step 1/3: Extracting category paths from product feed …")
    category_paths = extract_category_paths_from_df(product_df)
    log.info("  Found %d unique category paths.", len(category_paths))

    id_to_path = {"CAT_{:04d}".format(i+1): path for i, path in enumerate(category_paths)}

    # Step 2: Extract queries
    log.info("Step 2/3: Extracting queries from keywords …")
    queries = extract_queries_from_df(keywords_df, limit=limit)
    log.info("  Loaded %d queries.", len(queries))

    # Load crash-recovery progress
    if no_resume:
        progress = {}  # type: Dict[str, List]
        if os.path.exists(progress_path):
            os.remove(progress_path)
        log.info("  Fresh start — cleared previous progress.")
    else:
        progress = load_progress(progress_path)
        if progress:
            log.info("  Found %d queries from previous in-flight progress.", len(progress))

    pending_queries = [q for q in queries if q not in progress]

    if not pending_queries:
        log.info("  All %d queries already mapped. Nothing to do!", len(queries))
        return _build_result_df(queries, progress)

    # Step 3: Create cache & process
    log.info("Step 3/3: Creating Gemini context cache …")
    client = genai.Client(api_key=api_key)
    cache_name = create_cache(client, id_to_path, prompt_path)

    system_text = None
    if cache_name is None:
        system_text = load_prompt(prompt_path, id_to_path)
        log.info("  Using non-cached mode with system instruction per request.")

    total_batches = (len(pending_queries) + batch_size - 1) // batch_size
    log.info("  Processing %d NEW queries in %d batches (batch size = %d) …",
             len(pending_queries), total_batches, batch_size)

    newly_mapped_count = 0
    total_usage = {"total_input": 0, "output": 0, "cached": 0, "thoughts": 0}
    for batch_idx in range(0, len(pending_queries), batch_size):
        batch = pending_queries[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        log.info("  Batch %d/%d  (%d queries) …", batch_num, total_batches, len(batch))
        results, batch_usage = call_llm(client, cache_name, batch, id_to_path, system_text=system_text)

        for key in total_usage:
            total_usage[key] += batch_usage.get(key, 0)
        batch_cost = calculate_cost(batch_usage)
        log.info("    Tokens — new input: %d, cached: %d, output: %d | Cost: $%.6f",
                 max(0, batch_usage["total_input"] - batch_usage["cached"]),
                 batch_usage["cached"],
                 batch_usage["output"],
                 batch_cost["total_cost"])

        successful_results = [item for item in results if not item.get("failed")]
        failed_results = [item for item in results if item.get("failed")]

        if failed_results:
            failed_queries = [item["query"] for item in failed_results]
            log.warning("  %d queries failed in batch %d and will be retried on next run: %s",
                        len(failed_results), batch_num, failed_queries[:5])

        for item in successful_results:
            progress[item["query"]] = item["categories"]
            newly_mapped_count += 1

        save_progress(progress_path, progress)

    # Cleanup
    if cache_name:
        try:
            client.caches.delete(name=cache_name)
            log.info("Deleted context cache: %s", cache_name)
        except Exception as e:
            log.warning("Could not delete cache: %s", e)

    if os.path.exists(progress_path):
        os.remove(progress_path)

    # Log cost summary
    total_cost = calculate_cost(total_usage)
    log.info("=" * 60)
    log.info("TOKEN USAGE SUMMARY")
    log.info("  New Input tokens: %d", max(0, total_usage["total_input"] - total_usage["cached"]))
    log.info("  Cached tokens:    %d", total_usage["cached"])
    log.info("  Output tokens:    %d", total_usage["output"])
    log.info("COST SUMMARY (includes 50%% Batch API discount)")
    log.info("  Raw Total:      $%.6f", total_cost["raw_total_cost"])
    log.info("  Discounted:     $%.6f (Batch: -50%%)", total_cost["total_cost"])
    log.info("=" * 60)
    log.info("Done! %d total queries (%d newly mapped this run)", len(queries), newly_mapped_count)

    return _build_result_df(queries, progress)


def _build_result_df(queries, progress):
    # type: (List[str], Dict) -> pd.DataFrame
    """Build a results DataFrame from progress data."""
    rows = []
    for q in queries:
        cats = progress.get(q, [])
        if cats:
            for cat in cats:
                rows.append({"query": q, "category_path": cat["path"], "score": cat["score"]})
        else:
            rows.append({"query": q, "category_path": "", "score": ""})
    return pd.DataFrame(rows)
