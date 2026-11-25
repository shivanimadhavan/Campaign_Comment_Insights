"""Fetch aggregate adverse-event reaction counts from openFDA into SQLite.

Run from project root: python -m src.fda_client
No API key required. These are report counts, not adverse reaction rates or causality.
"""

import argparse
import re
import time
from datetime import datetime, timezone

import requests
from sqlalchemy import delete, select

from src.database import FDAReaction, Product, SessionLocal, init_db

API_URL = "https://api.fda.gov/drug/event.json"
REACTION_FIELD = "patient.reaction.reactionmeddrapt.exact"
TOP_REACTIONS = 10


def normalize_ingredient(raw: str) -> str:
    """Remove dosage/form strength, and expand common ingredient abbreviation."""
    value = (raw or "").strip()
    value = re.sub(r"\s+\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|%)\b.*$", "", value, flags=re.I)
    value = re.sub(r"\bHCl\b", "hydrochloride", value, flags=re.I)
    return " ".join(value.split())


def get_top_reactions(ingredient: str, limit: int = TOP_REACTIONS) -> list[dict]:
    """Use server-side counting; never download individual patient reports."""
    safe_ingredient = ingredient.replace('"', '').replace('\\', '')
    params = {
        "search": f'patient.drug.openfda.substance_name:"{safe_ingredient}"',
        "count": REACTION_FIELD,
        "limit": limit,
    }
    for attempt in range(3):
        try:
            response = requests.get(API_URL, params=params, timeout=25)
            if response.status_code == 404:
                raise LookupError(f"No openFDA reports matched ingredient {ingredient!r}")
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            data = response.json()
            results = data.get("results")
            if not isinstance(results, list):
                raise ValueError("openFDA response missing results list")
            cleaned = []
            for item in results:
                term, count = item.get("term"), item.get("count")
                if isinstance(term, str) and term.strip() and type(count) is int and count >= 0:
                    cleaned.append({"reaction": term.strip(), "report_count": count})
            if not cleaned:
                raise ValueError("openFDA returned no usable reaction counts")
            return cleaned
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("openFDA request failed after retries")


def refresh_fda_reactions(limit: int = TOP_REACTIONS) -> dict:
    """Update each product atomically; preserve stored results on API failures."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    init_db()
    with SessionLocal() as session:
        products = session.execute(select(Product.product_id, Product.product_name, Product.active_ingredient)).all()

    stats = {"updated_products": 0, "failed_products": 0, "reactions_saved": 0}
    if not products:
        print("No products found. First run: python main.py data/delivery_1")
        return stats

    for product_id, name, raw_ingredient in products:
        ingredient = normalize_ingredient(raw_ingredient)
        if not ingredient:
            stats["failed_products"] += 1
            print(f"SKIP {name}: no active ingredient")
            continue
        try:
            reactions = get_top_reactions(ingredient, limit)
            with SessionLocal.begin() as session:
                session.execute(delete(FDAReaction).where(FDAReaction.product_id == product_id))
                session.add_all([
                    FDAReaction(product_id=product_id, reaction=r["reaction"], report_count=r["report_count"])
                    for r in reactions
                ])
            stats["updated_products"] += 1
            stats["reactions_saved"] += len(reactions)
            print(f"SAVED {name} ({ingredient}): {len(reactions)} reaction counts")
        except (requests.RequestException, LookupError, ValueError) as exc:
            stats["failed_products"] += 1
            print(f"FAILED {name} ({ingredient}): {type(exc).__name__}: {exc}")
    print(f"Updated at {datetime.now(timezone.utc).isoformat()} (UTC)")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh openFDA reported reactions for each product")
    parser.add_argument("--limit", type=int, default=TOP_REACTIONS, help="Top reactions per ingredient (default 10)")
    args = parser.parse_args()
    print(refresh_fda_reactions(limit=args.limit))
