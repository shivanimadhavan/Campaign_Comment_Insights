"""WEFRA LIFE: process one CSV delivery through the complete pipeline.

Examples (run from the project root):
    python main.py data/delivery_1
    python main.py data/delivery_2
    python main.py data/delivery_1 --ingest-only
    python main.py data/delivery_2 --max-comments 5 --skip-fda
"""

import argparse
import os
import sys
from pathlib import Path

from src.pipeline import process_delivery


PROJECT_ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest one delivery, classify comments, fetch openFDA data, and export results."
    )
    parser.add_argument(
        "delivery_folder", type=Path,
        help="Delivery folder, e.g. data/delivery_1",
    )
    parser.add_argument(
        "--ingest-only", action="store_true",
        help="Only clean and load the delivery into SQLite; skip enrichment and export.",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip Gemini classification (existing classifications are preserved).",
    )
    parser.add_argument(
        "--skip-fda", action="store_true",
        help="Skip the openFDA refresh (existing reactions are preserved).",
    )
    parser.add_argument(
        "--max-comments", type=int, default=None, metavar="N",
        help="Limit NEW successful LLM classifications for a test run.",
    )
    args = parser.parse_args(argv)

    if args.max_comments is not None and args.max_comments < 0:
        parser.error("--max-comments must be zero or greater")

    folder = args.delivery_folder.expanduser()
    if not folder.is_absolute() and not folder.is_dir():
        folder = PROJECT_ROOT / folder
    if not folder.is_dir():
        parser.error(f"Delivery folder does not exist: {folder}")

    # Fail quickly if Gemini cannot be used. Otherwise classifier.py would retry
    # each individual comment several times, potentially thousands of times.
    needs_gemini = (
        not args.ingest_only
        and not args.skip_llm
        and args.max_comments != 0
    )
    if needs_gemini:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        if not os.getenv("GEMINI_API_KEY"):
            parser.error(
                "GEMINI_API_KEY is missing. Add it to .env, or run with "
                "--ingest-only / --skip-llm."
            )

    try:
        print(f"\n[1/4] Ingesting {folder.name} ...", flush=True)
        process_delivery(folder)

        if args.ingest_only:
            print("\nIngestion complete. LLM, openFDA, and CSV export were skipped.")
            return 0

        if args.skip_llm:
            print("\n[2/4] Gemini classification skipped (--skip-llm).")
        else:
            from src.classifier import classify_pending

            print("\n[2/4] Classifying pending comments with Gemini ...", flush=True)
            classification = classify_pending(max_comments=args.max_comments)
            print(f"Classification summary: {classification}")

        if args.skip_fda:
            print("\n[3/4] openFDA refresh skipped (--skip-fda).")
        else:
            from src.fda_client import refresh_fda_reactions

            print("\n[3/4] Refreshing openFDA reactions ...", flush=True)
            fda = refresh_fda_reactions()
            print(f"openFDA summary: {fda}")
            if fda["failed_products"]:
                print(
                    "WARNING: Some openFDA products could not be refreshed; "
                    "see errors above.",
                    file=sys.stderr,
                )

        from src.export import export_classifications

        print("\n[4/4] Exporting current classifications ...", flush=True)
        exported = export_classifications()
        print(f"\nExport saved to: {exported['path']}")

        if exported["pending_classification"]:
            print(
                f"WARNING: {exported['pending_classification']} nonblank comments "
                "are still awaiting a current classification.",
                file=sys.stderr,
            )
            # A full, unbounded run must not appear to have succeeded when
            # required classifications are missing. Limited/skip runs are
            # intentionally partial and are clearly reported as such.
            if not args.skip_llm and args.max_comments is None:
                return 1
        if exported["blank_text"]:
            print(
                f"NOTE: {exported['blank_text']} comments have blank text and "
                "are excluded from the classification export."
            )
        print("\nPipeline finished.")
        return 0

    except (OSError, ValueError, RuntimeError) as exc:
        print(f"\nPipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
