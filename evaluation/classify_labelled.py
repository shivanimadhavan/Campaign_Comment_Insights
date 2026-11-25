"""Classify only the IDs in evaluation/labelled_sample.csv.

Run from project root:
    python -m evaluation.classify_labelled
"""

import csv
from pathlib import Path

from sqlalchemy import select

from src.classifier import PROMPT_VERSION, classify_pending
from src.database import Comment, CommentAnalysis, SessionLocal

ROOT = Path(__file__).resolve().parent.parent
LABELLED_FILE = ROOT / "evaluation" / "labelled_sample.csv"


def main():
    with LABELLED_FILE.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    comment_ids = [row["comment_id"].strip() for row in rows]
    if not comment_ids or any(not cid for cid in comment_ids):
        raise ValueError("The labelled file has no IDs or contains a blank ID")
    if len(comment_ids) != len(set(comment_ids)):
        raise ValueError("The labelled file contains duplicate comment IDs")

    with SessionLocal() as session:
        comments = {
            comment.comment_id: comment
            for comment in session.scalars(
                select(Comment).where(Comment.comment_id.in_(comment_ids))
            )
        }
        missing = sorted(set(comment_ids) - set(comments))
        if missing:
            raise RuntimeError(f"Missing IDs in COMMENTS: {missing}")
        different_text = [
            row["comment_id"]
            for row in rows
            if (comments[row["comment_id"]].comment_text or "")
            != row["comment_text"]
        ]

    print(f"Labelled sample: {len(comment_ids)} comments")
    if different_text:
        print("NOTE: labelled text differs from current COMMENTS text for:")
        print(", ".join(different_text))
        print("Predictions will use current database text; the evaluation flags mismatches.")

    stats = classify_pending(comment_ids=comment_ids)
    print("Classification summary:", stats)

    # Verify all 20 have a current, saved prediction.
    with SessionLocal() as session:
        rows_db = session.execute(
            select(Comment, CommentAnalysis)
            .outerjoin(CommentAnalysis, Comment.comment_id == CommentAnalysis.comment_id)
            .where(Comment.comment_id.in_(comment_ids))
        ).all()
        incomplete = [
            comment.comment_id
            for comment, analysis in rows_db
            if analysis is None
            or analysis.analysed_hash != comment.text_hash
            or analysis.prompt_version != PROMPT_VERSION
            or analysis.topic is None
            or analysis.sentiment is None
            or analysis.side_effect_flag is None
            or not analysis.reason
        ]
    if incomplete:
        print("Still missing/stale classifications:", sorted(incomplete))
        raise SystemExit(1)
    print("SUCCESS: All labelled comments now have current classifications.")
    print("Next run: python -m evaluation.evaluate")


if __name__ == "__main__":
    main()
