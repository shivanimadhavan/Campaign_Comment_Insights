"""Bonus A: compare saved Gemini results with 20 human-labelled examples.

Run from project root:
    python -m evaluation.evaluate
Creates:
    evaluation/results/evaluation_metrics.json
    evaluation/results/comparison.csv
    evaluation/results/incorrect_predictions.csv
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import select

from src.classifier import PROMPT_VERSION, Classification
from src.database import Comment, CommentAnalysis, SessionLocal

ROOT = Path(__file__).resolve().parent.parent
LABELLED_FILE = ROOT / "evaluation" / "labelled_sample.csv"
RESULT_DIR = ROOT / "evaluation" / "results"


def parse_bool(value):
    value = str(value).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Invalid side_effect_flag: {value!r}")


def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def calculate_metrics(records):
    count = len(records)
    tp = sum(r["human_flag"] and r["pred_flag"] for r in records)
    fp = sum(not r["human_flag"] and r["pred_flag"] for r in records)
    fn = sum(r["human_flag"] and not r["pred_flag"] for r in records)
    tn = sum(not r["human_flag"] and not r["pred_flag"] for r in records)
    confusion = defaultdict(Counter)
    for r in records:
        confusion[r["human_topic"]][r["pred_topic"]] += 1
    return {
        "evaluated": count,
        "topic_accuracy": ratio(sum(r["human_topic"] == r["pred_topic"] for r in records), count),
        "sentiment_accuracy": ratio(sum(r["human_sentiment"] == r["pred_sentiment"] for r in records), count),
        "side_effect_precision": ratio(tp, tp + fp),
        "side_effect_recall": ratio(tp, tp + fn),
        "side_effect_f1": ratio(2 * tp, 2 * tp + fp + fn),
        "side_effect_confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "topic_confusion": {key: dict(val) for key, val in confusion.items()},
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with LABELLED_FILE.open(encoding="utf-8-sig", newline="") as file:
        labelled_rows = list(csv.DictReader(file))
    if not labelled_rows:
        raise ValueError("Labelled sample is empty")
    comment_ids = [r["comment_id"].strip() for r in labelled_rows]
    if len(comment_ids) != len(set(comment_ids)):
        raise ValueError("Duplicate labelled comment IDs")

    with SessionLocal() as session:
        db_rows = session.execute(
            select(Comment, CommentAnalysis)
            .outerjoin(CommentAnalysis, Comment.comment_id == CommentAnalysis.comment_id)
            .where(Comment.comment_id.in_(comment_ids))
        ).all()
        by_id = {comment.comment_id: (comment, analysis) for comment, analysis in db_rows}

    comparisons = []
    records = []
    incomplete = []
    for labelled in labelled_rows:
        cid = labelled["comment_id"].strip()
        pair = by_id.get(cid)
        if not pair:
            incomplete.append(f"{cid} (missing comment)")
            continue
        comment, analysis = pair
        if (
            analysis is None
            or analysis.analysed_hash != comment.text_hash
            or analysis.prompt_version != PROMPT_VERSION
            or analysis.topic is None
            or analysis.sentiment is None
            or analysis.side_effect_flag is None
            or not analysis.reason
        ):
            incomplete.append(f"{cid} (missing or stale prediction)")
            continue

        # Validate predictions; reject malformed results rather than counting them.
        predicted = Classification.model_validate({
            "topic": analysis.topic,
            "sentiment": analysis.sentiment,
            "side_effect_flag": analysis.side_effect_flag,
            "reason": analysis.reason,
        })
        human_flag = parse_bool(labelled["side_effect_flag"])
        human_topic = labelled["topic"].strip()
        human_sentiment = labelled["sentiment"].strip()
        if human_flag != (human_topic == "side_effect"):
            raise ValueError(f"Inconsistent human label for {cid}")
        exact = (comment.comment_text or "") == labelled["comment_text"]

        record = {
            "comment_id": cid,
            "text_exact_match": exact,
            "human_topic": human_topic,
            "pred_topic": predicted.topic,
            "human_sentiment": human_sentiment,
            "pred_sentiment": predicted.sentiment,
            "human_flag": human_flag,
            "pred_flag": predicted.side_effect_flag,
        }
        records.append(record)
        comparisons.append({
            "comment_id": cid,
            "text_exact_match": exact,
            "labelled_comment_text": labelled["comment_text"],
            "database_comment_text": comment.comment_text,
            "human_topic": human_topic,
            "predicted_topic": predicted.topic,
            "topic_match": human_topic == predicted.topic,
            "human_sentiment": human_sentiment,
            "predicted_sentiment": predicted.sentiment,
            "sentiment_match": human_sentiment == predicted.sentiment,
            "human_side_effect_flag": str(human_flag).lower(),
            "predicted_side_effect_flag": str(predicted.side_effect_flag).lower(),
            "side_effect_match": human_flag == predicted.side_effect_flag,
            "predicted_reason": predicted.reason,
        })

    if incomplete:
        print("Evaluation stopped: these IDs need current predictions:")
        print("\n".join(incomplete))
        raise SystemExit("Run: python -m evaluation.classify_labelled")

    metrics = calculate_metrics(records)
    exact_records = [r for r in records if r["text_exact_match"]]
    metrics["text_exact_match_count"] = len(exact_records)
    metrics["text_mismatch_ids"] = [r["comment_id"] for r in records if not r["text_exact_match"]]
    metrics["exact_text_subset"] = calculate_metrics(exact_records)
    metrics["note"] = (
        "All-sample agreement includes any text mismatches between the labelled "
        "file and current database. Exact-text subset excludes them. "
        "Review mismatched rows manually; 20 comments are too few for strong accuracy claims."
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(comparisons[0].keys())
    write_csv(RESULT_DIR / "comparison.csv", comparisons, fieldnames)
    incorrect = [
        r for r in comparisons
        if not (r["topic_match"] and r["sentiment_match"] and r["side_effect_match"])
    ]
    write_csv(RESULT_DIR / "incorrect_predictions.csv", incorrect, fieldnames)
    with (RESULT_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"Evaluated: {metrics['evaluated']}")
    print(f"Topic accuracy: {metrics['topic_accuracy']}")
    print(f"Sentiment accuracy: {metrics['sentiment_accuracy']}")
    print(f"Side-effect precision: {metrics['side_effect_precision']}")
    print(f"Side-effect recall: {metrics['side_effect_recall']}")
    print(f"Side-effect false negatives: {metrics['side_effect_confusion']['FN']}")
    print(f"Incorrect predictions: {len(incorrect)}")
    if metrics["text_mismatch_ids"]:
        print("WARNING: labelled/database text mismatch:", metrics["text_mismatch_ids"])
        print("See exact_text_subset in evaluation_metrics.json")
    print(f"Reports saved in: {RESULT_DIR}")


if __name__ == "__main__":
    main()
