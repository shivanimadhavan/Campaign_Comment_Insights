"""Validated Gemini comment classification, supporting targeted comment IDs.

Live usage requires GEMINI_API_KEY in .env.
"""

import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from src.database import SessionLocal, Comment, CommentAnalysis


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
PROMPT_PATH = BASE_DIR / "prompts" / "classify_comment.txt"
PROMPT_VERSION = "v1"  # Increase when you edit the prompt.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
REQUEST_DELAY = float(os.getenv("GEMINI_REQUEST_DELAY", "0"))


# --------------------------------------------------
# Classification schema
# --------------------------------------------------

class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: Literal[
        "effect", "side_effect", "price_availability",
        "usage_question", "other", "spam",
    ]
    sentiment: Literal["positive", "neutral", "negative"]
    side_effect_flag: bool
    reason: str = Field(min_length=3, max_length=600)

    @model_validator(mode="after")
    def consistent_side_effect(self):
        if self.side_effect_flag != (self.topic == "side_effect"):
            raise ValueError("side_effect topic and flag must agree")
        return self


# Gemini-compatible schema, separate from Pydantic's schema.
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "topic": {
            "type": "STRING",
            "enum": [
                "effect", "side_effect", "price_availability",
                "usage_question", "other", "spam",
            ],
        },
        "sentiment": {
            "type": "STRING",
            "enum": ["positive", "neutral", "negative"],
        },
        "side_effect_flag": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"},
    },
    "required": ["topic", "sentiment", "side_effect_flag", "reason"],
}


# --------------------------------------------------
# Gemini API
# --------------------------------------------------

@lru_cache(maxsize=1)
def _gemini_client():
    """Reuse one client instead of creating one per comment."""
    from google import genai

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing. Check .env.")
    return genai.Client(api_key=key)


def gemini_predict(comment_text, prompt):
    """Classify a single comment with Gemini."""
    from google.genai import types

    response = _gemini_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{prompt}\n\nCOMMENT TO CLASSIFY:\n{comment_text}",
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=GEMINI_RESPONSE_SCHEMA,
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned no classification")
    return Classification.model_validate_json(response.text)


def _retry_delay_seconds(exc):
    """Read the suggested wait time from a Gemini 429 error, if supplied."""
    message = str(exc)
    if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
        return None
    match = re.search(r"retry in\s+([0-9.]+)s", message, flags=re.I)
    if not match:
        match = re.search(r"retryDelay['\"]?\s*:\s*['\"]([0-9.]+)s", message)
    return max(1.0, float(match.group(1)) + 1.0) if match else 60.0


# --------------------------------------------------
# Classify pending comments
# --------------------------------------------------

def classify_pending(predict=gemini_predict, max_comments=None, comment_ids=None):
    """Classify new/edited comments, or only the IDs supplied in comment_ids.

    comment_ids=None: scan all comments (original pipeline behavior).
    comment_ids=[...]: examine only those IDs (Bonus A behavior).
    Existing current classifications are skipped in both cases.
    """
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Classification prompt not found: {PROMPT_PATH}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Classification prompt is empty")
    if max_comments is not None and max_comments < 0:
        raise ValueError("max_comments must be zero or greater")
    if REQUEST_DELAY < 0:
        raise ValueError("GEMINI_REQUEST_DELAY cannot be negative")

    stats = {"analyzed": 0, "cached": 0, "blank": 0, "failed": 0}

    with SessionLocal() as session:
        query = select(Comment.comment_id)
        if comment_ids is not None:
            # An empty list intentionally selects no comments.
            unique_ids = list(dict.fromkeys(comment_ids))
            query = query.where(Comment.comment_id.in_(unique_ids))
        ids = session.scalars(query.order_by(Comment.comment_id)).all()

    last_request_start = None
    for comment_id in ids:
        if max_comments is not None and stats["analyzed"] >= max_comments:
            break

        with SessionLocal() as session:
            comment = session.get(Comment, comment_id)
            if comment is None:
                continue
            if not comment.comment_text or not comment.comment_text.strip():
                stats["blank"] += 1
                continue
            prior = session.get(CommentAnalysis, comment_id)
            if (
                prior is not None
                and prior.analysed_hash == comment.text_hash
                and prior.prompt_version == PROMPT_VERSION
                and prior.topic is not None
                and prior.sentiment is not None
                and prior.reason
            ):
                stats["cached"] += 1
                continue
            text = comment.comment_text
            hash_value = comment.text_hash

        last_error = None
        result = None
        for attempt in range(3):
            # Pace every request, including retries. Not just successful calls.
            if REQUEST_DELAY > 0 and last_request_start is not None:
                wait = REQUEST_DELAY - (time.monotonic() - last_request_start)
                if wait > 0:
                    time.sleep(wait)
            last_request_start = time.monotonic()

            try:
                candidate = predict(text, prompt)
                result = (
                    candidate if isinstance(candidate, Classification)
                    else Classification.model_validate(candidate)
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    delay = _retry_delay_seconds(exc)
                    time.sleep(delay if delay is not None else 1 + attempt * 2)

        if result is None:
            summary = str(last_error).splitlines()[0][:250]
            print(f"FAILED comment {comment_id}: {type(last_error).__name__}: {summary}")
            stats["failed"] += 1
            continue

        with SessionLocal.begin() as session:
            comment = session.get(Comment, comment_id)
            if comment is None or comment.text_hash != hash_value:
                print(f"SKIP {comment_id}: comment changed during classification")
                continue
            analysis = session.get(CommentAnalysis, comment_id)
            if analysis is None:
                analysis = CommentAnalysis(comment_id=comment_id)
                session.add(analysis)
            analysis.topic = result.topic
            analysis.sentiment = result.sentiment
            analysis.side_effect_flag = result.side_effect_flag
            analysis.reason = result.reason
            analysis.analysed_hash = hash_value
            analysis.prompt_version = PROMPT_VERSION

        stats["analyzed"] += 1
        print(
            f"Classified {comment_id}: {result.topic} | {result.sentiment} | "
            f"side_effect={result.side_effect_flag}"
        )

    return stats
