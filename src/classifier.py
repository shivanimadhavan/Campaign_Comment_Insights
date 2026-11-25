
"""Validated Gemini comment classification. Requires GEMINI_API_KEY for live use."""

import os
import time
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
PROMPT_VERSION = "v1"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Optional delay between successful API calls (seconds).
# Configure using GEMINI_REQUEST_DELAY in .env.
REQUEST_DELAY = float(os.getenv("GEMINI_REQUEST_DELAY", "0"))


# --------------------------------------------------
# Classification Schema
# --------------------------------------------------

class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: Literal[
        "effect",
        "side_effect",
        "price_availability",
        "usage_question",
        "other",
        "spam",
    ]

    sentiment: Literal[
        "positive",
        "neutral",
        "negative",
    ]

    side_effect_flag: bool

    reason: str = Field(
        min_length=3,
        max_length=600,
    )

    @model_validator(mode="after")
    def consistent_side_effect(self):
        if self.side_effect_flag != (self.topic == "side_effect"):
            raise ValueError(
                "side_effect topic and flag must agree"
            )
        return self


# --------------------------------------------------
# Gemini-Compatible Response Schema
# --------------------------------------------------

# Keep the API response schema separate from the
# Pydantic validation model.
#
# This avoids sending additionalProperties=false,
# which caused the Gemini 400 INVALID_ARGUMENT error.

GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "topic": {
            "type": "STRING",
            "enum": [
                "effect",
                "side_effect",
                "price_availability",
                "usage_question",
                "other",
                "spam",
            ],
        },
        "sentiment": {
            "type": "STRING",
            "enum": [
                "positive",
                "neutral",
                "negative",
            ],
        },
        "side_effect_flag": {
            "type": "BOOLEAN",
        },
        "reason": {
            "type": "STRING",
        },
    },
    "required": [
        "topic",
        "sentiment",
        "side_effect_flag",
        "reason",
    ],
}


# --------------------------------------------------
# Gemini API
# --------------------------------------------------

def gemini_predict(comment_text, prompt):
    """Classify one comment using Gemini."""

    from google import genai
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check your .env file."
        )

    client = genai.Client(api_key=key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=(
            f"{prompt}\n\n"
            f"COMMENT TO CLASSIFY:\n{comment_text}"
        ),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=GEMINI_RESPONSE_SCHEMA,
        ),
    )

    if not response.text:
        raise ValueError(
            "Gemini returned no classification"
        )

    return Classification.model_validate_json(
        response.text
    )


# --------------------------------------------------
# Classify Pending Comments
# --------------------------------------------------

def classify_pending(
    predict=gemini_predict,
    max_comments=None,
):
    """
    Classify comments stored in SQLite.

    Rules:
    - Classify new comments.
    - Reclassify comments with changed text.
    - Reclassify when PROMPT_VERSION changes.
    - Skip already classified unchanged comments.
    - Skip blank comments.
    - Retry failed API requests.
    - Save only validated classifications.
    """

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Classification prompt not found: {PROMPT_PATH}"
        )

    prompt = PROMPT_PATH.read_text(
        encoding="utf-8"
    ).strip()

    if not prompt:
        raise ValueError(
            "Classification prompt is empty."
        )

    if max_comments is not None and max_comments < 0:
        raise ValueError(
            "max_comments must be zero or greater"
        )

    stats = {
        "analyzed": 0,
        "cached": 0,
        "blank": 0,
        "failed": 0,
    }

    # Fetch comment IDs from the database.
    with SessionLocal() as session:
        ids = session.scalars(
            select(Comment.comment_id).order_by(
                Comment.comment_id
            )
        ).all()

    for comment_id in ids:

        # Stop after the requested number of successful analyses.
        if (
            max_comments is not None
            and stats["analyzed"] >= max_comments
        ):
            break

        # ------------------------------------------
        # Read comment and check existing analysis
        # ------------------------------------------

        with SessionLocal() as session:

            comment = session.get(
                Comment,
                comment_id,
            )

            if comment is None:
                continue

            if (
                not comment.comment_text
                or not comment.comment_text.strip()
            ):
                stats["blank"] += 1
                continue

            prior = session.get(
                CommentAnalysis,
                comment_id,
            )

            # Skip unchanged, successfully analyzed comments.
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

        # ------------------------------------------
        # Call Gemini with retries
        # ------------------------------------------

        last_error = None
        result = None

        for attempt in range(3):

            try:
                candidate = predict(
                    text,
                    prompt,
                )

                result = (
                    candidate
                    if isinstance(candidate, Classification)
                    else Classification.model_validate(
                        candidate
                    )
                )

                break

            except Exception as exc:

                last_error = exc

                if attempt < 2:
                    time.sleep(
                        1 + attempt * 2
                    )

        # ------------------------------------------
        # Handle failed classifications
        # ------------------------------------------

        if result is None:

            print(
                f"FAILED comment {comment_id}: "
                f"{type(last_error).__name__}: "
                f"{last_error}"
            )

            stats["failed"] += 1

            continue

        # ------------------------------------------
        # Save validated result to SQLite
        # ------------------------------------------

        with SessionLocal.begin() as session:

            comment = session.get(
                Comment,
                comment_id,
            )

            if (
                comment is None
                or comment.text_hash != hash_value
            ):

                print(
                    f"SKIP {comment_id}: "
                    "comment changed during classification"
                )

                continue

            analysis = session.get(
                CommentAnalysis,
                comment_id,
            )

            if analysis is None:

                analysis = CommentAnalysis(
                    comment_id=comment_id
                )

                session.add(analysis)

            analysis.topic = result.topic
            analysis.sentiment = result.sentiment
            analysis.side_effect_flag = (
                result.side_effect_flag
            )
            analysis.reason = result.reason
            analysis.analysed_hash = hash_value
            analysis.prompt_version = PROMPT_VERSION

        stats["analyzed"] += 1

        print(
            f"Classified {comment_id}: "
            f"{result.topic} | "
            f"{result.sentiment} | "
            f"side_effect={result.side_effect_flag}"
        )

        # Respect optional request pacing.
        if REQUEST_DELAY > 0:
            time.sleep(REQUEST_DELAY)

    return stats
