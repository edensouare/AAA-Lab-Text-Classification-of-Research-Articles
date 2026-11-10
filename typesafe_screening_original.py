"""
TypeSafe semantic screening for the AAA Lab AI-trust / overreliance review.

Input: CSV containing at least Title and Abstract columns.
Output: original rows plus TypeSafe Noul probabilities and a conservative
screening-priority label.

Install:
    python -m pip install typesafe-sdk pandas

Set TYPESAFE_API_KEY in your environment before running.

Pilot:
    python typesafe_screening.py \
        --input "Systematic Review about AI - Duplicate check.csv" \
        --output "typesafe_pilot_results.csv" \
        --limit 25

Full original dataset:
    python typesafe_screening.py \
        --input "Systematic Review about AI - Duplicate check.csv" \
        --output "typesafe_results.csv"

This script deliberately does not auto-exclude papers. Thresholds are
triage defaults only and must be calibrated against human-reviewed records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient


MODEL = "jev-1.14.0"

# These questions are intentionally independent: one article can address
# trust, overreliance, and problematic use at the same time.
QUESTIONS = {
    "ai_is_central": Noul(
        instructions=(
            "Is an artificial intelligence system, AI-enabled tool, "
            "chatbot, conversational agent, or large language model a "
            "substantive subject of this study?"
        ),
        criteria=NoulCriteria(
            true=(
                "The research meaningfully investigates an AI system, "
                "people's interaction with it, its effects, or its use. "
                "AI is not merely incidental background."
            ),
            false=(
                "AI is absent, only mentioned in passing, or is not a "
                "meaningful subject of the research."
            ),
        ),
    ),
    "trust_in_ai": Noul(
        instructions=(
            "Does this study investigate people's trust in, trust "
            "calibration toward, or willingness to rely on an AI system?"
        ),
        criteria=NoulCriteria(
            true=(
                "The study examines trust, distrust, perceived "
                "trustworthiness, appropriate trust, trust calibration, "
                "or willingness to rely on an AI system."
            ),
            false=(
                "Trust in an AI system is not investigated. A generic "
                "mention of trust, or trust in a non-AI person/system, "
                "does not count."
            ),
        ),
    ),
    "ai_overreliance": Noul(
        instructions=(
            "Does this study investigate excessive or inappropriate "
            "reliance on AI-generated information, recommendations, "
            "predictions, or decisions?"
        ),
        criteria=NoulCriteria(
            true=(
                "The study investigates overreliance, automation bias, "
                "over-trust, uncritical acceptance, inappropriate "
                "delegation, or failure to verify AI outputs or decisions."
            ),
            false=(
                "The study concerns ordinary AI use or appropriate "
                "reliance without investigating excessive, inappropriate, "
                "or uncritical reliance."
            ),
        ),
    ),
    "problematic_ai_use": Noul(
        instructions=(
            "Does this study investigate problematic, compulsive, "
            "addictive, excessive, or dependent use of an AI system?"
        ),
        criteria=NoulCriteria(
            true=(
                "The study examines problematic or compulsive engagement "
                "with AI, AI addiction, excessive AI use, or psychological "
                "or behavioral dependence on an AI system."
            ),
            false=(
                "Dependence refers only to technical dependency, model "
                "dependence, statistical dependence, or general reliance "
                "without problematic user behavior; or the study does not "
                "investigate problematic AI use."
            ),
        ),
    ),
    "empirical_human_research": Noul(
        instructions=(
            "Does this paper report original empirical research involving "
            "human participants or human behavioral/interaction data?"
        ),
        criteria=NoulCriteria(
            true=(
                "The paper reports original observations, experiments, "
                "surveys, interviews, or analyses of human behavioral or "
                "interaction data."
            ),
            false=(
                "The paper is a commentary, conceptual discussion, "
                "technical system paper without human data, review, "
                "meta-analysis, or research protocol."
            ),
        ),
    ),
}


def make_state(title: Any, abstract: Any) -> dict[str, str]:
    title_text = "" if pd.isna(title) else str(title).strip()
    abstract_text = "" if pd.isna(abstract) else str(abstract).strip()
    return {
        "title": title_text,
        "abstract": abstract_text,
    }


def row_key(title: str, abstract: str, model: str) -> str:
    raw = f"{model}\n{title}\n{abstract}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def extract_probability(response: Any, question_id: str) -> float:
    """Read the documented Python SDK response shape."""
    answer = response.answers[question_id]
    value = float(answer.noul)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Unexpected probability for {question_id}: {value}")
    return value


def priority_label(scores: dict[str, float]) -> str:
    """
    Conservative triage only; not a final inclusion/exclusion decision.

    These starting thresholds are not validated and should be calibrated
    using an adjudicated human-reviewed sample.
    """
    ai = scores["ai_is_central"]
    target_scores = [
        scores["trust_in_ai"],
        scores["ai_overreliance"],
        scores["problematic_ai_use"],
    ]
    strongest_target = max(target_scores)

    if ai >= 0.70 and strongest_target >= 0.70:
        return "priority_human_review"
    if ai >= 0.35 and strongest_target >= 0.35:
        return "human_review"
    if ai < 0.20 and strongest_target < 0.20:
        return "lower_priority_review"
    return "uncertain_review"


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                cache[item["row_key"]] = item
            except (json.JSONDecodeError, KeyError):
                # Ignore an incomplete final line after an interrupted run.
                continue
    return cache


def append_cache(cache_path: Path, item: dict[str, Any]) -> None:
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default="typesafe_screening_results.csv")
    parser.add_argument("--cache", default="typesafe_screening_cache.jsonl")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N uncached rows (0 means all rows)",
    )
    parser.add_argument("--pause", type=float, default=0.0)
    args = parser.parse_args()

    if not os.getenv("TYPESAFE_API_KEY"):
        raise EnvironmentError(
            "TYPESAFE_API_KEY is not set. Set it in your shell; "
            "do not paste your API key into this script."
        )

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache)

    df = pd.read_csv(input_path)
    required = {"Title", "Abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    cache = load_cache(cache_path)
    records: list[dict[str, Any]] = []
    uncached_count = 0

    with TypeSafeClient(model=args.model) as client:
        for row_index, row in df.iterrows():
            state = make_state(row["Title"], row["Abstract"])
            key = row_key(state["title"], state["abstract"], args.model)

            if key in cache:
                result = dict(cache[key])
                result["source_row_index"] = int(row_index)
                records.append(result)
                continue

            if args.limit and uncached_count >= args.limit:
                # Keep rows not yet processed in the output with blank scores.
                records.append({
                    "row_key": key,
                    "source_row_index": int(row_index),
                    "model": args.model,
                    "status": "not_run_limit_reached",
                })
                continue

            uncached_count += 1
            try:
                response = client.system_one(
                    state=state,
                    questions=QUESTIONS,
                    model=args.model,
                )

                scores = {
                    question_id: extract_probability(response, question_id)
                    for question_id in QUESTIONS
                }

                result = {
                    "row_key": key,
                    "source_row_index": int(row_index),
                    "model": getattr(response, "model", args.model),
                    "input_tokens": getattr(
                        getattr(response, "usage", None), "input_tokens", None
                    ),
                    "output_tokens": getattr(
                        getattr(response, "usage", None), "output_tokens", None
                    ),
                    **scores,
                    "strongest_target_dimension": max(
                        ("trust_in_ai", "ai_overreliance", "problematic_ai_use"),
                        key=lambda name: scores[name],
                    ),
                    "screening_priority": priority_label(scores),
                    "status": "success",
                }
                append_cache(cache_path, result)
                result["source_row_index"] = int(row_index)
                records.append(result)

                print(
                    f"Processed row {row_index + 1}/{len(df)}: "
                    f"{result['screening_priority']}"
                )

                if args.pause > 0:
                    time.sleep(args.pause)

            except Exception as exc:
                error_result = {
                    "row_key": key,
                    "source_row_index": int(row_index),
                    "model": args.model,
                    "status": "error",
                    "error": repr(exc),
                }
                # Do not cache failures as completed results; rerunning the
                # script will retry this row.
                records.append(error_result)
                print(f"ERROR on row {row_index}: {exc}")

    results_df = pd.DataFrame(records)
    # Merge predictions back to the exact original rows by their source index.
    results_df = results_df.drop_duplicates(
        subset=["source_row_index"], keep="last"
    ).set_index("source_row_index")
    merged = df.join(results_df, how="left")
    merged.to_csv(output_path, index=False)

    print(f"\nInput rows: {len(df)}")
    print(f"Output: {output_path.resolve()}")
    print(f"Cache: {cache_path.resolve()}")
    if "screening_priority" in merged.columns:
        print("\nScreening priority counts:")
        print(merged["screening_priority"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
