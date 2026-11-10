"""
TypeSafe semantic screening for the AAA Lab AI-trust / overreliance review.

Input: CSV containing at least Title and Abstract columns.
Output: original rows plus 15 eligibility criterion scores and a conservative
screening recommendation (Include, Exclude, or Needs Review).

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

This script does not make final inclusion/exclusion decisions. Thresholds are
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

import pandas as pd  # pyright: ignore[reportMissingModuleSource]
from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient  # pyright: ignore[reportMissingImports]


MODEL = "jev-1.14.0"

# These questions are intentionally independent: one article can address
# trust, overreliance, and problematic use at the same time.
# The lab's protocol is encoded as separate, auditable binary questions.
# "true" means the criterion is met; for exclusion questions this is a
# reason to exclude. Read baseprompt.txt and eligibilitycriteria.txt alongside
# this script as the authoritative protocol; amend questions if the lab revises it.
CRITERIA = {
    "ex01_biological_outcomes": ("exclusion", "Does the article measure anatomical/structural, electrophysiological, or histological outcomes, including genomic, proteomic or metabolomic outcomes alongside those measures?", "The study meets the specified biological-outcome exclusion.", "The study does not meet this biological-outcome exclusion."),
    "ex02_industrial_technical": ("exclusion", "Is the article focused on agriculture, energy, resources, materials science, drug discovery, or another industrial/technical development?", "Its focus is one of these industrial or technical development areas.", "It is not primarily about industrial or technical development."),
    "ex03_preclinical_biomedical": ("exclusion", "Is the article neurotechnology or biomedical research restricted to laboratory or pre-clinical contexts?", "It is restricted to laboratory or pre-clinical neurotechnology or biomedical research.", "It is not restricted to those contexts."),
    "ex04_ai_algorithm_only": ("exclusion", "Is this an AI algorithm development, performance optimization, or hardware implementation study WITHOUT a health or mental-health focus?", "It is centered on AI algorithm/performance/hardware development without a health or mental-health focus.", "It does not meet that combined condition."),
    "ex05_logistics_efficiency": ("exclusion", "Is the article an AI application in logistics, resource management, or machine efficiency without clear relevance to health or mental health?", "It concerns logistics, resources, or machine efficiency without clear health or mental-health relevance.", "It does not meet that combined condition."),
    "ex06_non_english": ("exclusion", "Is the article NOT published in English? Use explicit language evidence, not author names or geographic location.", "There is explicit evidence the paper is not published in English.", "There is explicit evidence the paper is published in English."),
    "ex07_non_academic": ("exclusion", "Is this a non-academic research paper or non-academic publication?", "The record is explicitly a non-academic publication.", "The record is explicitly an academic research paper."),
    "in01_primary_data": ("inclusion", "Does this article provide primary data about AI dependency, AI overreliance, or AI addiction?", "Original data on AI dependency, overreliance, or addiction are presented.", "It does not present primary data on these topics."),
    "in02_problematic_dependence": ("inclusion", "Does it study problematic dependence, addiction, or excessive immersion leading to psychological or social dysfunction, in the context of AI use?", "It studies such problematic AI-related dependence or dysfunction.", "It does not study such problematic AI-related dependence or dysfunction."),
    "in03_ai_use_dependence": ("inclusion", "Does it research AI use and human dependence on AI?", "It addresses people's use of and dependence on AI.", "It does not address people's dependence on AI."),
    "in04_dependence_mechanisms": ("inclusion", "Does it analyze AI-related cognitive dependence, relational dependence, loss of autonomy, reduced self-efficacy, or social isolation?", "It analyzes one or more of these AI-related dependence mechanisms.", "It does not analyze these AI-related dependence mechanisms."),
    "in05_demographics": ("inclusion", "Does it compare demographic variation (age, gender, socioeconomic status, digital literacy, etc.) in AI use, AI overreliance, or AI dependence?", "It compares demographic variation in the relevant AI-related phenomena.", "It does not compare demographic variation in the relevant AI-related phenomena."),
    "in06_overreliance": ("inclusion", "Does it investigate excessive or uncritical human reliance on AI, algorithms, or automated systems, including automation bias, overtrust, or decision-support dependency?", "It investigates AI overreliance, automation bias, overtrust, or uncritical deferral to AI.", "It does not investigate these forms of AI overreliance."),
    "in07_addiction": ("inclusion", "Does it investigate compulsive or addictive AI use, excessive conversational-AI engagement, or loss of control/self-regulation in AI use?", "It investigates problematic or addictive patterns of AI use.", "It does not investigate problematic or addictive AI use."),
    "in08_dependency": ("inclusion", "Does it investigate human reliance on AI for everyday tasks or cognition, such as cognitive offloading, learned helplessness, or skill degradation?", "It investigates AI dependency in everyday tasks or cognitive functioning.", "It does not investigate this kind of AI dependency."),
}

QUESTIONS = {
    name: Noul(instructions=prompt, criteria=NoulCriteria(true=yes, false=no))
    for name, (_, prompt, yes, no) in CRITERIA.items()
}

# Conservative triage defaults; these are not validated decision thresholds.
HIGH = 0.85
LOW = 0.15


def eligibility_label(scores: dict[str, float]) -> str:
    """A screening recommendation, NOT an independently verified decision."""
    exclusions = [scores[k] for k, v in CRITERIA.items() if v[0] == "exclusion"]
    inclusions = [scores[k] for k, v in CRITERIA.items() if v[0] == "inclusion"]
    # Language/publication status is often unknowable from an abstract.
    # Never automatically exclude on those two model-inferred attributes.
    substantive_exclusions = [scores[k] for k in CRITERIA if k.startswith("ex0") and k not in ("ex06_non_english", "ex07_non_academic")]
    if any(value >= HIGH for value in substantive_exclusions) and max(inclusions) <= LOW:
        return "Exclude"
    if max(inclusions) >= HIGH and all(value <= LOW for value in exclusions):
        return "Include"
    return "Needs Review"


def make_state(title: Any, abstract: Any) -> dict[str, str]:
    title_text = "" if pd.isna(title) else str(title).strip()
    abstract_text = "" if pd.isna(abstract) else str(abstract).strip()
    return {
        "title": title_text,
        "abstract": abstract_text,
    }


def row_key(title: str, abstract: str, model: str) -> str:
    raw = f"eligibility-v1\n{model}\n{title}\n{abstract}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def extract_probability(response: Any, question_id: str) -> float:
    """Read the documented Python SDK response shape."""
    answer = response.answers[question_id]
    value = float(answer.noul)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Unexpected probability for {question_id}: {value}")
    return value




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
    parser.add_argument("--cache", default="typesafe_eligibility_cache.jsonl")
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
                    "screening_recommendation": eligibility_label(scores),
                    "status": "success",
                }
                append_cache(cache_path, result)
                result["source_row_index"] = int(row_index)
                records.append(result)

                print(
                    f"Processed row {row_index + 1}/{len(df)}: "
                    f"{result['screening_recommendation']}"
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
    if "screening_recommendation" in merged.columns:
        print("\nScreening recommendation counts:")
        print(merged["screening_recommendation"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
