
import pandas as pd

INPUT_FILE = "typesafe_eligibility_results.csv"
OUTPUT_FILE = "typesafe_rescored_results.csv"
REVIEW_FILE = "typesafe_human_review.csv"

HIGH = 0.85
LOW = 0.15

EXCLUSION_COLUMNS = [
    "ex01_biological_outcomes",
    "ex02_industrial_technical",
    "ex03_preclinical_biomedical",
    "ex04_ai_algorithm_only",
    "ex05_logistics_efficiency",
    "ex06_non_english",
    "ex07_non_academic",
]

INCLUSION_COLUMNS = [
    "in01_primary_data",
    "in02_problematic_dependence",
    "in03_ai_use_dependence",
    "in04_dependence_mechanisms",
    "in05_demographics",
    "in06_overreliance",
    "in07_addiction",
    "in08_dependency",
]


def rescore(row):
    inclusions = row[INCLUSION_COLUMNS]
    exclusions = row[EXCLUSION_COLUMNS]

    if inclusions.isna().any() or exclusions.isna().any():
        return "Needs Review"

    max_inclusion = inclusions.max()
    max_exclusion = exclusions.max()

    # Provisional exclusion based on low relevance.
    if max_inclusion <= LOW:
        return "Exclude Candidate"

    # Strong inclusion evidence and low exclusion scores.
    if max_inclusion >= HIGH and max_exclusion <= LOW:
        return "Include"

    return "Needs Review"


def main():
    df = pd.read_csv(INPUT_FILE)

    required = INCLUSION_COLUMNS + EXCLUSION_COLUMNS
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "status" in df.columns:
        if not df["status"].eq("success").all():
            raise ValueError("Some API results are incomplete.")

    df["max_inclusion_score"] = df[INCLUSION_COLUMNS].max(axis=1)
    df["max_exclusion_score"] = df[EXCLUSION_COLUMNS].max(axis=1)

    df["original_recommendation"] = df["screening_recommendation"]
    df["revised_recommendation"] = df.apply(rescore, axis=1)

    df["recommendation_changed"] = (
        df["original_recommendation"] != df["revised_recommendation"]
    )

    df.to_csv(OUTPUT_FILE, index=False)

    review = df[
        [
            "Title",
            "Abstract",
            "original_recommendation",
            "revised_recommendation",
            "max_inclusion_score",
            "max_exclusion_score",
        ]
    ].copy()

    review["human_decision"] = ""
    review["reviewer_notes"] = ""

    review.to_csv(REVIEW_FILE, index=False)

    print("\nOriginal recommendations:")
    print(df["original_recommendation"].value_counts())

    print("\nRevised recommendations:")
    print(df["revised_recommendation"].value_counts())

    print("\nArticles with changed recommendations:")
    changed = df[df["recommendation_changed"]]

    for _, row in changed.iterrows():
        print(
            f"- {row['Title']}: "
            f"{row['original_recommendation']} -> "
            f"{row['revised_recommendation']}"
        )

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Saved: {REVIEW_FILE}")


if __name__ == "__main__":
    main()
