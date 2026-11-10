# TypeSafe API Integration for AI Overreliance Literature Screening

## Overview

This branch explores the integration of the **TypeSafe AI API** into an existing machine learning pipeline for screening academic literature on human overreliance, dependence, and addiction related to artificial intelligence.

The original project uses natural language processing (NLP), TF-IDF vectorization, and traditional machine learning classifiers to identify potentially relevant research articles. This branch extends that workflow by testing whether an external AI-based eligibility assessment system can provide more context-sensitive screening decisions.

The objective is to reduce the manual effort required to screen large volumes of academic literature while preserving relevant studies and minimizing false exclusions.

**Current status:** Proof-of-concept integration completed. A 19-article pilot has been evaluated against manually assigned eligibility labels. Further validation is required before full-dataset deployment.

## Research Objectives

This integration investigates three questions:

1. Can TypeSafe AI distinguish research about **human dependence on AI** from research that merely mentions AI, addiction, or dependence in unrelated contexts?
2. Can automated eligibility assessments reduce the number of articles requiring manual screening?
3. Can the system identify irrelevant articles without incorrectly excluding studies that meet the systematic review's inclusion criteria?

The primary consideration is **screening sensitivity**: relevant articles should not be lost during automated filtering.

## Dataset

The original dataset contains **5,255 academic records** retrieved from research databases, including Scopus, PsycINFO, and Web of Science.

The initial TypeSafe pilot uses 19 unique articles selected to test the system's ability to distinguish relevant research from unrelated or misleadingly similar topics.

The pilot includes articles addressing AI overreliance, cognitive dependence, AI-assisted decision-making, addiction, software engineering, biomedical applications, and unrelated psychological research.

## Methodology

### 1. Eligibility Criteria

The screening system evaluates articles using 15 eligibility criteria:

**Seven exclusion criteria**

- Anatomical or genomic research unrelated to human AI dependence
- Industrial or technical AI applications without relevant human outcomes
- Preclinical neurological or biomedical studies
- Algorithm-focused research without relevant human or health implications
- Logistics or resource optimization without relevant human outcomes
- Non-English publications
- Nonacademic publications

**Eight inclusion criteria**

- Primary research on AI overreliance, dependence, or addiction
- Problematic AI dependence or associated social dysfunction
- AI use and dependence
- Cognitive or relational dependence and effects on autonomy
- Demographic differences in AI dependence
- AI overreliance, automation bias, or excessive trust
- AI addiction or compulsive use
- AI dependency and cognitive offloading

Each article is evaluated using its title and abstract.

### 2. TypeSafe API Integration

The Python integration uses the TypeSafe SDK to evaluate each article against the eligibility criteria.

The workflow:

1. Load article metadata from a CSV file using pandas.
2. Extract the article title and abstract.
3. Submit the article for assessment against the 15 criteria.
4. Retrieve criterion-level scores.
5. Apply decision rules to generate a screening recommendation.
6. Export results to CSV.
7. Cache API responses to support reproducibility and avoid unnecessary repeated requests.

The screening recommendations are:

| Recommendation | Interpretation |
|---|---|
| Include | Strong evidence of relevance under the decision rules |
| Exclude Candidate | Low apparent relevance; requires validation before definitive exclusion |
| Needs Review | Insufficient or ambiguous evidence for an automated decision |

### 3. Revised Decision Logic

The initial screening logic produced a large number of `Needs Review` recommendations, including for articles that were clearly outside the review's scope.

To investigate this behavior, a separate rescoring script was developed to apply revised thresholds to existing API responses.

The exploratory rules were:

- **Exclude Candidate:** All eight inclusion scores are at or below 0.15.
- **Include:** At least one inclusion score is at or above 0.85, and all exclusion scores are at or below 0.15.
- **Needs Review:** All remaining cases, including incomplete assessments.

This revision uses cached TypeSafe scores and does not require additional API requests.

These thresholds are experimental and have not been independently validated.

## Pilot Evaluation

### Initial Screening Results

The TypeSafe integration successfully processed all 19 pilot articles.

| Recommendation | Articles |
|---|---:|
| Include | 6 |
| Needs Review | 13 |
| Exclude | 0 |
| **Total** | **19** |

Although the API successfully returned structured eligibility assessments, the initial decision rules did not produce any definitive exclusions.

### Revised Screening Results

The rescoring procedure was evaluated against human-assigned labels consisting of nine excluded articles and ten included articles.

| Human decision | Predicted Exclude | Predicted Include | Needs Review |
|---|---:|---:|---:|
| Exclude (9) | 6 | 0 | 3 |
| Include (10) | 0 | 6 | 4 |
| **Total** | **6** | **6** | **7** |

### Key Findings

- **19/19 articles** were successfully processed.
- **12/19 articles (63.2%)** received correct definitive screening recommendations.
- **7/19 articles (36.8%)** remained unresolved and required human review.
- **6/9 confirmed exclusions (66.7%)** were correctly identified.
- **6/10 confirmed inclusions (60.0%)** were correctly identified.
- **0 false exclusions** were observed in the pilot.
- No definitive recommendations contradicted the assigned human labels.

The percentages for correctly identified inclusions and exclusions reflect the proportion receiving correct definitive recommendations, with `Needs Review` treated as unresolved.

### Interpretation

The pilot suggests that TypeSafe's principal limitation in this configuration is not necessarily incorrect classification, but its tendency to leave articles unresolved.

For example, research on AI-assisted addiction treatment may contain both "AI" and "addiction" without studying addiction to AI. Similarly, software engineering papers may discuss dependence on AI-based infrastructure without investigating human psychological or behavioral dependence.

These distinctions are important for determining eligibility.

The revised screening rules improved the proportion of articles receiving definitive recommendations while preserving all ten human-labeled inclusions from automatic exclusion in this small pilot.

However, the thresholds were explored using the same 19 articles on which they were evaluated. These results should therefore be interpreted as preliminary rather than as evidence of generalizable performance.

## Project Files

| File | Purpose |
|---|---|
| `typesafe_screening.py` | TypeSafe API integration and article screening |
| `rescore_typesafe.py` | Recalculates screening recommendations using existing criterion scores |
| `evaluate_typesafe.py` | Compares screening recommendations against human labels |
| `typesafe_pilot_19_articles.csv` | Pilot input dataset |
| `typesafe_eligibility_results.csv` | API-generated criterion scores and screening recommendations |
| `typesafe_rescored_results.csv` | Revised screening recommendations |
| `typesafe_human_review.csv` | Human-labeled evaluation dataset |
| `typesafe_eligibility_cache.jsonl` | Cached API assessments |
| `requirements.txt` | Python dependencies |

*File availability depends on the current branch contents. Generated outputs may be excluded from version control.*

## Running the Pilot

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Configure API authentication

Set the `TYPESAFE_API_KEY` environment variable using your preferred secret-management method.

Do not commit API credentials to the repository.

### Run the TypeSafe screening pipeline

```bash
python typesafe_screening.py \
  --input "typesafe_pilot_19_articles.csv" \
  --output "typesafe_eligibility_results.csv" \
  --limit 19
```

### Recalculate screening recommendations

```bash
python rescore_typesafe.py
```

### Evaluate against human labels

```bash
python evaluate_typesafe.py
```

The rescoring and evaluation scripts operate on existing CSV files and do not require new API requests.

## Limitations

Several limitations should be considered when interpreting the results:

- **Small sample size:** The pilot contains only 19 articles.
- **Non-independent evaluation:** The revised thresholds were assessed on the same articles used to explore the decision rules.
- **Limited generalizability:** The pilot may not represent the full diversity of the 5,255-record dataset.
- **Criterion ambiguity:** Some articles contain terminology associated with AI dependence without studying the phenomenon itself.
- **Unresolved classifications:** Seven articles still required human review after rescoring.
- **Uncalibrated scores:** Criterion scores should not be interpreted as validated probabilities of article eligibility.
- **Human oversight:** Automated recommendations are intended to support, not replace, systematic-review screening decisions.

## Future Work

The next phase will investigate whether general-purpose language models, including **Anthropic's Claude Haiku**, can improve semantic eligibility assessment.

The proposed comparison will evaluate:

1. The number of relevant articles incorrectly excluded.
2. The proportion of articles requiring human review.
3. Agreement with human-assigned eligibility labels.
4. Quality and consistency of exclusion explanations.
5. API usage and computational cost.

Both systems should ultimately be evaluated on an independent, human-labeled sample before being applied to the complete dataset.

## Research Context

This project is part of ongoing undergraduate research in the **Artificial Anthropomorphic Agent (AAA) Lab at Sungkyunkwan University**, investigating human interaction with artificial intelligence and patterns of AI overreliance.

The integration is an exploratory effort to improve the efficiency, transparency, and reproducibility of literature screening for systematic reviews.

---

**Branch:** `Typesafe-API-Integration`

**Status:** Pilot evaluated; independent validation pending.
