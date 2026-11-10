# TypeSafe semantic screening (experimental)

This integration supplements the original TF-IDF / Random Forest classifier; it does not replace or modify it. The original 5,255-record dataset is `Systematic Review about AI - Duplicate check.csv`.

## Setup

Use Python 3.10+ and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a TypeSafe API key and configure it locally. Never commit the key.

PowerShell:

```powershell
$env:TYPESAFE_API_KEY="your_api_key"
```

macOS/Linux:

```bash
export TYPESAFE_API_KEY="your_api_key"
```

## Run a 25-article pilot

```bash
python typesafe_screening.py --input "Systematic Review about AI - Duplicate check.csv" --output typesafe_pilot_results.csv --limit 25
```

For the full run, omit `--limit 25` and choose a new output path. Calls may incur charges. The script caches successful responses in `typesafe_screening_cache.jsonl` for resumable runs.

The output retains original CSV columns and adds semantic scores for AI centrality, trust in AI, overreliance, problematic AI use, and empirical human research. Screening priority is **for human review, not automatic exclusion**. Thresholds are preliminary and should be calibrated against human-reviewed labels.

## Evaluation cautions

The original classifier's labels were generated from keywords, not independent human relevance judgments. Its reported accuracy therefore measures agreement with those heuristics rather than actual review relevance. It also fits TF-IDF before the train/test split; a fair baseline should fit vectorization inside each training fold. To estimate real recall, manually adjudicate a sample including records excluded by the original model.

The default TypeSafe model is `jev-1.14.0`; verify availability and SDK compatibility against https://docs.typesafe.ai/. This script has not been validated against a live API account.
