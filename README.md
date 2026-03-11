# Predicting Team Success from Coordination Patterns

This repository analyzes SCIALOG team discussions using two annotation regimes:

- `data/`: legacy utterance-level CDP annotations used by the main funding-outcome analyses
- canonical v2 outputs in `../gemini_data_analysis/outputs/`: newer chunk-based behavioral annotations used for the active fuzzy-linkography workflow

The main finding from the current outcome pipeline is unchanged: concentrated coordination leadership predicts funding better than overall coordination diversity.

## Repository State

| Dataset | Scope | Status in this repo | Primary use |
|---|---|---|---|
| `data/` | 157 session JSONs across 8 conferences (2020-2022) | Complete legacy corpus | Main entropy/Gini/outcome analyses |
| `data-v2/` | Repo-local mirror/subset of the v2 chunk corpus | Local convenience copy | Old-vs-new comparison, local inspection |
| `../gemini_data_analysis/outputs` | 162 session directories and 1325 JSON files across 8 conferences | Canonical v2 annotation source | Active fuzzy linkography, v2 feature development, registry-backed analysis |
| `old-vs-new/` | Verification and comparison docs/scripts | Active bridge layer | Mapping CDP to v2 behavioral coding |

## Quick Start

```bash
pip install -e .
make all
```

Key outputs:

- Main analysis reports: `outputs/analysis/`
- Tables: `outputs/tables/`
- Figures: `figures/final/`
- Full project narrative: `docs/PROJECT_CONTEXT.md`

## Working With The Two Data Formats

### Legacy CDP data

`data/<conference>/session_data/*.json` contains utterance-level transcripts with:

- speaker identity
- timestamps
- transcript text
- `annotations["Coordination and Decision Practices"]`
- optional `when` labels (`beginning`, `middle`, `end`)

This is the dataset used by the current pipelines in `pipelines/` and the core loaders in `src/linkography_ai/`.

### New v2 data

The canonical v2 annotation source is `../gemini_data_analysis/outputs/<conference>/output_<session_id>/...json`. Repo-local `data-v2/` uses the same schema and can still be inspected directly.

These chunk-based behavioral annotations contain three main sections:

- `chunk_summary`
- `utterance_annotations`
- `session_state`

Representative `chunk_summary` fields include:

- `idea_trajectory`
- `decision_crystallization_level`
- `collective_engagement_level`
- `explicit_commitment_signal`
- `shared_vision_indicator`
- `pronoun_shift_flag`

Some session folders also contain `ATTN_*.json` sidecar files or malformed files, so v2 tooling should only treat files as chunks when they parse successfully and include `chunk_summary`.

### Active v2 methodology

The current v2 workflow is centered on fuzzy linkography:

- utterances are treated as sequential moves
- semantic similarity is computed across moves
- similarities are converted into fuzzy link weights
- meeting-level metrics summarize semantic continuity, cross-speaker linkage, and temporal backlink structure

Current implementation note:

- methodology: fuzzy linkography
- current similarity model: Latent Semantic Analysis (LSA)

Entry points:

- `pipelines/fuzzy_linkography_v2.py`
- `pipelines/merge_fuzzy_with_outcomes.py`
- `pipelines/fuzzy_linkography_outcomes.py`
- `docs/FUZZY_LINKOGRAPHY_V2_SUMMARY.md`

Path portability:

- Active pipelines default to repo-relative `data/` and `outputs/`. For v2 annotations, they prefer the canonical sibling path `../gemini_data_analysis/outputs` when it exists, and otherwise fall back to repo-local `data-v2/`.
- If one machine keeps those folders outside the repo, set:
  - `LINKOGRAPHY_AI_DATA_ROOT`
  - `LINKOGRAPHY_AI_DATA_V2_ROOT`
  - `LINKOGRAPHY_AI_OUTPUTS_ROOT`
- Example:
  `LINKOGRAPHY_AI_DATA_V2_ROOT="/Users/maxchalekson/Projects/NICO-Research/gemini_data_analysis/outputs" PYTHONPATH=src .venv/bin/python pipelines/fuzzy_linkography_v2.py`

## Project Structure

```text
linkography_ai/
├── docs/
│   └── PROJECT_CONTEXT.md
├── data/
├── data-v2/
├── old-vs-new/
│   ├── 2_conf_v2_verification/
│   └── 3_deep_annotation_comparison/
├── pipelines/
├── src/linkography_ai/
├── outputs/
└── figures/
```

## Main Analysis Pipeline

The published analysis flow still runs on `data/`:

```bash
make all
python pipelines/speaker_diversity_outcomes.py
python pipelines/meeting_profile_classifier.py
```

Core modules:

- `src/linkography_ai/io_sessions.py`
- `src/linkography_ai/entropy.py`
- `src/linkography_ai/segmentation.py`
- `src/linkography_ai/discovery.py`

## Old vs New Annotation Work

Use the `old-vs-new/` folder when you want to compare legacy CDP against the new annotation style mirrored in repo-local `data-v2/`.

Useful entry points:

- `old-vs-new/2_conf_v2_verification/README.md`
- `old-vs-new/3_deep_annotation_comparison/README.md`

Current bridge work includes:

- conference-level alignment checks between `data/` and repo-local `data-v2/`
- per-session trajectory comparisons
- code-mapping notes between CDP scores and v2 behavioral labels
- fuzzy-linkography feature extraction on the canonical v2 outputs

## Where To Start

- Read `docs/PROJECT_CONTEXT.md` for the full research and reproducibility context.
- Read `old-vs-new/3_deep_annotation_comparison/README.md` if you are working on the new annotation scheme.
- Use `data/` for the existing outcome models and the canonical v2 outputs for fuzzy linkography and new-style annotation development.
