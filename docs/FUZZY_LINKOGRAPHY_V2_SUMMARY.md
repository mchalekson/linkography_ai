# Fuzzy Linkography V2 Implementation Summary

**Date:** 2026-03-06  
**Environment used:** `gem_samp`  
**Data source:** `../gemini_data_analysis/outputs/` (canonical Gemini/Evey chunk JSON outputs; repo-local `data-v2/` is a mirror/subset)

---

## What Was Implemented

### 1) Fuzzy linkography feature extraction from updated JSONs

- Added core module: [src/linkography_ai/fuzzy_linkography.py](src/linkography_ai/fuzzy_linkography.py)
- Added pipeline: [pipelines/fuzzy_linkography_v2.py](pipelines/fuzzy_linkography_v2.py)

This pipeline:

- Recursively reads all `*_chunk*.json` files in the canonical v2 outputs directory
- Groups chunk files into meetings
- Converts each utterance into a move
- Computes semantic similarity between moves
- Uses those similarities inside a fuzzy-linkography inference step
- Converts similarities into fuzzy link strengths
- Exports chunk-level and meeting-level fuzzy metrics

### 2) Merge with funding outcomes

- Added pipeline: [pipelines/merge_fuzzy_with_outcomes.py](pipelines/merge_fuzzy_with_outcomes.py)

This pipeline:

- Joins fuzzy metrics to session outcomes, preferring `chunk-registry/chunk_registry_v1.csv` when available and falling back to `entropy_with_outcomes.csv`
- Produces meeting-level merged table
- Produces session-level aggregated table (weighted by move count)

### 3) First-pass outcome tests for fuzzy features

- Added pipeline: [pipelines/fuzzy_linkography_outcomes.py](pipelines/fuzzy_linkography_outcomes.py)

This pipeline runs:

- Mann-Whitney tests (funded vs unfunded)
- Spearman correlations (`any_funded`, `funded_rate`)
- Cohen’s d effect sizes

---

## Method Definition Used

For each meeting, utterances are treated as sequential design moves.

### Move text used

- Prefer `transcript` / `text` / `utterance` if present
- Otherwise use concatenated `codes[].evidence` snippets from updated schema

### Fuzzy-linkography inference

- Overall methodology: fuzzy linkography
- Current similarity model in this repo: Latent Semantic Analysis (LSA) cosine similarity over move text
- Threshold: $t = 0.35$
- Fuzzy link weight:

$$
w_{ij} = \max\left(0, \frac{s_{ij} - t}{1 - t}\right), \quad i<j
$$

where $s_{ij}$ is the semantic similarity score and $w_{ij}\in[0,1]$.

In other words: the repo uses the fuzzy-linkography framework to infer weighted links between utterances, and LSA is the current computational model used for the semantic-similarity step inside that framework.

### Fuzzy metrics exported

- `weighted_ldi`
- `mean_nonzero_weight`
- `forelink_weight_mean`
- `backlink_weight_mean`
- `cross_speaker_weight_ratio`
- `late_minus_early_backlink`
- `forelink_entropy`
- `backlink_entropy`
- `horizon_entropy`
- `overall_link_entropy`

---

## Output Files

### Fuzzy extraction outputs

- [outputs/tables/fuzzy_linkography_v2_by_chunk.csv](outputs/tables/fuzzy_linkography_v2_by_chunk.csv)
- [outputs/tables/fuzzy_linkography_v2_by_meeting.csv](outputs/tables/fuzzy_linkography_v2_by_meeting.csv)
- [outputs/analysis/fuzzy_linkography_v2_summary.txt](outputs/analysis/fuzzy_linkography_v2_summary.txt)

### Fuzzy + outcome merged outputs

- [outputs/tables/fuzzy_linkography_with_outcomes_by_meeting.csv](outputs/tables/fuzzy_linkography_with_outcomes_by_meeting.csv)
- [outputs/tables/fuzzy_linkography_with_outcomes_by_session.csv](outputs/tables/fuzzy_linkography_with_outcomes_by_session.csv)
- [outputs/analysis/fuzzy_linkography_with_outcomes_summary.txt](outputs/analysis/fuzzy_linkography_with_outcomes_summary.txt)

### Fuzzy outcome test outputs

- [outputs/tables/fuzzy_linkography_outcomes_tests.csv](outputs/tables/fuzzy_linkography_outcomes_tests.csv)
- [outputs/analysis/fuzzy_linkography_outcomes_summary.txt](outputs/analysis/fuzzy_linkography_outcomes_summary.txt)

---

## Current Findings (First Pass)

### Coverage

- Meetings analyzed from the canonical v2 outputs: **179**
- Chunks analyzed: **1219**
- Meeting rows matched to outcomes: **154 / 179 (86.0%)**
- Session rows matched to outcomes: **136 / 158 (86.1%)**

### Descriptive means (meeting-level)

- `weighted_ldi`: **1.0663**
- `mean_nonzero_weight`: **0.1823**
- `cross_speaker_weight_ratio`: **0.7440**
- `late_minus_early_backlink`: **1.4239**
- `overall_link_entropy`: **44.6974**

### Current interpretation (session-level)

From [outputs/analysis/fuzzy_linkography_outcomes_summary.txt](outputs/analysis/fuzzy_linkography_outcomes_summary.txt) and [outputs/analysis/fuzzy_linkography_model_increment.txt](outputs/analysis/fuzzy_linkography_model_increment.txt):

- No single trajectory-derived fuzzy metric is a strong standalone direct-effect result in the current LSA rerun.
- The stronger result is additive:
  - existing baseline model AUC: **0.7300**
  - baseline + fuzzy-linkography trajectory layer AUC: **0.8095**
  - improvement: **+0.0795**

So the current value of this layer is as a complementary meeting-trajectory signal rather than a single dominant predictor.

---

## How To Run

Using `gem_samp` environment:

```bash
PYTHONPATH=src conda run -n gem_samp python pipelines/fuzzy_linkography_v2.py --conference ALL --threshold 0.35
conda run -n gem_samp python pipelines/merge_fuzzy_with_outcomes.py
conda run -n gem_samp python pipelines/fuzzy_linkography_outcomes.py
```

Or via `Makefile` target commands (using project Python path setup):

```bash
make fuzzy_v2
make fuzzy_v2_merge
make fuzzy_v2_outcomes
```

---

## Interpretation for the project

This implementation adds the missing fuzzy-linkography layer on top of updated rich annotations:

- Updated JSON gives high-resolution **local utterance coding**
- Fuzzy linkography now adds **global semantic-relational structure over time**
- Existing repo strengths remain in **outcome modeling and trajectory testing**

So this serves as a practical bridge between Evey’s richer annotation schema and the repo’s downstream predictive analysis workflow.
