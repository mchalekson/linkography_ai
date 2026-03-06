# Fuzzy Linkography V2 Implementation Summary

**Date:** 2026-03-06  
**Environment used:** `gem_samp`  
**Data source:** `outputs-v2/` (updated Gemini/Evey chunk JSON schema)

---

## What Was Implemented

### 1) Fuzzy linkography feature extraction from updated JSONs

- Added core module: [src/linkography_ai/fuzzy_linkography.py](src/linkography_ai/fuzzy_linkography.py)
- Added pipeline: [pipelines/fuzzy_linkography_v2.py](pipelines/fuzzy_linkography_v2.py)

This pipeline:

- Recursively reads all `*_chunk*.json` files in `outputs-v2/`
- Groups chunk files into meetings
- Converts each utterance into a move
- Computes semantic similarity between moves
- Converts similarities into fuzzy link strengths
- Exports chunk-level and meeting-level fuzzy metrics

### 2) Merge with funding outcomes

- Added pipeline: [pipelines/merge_fuzzy_with_outcomes.py](pipelines/merge_fuzzy_with_outcomes.py)

This pipeline:

- Joins fuzzy metrics to `entropy_with_outcomes.csv`
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

### Semantic link inference

- Similarity model: TF-IDF cosine similarity over move text
- Threshold: $t = 0.35$
- Fuzzy link weight:

$$
w_{ij} = \max\left(0, \frac{s_{ij} - t}{1 - t}\right), \quad i<j
$$

where $s_{ij}$ is cosine similarity and $w_{ij}\in[0,1]$.

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

- Meetings analyzed from `outputs-v2`: **130**
- Chunks analyzed: **788**
- Meeting rows matched to outcomes: **102 / 130 (78.5%)**
- Session rows matched to outcomes: **81 / 102 (79.4%)**

### Descriptive means (meeting-level)

- `weighted_ldi`: **0.0216**
- `mean_nonzero_weight`: **0.1573**
- `cross_speaker_weight_ratio`: **0.5280**
- `late_minus_early_backlink`: **0.0201**
- `overall_link_entropy`: **1.1196**

### First inferential signal (session-level)

From [outputs/analysis/fuzzy_linkography_outcomes_summary.txt](outputs/analysis/fuzzy_linkography_outcomes_summary.txt):

- `mean_nonzero_weight` shows a significant difference between funded and unfunded sessions
  - Funded mean: **0.1603**
  - Unfunded mean: **0.1945**
  - Mann-Whitney $p=0.0215$
  - Spearman with `any_funded`: $r=-0.2576$, $p=0.0203$
  - Spearman with `funded_rate`: $r=-0.2724$, $p=0.0139$

Most other fuzzy metrics were not significant in this first-pass test.

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
