# CDP Entropy Analysis — Integration README

**Last updated**: February 25, 2026

---

## TL;DR

This repository analyzes **Coordination and Decision Practices (CDP)** score diversity in SCIALOG team discussions using Shannon entropy and temporal dynamics. The goal is to understand whether the **intensity of coordination behaviors** (measured as entropy over CDP score 1 vs score 2 distributions) varies across discussion phases, relates to session outcomes, and exhibits temporal patterns.

**Current state**: ✅ **P0 Complete** - Full pipeline executed on all 157 sessions across 8 conferences. Core findings: Entropy remains **stable** (0.733 → 0.745, no significant change), suggesting teams maintain a consistent mix of basic and advanced coordination throughout. 78.3% of sessions successfully matched with funding outcome data. ✅ **P1 Outcomes testing complete** - No significant entropy differences between funded vs unfunded sessions. ✅ **P1/P2 Extensions complete** - Batch convergence, time-binning comparison, time-pressure language, and outcome modeling executed. ✅ **P3 CDP Deep Dives COMPLETE** - Five new analyses executed: content analysis (utterance-level), speaker-level diversity, fine-grained timing (5-min bins), cohort comparison (years), and role analysis. ✅ **P4 Outcomes-Focused CDP Analyses COMPLETE** - Speaker diversity vs outcomes, timing patterns vs outcomes, and meeting-profile classifier executed.

**What it produces**:
- **Batch entropy table**: `outputs/tables/cdp_entropy_by_session_ALL_20260225_091354.csv` ✅ **GENERATED** (157 sessions)
- **Entropy trajectory analysis**: `outputs/analysis/entropy_trajectory_summary.txt` ✅ **GENERATED**
- **Entropy with outcomes**: `outputs/tables/entropy_with_outcomes.csv` ✅ **GENERATED** (123 matched sessions)
- **Outcomes stats report**: `outputs/analysis/entropy_outcomes_stats.txt` ✅ **GENERATED**
- **Outcomes group summary**: `outputs/tables/entropy_outcome_group_summary.csv` ✅ **GENERATED**
- **Convergence rates**: `outputs/tables/convergence_rates_by_session.csv` ✅ **GENERATED**
- **Convergence vs entropy**: `figures/final/convergence_vs_entropy_scatter.png` ✅ **GENERATED**
- **Time-binning comparison**: `outputs/tables/time_binning_comparison.csv` ✅ **GENERATED**
- **Time-binning summary**: `outputs/analysis/time_binning_comparison_summary.txt` ✅ **GENERATED**
- **Time-pressure language**: `outputs/tables/time_pressure_language_by_session.csv` ✅ **GENERATED**
- **Time-pressure summary**: `outputs/analysis/time_pressure_language_summary.txt` ✅ **GENERATED**
- **Entropy normalization comparison**: `outputs/analysis/entropy_normalization_comparison.txt` ✅ **GENERATED**
- **Raw vs normalized plot**: `figures/final/raw_vs_normalized_entropy_scatter.png` ✅ **GENERATED**
- **Outcome modeling**: `outputs/analysis/outcome_modeling_report.txt` ✅ **GENERATED**
- **Outcome model coefficients**: `outputs/tables/outcome_model_coefficients.csv` ✅ **GENERATED**
- **CDP content analysis**: `outputs/tables/cdp_content_analysis.csv`, `outputs/analysis/cdp_content_analysis_summary.txt` ✅ **GENERATED**
- **Speaker-level CDP**: `outputs/tables/speaker_level_cdp.csv`, `outputs/analysis/speaker_level_cdp_summary.txt` ✅ **GENERATED**
- **Fine-grained CDP timing**: `outputs/tables/cdp_fine_grained_entropy_300s.csv`, `outputs/analysis/cdp_fine_grained_summary_300s.txt` ✅ **GENERATED**
- **CDP by cohort**: `outputs/analysis/cdp_by_cohort_summary.txt` ✅ **GENERATED**
- **Speaker role CDP**: `outputs/tables/speaker_role_cdp.csv`, `outputs/analysis/speaker_role_cdp_summary.txt` ✅ **GENERATED**
- **Speaker diversity vs outcomes**: `outputs/tables/speaker_diversity_with_outcomes.csv`, `outputs/analysis/speaker_diversity_outcomes_summary.txt` ✅ **GENERATED**
- **Timing patterns vs outcomes**: `outputs/tables/timing_features_with_outcomes.csv`, `outputs/analysis/timing_patterns_outcomes_summary.txt` ✅ **GENERATED**
- **Meeting profile classifier**: `outputs/tables/meeting_profile_classifier_results.csv`, `outputs/analysis/meeting_profile_classifier_results.txt` ✅ **GENERATED**
- **Trajectory visualization**: `figures/final/entropy_trajectory.png` ✅ **GENERATED**
- **Session-level figures**: `figures/generated/slide*.png` (signals, convergence, entropy vs CD)
- **Logs**: `outputs/logs/slide*.txt` (callouts and metadata)

**Data coverage**: 8 SCIALOG conferences (2020NES, 2021ABI, 2021CMC, 2021MND, 2021MZT, 2021NES, 2021SLU, 2022MND) with annotated session transcripts in `data/`.

**Key Finding (P0)**: Entropy shows **no significant trajectory** (stable ~0.7), contradicting convergence hypothesis. Teams use a balanced mix of basic (score 1) and advanced (score 2) coordination throughout sessions.

**Key Findings (P3 — CDP Deep Dives)**:
- **Content Analysis**: Score 1 utterances 71% frequent but 2.6x shorter (19 tokens); score 2 are 29% frequent but 2.6x longer (49 tokens) → complex coordination ideas appear less frequently but in longer utterances
- **Speaker Diversity**: Score 2 CDP is more balanced across speakers (Gini=0.289) vs score 1 (Gini=0.418), suggesting advanced coordination involves broader speaker participation
- **Fine-Grained Timing**: Mean entropy per 5-min bin (0.418) with high variance (std=0.440) indicates dynamic shifts between basic and advanced coordination throughout meetings
- **Cohort Effects**: No significant year effect in beginning/end segments (p>0.05); middle segment shows trend (H=7.90, p~0.02) with 2022 teams showing lower entropy mid-meeting (0.427 vs 0.664 in 2021)
- **Speaker Roles**: Limited role metadata in sessions; analysis framework ready for role-enriched datasets

**Key Findings (P4 — Outcomes-Focused CDP Analyses)**:
- **Speaker Diversity vs Outcomes**: Funded sessions show **higher Gini** (more concentrated coordination) for both score 1 and score 2; participation rate not predictive.
- **Timing Patterns vs Outcomes**: No significant differences in phase rhythm, transition counts, or entropy trends between funded vs unfunded sessions.
- **Meeting Profile Classifier**: Combining speaker diversity + timing features improves ROC-AUC from **0.539 → 0.688** (~27.7% improvement) over entropy-only baseline.

**Next 3–5 analyses** (P1/P2/P3/P4):
1. ✅ **DONE** - Validated entropy trajectories (138 usable sessions; found stability, not convergence)
2. ✅ **DONE** - Merged entropy with funding outcomes (123/157 matched, 78.3% success rate)
3. ✅ **DONE** - Statistical testing: funded vs unfunded sessions show no significant entropy differences (Mann-Whitney p = 0.1925; d = -0.25)
4. ✅ **DONE** - Batch convergence detection across all sessions (rates + scatter vs entropy change)
5. ✅ **DONE** - Time-binned vs index-based thirds comparison (high agreement; middle most variable)
6. ✅ **DONE** - Time-pressure & decision-closure language scan (pressure peaks late)
7. ✅ **DONE** - Outcome modeling beyond entropy (low R^2; exploratory)
8. ✅ **DONE** - CDP content analysis: utterance-level patterns (score 1 vs score 2 characteristics)
9. ✅ **DONE** - Speaker-level CDP diversity and participation rates (Gini metrics)
10. ✅ **DONE** - Fine-grained CDP timing: 5-min bin entropy to detect inflection points
11. ✅ **DONE** - CDP patterns by cohort year with Kruskal-Wallis H-tests
12. ✅ **DONE** - Speaker diversity vs outcomes (Gini and coordination concentration)
13. ✅ **DONE** - Timing patterns vs outcomes (phase rhythm and transitions)
14. ✅ **DONE** - Meeting profile classifier (speaker + timing features)

---

## Goals and Research Questions

### Inferred Goals (update if wrong)

1. **Goal: Quantify CDP score diversity across discussion phases** ✅ **COMPLETE**
   - **Deliverable**: Per-session table with `entropy_beginning`, `entropy_middle`, `entropy_end`.
   - **Success metric**: Entropy values computed for all sessions; distributions visualized by phase.
   - **Status**: ✅ **DONE** - 157 sessions analyzed, results in `cdp_entropy_by_session_ALL_20260225_091354.csv`
   - **Key Finding**: Mean entropy stable (0.733 → 0.745), teams maintain consistent mix of score 1 vs score 2 coordination.

2. **Goal: Relate entropy trajectories to session outcomes** ✅ **COMPLETE**
   - **Deliverable**: Statistical comparison (t-test, regression) of entropy patterns for successful vs unsuccessful sessions.
   - **Success metric**: Significant difference or clear trend; reported in results table.
   - **Status**: ✅ Tests complete - no significant funded vs unfunded entropy differences (Mann-Whitney p = 0.1925)

3. **Goal: Detect temporal convergence signals** ✅ **COMPLETE**
   - **Deliverable**: Per-session convergence rate (utterances meeting strict convergence criteria).
   - **Success metric**: Convergence flagged in logs; visualized in time-series plots.
   - **Status**: ✅ Batch convergence rates computed (`convergence_rates_by_session.csv`).

4. **Goal: Validate structural wrap as meeting-management signal**
   - **Deliverable**: Per-bin structural wrap counts; correlation with entropy decay.
   - **Success metric**: Wrap increases in final third; negatively correlated with entropy.
   - **Status**: ✅ Regex implemented (`STRUCTURAL_WRAP_PAT`); systematic validation MISSING.

5. **Goal: Provide reproducible pipeline for entropy computation**
   - **Deliverable**: CLI tools in `pipelines/` with `--help` documentation; README usage examples.
   - **Success metric**: Another researcher can replicate entropy table with single command.
   - **Status**: ✅ Implemented; tested on example session.

6. **Goal: Explore time-binned dynamics vs index-based thirds** ✅ **COMPLETE**
   - **Deliverable**: Side-by-side comparison of both segmentation methods.
   - **Success metric**: Documented trade-offs; recommendation for each use case.
   - **Status**: ✅ Comparative analysis complete (`time_binning_comparison_summary.txt`).

---

## Key Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| **Notebook (exploratory)** | `notebooks/linkography-ai.ipynb` | Original slide analyses (Dec 2025); not executed. |
| **Batch entropy table** | `outputs/tables/cdp_entropy_by_session_ALL_20260225_091354.csv` | ✅ **GENERATED** - 157 sessions, all 8 conferences |
| **Entropy trajectory analysis** | `outputs/analysis/entropy_trajectory_summary.txt` | ✅ **GENERATED** - Statistical tests showing stability |
| **Entropy with outcomes** | `outputs/tables/entropy_with_outcomes.csv` | ✅ **GENERATED** - 123 matched sessions with funding data |
| **Outcomes stats report** | `outputs/analysis/entropy_outcomes_stats.txt` | ✅ **GENERATED** - Mann-Whitney, effect size, correlations |
| **Outcomes group summary** | `outputs/tables/entropy_outcome_group_summary.csv` | ✅ **GENERATED** - Group means/medians by funding |
| **Trajectory visualization** | `figures/final/entropy_trajectory.png` | ✅ **GENERATED** - Bar chart + individual trajectories |
| **Data validation report** | `outputs/logs/data_validation_report.txt` | ✅ **GENERATED** - All 157 sessions passed validation |
| **Convergence rates** | `outputs/tables/convergence_rates_by_session.csv` | ✅ **GENERATED** - Strict convergence + structural wrap rates |
| **Convergence vs entropy** | `figures/final/convergence_vs_entropy_scatter.png` | ✅ **GENERATED** - Scatter plot |
| **Time-binning comparison** | `outputs/tables/time_binning_comparison.csv` | ✅ **GENERATED** - Time vs index thirds |
| **Time-binning summary** | `outputs/analysis/time_binning_comparison_summary.txt` | ✅ **GENERATED** - Correlations + diffs |
| **Time-pressure language** | `outputs/tables/time_pressure_language_by_session.csv` | ✅ **GENERATED** - Per-session counts |
| **Time-pressure summary** | `outputs/analysis/time_pressure_language_summary.txt` | ✅ **GENERATED** - Segment means |
| **Normalization comparison** | `outputs/analysis/entropy_normalization_comparison.txt` | ✅ **GENERATED** - Raw vs normalized diff |
| **Raw vs normalized plot** | `figures/final/raw_vs_normalized_entropy_scatter.png` | ✅ **GENERATED** - Scatter plot |
| **Outcome modeling report** | `outputs/analysis/outcome_modeling_report.txt` | ✅ **GENERATED** - Linear models |
| **Outcome model coefficients** | `outputs/tables/outcome_model_coefficients.csv` | ✅ **GENERATED** - Coefficients table |
| **CDP by cohort summary** | `outputs/analysis/cdp_by_cohort_summary.txt` | ✅ **GENERATED** - Year-by-year entropy | 
| **CDP by cohort pairwise** | `outputs/analysis/cdp_by_cohort_pairwise.txt` | ✅ **GENERATED** - Pairwise Mann-Whitney U + Holm |
| **Speaker diversity vs outcomes** | `outputs/analysis/speaker_diversity_outcomes_summary.txt` | ✅ **GENERATED** - Gini/participation vs funding |
| **Speaker diversity + outcomes table** | `outputs/tables/speaker_diversity_with_outcomes.csv` | ✅ **GENERATED** - Merged session outcomes |
| **Timing patterns vs outcomes** | `outputs/analysis/timing_patterns_outcomes_summary.txt` | ✅ **GENERATED** - Timing features vs funding |
| **Timing features + outcomes table** | `outputs/tables/timing_features_with_outcomes.csv` | ✅ **GENERATED** - Per-session timing features |
| **Meeting profile classifier** | `outputs/analysis/meeting_profile_classifier_results.txt` | ✅ **GENERATED** - ROC-AUC comparisons |
| **Classifier results table** | `outputs/tables/meeting_profile_classifier_results.csv` | ✅ **GENERATED** - Model metrics |
| **Example session figures** | `figures/generated/slide1_2021_11_04_NES_S6.png` | Signals plot (CD + wrap). |
| | `figures/generated/slide2_2021_11_04_NES_S6.png` | Convergence detection plot. |
| | `figures/generated/slide3_2021_11_04_NES_S6.png` | Entropy vs CD dual-axis plot. |
| **Example session logs** | `outputs/logs/slide1_2021_11_04_NES_S6.txt` | Callout: longest commitment-coded utterance. |
| | `outputs/logs/slide2_2021_11_04_NES_S6.txt` | Callout: longest convergence utterance. |
| | `outputs/logs/slide3_2021_11_04_NES_S6.txt` | Session metadata + callout. |
| **Codebook** | `codebook/codebook.md` | CDP annotation definitions (including score 1 vs 2). |
| **Project README** | `README.md` | Installation, usage, data structure reference. |

**Note**: `figures/final/` now contains `entropy_trajectory.png`, `convergence_vs_entropy_scatter.png`, and `raw_vs_normalized_entropy_scatter.png`.

---

## Repository Map

### Entry Points (Pipelines)

| Script | Purpose | Slide Ref | Status |
|--------|---------|-----------|--------|
| `pipelines/signals.py` | Per-bin CD + structural wrap signals | Slide 1 | ✅ Tested |
| `pipelines/convergence.py` | Strict convergence detection | Slide 2 | ✅ Tested |
| `pipelines/entropy_vs_cd.py` | Entropy vs CD minutes dual plot | Slide 3 | ✅ Tested |
| `pipelines/run_cdp_entropy_all.py` | Batch entropy across all conferences | Slide 7 | ✅ Ready |
| `pipelines/validate_data_integrity.py` | Validate session JSON integrity | — | ✅ Ready |
| `pipelines/analyze_entropy_trajectories.py` | Analyze beginning→middle→end patterns | — | ✅ Ready |
| `pipelines/merge_entropy_with_outcomes.py` | Merge entropy with funding outcomes | — | ✅ Ready |
| `pipelines/test_entropy_outcomes.py` | Statistical testing: entropy vs outcomes | — | ✅ Ready |
| `pipelines/batch_convergence.py` | Batch strict convergence rates | — | ✅ Ready |
| `pipelines/compare_time_binning.py` | Time-based vs index-based thirds | — | ✅ Ready |
| `pipelines/compare_entropy_normalization.py` | Raw vs normalized entropy | — | ✅ Ready |
| `pipelines/time_pressure_language.py` | Time-pressure & closure language | — | ✅ Ready |
| `pipelines/outcome_modeling.py` | Outcome modeling beyond entropy | — | ✅ Ready |

### Core Modules (`src/linkography_ai/`)

| Module | Purpose |
|--------|---------|
| `entropy.py` | Shannon entropy: $H = -\sum_i p_i \log_2(p_i)$; optional normalization by $\log_2(K)$. |
| `segmentation.py` | Index-based thirds: `beginning`, `middle`, `end` via $\lfloor n/3 \rfloor$ logic. |
| `io_sessions.py` | Load session JSON; extract CDP annotations/scores from `annotations` field. |
| `slides.py` | Time-binned analysis; structural wrap regex; `compute_entropy_vs_cd()`. |
| `discovery.py` | Discover conference directories in `data/` with `session_data/` subdirs. |

---

## What We Measure: CDP Score Diversity

**Focus**: "Coordination and Decision Practices" (CDP) annotations only

**Scope**: We ignore all other annotation categories and only analyze **CDP**.

**What CDP Measures**: When an utterance has the CDP annotation, it includes a **score** field:
- **Score 1**: Basic coordination (structuring contributions, simple process management)
- **Score 2**: Advanced coordination (explicit agenda-setting, complex decision-making)

**What Entropy Measures**: The diversity of CDP **intensity levels** (score 1 vs score 2) within a segment.
- **High entropy** (~1.0): Mix of score 1 and score 2 utterances (varied coordination intensity)
- **Low entropy** (~0.0): All utterances have same score (uniform coordination level)

**Example**: 
- Segment with 5 score-1 and 5 score-2 CDP utterances → High entropy (diverse coordination)
- Segment with 10 score-1 CDP utterances → Low entropy (uniform coordination)

**Research Question**: Do teams start with a **mixed** use of CDP scores (high entropy) and converge to a **single** CDP score (low entropy) by the end?

---

## Understanding Entropy Values (CDP Scores Only)

**Normalized Entropy Range**: 0.0 to 1.0 (current pipeline uses `--normalize` flag)

Since we only use **two** CDP score levels (1 vs 2), interpretation is **binary**:

| Value | Interpretation |
|-------|----------------|
| **0.0** | All CDP utterances are the **same score** (all 1s or all 2s) |
| **~1.0** | **Balanced mix** of score 1 and score 2 CDP utterances |

**Observed Range in Data**: ~0.73 - 0.75 (stable mix of score 1 vs 2)

**The P0 Finding**: Entropy remains **stable** from 0.733 (beginning) → 0.745 (end)
- **Meaning**: Teams keep a **steady mix** of basic (score 1) and advanced (score 2) coordination.
- **Contradicts**: Initial hypothesis that teams would converge to **one** coordination intensity.
- **Possible explanations**:
   - Sessions require both basic and advanced coordination throughout.
   - Facilitators sustain a mix of process structuring and decision-making.

---

## P3 Findings: CDP Deep Dives

Since entropy alone does not predict outcomes, we shifted focus to **how** CDP is used across different contexts: What are the differences between score 1 and score 2 utterances? Do certain speakers drive advanced coordination? When do teams shift between basic and advanced practices? Do patterns differ by cohort year?

### P3.1 CDP Content Analysis

**Key Question**: What linguistic/conceptual differences exist between score 1 vs score 2 utterances?

**Method**: Extract all utterances with CDP annotations, group by score, compute token counts and sample excerpts.

**Results** (`cdp_content_analysis.csv`, `cdp_content_analysis_summary.txt`):
- **Score 1 prevalence**: 71% of all CDP utterances (mean 24.03 per session)
- **Score 2 prevalence**: 29% of all CDP utterances (mean 9.30 per session)
- **Score 1 length**: 19 tokens/utterance (mean)
- **Score 2 length**: 49 tokens/utterance (mean)

**Interpretation**: 
- Score 2 (advanced coordination) appears **2.6× less frequently** but in **2.6× longer utterances**
- Suggests advanced coordination involves **complex, sustained discussion** rather than frequent short phrases
- Basic coordination (score 1) provides frequent **structural scaffolding** while advanced coordination (score 2) tackles the **harder content**

---

### P3.2 Speaker-Level CDP Analysis

**Key Question**: Which speakers drive CDP adoption? Is advanced coordination more balanced across team members?

**Method**: Extract speaker identities, count CDP scores per speaker, compute Gini coefficient (concentration measure) for each score.

**Results** (`speaker_level_cdp.csv`, `speaker_level_cdp_summary.txt`):
- **Mean speakers with CDP**: 6.74 per session
- **Mean total speakers**: 13.28 per session
- **CDP participation rate**: 52% (half of all speakers contribute to CDP)
- **Gini (score 1)**: 0.418 (moderate concentration; few speakers dominate basic coordination)
- **Gini (score 2)**: 0.289 (balanced distribution; advanced coordination spreads across speakers)

**Interpretation**:
- **Score 2 is more "inclusive"**: Lower Gini (0.289 vs 0.418) means advanced coordination involves diverse speakers
- **Score 1 is more "concentrated"**: Few speakers repeat basic coordination phrases
- Hypothesis: Advanced coordination is a **team activity**, while basic coordination is driven by **key facilitators**
- Possible link to outcomes: Teams with more balanced speaker participation in advanced coordination might perform better (testable in future work)

---

### P3.3 Fine-Grained CDP Timing (5-Minute Bins)

**Key Question**: When do teams shift between basic and advanced coordination? Can we detect inflection points?

**Method**: Replace coarse "thirds" segmentation with fine-grained 5-minute time windows (configurable). Compute entropy per bin.

**Results** (`cdp_fine_grained_entropy_300s.csv`, `cdp_fine_grained_summary_300s.txt`):
- **Total bin observations**: 1,322 (across all sessions)
- **Mean entropy per bin**: 0.418 (std 0.440)
- **Entropy range**: [0.0, 1.0] (full spectrum represented)

**Interpretation**:
- Entropy varies **substantially within meetings** (std ≥ mean)
- Many bins show **pure score 1** (entropy ≈ 0) or **pure score 2** (entropy ≈ 0)
- Some bins show **mixed** use (entropy ≈ 0.7–1.0)
- **Inference**: Teams experience **dynamic shifts** in coordination mode; not a static mix throughout
- **Use case**: Identify bins where shift occurs; investigate what triggered the switch (new topic? decision point?)

---

### P3.4 CDP Patterns by Cohort Year

**Key Question**: Do team coordination patterns differ across conference cohorts (2020 vs 2021 vs 2022)?

**Method**: Segment entropy distributions by year, run Kruskal-Wallis H-tests per segment (non-parametric ANOVA).

**Results** (`cdp_by_cohort_summary.txt`):
- **Beginning segment**: No significant cohort effect (H = 0.95, p > 0.05)
- **Middle segment**: **Trend toward significance** (H = 7.90, p ≈ 0.02)
  - 2022: mean 0.427 (low entropy, focused coordination)
  - 2021: mean 0.664 (higher entropy, mixed coordination)
  - 2020: mean 0.717 (highest entropy)
- **End segment**: No significant cohort effect (H = 2.71, p > 0.05)

**Pairwise tests** (`cdp_by_cohort_pairwise.txt`):
- **Beginning**: No pairwise differences (Holm-adjusted p ≥ 0.97)
- **Middle**: 2020 vs 2022 (p = 0.028, Holm = 0.056) and 2021 vs 2022 (p = 0.022, Holm = 0.067) show **trend-level** separation; 2022 is consistently lower
- **End**: No pairwise differences (Holm-adjusted p ≥ 0.26)

**Interpretation**:
- Recent cohorts (2022) show **more structured decision-making** in the middle phase
- Possible explanation: Accumulated experience or tighter time constraints
- Most recent teams may shift to **pure score 1** (focused process) mid-meeting, then return to **mixed** mode at end
- Could indicate maturing team dynamics or changed facilitation style

---

### P3.5 Speaker Role Analysis

**Key Question**: Do specific roles (facilitator, fellow, mentor) drive CDP adoption?

**Method**: Extract speaker roles from session metadata, correlate with CDP scores.

**Results** (`speaker_role_cdp.csv`, `speaker_role_cdp_summary.txt`):
- **Sessions with identified facilitators**: 0 (metadata not populated in current dataset)
- Framework ready for datasets with explicit role assignments

**Note**: This analysis is **framework-complete** but awaits richer session metadata. If your dataset includes explicit role labels, this pipeline will immediately identify which roles drive score 1 vs score 2 utterances.

---

## Sample Outputs

### Batch Entropy CSV (cdp_entropy_by_session_ALL_*.csv)

**First 3 rows**:
```csv
conference,session_id,n_utterances,outcome,entropy_beginning,n_cdp_beginning,n_unique_cdp_beginning,entropy_middle,n_cdp_middle,n_unique_cdp_middle,entropy_end,n_cdp_end,n_unique_cdp_end
2020NES,2020_11_05_NES_S1,82,,0.804,47,2,0.912,53,2,0.963,46,2
2020NES,2020_11_05_NES_S2,67,,0.954,47,2,0.891,31,2,0.899,57,2
```

**Column Definitions**:
- `conference`: e.g., "2020NES", "2021MZT"
- `session_id`: Unique identifier (date_conference_session)
- `n_utterances`: Total utterances in session
- `outcome`: NULL (not used - see funded_status in outcomes CSV instead)
- `entropy_beginning/middle/end`: Normalized Shannon entropy (0-1 scale)
- `n_cdp_beginning/middle/end`: Total CDP annotation count (can exceed n_utterances due to multi-label)
- `n_unique_cdp_beginning/middle/end`: Number of distinct CDP **score levels** used (1 or 2)

### Entropy with Outcomes CSV (entropy_with_outcomes.csv)

**First 3 rows**:
```csv
conference,session_id,...,entropy_end,n_cdp_end,n_unique_cdp_end,funded_rate,any_funded,n_teams
2020NES,2020_11_05_NES_S1,...,0.963,46,2,0.0,0,1
2020NES,2020_11_05_NES_S3,...,0.921,38,2,0.333,1,3
2020NES,2020_11_05_NES_S4,...,0.964,23,2,1.0,1,1
```

**New Columns**:
- `funded_rate`: Proportion of teams funded (0.0 to 1.0)
- `any_funded`: Binary - did ANY team get funded? (0 or 1)
- `n_teams`: Number of teams formed in this session

---

## How to Load and Analyze Outputs (Python)

### Load Entropy Data
```python
import pandas as pd

# Load latest batch entropy results
df = pd.read_csv('outputs/tables/cdp_entropy_by_session_ALL_20260225_091354.csv')

# Quick stats
print(f"Total sessions: {len(df)}")
print(f"Mean beginning entropy: {df['entropy_beginning'].mean():.3f}")
print(f"Mean end entropy: {df['entropy_end'].mean():.3f}")

# Check for convergence (entropy decrease)
df['entropy_change'] = df['entropy_end'] - df['entropy_beginning']
print(f"Sessions with entropy decrease: {(df['entropy_change'] < 0).sum()}")
print(f"Mean entropy change: {df['entropy_change'].mean():.3f}")
```

### Analyze Outcomes Relationship
```python
# Load merged data
df = pd.read_csv('outputs/tables/entropy_with_outcomes.csv')

# Compare funded vs unfunded
funded = df[df['any_funded'] == 1]
unfunded = df[df['any_funded'] == 0]

print(f"Funded sessions (n={len(funded)}): entropy_end = {funded['entropy_end'].mean():.3f}")
print(f"Unfunded sessions (n={len(unfunded)}): entropy_end = {unfunded['entropy_end'].mean():.3f}")

# Correlation
corr = df[['entropy_end', 'funded_rate']].corr()
print(f"\nCorrelation (entropy_end vs funded_rate): {corr.iloc[0,1]:.3f}")
```

### Filter by Conference
```python
# Analyze specific conference
nes_2021 = df[df['conference'] == '2021NES']
print(f"2021NES sessions: {len(nes_2021)}")
print(f"Mean entropy: {nes_2021['entropy_end'].mean():.3f}")
```

---

## Troubleshooting

### Common Issues

**1. "No columns to parse" error**
- **Cause**: Batch entropy CSV is empty (no sessions processed)
- **Fix**: Check that `data/*/session_data/*.json` files exist
- **Verify**: `ls data/2020NES/session_data/ | wc -l` should return >0

**2. "ModuleNotFoundError: matplotlib"**
- **Cause**: Dependencies not installed in virtual environment
- **Fix**: `.venv/bin/python -m pip install matplotlib pandas numpy`

**3. Low match rate (<50%) in outcome merge**
- **Cause**: Session ID mismatch between entropy CSV and outcome JSONs
- **Check**: `cat outputs/logs/outcome_merge_report.txt` for details
- **Normal**: 78.3% match rate is expected (some sessions lack outcome data)

**4. All entropy values near 0.9**
- **Not a bug**: This is real data - it indicates a **balanced mix** of CDP score 1 and score 2.
- **Interpretation**: Sessions are using both coordination intensity levels rather than converging on just one.

### Verification Commands

```bash
# Check data integrity
make validate

# Verify outputs exist
ls -lh outputs/tables/cdp_entropy_by_session_ALL_*.csv
ls -lh outputs/analysis/entropy_trajectory_summary.txt
ls -lh figures/final/entropy_trajectory.png

# Count sessions per conference
.venv/bin/python -c "import pandas as pd; df = pd.read_csv('outputs/tables/cdp_entropy_by_session_ALL_20260225_091354.csv'); print(df['conference'].value_counts())"

# Quick stats
.venv/bin/python -c "import pandas as pd; df = pd.read_csv('outputs/tables/cdp_entropy_by_session_ALL_20260225_091354.csv'); print(df[['entropy_beginning', 'entropy_middle', 'entropy_end']].describe())"
```

---

### Core Modules (`src/linkography_ai/`)

| Module | Purpose |
|--------|---------|
| `entropy.py` | Shannon entropy: $H = -\sum_i p_i \log_2(p_i)$; optional normalization by $\log_2(K)$. |
| `segmentation.py` | Index-based thirds: `beginning`, `middle`, `end` via $\lfloor n/3 \rfloor$ logic. |
| `io_sessions.py` | Load session JSON; extract CDP annotations/scores from `annotations` field. |
| `slides.py` | Time-binned analysis; structural wrap regex; `compute_entropy_vs_cd()`. |
| `discovery.py` | Discover conference directories in `data/` with `session_data/` subdirs. |

### Data Structure

```
data/
├── 2020NES/
│   ├── session_data/              # Session transcripts (JSON)
│   ├── 2020NES_session_outcomes.json  # Outcome labels per session
│   ├── 2020NES_person_to_team.json    # Team membership (not used yet)
│   └── featurized data/           # MISSING — no current use
├── 2021ABI/
├── 2021CMC/
├── 2021MND/
├── 2021MZT/
├── 2021NES/
├── 2021SLU/
└── 2022MND/
```

**Session JSON format** (SCIALOG schema):
```json
{
  "all_speakers": ["Marcel Schreier", "Andrew Feig", ...],
  "total_speaking_length": 5667,
  "all_data": [
    {
      "speaker": "Marcel Schreier",
      "timestamp": "00:02-00:03",
      "transcript": "Hi everyone.",
      "start_time": "00:02",
      "end_time": "00:03",
      "annotations": {
        "Relational Climate": {
          "explanation": "...",
          "score": 1,
          "score_justification": "...",
          "when": "beginning"
        },
        "Coordination and Decision Practices": {
          "explanation": "...",
          "score": 2,
          "when": "middle"
        }
      },
      "role": "Scialog Fellow",
      "when": "beginning"
    }
  ]
}
```
**Note**: `annotations` is a nested dict where each key is an annotation category; we only use **Coordination and Decision Practices** entries (with `score` and `when`).

**Outcome JSON format** (actual structure):
```json
{
  "2021_11_04_NES_S5": {
    "all_speakers": ["Marcel Schreier", "Xiao Su", ...],
    "facilitators": ["Alissa Park", "Christopher Jones"],
    "teams": {
      "NES5": {
        "members": ["Haotian Wang", "Andrea Hicks"],
        "funded_status": 1
      },
      "NES22": {
        "members": ["Xiao Su", "Jimmy Jiang"],
        "funded_status": 0
      }
    }
  }
}
```
**Note**: No explicit `outcome` field exists. The pipeline currently tries to extract `outcome` but will find NULL. Consider using `funded_status` (1 = funded, 0 = not funded) as a proxy for session success.

### Output Locations

| Output Type | Path | Status | Created By |
|-------------|------|--------|------------|
| Batch tables | `outputs/tables/` | Auto-created | `run_cdp_entropy_all.py`, `merge_entropy_with_outcomes.py`, `batch_convergence.py`, `compare_time_binning.py`, `time_pressure_language.py`, `outcome_modeling.py` |
| Analysis outputs | `outputs/analysis/` | Auto-created | `analyze_entropy_trajectories.py`, `compare_time_binning.py`, `time_pressure_language.py`, `outcome_modeling.py` |
| Logs | `outputs/logs/` | ✅ Exists | All pipeline scripts |
| Generated figures | `figures/generated/` | ✅ Exists | Slide 1-3 pipelines |
| Final figures | `figures/final/` | Auto-created | `analyze_entropy_trajectories.py`, `batch_convergence.py` |

**Key Artifacts Generated:**

| File | Source | Content |
|------|--------|---------|
| `outputs/tables/cdp_entropy_by_session_ALL_*.csv` | `run_cdp_entropy_all.py` | Per-session entropy (beginning/middle/end) + CDP counts |
| `outputs/tables/entropy_with_outcomes.csv` | `merge_entropy_with_outcomes.py` | Entropy + `funded_rate`, `any_funded`, `n_teams` |
| `outputs/analysis/entropy_trajectory_summary.txt` | `analyze_entropy_trajectories.py` | Statistical tests + effect sizes for phase transitions |
| `figures/final/entropy_trajectory.png` | `analyze_entropy_trajectories.py` | Bar chart + individual trajectory lines |
| `outputs/logs/data_validation_report.txt` | `validate_data_integrity.py` | Data quality summary across all sessions |
| `outputs/logs/outcome_merge_report.txt` | `merge_entropy_with_outcomes.py` | Match/unmatch log for entropy-outcome merge |
| `outputs/analysis/entropy_outcomes_stats.txt` | `test_entropy_outcomes.py` | Funded vs unfunded statistical tests |
| `outputs/tables/entropy_outcome_group_summary.csv` | `test_entropy_outcomes.py` | Group means/medians |
| `outputs/tables/convergence_rates_by_session.csv` | `batch_convergence.py` | Strict convergence + structural wrap rates |
| `figures/final/convergence_vs_entropy_scatter.png` | `batch_convergence.py` | Convergence vs entropy change |
| `outputs/tables/time_binning_comparison.csv` | `compare_time_binning.py` | Time vs index thirds comparison |
| `outputs/analysis/time_binning_comparison_summary.txt` | `compare_time_binning.py` | Correlation + mean diffs |
| `outputs/tables/time_pressure_language_by_session.csv` | `time_pressure_language.py` | Time-pressure/closure counts |
| `outputs/analysis/time_pressure_language_summary.txt` | `time_pressure_language.py` | Segment-level summary |
| `outputs/analysis/entropy_normalization_comparison.txt` | `compare_entropy_normalization.py` | Raw vs normalized differences |
| `figures/final/raw_vs_normalized_entropy_scatter.png` | `compare_entropy_normalization.py` | Raw vs normalized scatter |
| `outputs/analysis/outcome_modeling_report.txt` | `outcome_modeling.py` | Exploratory linear models |
| `outputs/tables/outcome_model_coefficients.csv` | `outcome_modeling.py` | Model coefficients |

---

## What's Implemented (Concrete)

### 1. CDP Extraction (`io_sessions.py`)

**Purpose**: Load session JSON files and extract CDP annotations (including scores).

**Method**:
- Parse `all_data` array; extract `annotations` dict.
- Handle field name variations: `cdp`, `CDP`, `coordination_decision_practices`, `Coordination and Decision Practices`.
- Flatten nested lists; skip non-list/non-dict values.
- Return list of `Utterance(text, cdp)` objects.

**File**: [src/linkography_ai/io_sessions.py](../src/linkography_ai/io_sessions.py)  
**Function**: `load_session_utterances(path: Path) -> List[Utterance]`

---

### 2. Index-Based Segmentation (`segmentation.py`)

**Purpose**: Divide utterances into temporal thirds by index position.

**Method**:
- Compute boundaries: `a = n // 3`, `b = (2*n) // 3`.
- Assign labels: `[0, a) → "beginning"`, `[a, b) → "middle"`, `[b, n) → "end"`.
- No timestamp required.

**File**: [src/linkography_ai/segmentation.py](../src/linkography_ai/segmentation.py)  
**Function**: `segment_thirds(n: int) -> List[str]`

**Limitations**: Ignores actual duration; assumes even pacing.

---

### 3. Shannon Entropy (`entropy.py`)

**Purpose**: Compute information-theoretic diversity of CDP distributions.

**Method**:
```python
H = -sum(p_i * log2(p_i) for p_i in ps)
if normalize:
   H /= log2(K)  # K = number of unique score levels (1 or 2)
```

**File**: [src/linkography_ai/entropy.py](../src/linkography_ai/entropy.py)  
**Function**: `shannon_entropy_from_counts(counts: List[int], normalize: bool) -> float`

**Edge cases**: Returns `NaN` if total=0; returns 0.0 if K≤1 after normalization.

---

### 4. Time-Binned Signals (`slides.py`)

**Purpose**: Aggregate utterances into fixed-duration bins (default 60s) and compute per-bin CD minutes + structural wrap minutes.

**Method**:
- Parse `start_time`, `end_time` → seconds.
- Assign utterances to bins; compute duration per bin.
- Smooth with rolling window (default 3).
- Detect commitment codes (`DEFAULT_COMMITMENT_CODES`) and structural wrap (`DEFAULT_STRUCTURAL_WRAP_PAT`).

**File**: [src/linkography_ai/slides.py](../src/linkography_ai/slides.py)  
**Function**: `compute_signals_by_bin(session_path, bin_sec, smooth_window, last_third_only, exclude_structural) -> pd.DataFrame`

**Regex patterns**:
```python
DEFAULT_STRUCTURAL_WRAP_PAT = re.compile(
    r"(how much time|time limit|agenda|next steps|wrap|close out|summary|moving on)",
    re.IGNORECASE
)
```

---

### 5. Convergence Detection (`convergence.py`)

**Purpose**: Detect "strict convergence" utterances: agreement phrase + commitment code + NOT structural wrap.

**Method**:
```python
CONVERGENCE_PAT = re.compile(
    r"(we (?:all )?agree|consensus|settle on|we decide|let'?s go with)",
    re.IGNORECASE
)

is_convergence = (
    is_convergence_phrase AND
    is_commitment_code AND
    NOT is_structural_wrap_text
)
```

**File**: [pipelines/convergence.py](../pipelines/convergence.py)  
**Output**: Dual-line plot (CD minutes, structural wrap minutes) + log with longest convergence utterance.

**Limitation**: Regex may miss implicit agreement; false positives from casual language.

---

### 6. Entropy vs CD (`entropy_vs_cd.py`)

**Purpose**: Dual-axis plot of entropy and CD minutes over time bins.

**Method**:
- Compute per-bin CDP entropy using `shannon_entropy_from_counts()` on **score-level** distributions.
- Compute per-bin CD minutes (commitment-coded duration).
- Plot both on same timeline.

**File**: [pipelines/entropy_vs_cd.py](../pipelines/entropy_vs_cd.py)  
**Uses**: `compute_entropy_vs_cd()` and `plot_entropy_vs_cd()` from [src/linkography_ai/slides.py](../src/linkography_ai/slides.py).

---

### 7. Batch Entropy Pipeline (`run_cdp_entropy_all.py`)

**Purpose**: Compute per-session entropy for all conferences.

**Method**:
1. Discover conferences via `list_conferences()`.
2. For each session:
   - Segment into thirds (index-based).
   - Count CDP **score levels** per segment.
   - Compute entropy (beginning, middle, end).
3. Output CSV with columns: `conference`, `session_id`, `n_utterances`, `outcome`, `entropy_beginning`, `entropy_middle`, `entropy_end`, `n_cdp_*`, `n_unique_cdp_*`.

**File**: [pipelines/run_cdp_entropy_all.py](../pipelines/run_cdp_entropy_all.py)

**CLI**:
```bash
python pipelines/run_cdp_entropy_all.py --conference ALL --normalize --max_sessions 0
```

**Output columns**:
- `conference`, `session_id`, `n_utterances`, `outcome`
- `entropy_beginning`, `entropy_middle`, `entropy_end`
- `n_cdp_beginning`, `n_cdp_middle`, `n_cdp_end` (total counts)
- `n_unique_cdp_beginning`, `n_unique_cdp_middle`, `n_unique_cdp_end` (unique **score levels**: 1 or 2)

**Current status**: ✅ **COMPLETE** - Full run executed on 2026-02-25 09:13:54.

**Latest Output**: `outputs/tables/cdp_entropy_by_session_ALL_20260225_091354.csv` (157 sessions)

---

## How to Run / Reproduce

### Quick Start (Recommended)

**Run the full pipeline with one command:**
```bash
make all
```

This will:
1. Validate data integrity across all sessions
2. Compute batch entropy (all conferences, normalized)
3. Analyze entropy trajectories with statistical tests
4. Merge entropy with funding outcomes
5. Run funded vs unfunded outcome tests
6. Batch convergence detection
7. Compare time-based vs index-based thirds
8. Compare raw vs normalized entropy
9. Time-pressure & decision-closure language scan
10. Outcome modeling beyond entropy

**Individual steps:**
```bash
make validate        # Check data quality
make batch_entropy   # Generate entropy table
make analyze         # Statistical analysis + plots
make merge_outcomes  # Add funding outcomes
make test_outcomes   # Statistical tests vs outcomes
make batch_convergence # Batch convergence rates
make compare_binning # Time vs index thirds
make compare_normalization # Raw vs normalized entropy
make time_pressure   # Time-pressure language scan
make outcome_model   # Outcome modeling beyond entropy
```

**See all targets:**
```bash
make help
```

---

### Manual Setup (if not using Makefile)

1. **Clone repository**:
   ```bash
   git clone https://github.com/mchalekson/linkography_ai.git
   cd linkography_ai
   ```

2. **Install package** (Python ≥3.10):
   ```bash
   pip install -e .
   ```

3. **Dependencies**: `requirements.txt` (pandas, numpy, matplotlib).

### Run Pipelines

#### Slide 1: Signals by time bin
```bash
python pipelines/signals.py --session data/2021NES/session_data/2021_11_04_NES_S6.json
```
**Output**: `figures/generated/slide1_2021_11_04_NES_S6.png`, `outputs/logs/slide1_*.txt`

#### Slide 2: Convergence detection
```bash
python pipelines/convergence.py --session data/2021NES/session_data/2021_11_04_NES_S6.json --print-context
```
**Output**: `figures/generated/slide2_*.png`, `outputs/logs/slide2_*.txt` (with before/after context for convergence utterance)

#### Slide 3: Entropy vs CD
```bash
python pipelines/entropy_vs_cd.py --session data/2021NES/session_data/2021_11_04_NES_S6.json
```
**Output**: `figures/generated/slide3_*.png`, `outputs/logs/slide3_*.txt`

#### Batch Entropy (All Conferences)
```bash
python pipelines/run_cdp_entropy_all.py --conference ALL --normalize --max_sessions 0
```
**Output**: `outputs/tables/cdp_entropy_by_session_ALL_*.csv`, `outputs/logs/run_cdp_entropy_ALL_*.txt`

**Column interpretation**:
- `entropy_*`: Shannon entropy of CDP **score** distribution in that segment (higher = more mixed).
- `n_cdp_*`: Total CDP annotations (utterances labeled with CDP).
- `n_unique_cdp_*`: Number of distinct CDP **score levels** used (1 or 2).
- `outcome`: Session success label (⚠️ currently NULL; needs validation).

### Common Flags

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--session` | path | *required* | Path to session JSON file |
| `--bin-sec` | int | 60 | Bin duration (seconds) for time-based methods |
| `--smooth-window` | int | 3 | Rolling window for smoothing time series |
| `--last-third-only` | flag | False | Restrict analysis to final third of session |
| `--exclude-structural` | flag | False | Exclude structural wrap utterances from CD counts |
| `--print-context` | flag | False | Show before/after context for callout utterances |
| `--normalize` | flag | False | Normalize entropy by log2(K) |
| `--conference` | str | ALL | Conference code (e.g., 2021MZT) or ALL |
| `--max_sessions` | int | 0 | Limit sessions per conference (0 = all) |

---

## Concrete Analysis Plan (Next Steps)

### Priority 1: Validate Entropy Pipeline ✅ **COMPLETE**

**Research Question**: Does CDP entropy decrease from beginning → middle → end (indicating convergence)?

**Answer**: ❌ **NO** - Entropy remains **stable** from 0.733 → 0.745 (beginning → end), **no significant change**.

**Implementation**: `pipelines/analyze_entropy_trajectories.py`

**Results** (executed 2026-02-25):
- **Sessions analyzed**: 138
- **Beginning entropy**: 0.733 ± 0.259
- **Middle entropy**: 0.650 ± 0.375
- **End entropy**: 0.745 ± 0.202
- **Beginning → End change**: -0.011 [95% CI: -0.068, 0.044]

**Interpretation**: Teams maintain a **stable mix** of CDP score levels across phases, contradicting the convergence hypothesis. This may indicate:
- Teams sustain both basic and advanced coordination throughout
- Facilitators keep a balance of structuring and decision-making
- Final decision-making still requires both coordination levels

**Outputs**: 
- ✅ `outputs/analysis/entropy_trajectory_summary.txt`
- ✅ `figures/final/entropy_trajectory.png`

---

### Priority 2: Correlate Entropy with Outcomes ✅ **COMPLETE**

**Research Question**: Do successful sessions show lower final-third entropy (more focused coordination)?

**Implementation**: `pipelines/merge_entropy_with_outcomes.py`

**Results** (executed 2026-02-25):
- **Entropy sessions**: 157
- **Outcome sessions**: 123  
- **Matched**: 123 (78.3% match rate)
- **Sessions with any funded team**: 68
- **Mean funding rate**: 0.37
- **Funding distribution**: 55 sessions with 0% funding, 25 with 100% funding

**Statistical Tests** (executed 2026-02-25):
- **Sessions with outcomes used**: 120
- **Funded (any_funded=1)**: 67
- **Unfunded (any_funded=0)**: 53
- **Entropy_end mean (funded)**: 0.7169
- **Entropy_end mean (unfunded)**: 0.7727
- **Mann-Whitney U**: 1529.0, **p** = 0.1925 (ns)
- **Cohen's d**: -0.2512 (small)
- **Pearson r (funded_rate vs entropy_end)**: -0.1045 (95% CI [-0.2874, 0.0902])
- **Spearman rho**: -0.0840 (95% CI [-0.2625, 0.0970])

**Data Ready**: `outputs/tables/entropy_with_outcomes.csv` contains:
- All entropy metrics (beginning/middle/end)
- `funded_rate`: proportion of teams funded (0.0 to 1.0)
- `any_funded`: binary indicator (0 or 1)
- `n_teams`: team count per session

**Tests Implemented**: `pipelines/test_entropy_outcomes.py` runs:
- Mann-Whitney U test: `entropy_end` for `any_funded=1` vs `any_funded=0`
- Correlation analysis: `funded_rate` vs `entropy_end`
- Effect size calculation (Cohen's d)

**Outputs**:
- ✅ `outputs/tables/entropy_with_outcomes.csv`
- ✅ `outputs/logs/outcome_merge_report.txt`
- ✅ `outputs/analysis/entropy_outcomes_stats.txt`
- ✅ `outputs/tables/entropy_outcome_group_summary.csv`
 

---

### Priority 3: Systematic Convergence Detection ✅ **COMPLETE**

**Research Question**: What fraction of sessions exhibit strict convergence in the final third?

**Method**:
1. Batch-run strict convergence detection across all sessions.
2. Extract convergence and structural wrap rates (time-based).
3. Compare convergence rate vs entropy change (`entropy_end - entropy_beginning`).

**Inputs**: All session JSONs.

**Results** (executed 2026-02-25):
- **Sessions analyzed**: 157
- **Mean strict convergence rate (last third)**: 0.0030
- **Mean structural wrap rate (last third)**: 0.2195

**Outputs**:
- ✅ `outputs/tables/convergence_rates_by_session.csv`
- ✅ `figures/final/convergence_vs_entropy_scatter.png`

**Assumptions**: Convergence regex captures meaningful agreement signals.

**Risks**: High false-positive rate from casual language ("we agree this is hard").

---

### Priority 4: Compare Normalized vs Raw Entropy ✅ **COMPLETE**

**Research Question**: Does normalization by log2(K) improve interpretability or comparability?

**Method**:
1. Reconstruct raw entropy from normalized values using $\log_2(K)$ and compare.
2. Summarize differences by segment.

**Inputs**: Same as Priority 1.

**Results** (executed 2026-02-25):
- **Mean diff (raw - normalized)**: beginning 0.0943, middle 0.0559, end 0.0985
- **Median diff**: 0.0000 for all segments

**Outputs**:
- ✅ `outputs/analysis/entropy_normalization_comparison.txt`
- ✅ `figures/final/raw_vs_normalized_entropy_scatter.png`

**Assumptions**: With CDP-only analysis, $K=2$ (scores 1 vs 2).

**Risks**: Normalization likely has **no effect** when $K$ is constant (CDP-only scores means $K=2$).
**Note**: The codebook contains 8 categories overall, but this analysis uses only the CDP category and its two score levels.

---

### Priority 5: Time-Binned vs Index-Based Thirds ✅ **COMPLETE**

**Research Question**: How different are time-based thirds vs index-based thirds for CDP entropy?

**Method**:
1. Compute CDP entropy by index-based thirds (utterance count).
2. Compute CDP entropy by time-based thirds (meeting duration).
3. Compare correlations and mean differences.

**Inputs**: All session JSONs.

**Results** (executed 2026-02-25):
- **Sessions analyzed**: 157
- **Correlation (time vs index)**: beginning r=0.934, middle r=0.524, end r=0.911
- **Mean diff (time - index)**: beginning -0.0025, middle -0.0908, end 0.0023

**Outputs**:
- ✅ `outputs/tables/time_binning_comparison.csv`
- ✅ `outputs/analysis/time_binning_comparison_summary.txt`

**Assumptions**: Timestamps are accurate and comparable to index-based thirds.

**Risks**: Middle segment shows more variability due to uneven pacing within sessions.

---

### Priority 6: Time-Pressure & Decision-Closure Language ✅ **COMPLETE**

**Research Question**: Where does time-pressure or decision-closure language appear in meetings?

**Method**:
1. Scan utterances for time-pressure and decision-closure phrases.
2. Aggregate counts by time-based thirds (beginning/middle/end).

**Results** (executed 2026-02-25):
- **Sessions analyzed**: 157
- **Time-pressure mean count**: 0.59 (peaks in end; mean 0.38)
- **Decision-closure mean count**: 2.32 (highest in end; mean 0.99)

**Outputs**:
- ✅ `outputs/tables/time_pressure_language_by_session.csv`
- ✅ `outputs/analysis/time_pressure_language_summary.txt`

---

### Priority 7: Outcome Modeling Beyond Entropy ✅ **COMPLETE**

**Research Question**: Do convergence, structural wrap, or time-pressure improve outcome prediction beyond entropy?

**Method**:
1. Merge entropy, convergence, and time-pressure signals.
2. Fit exploratory linear models for `funded_rate` and `any_funded`.

**Results** (executed 2026-02-25):
- **Predictors**: entropy_end, entropy_change, strict_conv_rate_last_third, structural_wrap_rate_last_third, time_pressure_total, decision_closure_total
- **R^2 (funded_rate)**: 0.1298
- **R^2 (any_funded)**: 0.1976

**Outputs**:
- ✅ `outputs/analysis/outcome_modeling_report.txt`
- ✅ `outputs/tables/outcome_model_coefficients.csv`

---

### Priority 8: Structural Wrap Validation

**Research Question**: Does structural wrap increase in final third? Does it correlate with entropy decay?

**Method**:
1. Batch-run `signals.py` for all sessions; extract `structural_time_sec` per bin.
2. Aggregate structural wrap time by segment (beginning/middle/end).
3. Correlate `structural_wrap_end` with `entropy_end`.

**Inputs**: All session JSONs.

**Outputs**:
- `outputs/tables/structural_wrap_by_segment.csv` (MISSING)
- `figures/final/structural_wrap_vs_entropy.png`

**Assumptions**: Regex accurately captures wrap language; wrap is independent of CDP scores.

**Risks**: Wrap may be highly correlated with CDP **scores**, inflating correlations.

---

### Priority 9–14 (Lower Priority)

9. **Speaker-level entropy**: Compute entropy per speaker; identify high-diversity vs low-diversity contributors.
   - **Missing**: Speaker extraction from JSON; speaker-level aggregation logic.

10. **Score transition analysis**: Track switches between CDP score 1 and score 2 across time bins.
   - **Method**: Transition counts; chi-square test.

11. **Temporal autocorrelation**: Measure lag-1 autocorrelation in entropy time series.
   - **Output**: Autocorrelation table per session.

12. **Outcome prediction model**: Logistic regression predicting success from entropy features.
   - **Features**: `entropy_beginning`, `entropy_end`, `entropy_decay`, `score2_share_end`.
    - **MISSING**: Train/test split; cross-validation.

13. **Cross-conference comparison**: Compare entropy distributions across SCIALOG cohorts (2020 vs 2021 vs 2022).
    - **Method**: ANOVA or Kruskal-Wallis.

14. **Qualitative validation**: Select high/low entropy sessions; manually review transcripts for face validity.
    - **Output**: Case study writeup.

---

## Methodological Notes

### Open Questions

1. **Normalization**: Should we report $H$ or $H / \log_2(K)$?
   - **Current recommendation**: Report both; use normalized for cross-session comparison.

2. **Bin size**: 60s vs 30s vs index-based thirds?
   - **Trade-off**: Finer bins capture dynamics but increase noise.
   - **Current recommendation**: Use 60s for within-session plots; use thirds for cross-session stats.

3. **Score imbalance handling**: Should we down-weight segments with extreme score imbalance?
   - **Risk**: Highly imbalanced segments push entropy toward 0.0, masking nuanced shifts.
   - **Current status**: No adjustment implemented.

4. **Multi-label handling**: (Not applicable for CDP-only scores unless multiple CDP scores appear on one utterance.)
   - **Current**: Each CDP score counted once per utterance when present.

### Known Limitations

- **No speaker attribution**: Entropy aggregated at session level.
- **No sequential dependencies**: Entropy assumes independent utterances.
- **Subjective outcomes**: Labels may vary by rater.
- **Incomplete timestamps**: Some sessions have malformed `start_time`/`end_time`.

---