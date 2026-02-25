linkography_ai — SCIALOG CDP analysis

Purpose
-------
This repository implements reproducible analyses of Coordination & Decision Practices (CDP) in SCIALOG meeting transcripts. The code extracts CDP codes from per-utterance annotations, aggregates them into time bins, and computes information-theoretic signals (Shannon entropy) and simple count-based statistics per session and per time-bin.

This README documents reproducibility steps, repository layout, the signal definitions used in analyses, and how to run the batch pipeline that produces session-level CSV outputs.

Repository layout
-----------------
- **`src/linkography_ai`**: core code implementing IO, segmentation, and signal computations. See [src/linkography_ai](src/linkography_ai).
- **`pipelines/`**: analysis and batch processing pipelines:
  - `signals.py` — compute per-bin minutes of coordination/decision and structural wrap signals, with smoothing. (Slide 1)
  - `convergence.py` — detect strict convergence (agreement phrase + commitment code + not structural wrap) and plot convergence vs structural signals. (Slide 2)
  - `entropy_vs_cd.py` — compute and plot entropy vs coordination/decision minutes using time-binned analysis. (Slide 3)
  - `run_cdp_entropy_all.py` — batch runner that iterates datasets, computes per-session CDP counts and entropies, and writes tables and logs to `outputs/`. (Slide 7)
- **`data/`**: per-conference folders (e.g., `data/2020NES`) containing `session_data/` JSON files and session outcome files. Session JSONs are expected under `data/<CONFERENCE>/session_data/*.json`.

Reproducibility and installation
------------------------------
Minimum environment
- Python 3.10+ (project `pyproject.toml` specifies `requires-python = ">=3.10"`).
- Typical runtime dependency: `pandas` (used by the pipeline). Install other dependencies that your analysis requires.

Recommended install (isolated venv)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
pip install -r requirements.txt
```

Running individual slide pipelines
----------------------------------

Each slide pipeline can be run on a single session and produces a PNG figure and a log file with signal statistics.

**Slide 1: Signals by time bin**
```bash
python pipelines/signals.py --session data/2021NES/session_data/2021_11_04_NES_S6.json
```
Outputs: `figures/generated/slide1_<session>.png` and `outputs/logs/slide1_<session>.txt`

**Slide 2: Convergence detection**
```bash
python pipelines/convergence.py --session data/2021NES/session_data/2021_11_04_NES_S6.json --print-context
```
Outputs: `figures/generated/slide2_<session>.png` and `outputs/logs/slide2_<session>.txt`

**Slide 3: Entropy vs Coordination/Decision**
```bash
python pipelines/entropy_vs_cd.py --session data/2021NES/session_data/2021_11_04_NES_S6.json
```
Outputs: `figures/generated/slide3_<session>.png` and `outputs/logs/slide3_<session>.txt`

All slide pipelines support these common flags:
- `--bin-sec` : bin width in seconds (default 60)
- `--smooth-window` : rolling mean window size (default 3)
- `--last-third-only` : restrict to last third of meeting (default True)
- `--out-fig` : custom output figure path
- `--out-log` : custom output log path
- `--print-context` : print nearby utterances around callouts

Running the batch entropy pipeline
--------------------------
The batch entropy pipeline in `pipelines/run_cdp_entropy_all.py` processes entire conferences or all datasets. It provides these CLI flags:

- `--conference` : conference id (e.g., `2021MZT`) or `ALL` (default)
- `--normalize`  : compute normalized Shannon entropy (divide by log2(K))
- `--max_sessions`: limit sessions processed per conference (0 = all)

Example

```bash
python pipelines/run_cdp_entropy_all.py --conference 2021MZT --normalize --max_sessions 0
```

Outputs
- Tables: `outputs/tables/cdp_entropy_by_session_<CONFERENCE>_<TIMESTAMP>.csv`
- Logs:   `outputs/logs/run_cdp_entropy_<CONFERENCE>_<TIMESTAMP>.txt`

The output table contains these columns (produced per session):
- `conference`, `session_id`, `n_utterances`, `outcome`
- For each segment (`beginning`, `middle`, `end`):
	- `entropy_<segment>`: Shannon entropy for CDP **score** categories in that segment (score 1 vs score 2)
	- `n_cdp_<segment>`: total number of CDP annotations counted in that segment
	- `n_unique_cdp_<segment>`: number of unique CDP **score** categories observed in that segment

Data and expected JSON structure
--------------------------------
Session JSON files under `data/<CONFERENCE>/session_data/` can be either:
- an object with an `all_data` list (preferred), or
- a plain list of utterance objects.

Each utterance object should provide one of the text fields: `transcript`, `text`, or `utterance`. CDP annotations are read from nested annotation dicts under `annotation_dict` / `annotations` with the key `Coordination and Decision Practices`. The pipeline extracts the **score** field (1 or 2) and records it as `CDP_score_1` or `CDP_score_2`.

Files and functions that implement these behaviors are in `src/linkography_ai/io_sessions.py` (see `_extract_cdp_from_utterance_dict` and `load_session_utterances`).

Signal definitions (technical)
----------------------------
- Coordination & Decision Practices (CDP):
	- CDP are categorical **scores** attached to individual utterances (score 1 = basic, score 2 = advanced). The code extracts the score as `CDP_score_1` or `CDP_score_2` and treats each as a single count per annotated utterance.

- Time-binned aggregation (structural wrap):
	- Sessions are time-binned using a simple thirds segmentation implemented in `segment_thirds(n)`. Each utterance is assigned to one of `beginning`, `middle`, or `end` according to its index within the session; this is the repository's operationalization of structural wrap/time-bin.

- Entropy (Shannon):
	- For each time-bin, the pipeline counts occurrences of each CDP **score** category and computes Shannon entropy: H = -sum(p_i log2 p_i).
	- The implementation is `shannon_entropy_from_counts(counts, normalize=False)` in `src/linkography_ai/entropy.py`.
	- The `--normalize` flag divides H by log2(K) where K is the number of nonzero categories, yielding a value in [0,1] when K>1.

Notes and best practices for analysis
------------------------------------
- Inspect raw session JSONs before running large batch jobs to confirm the CDP field naming conventions used in your dataset.
- When comparing entropy across sessions with different numbers of observed categories, prefer `--normalize` to reduce scale effects.
- The simple thirds segmentation is intentionally coarse; for finer temporal analysis replace `segment_thirds` with a custom binning function.

Additional pipelines (batch analyses)
-----------------------------------

**Batch convergence detection**
```bash
python pipelines/batch_convergence.py
```
Outputs: `outputs/tables/convergence_rates_by_session.csv`, `figures/final/convergence_vs_entropy_scatter.png`

**Time-binned vs index-based thirds comparison**
```bash
python pipelines/compare_time_binning.py --normalize
```
Outputs: `outputs/tables/time_binning_comparison.csv`, `outputs/analysis/time_binning_comparison_summary.txt`

**Raw vs normalized entropy comparison**
```bash
python pipelines/compare_entropy_normalization.py
```
Outputs: `outputs/analysis/entropy_normalization_comparison.txt`, `figures/final/raw_vs_normalized_entropy_scatter.png`

**Time-pressure & decision-closure language**
```bash
python pipelines/time_pressure_language.py
```
Outputs: `outputs/tables/time_pressure_language_by_session.csv`, `outputs/analysis/time_pressure_language_summary.txt`

**Outcome modeling beyond entropy**
```bash
python pipelines/outcome_modeling.py
```
Outputs: `outputs/analysis/outcome_modeling_report.txt`, `outputs/tables/outcome_model_coefficients.csv`

CDP-Focused Deep Analysis Pipelines
------------------------------------
These pipelines move beyond aggregate entropy metrics to examine how CDP is actually used in meetings: what specific utterances are annotated with each score, which speakers drive CDP usage, when teams shift between basic and advanced coordination, and whether patterns differ across cohorts.

**CDP content analysis: utterance-level by score**
```bash
python pipelines/cdp_content_analysis.py
```
Analyzes what kinds of utterances are annotated with CDP score 1 (basic) vs score 2 (advanced). Computes token counts and samples representative text fragments.

Outputs:
- `outputs/tables/cdp_content_analysis.csv`: Session-level aggregates (count, percent, token length by score)
- `outputs/analysis/cdp_content_analysis_summary.txt`: Summary statistics (mean utterance counts and token lengths)

Key finding: Score 1 utterances are 2.6x more frequent (71% of all CDP) but score 2 utterances are 2.6x longer (49 vs 19 tokens), suggesting more complex coordination ideas appear in fewer, longer utterances.

**Speaker-level CDP analysis: participation and diversity**
```bash
python pipelines/speaker_level_cdp.py
```
Identifies which speakers drive CDP usage and tests whether balanced speaker participation correlates with outcomes. Computes Gini coefficients for score concentration across speakers.

Outputs:
- `outputs/tables/speaker_level_cdp.csv`: Session-level (Gini for each score, speaker participation rate)
- `outputs/analysis/speaker_level_cdp_summary.txt`: Summary statistics (mean Gini, mean participation)

Key finding: Score 2 (advanced CDP) is more balanced across speakers (Gini=0.289) than score 1 (Gini=0.418), suggesting advanced coordination involves broader participation.

**Fine-grained CDP timing: entropy in 5-10 minute windows**
```bash
python pipelines/fine_grained_cdp_timing.py --bin-sec 300
```
Replaces the coarse thirds segmentation with configurable time windows (default 5 minutes) to detect inflection points where teams shift between basic and advanced coordination practices.

Outputs:
- `outputs/tables/cdp_fine_grained_entropy_300s.csv`: Per-bin entropy and counts (bin start/end, entropy, n_cdp)
- `outputs/analysis/cdp_fine_grained_summary_300s.txt`: Summary (total bins, mean entropy, range)

Usage note: Change `--bin-sec` to 600 for 10-min bins, 180 for 3-min bins, etc.

Key finding: Mean entropy per 5-min bin (0.418) shows substantial variance (std=0.440) with range [0, 1], indicating dynamic shifts between score 1 and score 2 throughout meetings.

**CDP patterns by cohort and year**
```bash
python pipelines/cdp_by_cohort.py
```
Tests whether CDP entropy distributions differ across conference years (2020, 2021, 2022) using Kruskal-Wallis H tests. Identifies cohort-level trends in how teams coordinate.

Outputs:
- `outputs/analysis/cdp_by_cohort_summary.txt`: Segment-by-segment statistics (mean, median, std per year) and H-test p-values

Key finding: No significant year effect in beginning/end segments (p>0.05), but middle segment shows trend (H=7.90, p~0.02): 2022 teams show lower entropy in the middle (mean 0.427) vs 2021 (0.664), suggesting more focused decision-making mid-meeting in the most recent cohort.

**Speaker role analysis: role-based CDP patterns**
```bash
python pipelines/speaker_role_cdp.py
```
Extracts speaker roles (facilitator, fellow, mentor, participant) from session metadata and analyzes whether specific roles drive CDP adoption.

Outputs:
- `outputs/tables/speaker_role_cdp.csv`: Session-level role-CDP correlations
- `outputs/analysis/speaker_role_cdp_summary.txt`: Aggregate statistics

Note: This analysis depends on explicit role assignments in session metadata; output is limited if role data is sparse in your dataset.

Where to look in the codebase
-----------------------------
- IO and CDP extraction: [src/linkography_ai/io_sessions.py](src/linkography_ai/io_sessions.py)
- Segmentation (thirds): [src/linkography_ai/segmentation.py](src/linkography_ai/segmentation.py)
- Entropy implementation: [src/linkography_ai/entropy.py](src/linkography_ai/entropy.py)
- Batch runner: [pipelines/run_cdp_entropy_all.py](pipelines/run_cdp_entropy_all.py)
- Batch convergence: [pipelines/batch_convergence.py](pipelines/batch_convergence.py)
- Time-binning comparison: [pipelines/compare_time_binning.py](pipelines/compare_time_binning.py)
- Normalization comparison: [pipelines/compare_entropy_normalization.py](pipelines/compare_entropy_normalization.py)
- Time-pressure language: [pipelines/time_pressure_language.py](pipelines/time_pressure_language.py)
- Outcome modeling: [pipelines/outcome_modeling.py](pipelines/outcome_modeling.py)
- **CDP content analysis**: [pipelines/cdp_content_analysis.py](pipelines/cdp_content_analysis.py)
- **Speaker-level CDP**: [pipelines/speaker_level_cdp.py](pipelines/speaker_level_cdp.py)
- **Fine-grained CDP timing**: [pipelines/fine_grained_cdp_timing.py](pipelines/fine_grained_cdp_timing.py)
- **CDP by cohort**: [pipelines/cdp_by_cohort.py](pipelines/cdp_by_cohort.py)
- **Speaker role CDP**: [pipelines/speaker_role_cdp.py](pipelines/speaker_role_cdp.py)

Testing
-------
```bash
pip install -r requirements-dev.txt
pytest
```
