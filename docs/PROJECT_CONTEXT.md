# CDP Entropy Analysis — Integration README

**Last updated**: February 12, 2026

---

## TL;DR

This repository analyzes **Coordination and Decision Practices (CDP)** in SCIALOG team discussions using Shannon entropy and temporal dynamics. The goal is to understand whether the **diversity of coordination behaviors** (measured as entropy over CDP code distributions) varies across discussion phases, relates to session outcomes, and exhibits temporal patterns like convergence or structural wrap.

**Current state**: Exploratory. Core entropy and time-segmentation pipelines are functional. Outputs include per-session entropy tables and visualizations for select sessions, but systematic cross-conference validation is incomplete.

**What it produces**:
- **Batch entropy table**: `outputs/tables/cdp_entropy_by_session_*.csv` (via `run_cdp_entropy_all.py`)
- **Session-level figures**: `figures/generated/slide*.png` (signals, convergence, entropy vs CD)
- **Logs**: `outputs/logs/slide*.txt` (callouts and metadata)

**Data coverage**: 8 SCIALOG conferences (2020NES, 2021ABI, 2021CMC, 2021MND, 2021MZT, 2021NES, 2021SLU, 2022MND) with annotated session transcripts in `data/`.

**Next 3–5 analyses**:
1. Run `run_cdp_entropy_all.py` across all conferences; validate entropy trajectories (beginning → middle → end).
2. Correlate entropy with session outcomes (success/failure labels from `*_session_outcomes.json`).
3. Detect convergence patterns systematically across all sessions (extend Slide 2 logic).
4. Compare normalized vs raw entropy for interpretability.
5. Explore time-binned entropy dynamics (finer than thirds) to detect mid-session shifts.

---

## Goals and Research Questions

### Inferred Goals (update if wrong)

1. **Goal: Quantify CDP diversity across discussion phases**
   - **Deliverable**: Per-session table with `entropy_beginning`, `entropy_middle`, `entropy_end`.
   - **Success metric**: Entropy values computed for all sessions; distributions visualized by phase.
   - **Status**: ✅ Implemented (`run_cdp_entropy_all.py`), but not run on full dataset.

2. **Goal: Relate entropy trajectories to session outcomes**
   - **Deliverable**: Statistical comparison (t-test, regression) of entropy patterns for successful vs unsuccessful sessions.
   - **Success metric**: Significant difference or clear trend; reported in results table.
   - **Status**: ⚠️ MISSING — outcomes exist in JSON but analysis not performed.

3. **Goal: Detect temporal convergence signals**
   - **Deliverable**: Per-session convergence rate (utterances meeting strict convergence criteria).
   - **Success metric**: Convergence flagged in logs; visualized in time-series plots.
   - **Status**: ✅ Implemented for single sessions (`convergence.py`); batch version MISSING.

4. **Goal: Validate structural wrap as meeting-management signal**
   - **Deliverable**: Per-bin structural wrap counts; correlation with entropy decay.
   - **Success metric**: Wrap increases in final third; negatively correlated with entropy.
   - **Status**: ✅ Regex implemented (`STRUCTURAL_WRAP_PAT`); systematic validation MISSING.

5. **Goal: Provide reproducible pipeline for entropy computation**
   - **Deliverable**: CLI tools in `pipelines/` with `--help` documentation; README usage examples.
   - **Success metric**: Another researcher can replicate entropy table with single command.
   - **Status**: ✅ Implemented; tested on example session.

6. **Goal: Explore time-binned dynamics vs index-based thirds**
   - **Deliverable**: Side-by-side comparison of both segmentation methods.
   - **Success metric**: Documented trade-offs; recommendation for each use case.
   - **Status**: ⚠️ Both implemented separately; comparative analysis MISSING.

---

## Key Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| **Notebook (exploratory)** | `notebooks/linkography-ai.ipynb` | Original slide analyses (Dec 2025); not executed. |
| **Batch entropy table** | `outputs/tables/cdp_entropy_by_session_*.csv` | MISSING — needs first full run. |
| **Example session figures** | `figures/generated/slide1_2021_11_04_NES_S6.png` | Signals plot (CD + wrap). |
| | `figures/generated/slide2_2021_11_04_NES_S6.png` | Convergence detection plot. |
| | `figures/generated/slide3_2021_11_04_NES_S6.png` | Entropy vs CD dual-axis plot. |
| **Example session logs** | `outputs/logs/slide1_2021_11_04_NES_S6.txt` | Callout: longest commitment-coded utterance. |
| | `outputs/logs/slide2_2021_11_04_NES_S6.txt` | Callout: longest convergence utterance. |
| | `outputs/logs/slide3_2021_11_04_NES_S6.txt` | Session metadata + callout. |
| **Codebook** | `codebook/codebook.md` | Definitions for 8 CDP behavioral codes. |
| **Project README** | `README.md` | Installation, usage, data structure reference. |

**Note**: `figures/final/` exists but is empty (reserved for publication-ready outputs).

---

## Repository Map

### Entry Points (Pipelines)

| Script | Purpose | Slide Ref | Status |
|--------|---------|-----------|--------|
| `pipelines/signals.py` | Per-bin CD + structural wrap signals | Slide 1 | ✅ Tested |
| `pipelines/convergence.py` | Strict convergence detection | Slide 2 | ✅ Tested |
| `pipelines/entropy_vs_cd.py` | Entropy vs CD minutes dual plot | Slide 3 | ✅ Tested |
| `pipelines/run_cdp_entropy_all.py` | Batch entropy across all conferences | Slide 7 | ✅ Code ready; not run |

### Core Modules (`src/linkography_ai/`)

| Module | Purpose |
|--------|---------|
| `entropy.py` | Shannon entropy: $H = -\sum_i p_i \log_2(p_i)$; optional normalization by $\log_2(K)$. |
| `segmentation.py` | Index-based thirds: `beginning`, `middle`, `end` via $\lfloor n/3 \rfloor$ logic. |
| `io_sessions.py` | Load session JSON; extract CDP codes from `annotations` field. |
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
**Note**: `annotations` is a nested dict where each key is a CDP category containing `explanation`, `score`, `score_justification`, and `when` fields.

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

| Output Type | Path | Status |
|-------------|------|--------|
| Batch tables | `outputs/tables/` | Directory exists; no CSV yet |
| Logs | `outputs/logs/` | 3 example logs present |
| Generated figures | `figures/generated/` | 3 example figures present |
| Final figures | `figures/final/` | Empty (reserved) |

---

## What's Implemented (Concrete)

### 1. CDP Extraction (`io_sessions.py`)

**Purpose**: Load session JSON files and extract CDP codes from multi-label annotations.

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
    H /= log2(K)  # K = number of unique codes
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
- Compute per-bin CDP entropy using `shannon_entropy_from_counts()` on code distributions.
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
   - Count CDP codes per segment.
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
- `n_unique_cdp_beginning`, `n_unique_cdp_middle`, `n_unique_cdp_end` (unique codes)

**Current status**: Code ready; **MISSING** first full run.

---

## How to Run / Reproduce

### Environment Setup

1. **Clone repository**:
   ```bash
   git clone https://github.com/mchalekson/linkography_ai.git
   cd linkography_ai
   ```

2. **Install package** (Python ≥3.10):
   ```bash
   pip install -e .
   ```

3. **Dependencies**: pandas, numpy, matplotlib (no explicit requirements.txt; inferred from imports).

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
- `entropy_*`: Shannon entropy of CDP code distribution in that segment (higher = more diverse).
- `n_cdp_*`: Total CDP code instances (multi-label utterances counted per code).
- `n_unique_cdp_*`: Number of distinct CDP codes used.
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

### Priority 1: Validate Entropy Pipeline

**Research Question**: Does CDP entropy decrease from beginning → middle → end (indicating convergence)?

**Method**:
1. Run `run_cdp_entropy_all.py --conference ALL --normalize` to generate full table.
2. Compute mean entropy by segment: `df.groupby('segment')[['entropy_beginning', 'entropy_middle', 'entropy_end']].mean()`.
3. Paired t-tests: `beginning vs middle`, `middle vs end`.

**Inputs**: All session JSONs in `data/*/session_data/`.

**Outputs**: 
- `outputs/tables/cdp_entropy_by_session_ALL_20260212_*.csv`
- `outputs/analysis/entropy_trajectory_stats.txt` (MISSING — needs creation)

**Assumptions**: Entropy is well-defined for all sessions (no NaN due to missing codes).

**Risks**: If most sessions have constant entropy, hypothesis not supported.

---

### Priority 2: Correlate Entropy with Outcomes

**Research Question**: Do successful sessions show lower final-third entropy (more focused coordination)?

**Method**:
1. Parse `outcome` field from `*_session_outcomes.json` (⚠️ currently missing in some files).
2. Filter table to sessions with valid outcomes.
3. Compare `entropy_end` for `outcome == "success"` vs `outcome == "failure"`.
4. T-test or Mann-Whitney U.

**Inputs**: 
- `outputs/tables/cdp_entropy_by_session_ALL_*.csv`
- `data/*/*_session_outcomes.json` (needs validation)

**Outputs**:
- `outputs/analysis/entropy_outcome_comparison.txt`
- `figures/final/entropy_by_outcome_boxplot.png`

**Assumptions**: Outcomes are binary or ordinal; labeling is consistent across conferences.

**Risks**: Outcome field may be missing or inconsistently named; small N for some conferences.

---

### Priority 3: Systematic Convergence Detection

**Research Question**: What fraction of sessions exhibit strict convergence in the final third?

**Method**:
1. Batch-run `convergence.py` for all sessions (requires loop wrapper).
2. Extract convergence rate per session: `n_convergence_utterances / n_total_utterances`.
3. Compare convergence rate vs entropy decay (`entropy_end - entropy_beginning`).

**Inputs**: All session JSONs.

**Outputs**:
- `outputs/tables/convergence_rates_by_session.csv` (MISSING — needs script)
- `figures/final/convergence_vs_entropy_scatter.png`

**Assumptions**: Convergence regex captures meaningful agreement signals.

**Risks**: High false-positive rate from casual language ("we agree this is hard").

---

### Priority 4: Compare Normalized vs Raw Entropy

**Research Question**: Does normalization by log2(K) improve interpretability or comparability?

**Method**:
1. Run `run_cdp_entropy_all.py` twice: once with `--normalize`, once without.
2. Compute correlation between `entropy_end` (raw) and `n_unique_cdp_end`.
3. Repeat for normalized entropy.
4. If raw entropy strongly correlates with K, prefer normalized.

**Inputs**: Same as Priority 1.

**Outputs**:
- `outputs/analysis/entropy_normalization_comparison.txt`
- `figures/final/raw_vs_normalized_entropy_scatter.png`

**Assumptions**: K varies sufficiently across sessions to matter.

**Risks**: If K is constant (≈8 codes always present), normalization has no effect.

---

### Priority 5: Time-Binned Entropy Dynamics

**Research Question**: Do entropy shifts occur mid-session (not detectable in thirds)?

**Method**:
1. Extend `entropy_vs_cd.py` to output per-bin entropy to CSV.
2. Identify sessions with sharp entropy drops (>1.0 bit change within 5 minutes).
3. Manually inspect transcripts for those sessions to identify triggers.

**Inputs**: All session JSONs.

**Outputs**:
- `outputs/tables/per_bin_entropy_all_sessions.csv` (MISSING — needs modification to `entropy_vs_cd.py`)
- `outputs/analysis/entropy_shift_sessions.txt` (list of candidate sessions)

**Assumptions**: 60s bins are fine enough; timestamps are accurate.

**Risks**: Noisy bins due to sparse utterances (some bins may have <5 utterances).

---

### Priority 6: Structural Wrap Validation

**Research Question**: Does structural wrap increase in final third? Does it correlate with entropy decay?

**Method**:
1. Batch-run `signals.py` for all sessions; extract `structural_time_sec` per bin.
2. Aggregate structural wrap time by segment (beginning/middle/end).
3. Correlate `structural_wrap_end` with `entropy_end`.

**Inputs**: All session JSONs.

**Outputs**:
- `outputs/tables/structural_wrap_by_segment.csv` (MISSING)
- `figures/final/structural_wrap_vs_entropy.png`

**Assumptions**: Regex accurately captures wrap language; wrap is independent of CDP codes.

**Risks**: Wrap may be highly correlated with "Coordination and Decision Practices" code, inflating correlations.

---

### Priority 7–12 (Lower Priority)

7. **Speaker-level entropy**: Compute entropy per speaker; identify high-diversity vs low-diversity contributors.
   - **Missing**: Speaker extraction from JSON; speaker-level aggregation logic.

8. **Code co-occurrence analysis**: Build co-occurrence matrix for CDP codes; identify systematic multi-label patterns.
   - **Method**: Pair-wise counts; chi-square test.

9. **Temporal autocorrelation**: Measure lag-1 autocorrelation in entropy time series.
   - **Output**: Autocorrelation table per session.

10. **Outcome prediction model**: Logistic regression predicting success from entropy features.
    - **Features**: `entropy_beginning`, `entropy_end`, `entropy_decay`, `n_unique_cdp_end`.
    - **MISSING**: Train/test split; cross-validation.

11. **Cross-conference comparison**: Compare entropy distributions across SCIALOG cohorts (2020 vs 2021 vs 2022).
    - **Method**: ANOVA or Kruskal-Wallis.

12. **Qualitative validation**: Select high/low entropy sessions; manually review transcripts for face validity.
    - **Output**: Case study writeup.

---

## Methodological Notes

### Open Questions

1. **Normalization**: Should we report $H$ or $H / \log_2(K)$?
   - **Current recommendation**: Report both; use normalized for cross-session comparison.

2. **Bin size**: 60s vs 30s vs index-based thirds?
   - **Trade-off**: Finer bins capture dynamics but increase noise.
   - **Current recommendation**: Use 60s for within-session plots; use thirds for cross-session stats.

3. **Rare code filtering**: Should codes with <3 occurrences be excluded?
   - **Risk**: Rare codes inflate K, lowering normalized entropy.
   - **Current status**: No filtering implemented.

4. **Multi-label handling**: Should we weight by 1/n_codes per utterance?
   - **Current**: Each code counted once per utterance.
   - **Alternative**: Weight by $1 / \text{n_codes}$ to avoid inflating counts.

### Known Limitations

- **No speaker attribution**: Entropy aggregated at session level.
- **No sequential dependencies**: Entropy assumes independent utterances.
- **Subjective outcomes**: Labels may vary by rater.
- **Incomplete timestamps**: Some sessions have malformed `start_time`/`end_time`.

---

## Questions or Issues

Contact **Max Chalekson** or **Evey** for clarifications.

**Repository**: [https://github.com/mchalekson/linkography_ai](https://github.com/mchalekson/linkography_ai)
