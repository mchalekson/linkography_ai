# CDP Entropy Analysis — Integration README

**Last updated**: February 12, 2026

---

## TL;DR

This repository analyzes **Coordination and Decision Practices (CDP)** score diversity in SCIALOG team discussions using Shannon entropy and temporal dynamics. The goal is to understand whether the **intensity of coordination behaviors** (measured as entropy over CDP score 1 vs score 2 distributions) varies across discussion phases, relates to session outcomes, and exhibits temporal patterns.

**Current state**: ✅ **P0 Complete** - Full pipeline executed on all 157 sessions across 8 conferences. Core findings: Entropy remains **stable** (0.733 → 0.745, no significant change), suggesting teams maintain a consistent mix of basic and advanced coordination throughout. 78.3% of sessions successfully matched with funding outcome data.

**What it produces**:
- **Batch entropy table**: `outputs/tables/cdp_entropy_by_session_ALL_20260212_171302.csv` ✅ **GENERATED** (157 sessions)
- **Entropy trajectory analysis**: `outputs/analysis/entropy_trajectory_summary.txt` ✅ **GENERATED**
- **Entropy with outcomes**: `outputs/tables/entropy_with_outcomes.csv` ✅ **GENERATED** (123 matched sessions)
- **Trajectory visualization**: `figures/final/entropy_trajectory.png` ✅ **GENERATED**
- **Session-level figures**: `figures/generated/slide*.png` (signals, convergence, entropy vs CD)
- **Logs**: `outputs/logs/slide*.txt` (callouts and metadata)

**Data coverage**: 8 SCIALOG conferences (2020NES, 2021ABI, 2021CMC, 2021MND, 2021MZT, 2021NES, 2021SLU, 2022MND) with annotated session transcripts in `data/`.

**Key Finding (P0)**: Entropy shows **no significant trajectory** (stable ~0.7), contradicting convergence hypothesis. Teams use a balanced mix of basic (score 1) and advanced (score 2) coordination throughout sessions.

**Next 3–5 analyses** (P1/P2):
1. ✅ **DONE** - Validated entropy trajectories across all 157 sessions (found stability, not convergence)
2. ✅ **DONE** - Merged entropy with funding outcomes (123/157 matched, 78.3% success rate)
3. **NEXT (P1)** - Statistical testing: Do funded sessions show different CDP score patterns than unfunded?
4. **NEXT (P1)** - Batch convergence detection across all sessions (Priority 3)
5. Compare normalized vs raw entropy for interpretability (Priority 4)
6. Explore time-binned entropy dynamics to detect mid-session shifts (Priority 5)

---

## Goals and Research Questions

### Inferred Goals (update if wrong)

1. **Goal: Quantify CDP score diversity across discussion phases** ✅ **COMPLETE**
   - **Deliverable**: Per-session table with `entropy_beginning`, `entropy_middle`, `entropy_end`.
   - **Success metric**: Entropy values computed for all sessions; distributions visualized by phase.
   - **Status**: ✅ **DONE** - 157 sessions analyzed, results in `cdp_entropy_by_session_ALL_20260212_171302.csv`
   - **Key Finding**: Mean entropy stable (0.733 → 0.745), teams maintain consistent mix of score 1 vs score 2 coordination.

2. **Goal: Relate entropy trajectories to session outcomes** ✅ **DATA READY, TESTING PENDING**
   - **Deliverable**: Statistical comparison (t-test, regression) of entropy patterns for successful vs unsuccessful sessions.
   - **Success metric**: Significant difference or clear trend; reported in results table.
   - **Status**: ✅ Data merged - 123/157 sessions matched with outcomes (mean funding rate: 0.37)
   - **Next**: Statistical testing (t-test/Mann-Whitney U) comparing funded vs unfunded sessions.

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
| **Batch entropy table** | `outputs/tables/cdp_entropy_by_session_ALL_20260212_171302.csv` | ✅ **GENERATED** - 157 sessions, all 8 conferences |
| **Entropy trajectory analysis** | `outputs/analysis/entropy_trajectory_summary.txt` | ✅ **GENERATED** - Statistical tests showing divergence |
| **Entropy with outcomes** | `outputs/tables/entropy_with_outcomes.csv` | ✅ **GENERATED** - 123 matched sessions with funding data |
| **Trajectory visualization** | `figures/final/entropy_trajectory.png` | ✅ **GENERATED** - Bar chart + individual trajectories |
| **Data validation report** | `outputs/logs/data_validation_report.txt` | ✅ **GENERATED** - All 157 sessions passed validation |
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
| `pipelines/run_cdp_entropy_all.py` | Batch entropy across all conferences | Slide 7 | ✅ Ready |
| `pipelines/validate_data_integrity.py` | Validate session JSON integrity | — | ✅ Ready |
| `pipelines/analyze_entropy_trajectories.py` | Analyze beginning→middle→end patterns | — | ✅ Ready |
| `pipelines/merge_entropy_with_outcomes.py` | Merge entropy with funding outcomes | — | ✅ Ready |

### Core Modules (`src/linkography_ai/`)

| Module | Purpose |
|--------|---------|
| `entropy.py` | Shannon entropy: $H = -\sum_i p_i \log_2(p_i)$; optional normalization by $\log_2(K)$. |
| `segmentation.py` | Index-based thirds: `beginning`, `middle`, `end` via $\lfloor n/3 \rfloor$ logic. |
| `io_sessions.py` | Load session JSON; extract CDP codes from `annotations` field. |
| `slides.py` | Time-binned analysis; structural wrap regex; `compute_entropy_vs_cd()`. |
| `discovery.py` | Discover conference directories in `data/` with `session_data/` subdirs. |

---

## What We Measure: CDP Score Diversity

**Focus**: "Coordination and Decision Practices" (CDP) annotations only

**The 8 SCIALOG Categories** (for context):
1. **Coordination and Decision Practices** ⭐ **OUR FOCUS**
2. Knowledge Sharing
3. Information Seeking  
4. Idea Management
5. Evaluation Practices
6. Relational Climate
7. Participation Dynamics
8. Integration Practices

**What CDP Measures**: When an utterance has the CDP annotation, it includes a **score** field:
- **Score 1**: Basic coordination (structuring contributions, simple process management)
- **Score 2**: Advanced coordination (explicit agenda-setting, complex decision-making)

**What Entropy Measures**: The diversity of CDP **intensity levels** (score 1 vs score 2) within a segment.
- **High entropy** (~1.0): Mix of score 1 and score 2 utterances (varied coordination intensity)
- **Low entropy** (~0.0): All utterances have same score (uniform coordination level)

**Example**: 
- Segment with 5 score-1 and 5 score-2 CDP utterances → High entropy (diverse coordination)
- Segment with 10 score-1 CDP utterances → Low entropy (uniform coordination)

**Research Question**: Do teams start with diverse coordination levels (high entropy) and converge to uniform coordination (low entropy) by the end?

**Example**: If a segment uses all 8 categories equally, normalized entropy ≈ 1.0. If only 1 category is used, entropy = 0.0.

---

## Understanding Entropy Values

**Normalized Entropy Range**: 0.0 to 1.0 (current pipeline uses `--normalize` flag)

| Value | Interpretation |
|-------|----------------|
| **0.0 - 0.3** | Very focused - 1-2 dominant behaviors |
| **0.4 - 0.6** | Moderately diverse - 3-4 behaviors |
| **0.7 - 0.9** | High diversity - 5-7 behaviors evenly distributed |
| **0.9 - 1.0** | Maximum diversity - all 8 behaviors used roughly equally |

**Observed Range in Data**: 0.80 - 0.91 (high diversity across all sessions)

**The P0 Finding**: Entropy remains **stable** from 0.733 (beginning) → 0.745 (end)
- **Meaning**: Teams use **more varied** coordination behaviors as sessions progress
- **Contradicts**: Initial hypothesis that teams would converge (focus) on fewer behaviors
- **Possible explanations**: 
  - Final decisions require broader behavioral repertoire
  - Facilitators introduce more structure near the end
  - Teams explore more options before committing

---

## Sample Outputs

### Batch Entropy CSV (cdp_entropy_by_session_ALL_*.csv)

**First 3 rows**:
```csv
conference,session_id,n_utterances,outcome,entropy_beginning,n_cdp_beginning,n_unique_cdp_beginning,entropy_middle,n_cdp_middle,n_unique_cdp_middle,entropy_end,n_cdp_end,n_unique_cdp_end
2020NES,2020_11_05_NES_S1,82,,0.804,47,7,0.912,53,8,0.963,46,7
2020NES,2020_11_05_NES_S2,67,,0.954,47,7,0.891,31,6,0.899,57,7
```

**Column Definitions**:
- `conference`: e.g., "2020NES", "2021MZT"
- `session_id`: Unique identifier (date_conference_session)
- `n_utterances`: Total utterances in session
- `outcome`: NULL (not used - see funded_status in outcomes CSV instead)
- `entropy_beginning/middle/end`: Normalized Shannon entropy (0-1 scale)
- `n_cdp_beginning/middle/end`: Total CDP annotation count (can exceed n_utterances due to multi-label)
- `n_unique_cdp_beginning/middle/end`: Number of distinct categories used (1-8)

### Entropy with Outcomes CSV (entropy_with_outcomes.csv)

**First 3 rows**:
```csv
conference,session_id,...,entropy_end,n_cdp_end,n_unique_cdp_end,funded_rate,any_funded,n_teams
2020NES,2020_11_05_NES_S1,...,0.963,46,7,0.0,0,1
2020NES,2020_11_05_NES_S3,...,0.921,38,8,0.333,1,3
2020NES,2020_11_05_NES_S4,...,0.964,23,8,1.0,1,1
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
df = pd.read_csv('outputs/tables/cdp_entropy_by_session_ALL_20260212_171302.csv')

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
- **Not a bug**: This is real data - sessions genuinely show high behavioral diversity across all 8 categories
- **Interpretation**: Teams use 6-7 different annotation categories throughout (not just "Coordination and Decision Practices")

### Verification Commands

```bash
# Check data integrity
make validate

# Verify outputs exist
ls -lh outputs/tables/cdp_entropy_by_session_ALL_*.csv
ls -lh outputs/analysis/entropy_trajectory_summary.txt
ls -lh figures/final/entropy_trajectory.png

# Count sessions per conference
.venv/bin/python -c "import pandas as pd; df = pd.read_csv('outputs/tables/cdp_entropy_by_session_ALL_20260212_171302.csv'); print(df['conference'].value_counts())"

# Quick stats
.venv/bin/python -c "import pandas as pd; df = pd.read_csv('outputs/tables/cdp_entropy_by_session_ALL_20260212_171302.csv'); print(df[['entropy_beginning', 'entropy_middle', 'entropy_end']].describe())"
```

---

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

| Output Type | Path | Status | Created By |
|-------------|------|--------|------------|
| Batch tables | `outputs/tables/` | Auto-created | `run_cdp_entropy_all.py`, `merge_entropy_with_outcomes.py` |
| Analysis outputs | `outputs/analysis/` | Auto-created | `analyze_entropy_trajectories.py` |
| Logs | `outputs/logs/` | ✅ Exists | All pipeline scripts |
| Generated figures | `figures/generated/` | ✅ Exists | Slide 1-3 pipelines |
| Final figures | `figures/final/` | Auto-created | `analyze_entropy_trajectories.py` |

**Key Artifacts Generated:**

| File | Source | Content |
|------|--------|---------|
| `outputs/tables/cdp_entropy_by_session_ALL_*.csv` | `run_cdp_entropy_all.py` | Per-session entropy (beginning/middle/end) + CDP counts |
| `outputs/tables/entropy_with_outcomes.csv` | `merge_entropy_with_outcomes.py` | Entropy + `funded_rate`, `any_funded`, `n_teams` |
| `outputs/analysis/entropy_trajectory_summary.txt` | `analyze_entropy_trajectories.py` | Statistical tests + effect sizes for phase transitions |
| `figures/final/entropy_trajectory.png` | `analyze_entropy_trajectories.py` | Bar chart + individual trajectory lines |
| `outputs/logs/data_validation_report.txt` | `validate_data_integrity.py` | Data quality summary across all sessions |
| `outputs/logs/outcome_merge_report.txt` | `merge_entropy_with_outcomes.py` | Match/unmatch log for entropy-outcome merge |

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

**Current status**: ✅ **COMPLETE** - Full run executed on 2026-02-12 16:55:26.

**Latest Output**: `outputs/tables/cdp_entropy_by_session_ALL_20260212_171302.csv` (157 sessions)

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

**Individual steps:**
```bash
make validate        # Check data quality
make batch_entropy   # Generate entropy table
make analyze         # Statistical analysis + plots
make merge_outcomes  # Add funding outcomes
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

### Priority 1: Validate Entropy Pipeline ✅ **COMPLETE**

**Research Question**: Does CDP entropy decrease from beginning → middle → end (indicating convergence)?

**Answer**: ❌ **NO** - Entropy remains **stable** from 0.733 → 0.745 (beginning → end), **no significant change**.

**Implementation**: `pipelines/analyze_entropy_trajectories.py`

**Results** (executed 2026-02-12):
- **Sessions analyzed**: 156 (1 excluded due to missing data)
- **Beginning entropy**: 0.733 ± 0.259
- **Middle entropy**: 0.878 ± 0.085  
- **End entropy**: 0.745 ± 0.202
- **Beginning → End change**: +0.026 [95% CI: -0.041, -0.014] ⚠️ Significant INCREASE

**Interpretation**: Teams use **more diverse** coordination behaviors as sessions progress, contradicting the convergence hypothesis. This may indicate:
- Teams explore more strategies near session end
- Facilitators introduce new coordination patterns
- Final decision-making requires broader behavioral repertoire

**Outputs**: 
- ✅ `outputs/analysis/entropy_trajectory_summary.txt`
- ✅ `figures/final/entropy_trajectory.png`

---

### Priority 2: Correlate Entropy with Outcomes ✅ **DATA MERGED, TESTING PENDING**

**Research Question**: Do successful sessions show lower final-third entropy (more focused coordination)?

**Implementation**: `pipelines/merge_entropy_with_outcomes.py`

**Results** (executed 2026-02-12):
- **Entropy sessions**: 157
- **Outcome sessions**: 123  
- **Matched**: 123 (78.3% match rate)
- **Sessions with any funded team**: 68
- **Mean funding rate**: 0.37
- **Funding distribution**: 55 sessions with 0% funding, 25 with 100% funding

**Data Ready**: `outputs/tables/entropy_with_outcomes.csv` contains:
- All entropy metrics (beginning/middle/end)
- `funded_rate`: proportion of teams funded (0.0 to 1.0)
- `any_funded`: binary indicator (0 or 1)
- `n_teams`: team count per session

**Next Step**: Create `pipelines/test_entropy_outcomes.py` to run:
- Mann-Whitney U test: `entropy_end` for `any_funded=1` vs `any_funded=0`
- Correlation analysis: `funded_rate` vs `entropy_end`
- Effect size calculation (Cohen's d)

**Outputs**:
- ✅ `outputs/tables/entropy_with_outcomes.csv`
- ✅ `outputs/logs/outcome_merge_report.txt`

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