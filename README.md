# Predicting Team Success from Coordination Patterns

**TL;DR**: This project analyzes 157 scientific team meetings to predict funding success. We found that **who coordinates** (concentrated leadership) predicts outcomes better than **how much** or **when** coordination happens. Teams with clearer coordination leadership are 27.7% more likely to receive funding.

---

## What This Project Does

**Problem**: How do you know if a collaborative team meeting will lead to successful outcomes?

**Approach**: Analyze behavioral patterns in transcribed team discussions using coordination metrics (entropy, Gini coefficient, speaker diversity).

**Key Finding**: Teams with **concentrated coordination leadership** (1-2 speakers driving decision-making) have significantly higher funding rates than teams with distributed coordination.

**Impact**: Facilitators can identify at-risk sessions early and intervene to improve outcomes.

---

## Quick Start

### Installation
```bash
# Clone and install
git clone https://github.com/mchalekson/linkography_ai.git
cd linkography_ai
pip install -e .

# Run full analysis pipeline
make all
```

**Outputs**: 
- Analysis results in `outputs/analysis/`
- Figures in `figures/final/`
- Complete project documentation in `docs/PROJECT_CONTEXT.md`

### See Key Results
```bash
# Main finding: Speaker diversity predicts funding
cat outputs/analysis/speaker_diversity_effect_sizes.txt

# Visualization: Gini coefficient by funding status
open figures/final/gini_by_funding.png

# Full statistical report
cat outputs/analysis/speaker_diversity_outcomes_summary.txt
```

---

## Project Structure

```
linkography_ai/
├── docs/
│   └── PROJECT_CONTEXT.md          ⭐ START HERE - Complete reproducibility guide
├── data/                            📊 157 meeting transcripts (8 conferences, 2020-2022)
├── pipelines/                       🔬 14 analysis pipelines
│   ├── speaker_diversity_outcomes.py  (KEY: Q3 - predicts funding)
│   ├── timing_patterns_outcomes.py    (Q4 - timing doesn't predict)
│   ├── meeting_profile_classifier.py  (Q5 - 27.7% ROC improvement)
│   └── ... (11 other analyses)
├── outputs/
│   ├── analysis/                    📈 Statistical reports
│   ├── tables/                      📋 Session-level data
│   └── figures/                     📊 Visualizations
├── src/linkography_ai/              💻 Core code (entropy, Gini, I/O)
└── Makefile                         ⚙️  Run everything with `make all`
```

**If you only read one file**: `docs/PROJECT_CONTEXT.md` — complete STEP 1-5 reproducibility guide

---

## Key Findings

### ✅ What Predicts Funding Success

| Finding | Effect Size | p-value |
|---------|-------------|---------|
| **Concentrated coordination leadership** (high Gini) | Cohen's d = 0.591 | p = 0.0006 |
| Advanced coordination concentration | Diff = 0.088 [95% CI: 0.034, 0.138] | Very strong |
| Combined speaker + timing features | +27.7% ROC-AUC improvement | ROC = 0.688 |

**Interpretation**: Teams where 1-2 speakers drive decision-making (high Gini score-2) are significantly more likely to get funded.

### ❌ What Doesn't Predict Funding

- **Coordination diversity** (entropy): No difference between funded vs unfunded (p = 0.193)
- **Timing patterns**: No difference in phase rhythm, transitions, or convergence (all p > 0.14)
- **Speaker participation rate**: No difference (p = 0.493)

---

## How to Use This Repo

### For Researchers
1. **Read the full story**: `docs/PROJECT_CONTEXT.md` (STEP 1-5 structure)
2. **Reproduce key findings**: `make all` (runs all 14 analyses)
3. **Check specific results**: See pipeline outputs in `outputs/analysis/`

### For Data Scientists
1. **Explore the data**: `data/*/session_data/*.json` (timestamped transcripts with behavioral codes)
2. **Run individual analyses**: 
   ```bash
   python pipelines/speaker_diversity_outcomes.py
   python pipelines/meeting_profile_classifier.py
   ```
3. **Modify pipelines**: Core code in `src/linkography_ai/`

### For Facilitators
1. **See the practical implications**: Section "STEP 5: Overall Story" in `docs/PROJECT_CONTEXT.md`
2. **Monitor Gini in real-time**: Use `pipelines/speaker_level_cdp.py` on live data
3. **Identify at-risk sessions**: Sessions with Gini < 0.25 may need intervention

---

## Data Overview

| Metric | Value |
|--------|-------|
| **Total sessions** | 157 |
| **Conferences** | 8 (2020-2022) |
| **Sessions with funding outcomes** | 123 (78.3%) |
| **Funded sessions** | 68 |
| **Mean utterances/session** | 66.5 |
| **Mean speakers/session** | 13.3 |

**What's in the data**: Timestamped meeting transcripts with human-annotated "Coordination and Decision Practices" (CDP) codes. Each utterance labeled with:
- **Score 1**: Basic coordination (structuring, turn-taking)
- **Score 2**: Advanced coordination (decision-making, synthesis)

---

## Key Metrics Explained

### Shannon Entropy (0-1)
**What it measures**: Diversity of coordination strategies (Score 1 vs Score 2 mix)
- 0.0 = All utterances same score (uniform)
- 1.0 = Perfect 50/50 mix
- ~0.73 = Observed average (70% Score 1, 30% Score 2)

**Finding**: Entropy is **stable** across meeting phases (doesn't predict outcomes)

### Gini Coefficient (0-1)
**What it measures**: Concentration of coordination across speakers
- 0.0 = Perfect equality (all speakers contribute equally)
- 1.0 = Perfect inequality (one speaker dominates)
- 0.33 = Funded session average
- 0.24 = Unfunded session average

**Finding**: Higher Gini (concentrated leadership) **strongly predicts funding** (p = 0.0006)

---

## Research Questions Answered

| Question | Method | Result |
|----------|--------|--------|
| **Q1**: Do teams converge to focused coordination? | Entropy trajectory analysis | ❌ No - stable across phases |
| **Q2**: Does entropy predict funding? | Mann-Whitney U | ❌ No - p = 0.193 |
| **Q3**: Does speaker diversity predict funding? | Gini + effect sizes | ✅ **Yes** - d = 0.591, p = 0.0006 |
| **Q4**: Do timing patterns predict funding? | Temporal features | ❌ No - robust null across bin sizes |
| **Q5**: Can we combine features for prediction? | Logistic regression | ✅ **Yes** - 27.7% ROC improvement |
| **Q6**: Do patterns differ by cohort year? | Kruskal-Wallis H | Partial - 2022 more focused mid-meeting |
| **Q7**: What do transcripts reveal? | Qualitative validation | High-Gini = decisive synthesis |

---

## Running Analyses

### Full Pipeline
```bash
make all  # Runs all 14 analyses sequentially
```

### Key Individual Pipelines
```bash
# Q3: Speaker diversity vs outcomes (KEY FINDING)
python pipelines/speaker_diversity_outcomes.py
python pipelines/posthoc_analyses.py  # Effect sizes + visualization

# Q5: Meeting profile classifier
python pipelines/meeting_profile_classifier.py

# Q6: Cohort analysis
python pipelines/cdp_by_cohort_pairwise.py

# Q7: Transcript validation
python pipelines/cdp_transcript_validation.py
```

### Outputs
- **Statistical reports**: `outputs/analysis/*.txt`
- **Data tables**: `outputs/tables/*.csv`
- **Visualizations**: `figures/final/*.png`

---

## Code Organization

### Core Modules (`src/linkography_ai/`)
- `entropy.py` - Shannon entropy computation
- `io_sessions.py` - JSON loading + CDP extraction
- `segmentation.py` - Temporal segmentation (thirds, bins)
- `discovery.py` - Conference/session discovery

### Key Pipelines (`pipelines/`)
- `run_cdp_entropy_all.py` - Batch entropy computation (foundation)
- `speaker_diversity_outcomes.py` - **Main result** (Q3)
- `meeting_profile_classifier.py` - Predictive model (Q5)
- `posthoc_analyses.py` - Effect sizes + visualization
- `timing_patterns_outcomes_bins.py` - Robustness checks (Q4)

---

## Dependencies

**Required**:
- Python ≥ 3.10
- pandas, numpy, matplotlib, scipy, scikit-learn

**Install**:
```bash
pip install -e .
```

**Test**:
```bash
pip install -r requirements-dev.txt
pytest
```

---

## Documentation

| File | Purpose | When to Read |
|------|---------|--------------|
| **`docs/PROJECT_CONTEXT.md`** | Complete reproducibility guide (STEP 1-5) | **START HERE** |
| `README.md` (this file) | Quick overview + navigation
| `codebook/codebook.md` | CDP annotation definitions | Understanding behavioral codes |
| `Makefile` | Pipeline orchestration | Running batch analyses |

---

## Citation

If you use this code or findings:

```
Huang, E., Chalekson, M. (2026). Predicting Team Success from Coordination Patterns: 
Analysis of SCIALOG Collaborative Meetings. Northwestern University.
```

---

## Contact

**Max Chalekson**  
Northwestern University  
NICO (Northwestern Institute on Complex Systems)

**Questions?** See `docs/PROJECT_CONTEXT.md` for detailed documentation or open an issue.

---

## Quick Links

- 📖 **Full documentation**: [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)
- 📊 **Key finding**: [outputs/analysis/speaker_diversity_effect_sizes.txt](outputs/analysis/speaker_diversity_effect_sizes.txt)
- 📈 **Visualization**: [figures/final/gini_by_funding.png](figures/final/gini_by_funding.png)
- 🔬 **All pipelines**: [pipelines/](pipelines/)
- 💻 **Core code**: [src/linkography_ai/](src/linkography_ai/)
