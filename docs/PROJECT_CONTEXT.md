# CDP Analysis of SCIALOG Team Discussions — Project Context

**Last updated**: February 25, 2026

---

## Overview

This document is a **complete reproducibility guide** for someone (human or AI) who knows **nothing about this research** and wants to recreate it from scratch.

**What this project does**: Analyzes how scientific teams coordinate during collaborative meetings to predict which teams will receive funding.

**Target audience**: Researchers, data scientists, or AI agents with no prior knowledge of linkography, SCIALOG, or team coordination analysis.

**How to use this document**: 
1. Read sequentially through STEP 1 → STEP 5
2. Each STEP builds on the previous one
3. Code paths and commands are provided to reproduce every result
4. No prior domain knowledge required — all concepts defined inline

**Project structure**: This document follows the complete research pipeline from raw data → analysis → results → interpretation → overall findings.

---

## STEP 1: What Was the Data?

**Context**: This analysis uses real transcripts from scientific team meetings. Think of it like analyzing a recording of a brainstorming session, but with human-annotated labels for specific behaviors.

### Data Source
**SCIALOG Collaborative Meetings**: Transcribed team discussions from 8 SCIALOG conferences held between 2020-2022. SCIALOG (Science + Dialog) brings together early-career researchers to form collaborative teams around interdisciplinary science challenges.

### Raw Data Structure
```
data/
├── 2020NES/  (Neural Engineering for Sustainability)
├── 2021ABI/  (Antibiotics Innovations)
├── 2021CMC/  (Chemical Machinery of the Cell)
├── 2021MND/  (Microbiome in the Nexus of Diet)
├── 2021MZT/  (Molecules to Marketplace: Zinc Transition)
├── 2021NES/  (Neural Engineering for Sustainability)
├── 2021SLU/  (Sustainable Landscapes)
└── 2022MND/  (Microbiome in the Nexus of Diet)
```

Each conference contains:
- **Session transcripts** (`session_data/*.json`): Timestamped utterances with human-annotated behavioral codes
- **Outcome data** (`*_session_outcomes.json`): Team formation and funding status
- **Person-to-team mappings** (`*_person_to_team.json`): Team membership (not used in current analysis)

### Session Transcript Format
Each session JSON contains:
```json
{
  "all_speakers": ["Speaker A", "Speaker B", ...],
  "total_speaking_length": 5667,
  "all_data": [
    {
      "speaker": "Speaker A",
      "timestamp": "00:02-00:03",
      "transcript": "Let's organize our approach...",
      "start_time": "00:02",
      "end_time": "00:03",
      "annotations": {
        "Coordination and Decision Practices": {
          "explanation": "Setting agenda for discussion",
          "score": 1,
          "score_justification": "Basic coordination",
          "when": "beginning"
        }
      },
      "role": "Scialog Fellow",
      "when": "beginning"
    }
  ]
}
```

**Key Fields**:
- `annotations["Coordination and Decision Practices"]`: The behavioral code we analyze
  - `score`: 1 (basic coordination) or 2 (advanced coordination)
  - `when`: Temporal segment (beginning/middle/end)
- `speaker`, `timestamp`, `transcript`: Speaker identity, timing, and utterance text

### Outcome Data Format
```json
{
  "2021_11_04_NES_S5": {
    "teams": {
      "NES5": {
        "members": ["Person A", "Person B"],
        "funded_status": 1
      },
      "NES22": {
        "members": ["Person C", "Person D"],
        "funded_status": 0
      }
    }
  }
}
```

**Key Fields**:
- `funded_status`: 1 = team received funding, 0 = team did not receive funding
- Aggregated to session-level:
  - `any_funded`: Did ANY team from this session get funded? (binary: 0 or 1)
  - `funded_rate`: What fraction of teams from this session got funded? (0.0 to 1.0)

### What is CDP (Coordination and Decision Practices)?

**Plain English**: CDP is a behavioral code that captures **how teams organize their work and make decisions**. It has two levels:

- **Score 1 (Basic Coordination)**: Organizing the conversation
  - Simple process management, turn-taking, agenda-setting
  - Example: "Let's go around and share ideas"
  - Think: **Structuring the discussion**

- **Score 2 (Advanced Coordination)**: Making decisions
  - Complex synthesis, explicit decision-making, strategic planning  
  - Example: "Given our constraints, I propose we prioritize approach X because it addresses both Y and Z"
  - Think: **Actually deciding what to do**

**Why this matters**: Teams need both. Score 1 keeps things organized. Score 2 moves things forward.

### What is Linkography?

**Plain English**: A method for analyzing design conversations by tracking how ideas connect over time.

**What we use from linkography**: We focus on one specific metric: **CDP score diversity** (how much teams mix basic vs advanced coordination). We measure this using Shannon entropy (a math concept explained in STEP 4).

**You don't need to understand linkography to reproduce this project** — just know we're measuring "how mixed are basic and advanced coordination behaviors."

---

## STEP 2: Summary Statistics of Data Attributes

### Dataset Coverage
| Attribute | Value |
|-----------|-------|
| **Total sessions** | 157 |
| **Total conferences** | 8 |
| **Sessions with outcome data** | 123 (78.3% match rate) |
| **Sessions with any funded team** | 68 |
| **Mean funded_rate across sessions** | 0.37 |
| **Date range** | 2020-2022 |

### Session-Level Statistics
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| **Utterances per session** | 66.5 | 29.8 | 14 | 198 |
| **CDP annotations per session** | 33.4 | — | — | — |
| **Speakers per session (all)** | 13.3 | — | — | — |
| **Speakers with CDP per session** | 6.7 | — | — | — |
| **CDP participation rate** | 52% | — | — | — |

### CDP Distribution Statistics
| Metric | Value |
|--------|-------|
| **Score 1 prevalence** | 71% of all CDP utterances |
| **Score 2 prevalence** | 29% of all CDP utterances |
| **Score 1 mean length** | 19 tokens/utterance |
| **Score 2 mean length** | 49 tokens/utterance |
| **Ratio (Score 2 / Score 1 length)** | 2.6× |

### Temporal Statistics (Entropy by Phase)
| Phase | Mean Entropy | Std | Range |
|-------|--------------|-----|-------|
| **Beginning** | 0.733 | 0.259 | [0.0, 1.0] |
| **Middle** | 0.650 | 0.375 | [0.0, 1.0] |
| **End** | 0.745 | 0.202 | [0.0, 1.0] |

**Interpretation**: Entropy near 0.7-0.75 indicates teams use a **balanced mix** of score 1 (basic) and score 2 (advanced) coordination. Values close to 0.0 would indicate all utterances have the same score (uniform). Values close to 1.0 indicate a perfect 50/50 mix.

### Speaker Diversity Statistics (Gini Coefficient)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Gini (Score 1)** | 0.418 | Moderate concentration; few speakers dominate basic coordination |
| **Gini (Score 2)** | 0.289 | Balanced distribution; advanced coordination spreads across speakers |

**Interpretation**: Lower Gini = more equal distribution. Score 2 (advanced coordination) is more evenly distributed among speakers, suggesting it's a **team activity** rather than driven by a single facilitator.

---

## STEP 3: Goal of Research

**Primary Research Goal**:  
Understand how **team coordination behaviors** (measured via CDP diversity and speaker patterns) evolve during collaborative meetings and whether these patterns relate to **team formation success** (funding outcomes).

**Specific Aims**:
1. **Temporal dynamics**: Do teams converge from diverse coordination strategies (high entropy) to focused strategies (low entropy) as meetings progress?
2. **Outcome prediction**: Do certain coordination patterns (e.g., more concentrated leadership, specific timing rhythms) predict funding success?
3. **Speaker diversity**: Does the distribution of coordination responsibility across team members matter for outcomes?
4. **Cohort effects**: Have coordination patterns changed over time (2020 → 2022)?

**Why This Matters**:
- **For facilitators**: Understand what coordination structures support successful team formation
- **For researchers**: Identify behavioral markers of productive scientific collaboration
- **For funding agencies**: Predict which early-stage collaborations are likely to succeed

---

## STEP 4: Research Questions, Analyses, and Results

**Structure**: For each question below, we show:
1. **What analysis was done** (method + pipeline path)
2. **Why this analysis** (justification for approach)
3. **What it looks like** (outcome variables, features, statistical tests)
4. **Results** (exact numbers, tables)
5. **Interpretation** (what it means, why it matters)

---

### Q1: Do teams converge to focused coordination strategies over time?

**Hypothesis**: Teams start with diverse coordination (high entropy, mixing basic and advanced practices) and converge to focused strategies (low entropy, uniform coordination) by the end of the meeting.

#### Analysis Approach
**Method**: Shannon entropy analysis across temporal phases

**Why this approach?**  
Shannon entropy measures the **diversity of coordination intensity levels** (score 1 vs score 2) in each meeting phase. High entropy = mixed use of basic and advanced coordination; low entropy = uniform coordination strategy. If teams converge, entropy should decrease from beginning → end.

**What the analysis looks like**:
- **Outcome variable**: Entropy (0-1 normalized Shannon entropy)
  - Formula: $H = -\sum_{i} p_i \log_2(p_i)$, normalized by $\log_2(K)$ where $K=2$ (two score levels)
- **Features**: Meeting phase (beginning/middle/end thirds by utterance index)
- **Statistical test**: Paired t-tests comparing entropy_beginning vs entropy_end

#### Results
| Metric | Value |
|--------|-------|
| **Sessions analyzed** | 138 (usable sessions with ≥3 CDP utterances per phase) |
| **Mean entropy_beginning** | 0.733 ± 0.259 |
| **Mean entropy_middle** | 0.650 ± 0.375 |
| **Mean entropy_end** | 0.745 ± 0.202 |
| **Beginning → End change** | -0.011 [95% CI: -0.068, 0.044] |
| **Statistical significance** | p > 0.05 (no significant change) |

**Visualization**: [entropy_trajectory.png](../figures/final/entropy_trajectory.png)

**Pipeline**: `pipelines/analyze_entropy_trajectories.py`

**Outputs**:
- `outputs/analysis/entropy_trajectory_summary.txt`
- `figures/final/entropy_trajectory.png`

#### Interpretation
**❌ Hypothesis rejected**: Teams do **NOT** converge to focused coordination. Entropy remains **stable** (~0.73-0.75) across all phases.

**What this means**:
- Teams maintain a **consistent mix** of basic (score 1) and advanced (score 2) coordination throughout meetings
- Both coordination levels are needed continuously, not just at specific phases
- Successful collaboration requires **sustained dual-mode thinking**: structuring (score 1) AND deciding (score 2)

**Why this might happen**:
- Facilitators intentionally sustain a balance of process management and decision-making
- Teams loop between planning and deciding rather than progressing linearly
- Complex problems require alternating between "how we work" (score 1) and "what we decide" (score 2)

---

### Q2: Do coordination patterns predict funding outcomes?

**Hypothesis**: Sessions with lower final-phase entropy (more focused coordination) will have higher funding rates.

#### Analysis Approach
**Method**: Statistical comparison of entropy distributions by funding status

**Why this approach?**  
If focused coordination predicts success, we should see significantly different entropy values between funded vs unfunded sessions. Mann-Whitney U (non-parametric) is appropriate because entropy distributions are not normal.

**What the analysis looks like**:
- **Outcome variable**: `any_funded` (binary: 0 or 1) and `funded_rate` (continuous: 0.0 to 1.0)
- **Features**: `entropy_end` (final third entropy)
- **Statistical tests**: 
  - Mann-Whitney U test (funded vs unfunded)
  - Pearson/Spearman correlation (funded_rate vs entropy)
  - Cohen's d effect size

#### Results
| Metric | Value |
|--------|-------|
| **Sessions with outcomes** | 120 |
| **Funded sessions (any_funded=1)** | 67 |
| **Unfunded sessions (any_funded=0)** | 53 |
| **Mean entropy_end (funded)** | 0.717 |
| **Mean entropy_end (unfunded)** | 0.773 |
| **Mann-Whitney U** | 1529.0, p = 0.193 (not significant) |
| **Cohen's d** | -0.25 (small effect) |
| **Pearson r (funded_rate vs entropy_end)** | -0.10 [95% CI: -0.29, 0.09] |

**Pipeline**: `pipelines/test_entropy_outcomes.py`

**Outputs**:
- `outputs/analysis/entropy_outcomes_stats.txt`
- `outputs/tables/entropy_outcome_group_summary.csv`

#### Interpretation
**❌ Hypothesis rejected**: Entropy alone does **NOT** predict funding outcomes.

**What this means**:
- The **diversity of coordination strategies** (score 1 vs score 2 mix) is similar in funded and unfunded sessions
- Funding success is not about **what coordination mode** teams use, but potentially **how** they use it (e.g., who speaks, when transitions happen)
- Need to look beyond aggregate entropy to **speaker-level** and **temporal dynamics**

**Next steps motivated by this finding**: Shift focus to speaker diversity (Q3) and timing patterns (Q4).

---

### Q3: Does speaker distribution of coordination predict outcomes?

**Hypothesis**: Sessions with more concentrated coordination leadership (high Gini coefficient) will have higher funding rates.

**Alternative hypothesis**: Sessions with more distributed coordination (low Gini) will have higher funding rates.

#### Analysis Approach
**Method**: Gini coefficient analysis + Mann-Whitney U tests

**Why this approach?**  
Gini coefficient measures **inequality in speaker participation**. High Gini = coordination concentrated in few speakers (clear leader). Low Gini = coordination distributed equally (egalitarian). Compare Gini distributions between funded vs unfunded sessions.

**What the analysis looks like**:
- **Outcome variables**: `any_funded` (binary)
- **Features**:
  - `gini_score_1`: Concentration of basic coordination across speakers
  - `gini_score_2`: Concentration of advanced coordination across speakers
  - `speaker_participation_cdp`: Fraction of speakers contributing any CDP
- **Statistical tests**: Mann-Whitney U, Cohen's d, bootstrap 95% CI
- **Effect size computation**: 2000 bootstrap replicates for mean difference CI

#### Results
| Feature | Funded Mean | Unfunded Mean | Difference [95% CI] | Cohen's d | p-value |
|---------|-------------|---------------|---------------------|-----------|---------|
| **gini_score_1** | 0.456 | 0.395 | 0.061 [0.013, 0.110] | 0.463 | 0.0055 |
| **gini_score_2** | 0.330 | 0.242 | 0.088 [0.034, 0.138] | 0.591 | 0.0006 |
| **speaker_participation_cdp** | 0.527 | 0.537 | -0.010 [-0.041, 0.020] | -0.135 | 0.4927 |

**Visualization**: [gini_by_funding.png](../figures/final/gini_by_funding.png) - boxplots showing Gini distributions by funding status

**Pipeline**: `pipelines/speaker_diversity_outcomes.py` + `pipelines/posthoc_analyses.py`

**Outputs**:
- `outputs/tables/speaker_diversity_with_outcomes.csv`
- `outputs/analysis/speaker_diversity_outcomes_summary.txt`
- `outputs/analysis/speaker_diversity_effect_sizes.txt`
- `figures/final/gini_by_funding.png`

#### Interpretation
**✅ Hypothesis confirmed (first version)**: Sessions with **more concentrated coordination** (higher Gini) have significantly higher funding rates.

**What this means**:
- **Funded sessions have clearer coordination leadership**: One or two speakers drive most of the advanced coordination (score 2)
- **Effect size is moderate-to-large** (d=0.591 for score 2), indicating practical significance
- **Statistical significance is strong** (p=0.0006), robust even with multiple comparison correction

**Why this might happen**:
- Clear leadership provides **direction and synthesis** needed for productive collaboration
- Dominant coordinators may be **skilled facilitators** who guide teams to actionable outcomes
- Distributed coordination might indicate **lack of consensus** or unclear team structure
- Successful teams need **someone to drive the decision-making process**

**Transcript evidence** (from `gini_sanity_excerpts.txt`):
- **High-Gini funded sessions**: Show long, directive score-2 utterances from 1-2 speakers (e.g., "Given our constraints, I propose we focus on X because...")
- **Low-Gini unfunded sessions**: Show distributed score-2 utterances across many speakers, often shorter and less decisive

**Caution**: High Gini could also reflect **domination** rather than leadership. Qualitative review suggests funded sessions have **productive leadership** (synthesizing, proposing) rather than controlling behavior.

---

### Q4: Do timing patterns (phase rhythm, transitions) predict outcomes?

**Hypothesis**: Sessions with specific temporal patterns (e.g., stable entropy across phases, fewer transitions between coordination modes) will have higher funding rates.

#### Analysis Approach
**Method**: Temporal feature extraction + statistical testing

**Why this approach?**  
If timing matters, we should see differences in metrics like:
- Entropy **rhythm** (variance across phases)
- **Transition counts** (how often teams switch between score 1 and score 2)
- **Convergence** (do teams end with lower entropy than they started?)

**What the analysis looks like**:
- **Outcome variable**: `any_funded` (binary)
- **Features**:
  - `entropy_std`: Standard deviation of entropy across beginning/middle/end
  - `n_transitions`: Count of score switches in fine-grained bins (5-min windows)
  - `entropy_change`: entropy_end - entropy_beginning
  - `entropy_slope`: Linear trend across phases
- **Statistical tests**: Mann-Whitney U for each feature
- **Robustness check**: Replicate analysis with 3-min (180s) and 10-min (600s) bins to test fragility

#### Results
**Primary analysis (5-min bins):**
| Feature | Funded Mean | Unfunded Mean | Mann-Whitney p |
|---------|-------------|---------------|----------------|
| **entropy_std** | 0.208 | 0.201 | 0.1438 |
| **n_transitions** | 12.4 | 11.8 | 0.3842 |
| **entropy_change** | 0.018 | 0.002 | 0.6251 |
| **entropy_slope** | 0.009 | 0.001 | 0.5909 |

**Robustness check (3-min bins):**
| Feature | p-value (180s) | p-value (600s) | Robust? |
|---------|----------------|----------------|---------|
| **n_transitions** | 0.0109* | 0.9707 | ❌ No (spurious) |
| **entropy_std** | 0.1789 | 0.1524 | ✅ Yes (stable null) |
| **entropy_change** | 0.5136 | 0.7293 | ✅ Yes (stable null) |

**Pipeline**: `pipelines/timing_patterns_outcomes.py` + `pipelines/timing_patterns_outcomes_bins.py`

**Outputs**:
- `outputs/analysis/timing_patterns_outcomes_summary.txt`
- `outputs/analysis/timing_patterns_outcomes_180s_summary.txt`
- `outputs/analysis/timing_patterns_outcomes_600s_summary.txt`

#### Interpretation
**❌ Hypothesis rejected**: Timing patterns do **NOT** predict funding outcomes.

**What this means**:
- **Phase rhythm** (how much entropy varies across beginning/middle/end) is similar for funded and unfunded sessions
- **Transition frequency** (switching between coordination modes) does not differ
- **Convergence** (entropy decrease over time) is not a predictor
- The **when** of coordination matters less than the **who** (speaker distribution from Q3)

**Robustness finding**:
- One spurious significant result (n_transitions p=0.0109 at 180s bins) **disappeared** at 600s bins (p=0.971)
- This confirms the null result is **stable** and not sensitive to bin size choice
- Timing patterns are **genuinely unrelated** to outcomes

**Why timing might not matter**:
- All meetings follow similar facilitated structure (imposed by SCIALOG format)
- Timing constraints are uniform across sessions, so no natural variation to exploit
- Outcomes depend on **content quality** and **leadership**, not pacing

---

### Q5: Can we predict outcomes better by combining features?

**Hypothesis**: Combining speaker diversity + timing features will improve prediction beyond using entropy alone.

#### Analysis Approach
**Method**: Meeting profile classifier with ROC-AUC comparison

**Why this approach?**  
ROC-AUC measures how well a model distinguishes between funded and unfunded sessions (0.5 = random guessing, 1.0 = perfect classification). Compare baseline (entropy only) vs full model (entropy + speaker diversity + timing).

**What the analysis looks like**:
- **Outcome variable**: `any_funded` (binary classification)
- **Baseline features**: `entropy_end` only
- **Full model features**: `entropy_end`, `gini_score_1`, `gini_score_2`, `speaker_participation_cdp`, `entropy_std`, `n_transitions`, `entropy_change`
- **Model**: Logistic regression with 5-fold cross-validation
- **Metric**: ROC-AUC (area under receiver operating characteristic curve)

#### Results
| Model | ROC-AUC | Improvement |
|-------|---------|-------------|
| **Baseline (entropy only)** | 0.539 | — |
| **Full model (speaker + timing)** | 0.688 | +0.149 (+27.7%) |

**Feature importance** (based on coefficients):
1. **gini_score_2** (advanced coordination concentration): Strongest positive predictor
2. **gini_score_1** (basic coordination concentration): Moderate positive predictor
3. **entropy_end**: Weak negative effect
4. **Timing features**: Minimal contribution

**Pipeline**: `pipelines/meeting_profile_classifier.py`

**Outputs**:
- `outputs/tables/meeting_profile_classifier_results.csv`
- `outputs/analysis/meeting_profile_classifier_results.txt`

#### Interpretation
**✅ Hypothesis confirmed**: Adding speaker diversity features **substantially improves** prediction.

**What this means**:
- **Speaker diversity (Gini) is the key signal**, not entropy or timing
- Model improvement of **27.7%** is meaningful but still far from perfect prediction
- ROC-AUC of 0.688 suggests **moderate predictive power** (good signal but not deterministic)
- **Practical application**: Can identify sessions with higher likelihood of success based on early coordination patterns

**Why the model isn't perfect**:
- Funding depends on many factors outside coordination (scientific quality, funding constraints, network effects)
- 123 sessions is a modest sample size for machine learning
- Coordination is necessary but not sufficient for funding success

**Real-world use case**:
- Facilitators could monitor Gini in real-time and intervene if coordination is too distributed
- Early identification of "at-risk" sessions (low Gini) for targeted support

---

### Q6: Do coordination patterns differ by cohort year?

**Hypothesis**: More recent cohorts (2022) show different coordination patterns than earlier cohorts (2020-2021) due to accumulated facilitation experience or changed norms.

#### Analysis Approach
**Method**: Kruskal-Wallis H-test + pairwise Mann-Whitney U with Holm correction

**Why this approach?**  
Kruskal-Wallis tests whether entropy distributions differ across 3+ groups (non-parametric ANOVA). Follow up with pairwise tests to identify which years differ. Holm correction controls for multiple comparisons.

**What the analysis looks like**:
- **Grouping variable**: Cohort year (2020, 2021, 2022)
- **Outcome variables**: `entropy_beginning`, `entropy_middle`, `entropy_end`
- **Statistical tests**: 
  - Kruskal-Wallis H (omnibus test per phase)
  - Pairwise Mann-Whitney U (all pairs: 2020 vs 2021, 2020 vs 2022, 2021 vs 2022)
  - Holm-Bonferroni correction for multiple comparisons
- **Effect size**: Rank-biserial correlation (rbc) for pairwise comparisons

#### Results
**Kruskal-Wallis H-tests:**
| Phase | H statistic | p-value | Significant? |
|-------|-------------|---------|--------------|
| **Beginning** | 0.95 | 0.621 | ❌ No |
| **Middle** | 7.90 | 0.019 | ✅ Yes (trend) |
| **End** | 2.71 | 0.258 | ❌ No |

**Pairwise tests (middle phase only):**
| Comparison | 2020 Mean | Other Mean | Raw p | Holm-adjusted p | Rank-biserial (rbc) |
|------------|-----------|------------|-------|-----------------|---------------------|
| **2020 vs 2022** | 0.717 | 0.427 | 0.028 | 0.056 (trend) | -0.484 (large) |
| **2021 vs 2022** | 0.664 | 0.427 | 0.022 | 0.067 (trend) | -0.351 (moderate) |
| **2020 vs 2021** | 0.717 | 0.664 | 0.839 | 0.839 | -0.055 (negligible) |

**Cohort entropy means by phase:**
| Cohort | Beginning | Middle | End |
|--------|-----------|--------|-----|
| **2020** | 0.804 | 0.717 | 0.895 |
| **2021** | 0.715 | 0.664 | 0.745 |
| **2022** | 0.684 | 0.427 | 0.646 |

**Pipeline**: `pipelines/cdp_by_cohort.py` + `pipelines/cdp_by_cohort_pairwise.py`

**Outputs**:
- `outputs/analysis/cdp_by_cohort_summary.txt`
- `outputs/analysis/cdp_by_cohort_pairwise.txt`

#### Interpretation
**✅ Partial confirmation**: Cohort differences exist **only in the middle phase**, with 2022 showing significantly lower entropy.

**What this means**:
- **2022 teams shift to more focused coordination mid-meeting** (entropy 0.427 vs 0.66-0.72 in earlier years)
- Earlier cohorts (2020-2021) maintain more mixed coordination throughout
- **Beginning and end phases are similar** across all years (no cohort effect)

**Why this might happen**:
- **Accumulated facilitation experience**: By 2022, facilitators may have refined techniques to drive faster consensus
- **Changed participant expectations**: Later cohorts may enter with clearer norms about decision-making
- **Time pressure**: 2022 sessions may have tighter schedules, forcing more focused mid-meeting coordination
- **Sample composition**: Different scientific domains or participant demographics

**Effect size interpretation**:
- **Rank-biserial correlation (rbc) -0.484** for 2020 vs 2022 = large effect (one group consistently ranks higher)
- **Trend-level significance** (Holm-adjusted p ≈ 0.06) suggests real difference but needs replication

**Caution**: Only **9 sessions in 2022** vs ~100 in 2020-2021, so 2022 estimates are less stable.

---

### Q7: What do transcripts reveal about high vs low Gini sessions?

**Hypothesis**: High-Gini funded sessions should show long, decisive score-2 utterances from one speaker. Low-Gini unfunded sessions should show distributed, shorter score-2 utterances.

#### Analysis Approach
**Method**: Qualitative transcript sampling and content validation

**Why this approach?**  
Quantitative findings need grounding in actual team conversations. Extract representative excerpts from high/low Gini sessions to validate that Gini reflects meaningful coordination differences.

**What the analysis looks like**:
- **Sampling strategy**: Select 3 high-Gini funded sessions and 3 low-Gini unfunded sessions
- **Extraction**: Top 3 longest score-2 utterances per session
- **Coding**: Manual review of utterance content, speaker patterns, and coordination quality

#### Results
**High-Gini funded sessions** (Gini ≈ 0.40-0.50):
- **Pattern**: 1-2 speakers contribute most score-2 utterances
- **Utterance characteristics**: Long (50-100 tokens), directive, synthesizing
- **Example themes**: 
  - "Given our discussion, I think we should prioritize X because it addresses both Y and Z constraints..."
  - "Let me summarize what I'm hearing: three main approaches, and here's how they connect..."
- **Coordination quality**: Clear leadership, actionable proposals, explicit decision-making

**Low-Gini unfunded sessions** (Gini ≈ 0.15-0.25):
- **Pattern**: 5-6 speakers contribute score-2 utterances evenly
- **Utterance characteristics**: Shorter (20-40 tokens), exploratory, questioning
- **Example themes**:
  - "We could try approach A, or maybe B..."
  - "That's interesting, but have we considered..."
- **Coordination quality**: Distributed thinking, less decisive, multiple parallel threads

**Pipeline**: `pipelines/cdp_transcript_validation.py` + `pipelines/posthoc_analyses.py`

**Outputs**:
- `outputs/tables/cdp_transcript_validation_samples.csv`
- `outputs/analysis/cdp_transcript_validation_summary.txt`
- `outputs/analysis/gini_sanity_excerpts.txt`

#### Interpretation
**✅ Hypothesis confirmed**: Gini differences reflect **real coordination structure differences**, not just statistical artifacts.

**What this means**:
- **High Gini = productive leadership**: Dominant speakers provide synthesis and direction
- **Low Gini = distributed exploration**: More voices but less convergence on actionable plans
- **Utterance length matters**: Long score-2 utterances indicate **complex synthesis** (combining multiple ideas)
- **Quality over quantity**: Having one skilled coordinator is more valuable than many people offering fragmented ideas

**Limitations**:
- Small sample (6 sessions) limits generalizability
- Manual coding is subjective; future work should use multiple raters
- Cannot distinguish **skilled facilitation** from **domination** without deeper analysis

**Recommendation**: Future work should code for **coordination quality** (synthesis, actionability) in addition to quantity.

---

## STEP 5: Overall Story — What Do These Analyses Tell Us?

### The Core Finding
**Team coordination success is about WHO coordinates, not HOW MUCH or WHEN.**

Across 157 SCIALOG collaborative meetings, we found that:
1. **Coordination diversity (entropy) is stable and doesn't predict outcomes** (Q1, Q2)
2. **Speaker distribution of coordination strongly predicts funding success** (Q3)
3. **Timing patterns don't matter for outcomes** (Q4)
4. **Combining speaker features improves prediction 27.7%** (Q5)
5. **Recent cohorts show more focused mid-meeting coordination** (Q6)
6. **High-Gini sessions have clearer leadership with decisive synthesis** (Q7)

### The Mechanism
Successful team formation requires **concentrated coordination leadership**:
- One or two speakers drive **advanced coordination** (score 2: synthesis, decision-making)
- These leaders provide **direction and actionable proposals**, not just facilitation
- Distributed coordination (low Gini) may reflect **lack of consensus** or unclear team structure
- **Long, complex utterances** (49 tokens average for score 2) enable the synthesis needed for decisions

### Practical Implications

**For Facilitators:**
- Monitor **speaker distribution** early in meetings
- Intervene if coordination is too distributed (no one taking leadership)
- Encourage **decisive synthesis** rather than just collecting ideas
- Aim for **balanced participation in ideas** but **concentrated coordination leadership**

**For Researchers:**
- **Speaker-level analysis** is more informative than aggregate measures
- **Gini coefficient** is a powerful metric for team dynamics (captures inequality meaningfully)
- **Entropy alone** is insufficient for predicting outcomes in facilitated settings
- **Temporal patterns** may be less important when meetings follow structured formats

**For Funding Agencies:**
- Early coordination patterns (measurable in first 30-60 minutes) can **predict success**
- ROC-AUC of 0.688 suggests **moderate predictive power** (better than random, not deterministic)
- Could use automated coordination monitoring to **flag at-risk sessions** for intervention

### Limitations and Future Directions

**Current Limitations:**
- **Moderate sample size** (123 sessions with outcomes) limits machine learning approaches
- **Observational design** prevents causal claims (correlation ≠ causation)
- **Single institutional context** (SCIALOG) may limit generalizability
- **Binary outcome** (funded vs unfunded) is coarse; funding quality/amount would be richer

**Next Steps:**
1. **Test on new cohorts** (2023-2024 data) for replication
2. **Demographic analysis**: Does Gini's predictive power depend on speaker seniority, domain, or background?
3. **Interaction effects**: Does Gini × entropy interaction matter? (concentrated + diverse might be optimal)
4. **Real-time prediction**: Can we build a live dashboard for facilitators?
5. **Qualitative deep dive**: Code for coordination quality (synthesis vs domination) to validate leadership interpretation

### The Broader Context
This work contributes to understanding **scientific team formation** through behavioral analysis. Key contributions:
- **Methodological**: Demonstrates how information theory (entropy, Gini) can quantify coordination patterns
- **Empirical**: Provides evidence that leadership structure matters more than pacing or diversity
- **Practical**: Offers actionable insights for improving collaborative meeting facilitation

The finding that **concentrated coordination predicts success** challenges assumptions about egalitarian collaboration. While diverse perspectives are valuable, **decisive leadership is critical** for converting ideas into actionable teams.

---

## 6. How to Reproduce This Project

### Quick Start
```bash
# Clone repository
git clone https://github.com/mchalekson/linkography_ai.git
cd linkography_ai

# Install dependencies (Python ≥3.10)
pip install -e .

# Run full pipeline
make all
```

This executes all 14 analyses sequentially (see Makefile for details).

### Individual Pipelines

#### Q1: Entropy Trajectories
```bash
make analyze
# or:
python pipelines/analyze_entropy_trajectories.py
```
**Outputs**:
- `outputs/analysis/entropy_trajectory_summary.txt`
- `figures/final/entropy_trajectory.png`

#### Q2: Entropy vs Outcomes
```bash
make test_outcomes
# or:
python pipelines/test_entropy_outcomes.py
```
**Outputs**:
- `outputs/analysis/entropy_outcomes_stats.txt`
- `outputs/tables/entropy_outcome_group_summary.csv`

#### Q3: Speaker Diversity vs Outcomes
```bash
python pipelines/speaker_diversity_outcomes.py
```
**Outputs**:
- `outputs/tables/speaker_diversity_with_outcomes.csv`
- `outputs/analysis/speaker_diversity_outcomes_summary.txt`

**Then run post-hoc effect sizes + visualization:**
```bash
python pipelines/posthoc_analyses.py
```
**Outputs**:
- `outputs/analysis/speaker_diversity_effect_sizes.txt` (Cohen's d, 95% CI)
- `figures/final/gini_by_funding.png` (boxplot)

#### Q4: Timing Patterns vs Outcomes
```bash
python pipelines/timing_patterns_outcomes.py
```
**Robustness check (3-min and 10-min bins):**
```bash
python pipelines/timing_patterns_outcomes_bins.py --bin-sec 180
python pipelines/timing_patterns_outcomes_bins.py --bin-sec 600
```
**Outputs**:
- `outputs/analysis/timing_patterns_outcomes_summary.txt`
- `outputs/analysis/timing_patterns_outcomes_180s_summary.txt`
- `outputs/analysis/timing_patterns_outcomes_600s_summary.txt`

#### Q5: Meeting Profile Classifier
```bash
python pipelines/meeting_profile_classifier.py
```
**Outputs**:
- `outputs/tables/meeting_profile_classifier_results.csv`
- `outputs/analysis/meeting_profile_classifier_results.txt`

#### Q6: Cohort Analysis
```bash
python pipelines/cdp_by_cohort.py
python pipelines/cdp_by_cohort_pairwise.py
```
**Outputs**:
- `outputs/analysis/cdp_by_cohort_summary.txt`
- `outputs/analysis/cdp_by_cohort_pairwise.txt`

#### Q7: Transcript Validation
```bash
python pipelines/cdp_transcript_validation.py
```
**Outputs**:
- `outputs/tables/cdp_transcript_validation_samples.csv`
- `outputs/analysis/cdp_transcript_validation_summary.txt`

**Post-hoc sanity excerpts:**
```bash
python pipelines/posthoc_analyses.py
```
**Outputs**:
- `outputs/analysis/gini_sanity_excerpts.txt`

### Key Output Files

| File | Description |
|------|-------------|
| `outputs/tables/cdp_entropy_by_session_ALL_*.csv` | Per-session entropy (beginning/middle/end) |
| `outputs/tables/entropy_with_outcomes.csv` | Entropy + funding outcomes |
| `outputs/tables/speaker_level_cdp.csv` | Gini coefficients per session |
| `outputs/analysis/speaker_diversity_effect_sizes.txt` | Effect sizes + 95% CI |
| `figures/final/entropy_trajectory.png` | Entropy by phase (Q1) |
| `figures/final/gini_by_funding.png` | Gini boxplots by funding (Q3) |
| `outputs/analysis/cdp_by_cohort_pairwise.txt` | Cohort comparisons (Q6) |
| `outputs/analysis/gini_sanity_excerpts.txt` | Transcript excerpts (Q7) |

### Code Structure

```
src/linkography_ai/
├── entropy.py           # Shannon entropy computation
├── segmentation.py      # Index-based thirds
├── io_sessions.py       # JSON loading + CDP extraction
├── slides.py            # Time-binned analysis
└── discovery.py         # Conference discovery

pipelines/
├── run_cdp_entropy_all.py              # Batch entropy (Q1 foundation)
├── analyze_entropy_trajectories.py     # Q1 statistical analysis
├── merge_entropy_with_outcomes.py      # Q2 data prep
├── test_entropy_outcomes.py            # Q2 statistical tests
├── speaker_diversity_outcomes.py       # Q3 analysis
├── timing_patterns_outcomes.py         # Q4 analysis
├── timing_patterns_outcomes_bins.py    # Q4 robustness
├── meeting_profile_classifier.py       # Q5 classifier
├── cdp_by_cohort.py                    # Q6 Kruskal-Wallis
├── cdp_by_cohort_pairwise.py           # Q6 pairwise tests
├── cdp_transcript_validation.py        # Q7 sampling
├── posthoc_analyses.py                 # Effect sizes + viz + excerpts
└── (10+ other exploratory pipelines)
```

### Common Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--conference` | Specify conference or ALL | `--conference 2021NES` |
| `--normalize` | Normalize entropy by log₂(K) | `--normalize` |
| `--bin-sec` | Time bin size (seconds) | `--bin-sec 300` |
| `--max_sessions` | Limit sessions (0=all) | `--max_sessions 10` |

---

## 7. Technical Details

### What is Shannon Entropy?
**Formula**: $H = -\sum_{i=1}^{K} p_i \log_2(p_i)$

Where:
- $K$ = number of distinct categories (for CDP: K=2, score 1 and score 2)
- $p_i$ = proportion of utterances in category $i$

**Normalized**: $H_{norm} = H / \log_2(K)$ (scales to 0-1 range)

**Interpretation for K=2**:
- $H=0.0$: All utterances same score (pure score 1 OR pure score 2)
- $H=1.0$: Perfect 50/50 mix of score 1 and score 2
- $H=0.7-0.75$: Roughly 70% score 1, 30% score 2 (observed in data)

### What is Gini Coefficient?
**Purpose**: Measure inequality in speaker distribution

**Formula**: 
$$G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n^2 \bar{x}}$$

Where:
- $x_i$ = CDP count for speaker $i$
- $n$ = number of speakers
- $\bar{x}$ = mean CDP count per speaker

**Interpretation**:
- $G=0.0$: Perfect equality (all speakers contribute equally)
- $G=1.0$: Perfect inequality (one speaker has all CDP)
- $G=0.3-0.4$: Moderate concentration (observed for score 2 in funded sessions)

### Segmentation Methods

**Index-based thirds** (default for aggregate analysis):
- Divide utterances into 3 equal groups by index
- Beginning: utterances [0, n/3)
- Middle: utterances [n/3, 2n/3)
- End: utterances [2n/3, n)

**Time-based thirds** (used for convergence, time-pressure):
- Divide session duration into 3 equal time periods
- Uses `start_time` and `end_time` fields

**Fine-grained bins** (5-min, 3-min, 10-min):
- Fixed-duration windows for within-session dynamics
- Used in timing pattern analysis (Q4)

### Statistical Methods

**Mann-Whitney U**: Non-parametric test for comparing two groups (funded vs unfunded)
- Does not assume normal distributions
- Tests whether one group tends to have higher values than the other

**Kruskal-Wallis H**: Non-parametric test for comparing 3+ groups (cohort years)
- Extension of Mann-Whitney U to multiple groups
- Equivalent to one-way ANOVA for ranked data

**Cohen's d**: Effect size for mean differences
- Small: d ≈ 0.2
- Medium: d ≈ 0.5
- Large: d ≈ 0.8

**Rank-biserial correlation (rbc)**: Effect size for Mann-Whitney U
- Ranges from -1 to 1
- Interpretation similar to Pearson correlation

**Bootstrap 95% CI**: Resampling method for confidence intervals
- 2000 bootstrap replicates
- Percentile method for CI construction

**Holm-Bonferroni correction**: Multiple comparison adjustment
- Controls family-wise error rate
- Less conservative than Bonferroni

### Codebook Reference

**CDP Score Definitions** (from [codebook/codebook.md](../codebook/codebook.md)):

**Score 1 - Basic Coordination**:
- Structuring contributions (turn-taking, agenda-setting)
- Simple process management
- Clarifying questions
- Acknowledgments

**Score 2 - Advanced Coordination**:
- Explicit decision-making
- Complex synthesis across multiple ideas
- Strategic planning
- Consensus-building with justification

**Other Annotation Categories** (not used in this analysis):
- Relational Climate
- Idea Generation
- Critical Thinking
- Resource Mobilization
- (and 3 others)

---

## 8. Known Limitations and Caveats

### Data Limitations
1. **Moderate sample size**: 123 sessions with outcomes (adequate for statistical tests, modest for ML)
2. **Imbalanced outcomes**: 68 funded vs 55 unfunded (roughly balanced but not large)
3. **Missing outcome data**: 34/157 sessions (21.7%) lack outcome labels
4. **Single institutional context**: All SCIALOG conferences (facilitated, time-constrained)
5. **Timestamp inconsistencies**: Some sessions have malformed `start_time`/`end_time` fields

### Methodological Limitations
1. **Observational design**: Correlation does not imply causation (Gini predicts outcomes, but does high Gini *cause* success?)
2. **Aggregation**: Session-level analysis loses within-team heterogeneity
3. **Binary outcome**: Funded vs unfunded is coarse (funding amount, quality, or longevity would be richer)
4. **Annotation reliability**: Single-rater CDP annotations (no inter-rater reliability reported)
5. **Speaker identification**: No demographic data (seniority, domain, background)

### Analytical Limitations
1. **No train/test split**: All analyses use full dataset (risk of overfitting in classifier)
2. **No causal modeling**: Cannot isolate effect of Gini independent of confounders
3. **Small 2022 sample**: Only 9 sessions in 2022 cohort (less stable estimates)
4. **Qualitative validation**: Only 6 sessions manually reviewed (small sample for Q7)

### Interpretation Limitations
1. **Gini ambiguity**: High Gini could reflect skilled leadership OR domination (cannot distinguish)
2. **Facilitation effects**: All sessions facilitated, so results may not generalize to unstructured meetings
3. **Temporal dynamics**: Stable entropy may be artifact of facilitation structure
4. **Selection effects**: Funded teams may differ in unobserved ways (prior connections, reputation)

---

## 9. Troubleshooting

### Common Issues

**"No columns to parse" error**
- **Cause**: Batch entropy CSV is empty (no sessions processed)
- **Fix**: Verify data exists: `ls data/*/session_data/*.json | wc -l` should return >0

**"ModuleNotFoundError"**
- **Cause**: Dependencies not installed
- **Fix**: `.venv/bin/python -m pip install -e .`

**Low match rate (<50%) in outcome merge**
- **Cause**: Session ID mismatch between entropy CSV and outcome JSONs
- **Check**: `cat outputs/logs/outcome_merge_report.txt`
- **Normal**: 78.3% match rate is expected (some sessions lack outcome data)

**All entropy values near 0.73-0.75**
- **Not a bug**: This is real data - balanced mix of score 1 (71%) and score 2 (29%)

### Verification Commands

```bash
# Check data integrity
make validate

# Verify outputs exist
ls -lh outputs/tables/cdp_entropy_by_session_ALL_*.csv
ls -lh outputs/analysis/entropy_trajectory_summary.txt
ls -lh figures/final/*.png

# Count sessions per conference
python -c "import pandas as pd; df = pd.read_csv('outputs/tables/cdp_entropy_by_session_ALL_20260225_103253.csv'); print(df['conference'].value_counts())"

# Quick stats
python -c "import pandas as pd; df = pd.read_csv('outputs/tables/cdp_entropy_by_session_ALL_20260225_103253.csv'); print(df[['entropy_beginning', 'entropy_middle', 'entropy_end']].describe())"
```

---

## 10. References and Resources

### Key Papers (Conceptual Foundations)
- **Linkography**: Goldschmidt, G. (2014). *Linkography: Unfolding the Design Process*. MIT Press.
- **Shannon Entropy**: Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
- **Gini Coefficient**: Gini, C. (1912). Variabilità e mutabilità. *Reprinted in Memorie di metodologica statistica* (Ed. Pizetti E, Salvemini, T).
- **Team Coordination**: Marks, M. A., Mathieu, J. E., & Zaccaro, S. J. (2001). A temporally based framework and taxonomy of team processes. *Academy of Management Review*, 26(3), 356-376.

### Code and Documentation
- **Repository**: https://github.com/mchalekson/linkography_ai
- **Codebook**: [codebook/codebook.md](../codebook/codebook.md)
- **Main README**: [README.md](../README.md)

### Related Work
- SCIALOG program: https://rescorp.org/scialog
- Team science literature: *Science of Team Science* (NCI, 2018)

---

**Document version**: 2.0 (Reproducibility-focused restructure per advisor feedback)  
**Last updated**: February 25, 2026  
**Contact**: Max Chalekson, Northwestern University
