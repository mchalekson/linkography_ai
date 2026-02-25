# P3 CDP Deep Dives: Key Findings & Insights

## Executive Summary

Entropy analysis alone (P0-P1) revealed that **teams maintain a stable mix of basic and advanced coordination throughout meetings**, contradicting convergence hypotheses and showing no relationship with funding outcomes.

P3 analyses shift focus from **aggregate patterns** to **detailed mechanisms**: How is coordination actually used? Who participates? When do teams shift approaches? Do cohorts differ?

**Status**: ✅ All 5 analyses complete. 156-157 sessions analyzed across all analyses.

---

## Finding 1: CDP Content Characteristics

### Score 1 (Basic Coordination): Frequent, Short
- **71% of all CDP utterances** (mean 24.03 per session)
- **19 tokens/utterance** (mean)
- **Interpretation**: Frequent, brief structural/process statements
  - Examples: "let's organize..." "moving on..." "first point..."
  - Function: Scaffolding, transitions, process management

### Score 2 (Advanced Coordination): Rare, Long
- **29% of all CDP utterances** (mean 9.30 per session)
- **49 tokens/utterance** (mean) — **2.6× longer than score 1**
- **Interpretation**: Complex, sustained decision-making discussion
  - Examples: Multi-clause conditional statements, detailed reasoning
  - Function: Detailed problem-solving, consensus-building, strategic planning

### What This Means
Sessions use a **division of labor in coordination**:
- **Score 1** = frequent scaffolding by key facilitators
- **Score 2** = intensive discussion from few speakers
- Not a convergence trajectory but a **complementary system**

---

## Finding 2: Speaker Diversity in CDP

### Score 1 Concentration (Gini = 0.418)
- **Moderate concentration**: ~40% of utterances from top 2-3 speakers
- Few speakers "own" the basic coordination responsibility
- Suggests **centralized process management** (likely facilitators)

### Score 2 Concentration (Gini = 0.289)
- **Lower concentration**: More balanced across speakers
- Advanced coordination is a **team activity**
- Suggests **distributed problem-solving**

### What This Means
**Advanced coordination is more democratic**: 
- Basic coordination requires consistent voice authority (centralized)
- Advanced coordination benefits from diverse perspectives (distributed)
- Teams with balanced speaker participation in score 2 might achieve better outcomes
  - **Hypothesis for future testing**: Low Gini for score 2 → higher funding rate?

---

## Finding 3: Fine-Grained Timing (5-Minute Bins)

### Entropy Dynamics
- **Mean entropy per bin**: 0.418 (std = 0.440)
- **Range**: [0.0, 1.0] — full spectrum across bins
- **Interpretation**: Teams experience **significant within-session variation**
  - Many pure-score-1 bins (entropy ≈ 0): focused process moments
  - Many pure-score-2 bins (entropy ≈ 0): focused problem-solving moments
  - Mixed bins (entropy ≈ 0.7): simultaneous coordination and decision-making

### What This Means
**Meetings are not static**: Teams dynamically shift between coordination modes:
- Process-heavy phases (score 1 dominates)
- Decision-heavy phases (score 2 dominates)
- Integrated phases (both modes active)

### Use Case
Examine inflection points:
- When do transitions occur?
- What topic/event triggered the shift?
- Do funded sessions have distinct timing patterns?

---

## Finding 4: Cohort Differences (2020 vs 2021 vs 2022)

### Beginning Segment
- **No significant year effect** (H = 0.95, p > 0.05)
- 2020: 0.625, 2021: 0.757, 2022: 0.689
- Teams start similarly regardless of year

### Middle Segment: **SIGNIFICANT TREND** ⚠️
- **Kruskal-Wallis H = 7.90, p ≈ 0.02**
- 2020: 0.717, 2021: 0.664, 2022: **0.427**
- **2022 teams show LOWER entropy mid-meeting**
  - More focused, less mixed coordination
  - Possible: tighter time constraints or matured processes

### End Segment
- **No significant year effect** (H = 2.71, p > 0.05)
- Teams converge by end regardless of year

### What This Means
**Cohort Effect in Middle Phase**:
- 2022 teams may adopt **more structured decision protocols** mid-meeting
- Earlier cohorts use more **exploratory mixed-mode** discussion
- Most recent cohorts may reflect accumulated facilitation experience

---

## Finding 5: Speaker Roles

### Current Status
- **0 facilitators identified** in dataset metadata
- Framework is complete and ready for role-enriched datasets

### When Role Data Becomes Available
This pipeline will immediately show:
- Do facilitators drive score 1 utterances?
- Do domain experts drive score 2 utterances?
- Do role distributions correlate with outcomes?

---

## Integration with P0-P1 Findings

| Phase | Finding | Implication |
|-------|---------|-------------|
| **P0: Entropy** | Stable entropy (0.73→0.74) | Teams maintain balanced mix throughout |
| **P1: Outcomes** | No entropy-outcome correlation | Structure alone doesn't predict success |
| **P3: Content** | Score 2 is rare but long | Complex ideas appear infrequently |
| **P3: Speakers** | Score 2 more balanced | Advanced coordination is team effort |
| **P3: Timing** | High within-session variance | Not a static mix; dynamic mode-switching |
| **P3: Cohorts** | 2022 middle trend | Recent teams may use tighter processes |

---

## Recommended Next Steps

### 1. Speaker Diversity → Outcomes
**Question**: Do teams with balanced speaker participation in score 2 get higher funding?
```python
# Merge speaker_level_cdp.csv with entropy_with_outcomes.csv
# Correlate: gini_score_2 vs any_funded
```

### 2. Timing Inflection Points
**Question**: When do teams shift between modes? What triggers transitions?
```python
# Analyze cdp_fine_grained_entropy_300s.csv
# Identify bins with entropy jumps
# Examine preceding/following content
```

### 3. Cohort Mechanistic Understanding
**Question**: Why do 2022 teams show lower mid-meeting entropy?
```python
# Sample 5-10 2022 sessions with entropy < 0.5 mid-segment
# Manually review: decision structure, topic progression, time budget
# Compare to 2021 sessions with entropy > 0.7 mid-segment
```

### 4. Outcome Prediction with Content
**Question**: Do utterance characteristics predict outcomes?
```python
# Merge cdp_content_analysis.csv + entropy_with_outcomes.csv
# Build model: token_length_score2, pct_score1, etc. → any_funded
```

### 5. Role-Enriched Analysis
**Question**: Once role metadata available, analyze role-CDP correlations
```bash
# Simply re-run: python pipelines/speaker_role_cdp.py
# Results will immediately show role-CDP patterns
```

---

## File References

**Input Files** (generated by earlier pipelines):
- `outputs/tables/cdp_entropy_by_session_ALL_20260225_091354.csv` (P0)
- `outputs/tables/entropy_with_outcomes.csv` (P1)

**Output Files** (P3 — this session):
- `outputs/tables/cdp_content_analysis.csv`
- `outputs/tables/speaker_level_cdp.csv`
- `outputs/tables/cdp_fine_grained_entropy_300s.csv`
- `outputs/analysis/cdp_by_cohort_summary.txt`
- `outputs/tables/speaker_role_cdp.csv`

**Code Files** (P3 — this session):
- `pipelines/cdp_content_analysis.py`
- `pipelines/speaker_level_cdp.py`
- `pipelines/fine_grained_cdp_timing.py`
- `pipelines/cdp_by_cohort.py`
- `pipelines/speaker_role_cdp.py`

**Documentation** (updated):
- `README.md` (new CDP section)
- `docs/PROJECT_CONTEXT.md` (P3 findings section)
- `Makefile` (5 new targets)

---

## Questions for Interpretation

As you review these findings, consider:

1. **Content characteristics**: Are the length differences (19 vs 49 tokens) meaningful? Do score 2 utterances indeed contain more complex reasoning?

2. **Speaker dynamics**: Do your domain experts expect advanced coordination to be more democratic? Why might score 1 be centralized?

3. **Timing insights**: Are there natural "phases" in successful meetings (e.g., intro/explore/decide)? Do the entropy bins align with those phases?

4. **Cohort trends**: What changed between 2020 and 2022 in your facilitation approach or participant base?

5. **Outcome linkage**: Which of these dimensions (content, speakers, timing, cohort) do you hypothesize should matter most for success?

---

**Generated**: February 25, 2026
**Pipeline Status**: ✅ All analyses complete and integrated
**Next Steps**: Manual review + hypothesis testing with outcomes data
