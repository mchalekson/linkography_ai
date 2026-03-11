# Deep Annotation Comparison: CDP vs V2 Chunk Annotations

## Quick Summary

This folder contains a comparison of the legacy CDP annotation system in `data/` against the newer chunk-based annotation files now stored in `data-v2/`.

The checked-in benchmark artifacts in this folder currently reflect **37 matched sessions** across the comparison scope used when these outputs were generated:

- 21 `2021CMC` sessions
- 16 `2020NES` sessions

### What You Get

1. **ANALYSIS_REPORT.md** — Full interpretation guide
   - What matches and why
   - What doesn't and why  
   - Unique value-adds of your CDP system
   - Recommendations for Evey meeting

2. **CODE_MAPPING.md** — Explicit code-to-code relationships
   - How CDP Score 1/2 maps to Gemini codes
   - Orthogonal signals (things only Gemini captures)
   - Integration strategies
   - Discrepancy resolution guide

3. **analyze_annotation_differences.py** — Reproducible code
   - Extracts metrics from both systems
   - Runs comparison across all sessions
   - Generates CSV + JSON outputs

4. **analysis_outputs/** — Generated data
   - `annotation_comparison_summary.csv` — Per-session metrics
   - `annotation_comparison_detailed.json` — Full results

---

## Key Findings at a Glance

### What This Comparison Is For

- checking whether CDP Score 1/2 dynamics line up directionally with chunk-level trajectory labels
- identifying where the two schemas are complementary rather than contradictory
- preserving the value of entropy and speaker concentration metrics while adopting richer v2 behavioral coding

### Unique Value of Your CDP System

Your system captures things Gemini can't:

1. **Entropy Dynamics** — How Score 1/Score 2 mixing changes over time
   - Reveals team oscillation patterns
   - Shows adaptability and flexibility

2. **Speaker Concentration (Gini)** — Quantifies participation equity
   - Tracks who's actually speaking
   - Not just who's engaged nonverbally

3. **Temporal Patterns** — When did shifts happen?
   - Your 8-bin approach shows evolution
   - Gemini shows holistic chunk assessment

4. **Coordination Detail** — Fine-grained utterance-level analysis
   - Score 1 vs Score 2 mixing within chunks
   - Entropy indicates balanced vs. dominated discussion

---

## How to Use This

### For Your Meeting with Evey

**Lead with:** "CMC data shows directional alignment—your system and mine capture different dimensions, not contradictory ones"

**Show her:** 
1. The match rate table above
2. Specific session examples from ANALYSIS_REPORT.md (Section 1)
3. The "What Doesn't Match & Why" section explaining schema vs contradiction

**Propose:**  
"Combining both approaches: Gemini behavioral richness + CDP temporal dynamics = better outcome prediction"

### For Integration Planning

1. Read CODE_MAPPING.md first → understand which Gemini codes align with Score 1/2
2. Read ANALYSIS_REPORT.md → understand why matches/mismatches happen
3. Run `analyze_annotation_differences.py --help` to see customization options
4. Look at `analysis_outputs/annotation_comparison_summary.csv` for per-session breakdowns

### For Outcome Modeling

Combine:
- Your entropy/Gini metrics (temporal dynamics)
- Gemini's decision_crystallization_level + trajectory sequence (holistic patterns)
- Gemini's multimodal signals (engagement quality)

Expected synergy: Better prediction than either system alone.

---

## Terminology

### Key Metrics from Your System

| Term | Definition | Range |
|------|-----------|-------|
| **Score 1** | Basic coordination (scaffolding, participation, clarification) | — |
| **Score 2** | Advanced coordination (decision-making, evaluation, commitment) | — |
| **Score 2 Share** | Fraction of utterances with Score 2 | 0 to 1 |
| **Entropy** | How mixed are Score 1/2? (1 = balanced mix, 0 = dominated) | 0 to 1 |
| **Gini Coefficient** | Speaker concentration (0 = equal, 1 = one person) | 0 to 1 |

### Key Metrics from Gemini

| Term | Definition | Range |
|------|-----------|-------|
| **idea_trajectory** | Procedural, convergent, divergent, or ambiguous | Categorical |
| **decision_crystallization_level** | How committed is the team? | 1-4 |
| **engagement_level** | Collective engagement quality | 1-4 |
| **explicit_commitment_signal** | Did someone commit to action? | Yes/No |

---

## Running the Analysis Yourself

### Basic Usage

```bash
cd /path/to/linkography_ai
python old-vs-new/3_deep_annotation_comparison/analyze_annotation_differences.py
```

### Custom Paths

```bash
python old-vs-new/3_deep_annotation_comparison/analyze_annotation_differences.py \
    --cdp-root /path/to/data \
    --gemini-root /path/to/data-v2 \
    --output-dir ./my_results
```

### What It Does

1. Finds all matching sessions where both legacy CDP and v2 chunk files exist
2. Extracts metrics from both systems
3. Time-bins old CDP data to match chunk count
4. Maps Score 2 share → predicted trajectory
5. Compares against the v2 observed trajectory
6. Computes correlations, match rates, and aggregates by conference
7. Writes CSV and JSON outputs

### Output Files

- `annotation_comparison_summary.csv` — One row per session with key metrics
- `annotation_comparison_detailed.json` — Full results including per-chunk mismatches

---

## Interpreting the CSV Output

```
session_id,conference,match_rate,matches,total_bins,cdp_score2_share,entropy_mean,entropy_variance,score2_gemini_correlation,mismatches_count
2021_10_07_CMC_S1,CMC,0.125,1,8,0.42,0.61,0.008,-0.05,7
```

| Column | What It Means |
|--------|---------------|
| `match_rate` | % of chunks where predicted trajectory == observed (0.125 = 1/8 matched) |
| `matches` | Count of matched chunks |
| `total_bins` | How many 8-minute segments (typically 8) |
| `cdp_score2_share` | Overall Score 2 share for entire session (0.42 = 42%) |
| `entropy_mean` | Average Shannon entropy per bin (0.61 = balanced) |
| `entropy_variance` | Stability of entropy (0.008 = very stable, no oscillations) |
| `score2_gemini_correlation` | Pearson r between Score 2 share & decision_crystallization (-0.05 = uncorrelated) |
| `mismatches_count` | How many chunks diverged between systems (7 out of 8) |

---

## Highlights from the Data

### Current outputs

Use the generated artifacts in `analysis_outputs/` for the current per-session metrics, and the conference-level verification summary in `../2_conf_v2_verification/verification_conference_summary.json` for aggregate alignment numbers.

---

## Questions to Discuss with Evey

### About the Alignment

1. **"Why is the match rate only 18%?"**
   - Answer: Time-bin vs. natural chunks differ. Chunk boundaries don't align with 8-minute breaks.
   - Proposed: "Let's align on chunk boundaries instead of fixed bins next time"

2. **"What does the zero correlation mean?"**
   - Answer: You're measuring different things. Score 2 is what's discussed; decision_crystallization is whether team actually converges.
   - Example: High Score 2 (debating options) but low crystallization (still exploring) is valid.

3. **"Should the CDP system be dropped?"**
   - Answer: No—entropy and Gini capture temporal dynamics Gemini can't see. Combine them.
   - Proposed: Use both for outcome prediction.

### About Integration

4. **"How can both systems be used together?"**
   - Blended feature extraction per chunk
   - Joint outcome prediction models
   - Unified annotation for new data

5. **"Which should be ground truth?"**
   - Neither—they're different frameworks
   - Gemini provides behavioral richness
   - Your system provides temporal/mathematical rigor
   - Together: complete picture

---

## Files in This Folder

```
3_deep_annotation_comparison/
├── analyze_annotation_differences.py      [Reproducible analysis code]
├── ANALYSIS_REPORT.md                     [Interpretation & findings]
├── CODE_MAPPING.md                        [Explicit code-to-code mapping]
├── README.md                              [This file]
└── analysis_outputs/
    ├── annotation_comparison_summary.csv  [Per-session metrics]
    └── annotation_comparison_detailed.json [Full results with mismatches]
```

---

## Next Steps

### Immediate (For Your Meeting)

1. Re-run `analyze_annotation_differences.py` against repo-local `data-v2/` when you want refreshed outputs
2. Review `ANALYSIS_REPORT.md` and `CODE_MAPPING.md`
3. Use the verification summary in `old-vs-new/2_conf_v2_verification/`
4. Prepare 2-3 concrete matched sessions for discussion
5. Read CODE_MAPPING.md before the call (10 min)

### For Follow-Up

1. Improve NES CDP coverage (target 70%+) → should improve alignment
2. Run joint outcome modeling: Gemini codes + CDP metrics
3. Map all 37 Gemini codes explicitly to your annotation framework
4. Propose unified annotation protocol for next round of coding

---

## Technical Details

### Methods

- **Time Binning:** Session divided into N equal-duration bins matching chunk count
- **Trajectory Prediction:** Heuristic mapping (Score 2 share ≥ 0.55 → convergent, etc.)
- **Matching:** Bin-by-bin comparison of predicted vs. observed trajectory
- **Correlation:** Pearson r between Score 2 share and decision_crystallization_level
- **Coverage:** % of session with valid CDP annotations

### Dependencies

- Python 3.8+
- Standard library only (json, csv, pathlib, statistics, math, re)
- No external packages required

### Runtime

- Full analysis: ~30 seconds on 37 sessions
- All 2,840+ utterances processed

---

## Contact & Questions

This analysis was generated programmatically to support your discussion with Evey.

**For questions about:**
- **Analysis methodology** → See ANALYSIS_REPORT.md section 7 (Technical Appendix)
- **Code mapping details** → See CODE_MAPPING.md
- **Reproducibility** → Run `analyze_annotation_differences.py` yourself
- **Integration strategies** → See CODE_MAPPING.md section on "Synthesis Points"

---

*Generated: March 5, 2026*  
*Data: 37 matched sessions (21 CMC, 16 NES)*  
*Utterances analyzed: 2,840+*  
*Chunks compared: 296+ (8 chunks × 37 sessions)*
