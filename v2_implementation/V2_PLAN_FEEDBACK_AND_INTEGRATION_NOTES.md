# V2 Plan: How Current Repo Aligns

**Date**: March 5, 2026

---

## Overview

The v2 plan has two phases that work together:

**Measurement Phase (Stages 0–3)**: Build and validate behavioral annotations from video/audio/transcript using multimodal AI + human expert validation. Only codes passing human-AI reliability (kappa ≥ 0.60) move forward.

**Modeling Phase (Stages 4–8)**: Use validated codes to build predictive ML models for team outcomes. This is where the current ML work (`outcome_modeling.py`, `speaker_diversity_outcomes.py`) can directly contribute.

---

## How Current Work Aligns

The repo is already answering a core v2 question: **When scientists converge on ideas and who drives that convergence predicts team success.** The findings map directly to v2's structural codes:

**Current finding**: Speaker concentration in advanced coordination (high Gini for Score 2) predicts 27.7% better funding outcomes.
- **V2 parallel**: `decision_crystallization_level` tracks when shared direction crystallizes + `Idea Management`/`Integration Practices` codes track who drives decision-making
- **Alignment**: Gini analysis of Score 2 speakers = v2's speaker concentration analysis of who drives `decision_crystallization` and synthesis

**Current finding**: Teams maintain stable convergence (entropy ~0.73-0.75) rather than progressing linearly; they loop between structuring and deciding.
- **V2 parallel**: `idea_trajectory` (divergent ↔ convergent) doesn't move one direction; meetings cycle between exploring and committing
- **Alignment**: Entropy dynamics = v2's tracking of when teams shift between `divergent` and `convergent` modes

**Current finding**: Fine-grained timing shows dynamic mode-switching (5-min entropy ranges 0.0–1.0).
- **V2 parallel**: `chunk_position` (beginning/middle/end) + temporal analysis of whether early convergence predicts better than late
- **Alignment**: Inflection-point analysis = v2's RQ4 (does trajectory predict outcomes beyond snapshots?)

The key difference isn't concept—it's **validation and granularity**. V2 uses AI + human validation to annotate these patterns using 37 rich codes (instead of the 2-level CDP), and adds multimodal signals (nods, shared affect, vocal enthusiasm) that amplify what's already visible in coordination structure.

---

## Alignment by Research Question

**RQ1 (Thin-slice)**: Do early-meeting behavioral signals predict as well as full-session signals?
- Current work: entropy analysis shows stable patterns across beginning/middle/end; proves early behavior carries signal
- V2 maps to this via: decision_crystallization and commitment signals in beginning vs. full session

**RQ3 (Predictive generalization)**: Do patterns hold across different conferences (LOCO validation)?
- Current work: discovered cohort effects (2022 teams show tighter mid-meeting entropy); suggests conference-level variation is real
- V2 maps to this via: LOCO testing that v2's codes generalize across the 8 Scialog conferences

**RQ4 (Trajectory effects)**: Does how direction/commitment evolves predict better than final snapshot?
- Current work: this is the core finding—concentration and trajectory matter; speaker Gini delta predicts better than static Gini
- V2 maps to this via: decision_crystallization_slope, problem_specificity_delta, engagement_trajectory across beginning/middle/end

**Current codec structure**: Score 1 (basic coordination) and Score 2 (advanced coordination) split directly maps to v2's behavioral dimensions:
- Score 1 = scaffolding, structure, turn-taking → maps to v2's `Participation Dynamics`, `Coordination and Decision Practices`
- Score 2 = synthesis, decision-making → maps to v2's `Integration Practices`, `Evaluation Practices`, `Idea Management`

---

## What Current Repo Provides vs. What V2 Adds

| Component | Current Repo | V2 Adds | Integration |
|---|---|---|---|
| **Concept** | Speaker concentration in Score 2 predicts outcomes | Same concept, validated via 37 rich codes + multimodal signals | Direct: apply Gini methodology to v2's speaker distributions |
| **Annotation** | Hand-coded CDP (2 levels) | AI-annotated + human validation (21 chunk + 16 utterance codes) | Current findings are proof-of-concept; v2 validates with stricter reliability |
| **Temporal analysis** | Entropy trajectories (beginning/middle/end) | Decision crystallization, problem specificity, engagement trajectories | Method (compute slope, test predictiveness) transfers directly |
| **Signals** | Transcript only | Transcript + video (nods, shared affect, vocal enthusiasm, backchannel) | Multimodal amplifies existing signals; test if video adds beyond transcript |
| **Validation** | No external validation; uses codes as-is | Human-AI agreement (kappa ≥ 0.60 gating) | V2 ensures codes are reliable; methodology works on any validated codes |
| **Modeling** | Logistic regression + ensemble, random CV | Same models + LOSO/LOCO (stricter CV) | Modeling framework can be adopted; enforce v2's stricter validation approach |

---

## Key Changes to Scripts

1. **Feature source**: Gini and entropy analysis works directly on v2's codes. After Stage 3 validation:
   - Compute Gini per speaker for codes that passed validation (e.g., Gini of `Integration Practices` speakers)
   - Compute entropy of `Idea Management` score diversity (similar to Score 1 vs Score 2 split)
   - Apply slope analysis to `decision_crystallization_level` and `problem_specificity_level` ratings

2. **Validation approach**: Adopt LOSO (157 folds, one session held out) + LOCO (8 folds, one conference held out) instead of random splits
   - speaker_diversity_outcomes.py can stay mostly intact; just change the CV strategy
   - This matches the intuition that cohort effects matter (already discovered in current data)

3. **Trajectory metrics**: Compute on v2's validated codes:
   - decision_crystallization_slope = (final_chunk_rating - first_chunk_rating) / num_chunks
   - problem_specificity_slope = same pattern
   - speaker_concentration_trajectory = (Gini_end - Gini_beginning) for behavioral codes
   - These replace entropy_slope; same methodology, richer input

4. **Multimodal feature layer**: Add engagement aggregates from Stage 4:
   - nod_rate per chunk, shared_affect presence, backchannel vocalizations
   - Test whether video signals amplify what's already visible in speaker concentration
   - This is additive; transcript-based analysis still runs in parallel

5. **Conference clustering**: Models already account for conference effects; formalize as mixed-effects with (1 | conference) random intercept to match v2's requirement

---

## Constraints & Opportunities

- **Stage 3 dependency**: Cannot start full modeling until Stage 3 releases validated codes and kappa table (shows which codes passed reliability threshold)
- **Code attrition risk**: If 2–3 codes fail validation, some features disappear; pipeline must handle missing features gracefully
- **Better AUC expected**: V2's richer annotations + multimodal signals should improve on current 0.688 ROC
- **Methodology is sound**: Current findings (concentration matters, trajectory matters, cohort variation exists) will likely replicate with v2's codes—this validates both approaches
- **Opportunity**: Test whether hand-coded findings transfer to AI-validated codes; if they do, it strengthens the case for both approaches

---

## Bottom Line

The core insight—**who drives decision-making and how that crystallizes over time predicts team success**—is exactly what v2's measurement pipeline is built to validate at scale with human reliability checks and multimodal depth. 

The Gini, entropy, and trajectory methodology transfers directly. Once v2 completes Stages 0–3, apply the proven analytical framework to v2's richer annotation set. Same questions, better evidence.

