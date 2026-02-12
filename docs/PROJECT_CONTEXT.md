# CDP Entropy Analysis Guide

## Overview

This repository explores Coordination & Decision Practices (CDP) annotations from SCIALOG scientific team meetings.

The current implementation focuses on understanding how coordination structure evolves over time using:

- Segment-level CDP distributions (beginning / middle / end)
- Shannon entropy as a measure of coordination diversity
- Visual analysis of entropy patterns across segments


This repository reflects the current exploratory state of the analysis.

---

## What the Notebook Currently Implements

### 1. Data Loading

- Loads CDP-coded behavioral data from `data/<conference>/`
- Organizes utterances by session and segment

Raw data is not modified.

---

### 2. Segment-Level CDP Distribution

- Computes frequency and proportion of CDP codes within:
  - Beginning
  - Middle
  - End

Purpose:
To observe how coordination behavior changes across meeting phases.

---

### 3. CDP Entropy (Primary Structural Signal)

- Computes Shannon entropy over CDP distributions per segment
- Optionally normalizes entropy

Interpretation:
- Higher entropy → broader mix of coordination behaviors
- Lower entropy → concentrated coordination mode

In the current analysis, entropy is used as a candidate indicator of:
- Coordination diversity
- Structural uncertainty
- Potential convergence toward decision closure

Entropy interpretation remains exploratory and requires further validation.

---

## Outputs

Derived artifacts are written to:

outputs/
  figures/
  tables/
  logs/

Typical outputs:
- CDP distribution plots
- Entropy-by-segment plots
- Session-level summary tables

---

## Current Scope

This repository currently implements:

- Distributional CDP analysis
- Segment-level entropy computation
- Visual inspection of structural trends

Future work may include:
- Transition modeling
- Graph-based structural metrics
- Statistical validation of entropy patterns

---

## Research Direction

The present analysis investigates:

- Whether entropy decreases toward the end of meetings
- Whether coordination diversity relates to decision closure
- Whether entropy can serve as a structural summary signal

This documentation reflects the current implementation and will evolve as the modeling framework becomes more formalized.