**PROJECT CONTEXT — Conceptual framing for CDP entropy analysis**

1. Purpose and scope
--------------------
This document provides the conceptual background for the repository's current implementation of Coordination & Decision Practices (CDP) analyses. It explains the theoretical motivation for using entropy as a structural signal, the precise operationalization used in the code, the assumptions that follow from those choices, and open methodological questions that a collaborator should consider before extending or interpreting results. It is intentionally focused on concepts and methods rather than installation or usage details.

2. Theoretical motivation
-------------------------
Group meetings instantiate a set of coordination behaviors: proposals, commitments, requests for clarification, scheduling, etc. From a systems perspective, the distribution of those behaviors across time encodes information about how a group is organizing work and converging (or not) on decisions. Shannon entropy provides a parsimonious, distribution-level summary of categorical diversity and unpredictability. In this repository, entropy is used as a descriptive signal to summarize the heterogeneity of CDP labels within bounded temporal windows of a meeting.

Key points about the motivation:
- Entropy quantifies variety and evenness in a categorical distribution; it does not by itself indicate causal direction, effectiveness, or normative quality of coordination.
- Using entropy as a structural signal is useful when one wants a low-dimensional summary of how mixed or concentrated coordination behaviors are at a given time.

3. What “coordination entropy” operationalizes here
--------------------------------------------------
In the codebase, coordination entropy is computed from per-utterance CDP labels as follows:

- Extraction: each utterance in a session yields a list of CDP labels (see [src/linkography_ai/io_sessions.py](../src/linkography_ai/io_sessions.py)). Multiple labels on the same utterance are treated as multiple counts.
- Aggregation: utterance-level labels are aggregated into a bin (either an index-based tertile or a fixed-duration time bin; see next section).
- Distribution: within each bin, counts per CDP category are converted to proportions $p_i$.
- **Entropy (Shannon):** Shannon entropy is computed as:

  $$H = -\sum_i p_i \log_2(p_i)$$

  where $p_i$ is the proportion of CDP category $i$ within the bin.

  An optional normalization divides $H$ by $\log_2(K)$, where $K$ is the number of observed (nonzero) categories, producing values bounded (approximately) in $[0,1]$ when $K > 1$. The implementation is in [src/linkography_ai/entropy.py](../src/linkography_ai/entropy.py).

Interpretive scope for this operationalization:
- High entropy indicates a broad mix of CDP categories in the bin (more behavioral variety / unpredictability in label identity).
- Low entropy indicates concentration in a small set of CDP categories (less behavioral variety in that window).
- Entropy is a descriptive indicator: it abstracts from semantics of individual codes and from speaker identity or turn-taking structure.

4. Time segmentation implementations in this repository
----------------------------------------------------
The repository contains two operational patterns for temporal aggregation, each implemented for different analysis needs:

- Index-based thirds (coarse structural wrap):
  - Implemented in [src/linkography_ai/segmentation.py](../src/linkography_ai/segmentation.py) and used by the batch pipeline [pipelines/run_cdp_entropy_all.py](../pipelines/run_cdp_entropy_all.py).
  - Behavior: given $n$ utterances in a session, utterance indices are partitioned into three contiguous bins by integer division: first $\lfloor n/3 \rfloor$ indices → `beginning`, next $\lfloor n/3 \rfloor$ indices → `middle`, remainder → `end`.
  - Rationale: simple, annotation-agnostic segmentation that avoids relying on timing metadata.

- Time-based bins and last-third windows (fine-grained, timeline-aware):
  - Implemented in [src/linkography_ai/slides.py](../src/linkography_ai/slides.py).
  - Behavior: session JSONs that contain `start_time` / `end_time` fields are parsed; times are converted to seconds and “unwrapped” if timers reset. Utterances are binned into fixed-duration windows (`bin_sec`, default 60s). There are options to restrict analysis to the last third of meeting duration (a meeting-relative time window) and to smooth per-bin series.
  - This module also implements a separate pipeline that pairs a coordination-minute signal (minutes of CDP-coded utterances per bin) with token-level entropy of free-text (optionally excluding structural wrap utterances) for plotting and local analyses.

Differences and coexistence:
- The batch pipeline (index thirds) is robust to missing or malformed timing metadata and supports dataset-level batch processing.
- The time-bin routines enable timeline-aware analyses (e.g., align signals to absolute minutes) but require reliable time annotations and additional preprocessing (unwrapping, sorting).

5. Assumptions implied by the operationalization
-------------------------------------------------
- Label validity: the approach assumes CDP labels attached to utterances reliably reflect discrete coordination behaviors. The code provides flexible extraction logic but does not validate label quality or inter-rater reliability.
- Independence within bins: entropy is computed from aggregated counts; the measure discards utterance ordering and temporal sequences within a bin.
- Equal weighting of counts: each CDP label occurrence contributes equally to the distribution (no speaker-weighting, no duration-weighting in the index-based pipeline). In the time-bin pipeline, ‘minutes of CDP’ is computed separately as a duration-based signal.
- Bin choice matters: index-based thirds assume uniform phase length in terms of number of utterances; fixed-duration bins assume timestamps accurately represent meeting progress. Both are coarse approximations of the latent “phases” of interaction.
- Multiple labels per utterance: when an utterance contains multiple CDP labels, the implementation counts each label. This treats multi-labeled turns as evidence of multiple simultaneous coordination behaviors rather than a single composite state.
- Normalization caveats: normalized entropy rescales $H$ by $\log_2(K)$; when $K$ is small (e.g., $K \leq 1$) normalization is degenerate and handled in code by returning 0.0 in that case.

6. What this repository currently does NOT attempt
-----------------------------------------------
- It does not validate CDP coding (no inter-rater reliability assessment or automated label adjudication).
- It does not implement causal inference or claim causal relationships between entropy and downstream outcomes.
- It does not model fine-grained temporal sequences (e.g., Markov or transition models) in the batch pipeline; ordering inside bins is ignored for entropy computations.
- It does not perform statistical hypothesis testing, correction for multiple comparisons, or effect-size estimation as part of automated runs.
- It does not disaggregate signals by speaker or role (speaker-aware analyses would require per-utterance speaker metadata and additional aggregation logic).
- It does not provide a single canonical “structure detector” — structural wrap is detected heuristically in `slides.py` via a regular expression; that detector is a convenience, not a validated instrument.

7. Open methodological questions and recommended next steps
---------------------------------------------------------
The following items are presented as open methodological questions or concrete next steps a collaborator might pursue before interpreting or extending analyses:

- Validation of code-level assumptions:
  - Assess inter-rater reliability of CDP labels and explore automated or semi-automated adjudication strategies.
  - Test sensitivity of entropy signals to the treatment of multi-label utterances (e.g., count once per utterance vs count each label).

- Sensitivity to temporal binning:
  - Compare index-based thirds, fixed-duration bins, and adaptive phase detection (e.g., changepoint detection) to see how bin choice affects entropy dynamics.
  - Evaluate how smoothing parameters and bin width ($\text{bin}_{\text{sec}}$) influence observed patterns.

- Alternative summary statistics and null models:
  - Compare Shannon entropy to alternative diversity measures (Simpson, Gini, richness) and to resampling-based null models that preserve marginal counts.
  - Develop null baselines that account for session length and label cardinality to better interpret effect sizes.

- Sequence-sensitive analyses:
  - Consider sequence or transition models (e.g., Markov chains, survival models for closure events) to capture ordering information that entropy omits.

- Linking signals to outcomes:
  - If outcome labels exist (session success/failure), pre-specify statistical models and control variables before exploratory regressions to avoid post-hoc interpretation.

- Structural wrap and semantics:
  - Validate the structural wrap detector against human judgments; consider a hybrid approach where wrap-up signals are annotated and used to train a lightweight classifier.

8. Implementation pointers (where to look in code)
------------------------------------------------
- CDP extraction and session IO: [src/linkography_ai/io_sessions.py](../src/linkography_ai/io_sessions.py)
- Index-based segmentation (thirds): [src/linkography_ai/segmentation.py](../src/linkography_ai/segmentation.py)
- Shannon entropy routine: [src/linkography_ai/entropy.py](../src/linkography_ai/entropy.py)
- Batch pipeline (index-based, dataset-level): [pipelines/run_cdp_entropy_all.py](../pipelines/run_cdp_entropy_all.py)
- Timeline-aware analyses, structural wrap heuristics, and plotting helpers: [src/linkography_ai/slides.py](../src/linkography_ai/slides.py)

9. Short guidance for collaborators
----------------------------------
- Treat entropy values as descriptive, hypothesis-generating signals rather than evidence of causality.
- Before interpreting cross-session differences, run sensitivity checks for binning, normalization, and label-handling rules described above.
- Document any changes to CDP extraction or binning choices and re-run the same preprocessing pipeline to maintain reproducibility.

This file is intended to communicate the repository's conceptual framing; it should be updated as the analytic framework and validation evidence evolve.