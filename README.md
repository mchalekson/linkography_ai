linkography_ai — SCIALOG CDP analysis

Purpose
-------
This repository implements reproducible analyses of Coordination & Decision Practices (CDP) in SCIALOG meeting transcripts. The code extracts CDP codes from per-utterance annotations, aggregates them into time bins, and computes information-theoretic signals (Shannon entropy) and simple count-based statistics per session and per time-bin.

This README documents reproducibility steps, repository layout, the signal definitions used in analyses, and how to run the batch pipeline that produces session-level CSV outputs.

Repository layout
-----------------
- **`src/linkography_ai`**: core code implementing IO, segmentation, and signal computations. See [src/linkography_ai](src/linkography_ai).
- **`pipelines/make_slide7_run_cdp_entropy_all.py`**: batch runner that iterates datasets, computes per-session CDP counts and entropies, and writes tables and logs to `outputs/`. See [pipelines/make_slide7_run_cdp_entropy_all.py](pipelines/make_slide7_run_cdp_entropy_all.py).
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
# then install runtime libs, e.g.:
pip install pandas
```

Running the batch pipeline
--------------------------
The pipeline discovered in `pipelines/make_slide7_run_cdp_entropy_all.py` provides these CLI flags:

- `--conference` : conference id (e.g., `2021MZT`) or `ALL` (default)
- `--normalize`  : compute normalized Shannon entropy (divide by log2(K))
- `--max_sessions`: limit sessions processed per conference (0 = all)

Example

```bash
python pipelines/make_slide7_run_cdp_entropy_all.py --conference 2021MZT --normalize --max_sessions 0
```

Outputs
- Tables: `outputs/tables/cdp_entropy_by_session_<CONFERENCE>_<TIMESTAMP>.csv`
- Logs:   `outputs/logs/run_cdp_entropy_<CONFERENCE>_<TIMESTAMP>.txt`

The output table contains these columns (produced per session):
- `conference`, `session_id`, `n_utterances`, `outcome`
- For each segment (`beginning`, `middle`, `end`):
	- `entropy_<segment>`: Shannon entropy for CDP categories in that segment
	- `n_cdp_<segment>`: total number of CDP annotations counted in that segment
	- `n_unique_cdp_<segment>`: number of unique CDP categories observed in that segment

Data and expected JSON structure
--------------------------------
Session JSON files under `data/<CONFERENCE>/session_data/` can be either:
- an object with an `utterances` list, or
- a plain list of utterance objects.

Each utterance object should provide one of the text fields: `transcript`, `text`, or `utterance`. CDP annotations may appear in several forms; the IO routines look for:

- top-level lists named `cdp`, `CDP`, or `coordination_decision_practices`
- nested annotation dicts under `annotation_dict` / `annotations` with keys like `Coordination and Decision Practices`, `CDP`, or `cdp`

Files and functions that implement these behaviors are in `src/linkography_ai/io_sessions.py` (see `_extract_cdp_from_utterance_dict` and `load_session_utterances`).

Signal definitions (technical)
----------------------------
- Coordination & Decision Practices (CDP):
	- CDP are categorical labels attached to individual utterances. The code extracts CDP labels as a list of strings per utterance and treats multiple labels per utterance as multiple counts.

- Time-binned aggregation (structural wrap):
	- Sessions are time-binned using a simple thirds segmentation implemented in `segment_thirds(n)`. Each utterance is assigned to one of `beginning`, `middle`, or `end` according to its index within the session; this is the repository's operationalization of structural wrap/time-bin.

- Entropy (Shannon):
	- For each time-bin, the pipeline counts occurrences of each CDP category and computes Shannon entropy: H = -sum(p_i log2 p_i).
	- The implementation is `shannon_entropy_from_counts(counts, normalize=False)` in `src/linkography_ai/entropy.py`.
	- The `--normalize` flag divides H by log2(K) where K is the number of nonzero categories, yielding a value in [0,1] when K>1.

Notes and best practices for analysis
------------------------------------
- Inspect raw session JSONs before running large batch jobs to confirm the CDP field naming conventions used in your dataset.
- When comparing entropy across sessions with different numbers of observed categories, prefer `--normalize` to reduce scale effects.
- The simple thirds segmentation is intentionally coarse; for finer temporal analysis replace `segment_thirds` with a custom binning function.

Where to look in the codebase
----------------------------
- IO and CDP extraction: [src/linkography_ai/io_sessions.py](src/linkography_ai/io_sessions.py)
- Segmentation (thirds): [src/linkography_ai/segmentation.py](src/linkography_ai/segmentation.py)
- Entropy implementation: [src/linkography_ai/entropy.py](src/linkography_ai/entropy.py)
- Batch runner: [pipelines/make_slide7_run_cdp_entropy_all.py](pipelines/make_slide7_run_cdp_entropy_all.py)
