# 2_conf_v2_verification

Cross-check legacy CDP session JSONs in `data/` against repo-local v2 chunk JSONs in `data-v2/`.

## Script

- [verify_v2_vs_old_json.py](verify_v2_vs_old_json.py)

## Default behavior

Runs on:

- `2021CMC`
- `2020NES`

Using:

- old JSONs: `data/<conference>/session_data/*.json`
- new JSONs: `data-v2/<conference>/output_<session_id>/...chunk*.json`

## Outputs

Written in this folder:
- `verification_session_metrics.csv`
- `verification_conference_summary.json`

## Run

From repo root:

```bash
python old-vs-new/2_conf_v2_verification/verify_v2_vs_old_json.py
```

Optional custom paths:

```bash
python old-vs-new/2_conf_v2_verification/verify_v2_vs_old_json.py \
  --old-root "/path/to/linkography_ai/data" \
  --gem-root "/path/to/linkography_ai/data-v2" \
  --conferences 2021CMC 2020NES
```

## Current benchmark outputs

The checked-in summary files in this folder currently report:

- `2021CMC`: 21 matched sessions, mean CDP coverage `0.764`, mean heuristic match rate `0.348`
- `2020NES`: 16 matched sessions, mean CDP coverage `0.460`, mean heuristic match rate `0.219`

See:

- `verification_session_metrics.csv`
- `verification_conference_summary.json`
