# 2_conf_v2_verification

Cross-check old CDP session JSONs against newer Gemini chunk JSONs for conference-level alignment.

## Script

- [verify_v2_vs_old_json.py](verify_v2_vs_old_json.py)

## Default behavior

Runs on:
- `2021CMC`
- `2020NES`

Using:
- old JSONs: `linkography_ai/data/<conference>/session_data/*.json`
- new JSONs: `gemini_data_analysis/outputs/<conference>/output_<session_id>/...chunk*.json`

## Outputs

Written in this folder:
- `verification_session_metrics.csv`
- `verification_conference_summary.json`

## Run

From repo root:

```bash
"/Users/maxchalekson/Northwestern University/Summer-2025/NICO/NICO Research/linkography_ai/.venv/bin/python" 2_conf_v2_verification/verify_v2_vs_old_json.py
```

Optional custom paths:

```bash
"/Users/maxchalekson/Northwestern University/Summer-2025/NICO/NICO Research/linkography_ai/.venv/bin/python" 2_conf_v2_verification/verify_v2_vs_old_json.py \
  --old-root "/path/to/linkography_ai/data" \
  --gem-root "/path/to/gemini_data_analysis/outputs" \
  --conferences 2021CMC 2020NES
```
