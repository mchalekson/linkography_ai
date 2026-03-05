# Deep Annotation Comparison: Index

## Start Here

For your meeting prep:
1. Read [SUMMARY_ANNOTATION_COMP.md](SUMMARY_ANNOTATION_COMP.md)
2. Use [README.md](README.md) as the quick reference
3. Use [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) for technical detail
4. Use [CODE_MAPPING.md](CODE_MAPPING.md) for explicit mapping

## Files

- [SUMMARY_ANNOTATION_COMP.md](SUMMARY_ANNOTATION_COMP.md): Executive summary
- [README.md](README.md): Practical usage and interpretation guide
- [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md): Full analysis narrative
- [CODE_MAPPING.md](CODE_MAPPING.md): CDP to Gemini code mapping
- [analyze_annotation_differences.py](analyze_annotation_differences.py): Reproducible script
- analysis_outputs/annotation_comparison_summary.csv: Session-level metrics
- analysis_outputs/annotation_comparison_detailed.json: Detailed per-session outputs

## Key Takeaways

- CMC shows meaningful directional alignment.
- NES mismatch is mostly driven by sparse old-data coverage.
- The two systems are complementary rather than contradictory.
- Your CDP metrics add temporal dynamics that Gemini chunk labels do not fully capture.

## Recommended Next Steps

1. Bring 2-3 concrete session examples to the meeting.
2. Discuss combining Gemini and CDP metrics in joint outcome modeling.
3. If possible, improve NES annotation coverage and rerun comparison.
