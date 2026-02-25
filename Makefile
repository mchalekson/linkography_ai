.PHONY: help validate batch_entropy analyze merge_outcomes test_outcomes batch_convergence compare_binning compare_normalization time_pressure outcome_model cdp_content speaker_cdp fine_grained cohort speaker_role all clean

PYTHON := .venv/bin/python

# Default target
help:
	@echo "================================================"
	@echo "CDP Entropy Analysis Pipeline"
	@echo "================================================"
	@echo ""
	@echo "Core Pipeline:"
	@echo "  make validate        - Validate data integrity across all sessions"
	@echo "  make batch_entropy   - Run batch entropy computation (all conferences)"
	@echo "  make analyze         - Analyze entropy trajectories (requires batch_entropy)"
	@echo "  make merge_outcomes  - Merge entropy with funding outcomes"
	@echo "  make test_outcomes   - Statistical tests: entropy vs funding outcomes"
	@echo ""
	@echo "Structural Analysis:"
	@echo "  make batch_convergence - Batch convergence detection"
	@echo "  make compare_binning - Compare time-based vs index-based thirds"
	@echo "  make compare_normalization - Compare raw vs normalized entropy"
	@echo "  make time_pressure   - Time-pressure language analysis"
	@echo ""
	@echo "CDP-Focused Deep Dives:"
	@echo "  make cdp_content     - Analyze CDP score 1 vs score 2 utterance content"
	@echo "  make speaker_cdp     - Speaker-level CDP usage and diversity"
	@echo "  make fine_grained    - Fine-grained CDP entropy (5-10 min bins)"
	@echo "  make cohort          - Compare CDP patterns across years"
	@echo "  make speaker_role    - Correlate speaker roles with CDP usage"
	@echo ""
	@echo "Integration:"
	@echo "  make outcome_model   - Outcome modeling beyond entropy"
	@echo "  make all             - Run full pipeline (core + structural + CDP + integration)"
	@echo "  make clean           - Remove generated outputs"
	@echo ""
	@echo "Quick start:"
	@echo "  make all             # Run everything"
	@echo ""

# Validate data integrity
validate:
	@echo "==> Validating data integrity..."
	$(PYTHON) pipelines/validate_data_integrity.py

# Run batch entropy computation
batch_entropy:
	@echo "==> Running batch entropy computation..."
	$(PYTHON) pipelines/run_cdp_entropy_all.py --conference ALL --normalize

# Analyze entropy trajectories
analyze:
	@echo "==> Analyzing entropy trajectories..."
	$(PYTHON) pipelines/analyze_entropy_trajectories.py

# Merge with outcomes
merge_outcomes:
	@echo "==> Merging entropy with outcomes..."
	$(PYTHON) pipelines/merge_entropy_with_outcomes.py

# Test entropy vs outcomes
test_outcomes:
	@echo "==> Testing entropy vs outcomes..."
	$(PYTHON) pipelines/test_entropy_outcomes.py

# Batch convergence detection
batch_convergence:
	@echo "==> Running batch convergence detection..."
	$(PYTHON) pipelines/batch_convergence.py

# Compare time-based vs index-based thirds
compare_binning:
	@echo "==> Comparing time-based vs index-based thirds..."
	$(PYTHON) pipelines/compare_time_binning.py --normalize

# Compare raw vs normalized entropy
compare_normalization:
	@echo "==> Comparing raw vs normalized entropy..."
	$(PYTHON) pipelines/compare_entropy_normalization.py

# Time-pressure language analysis
time_pressure:
	@echo "==> Analyzing time-pressure language..."
	$(PYTHON) pipelines/time_pressure_language.py

# Outcome modeling beyond entropy
outcome_model:
	@echo "==> Running outcome modeling..."
	$(PYTHON) pipelines/outcome_modeling.py

# CDP Content Analysis (utterance-level)
cdp_content:
	@echo "==> Analyzing CDP content (score 1 vs score 2 utterances)..."
	$(PYTHON) pipelines/cdp_content_analysis.py

# Speaker-level CDP Analysis
speaker_cdp:
	@echo "==> Analyzing speaker-level CDP usage..."
	$(PYTHON) pipelines/speaker_level_cdp.py

# Fine-grained CDP timing (5-10 min bins)
fine_grained:
	@echo "==> Computing fine-grained CDP entropy (5-10 min bins)..."
	$(PYTHON) pipelines/fine_grained_cdp_timing.py

# CDP by cohort (2020/2021/2022)
cohort:
	@echo "==> Comparing CDP patterns across conference years..."
	$(PYTHON) pipelines/cdp_by_cohort.py

# Speaker role and CDP
speaker_role:
	@echo "==> Analyzing speaker roles and CDP usage..."
	$(PYTHON) pipelines/speaker_role_cdp.py

# Speaker diversity outcomes correlation
speaker_diversity_outcomes:
	@echo "==> Correlating speaker diversity metrics with outcomes..."
	$(PYTHON) pipelines/speaker_diversity_outcomes.py

# Timing patterns outcomes analysis
timing_patterns_outcomes:
	@echo "==> Analyzing timing patterns vs outcomes..."
	$(PYTHON) pipelines/timing_patterns_outcomes.py

# Meeting profile classifier
meeting_profile:
	@echo "==> Building meeting profile classifier..."
	$(PYTHON) pipelines/meeting_profile_classifier.py

# Run full pipeline
all: validate batch_entropy analyze merge_outcomes test_outcomes batch_convergence compare_binning compare_normalization time_pressure outcome_model cdp_content speaker_cdp fine_grained cohort speaker_role speaker_diversity_outcomes timing_patterns_outcomes meeting_profile
	@echo ""
	@echo "================================================"
	@echo "Pipeline complete!"
	@echo "================================================"
	@echo "Core Outputs:"
	@echo "  - outputs/logs/data_validation_report.txt"
	@echo "  - outputs/tables/cdp_entropy_by_session_ALL_*.csv"
	@echo "  - outputs/analysis/entropy_trajectory_summary.txt"
	@echo "  - outputs/tables/entropy_with_outcomes.csv"
	@echo "  - figures/final/entropy_trajectory.png"
	@echo ""
	@echo "CDP-Focused Analysis Outputs:"
	@echo "  - outputs/tables/cdp_content_analysis.csv"
	@echo "  - outputs/analysis/cdp_content_analysis_summary.txt"
	@echo "  - outputs/tables/speaker_level_cdp.csv"
	@echo "  - outputs/analysis/speaker_level_cdp_summary.txt"
	@echo "  - outputs/tables/cdp_fine_grained_entropy_300s.csv"
	@echo "  - outputs/analysis/cdp_fine_grained_summary_300s.txt"
	@echo "  - outputs/analysis/cdp_by_cohort_summary.txt"
	@echo ""
	@echo "Outcome Prediction Models:"
	@echo "  - outputs/tables/speaker_diversity_with_outcomes.csv"
	@echo "  - outputs/analysis/speaker_diversity_outcomes_summary.txt"
	@echo "  - outputs/tables/timing_features_with_outcomes.csv"
	@echo "  - outputs/analysis/timing_patterns_outcomes_summary.txt"
	@echo "  - outputs/tables/meeting_profile_classifier_results.csv"
	@echo "  - outputs/analysis/meeting_profile_classifier_results.txt"
	@echo ""

# Clean outputs
clean:
	@echo "==> Cleaning generated outputs..."
	rm -rf outputs/tables/*.csv
	rm -rf outputs/analysis/*.txt
	rm -rf outputs/logs/*.txt
	rm -rf figures/final/*.png
	@echo "Done!"
