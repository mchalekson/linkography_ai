.PHONY: help validate batch_entropy analyze merge_outcomes all clean

PYTHON := .venv/bin/python

# Default target
help:
	@echo "================================================"
	@echo "CDP Entropy Analysis Pipeline"
	@echo "================================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make validate        - Validate data integrity across all sessions"
	@echo "  make batch_entropy   - Run batch entropy computation (all conferences)"
	@echo "  make analyze         - Analyze entropy trajectories (requires batch_entropy)"
	@echo "  make merge_outcomes  - Merge entropy with funding outcomes"
	@echo "  make all             - Run full pipeline (validate → batch → analyze → merge)"
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

# Run full pipeline
all: validate batch_entropy analyze merge_outcomes
	@echo ""
	@echo "================================================"
	@echo "Pipeline complete!"
	@echo "================================================"
	@echo "Outputs:"
	@echo "  - outputs/logs/data_validation_report.txt"
	@echo "  - outputs/tables/cdp_entropy_by_session_ALL_*.csv"
	@echo "  - outputs/analysis/entropy_trajectory_summary.txt"
	@echo "  - outputs/tables/entropy_with_outcomes.csv"
	@echo "  - figures/final/entropy_trajectory.png"
	@echo ""

# Clean outputs
clean:
	@echo "==> Cleaning generated outputs..."
	rm -rf outputs/tables/*.csv
	rm -rf outputs/analysis/*.txt
	rm -rf outputs/logs/*.txt
	rm -rf figures/final/*.png
	@echo "Done!"
