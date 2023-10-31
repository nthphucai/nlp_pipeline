.DEFAULT_GOAL := quality
.PHONY: quality format clean

CHECK_DIRS := nlp_pipeline

# Format source code automatically and check is there are any problems left that need manual fixing

format:
	black $(CHECK_DIRS)
	isort $(CHECK_DIRS)

# Check that source code meets quality standards
quality:
	black --check $(CHECK_DIRS)
	isort --check-only $(CHECK_DIRS)
	flake8 $(CHECK_DIRS)

clean:
	find . | grep -E '(\.mypy_cache|__pycache__|.DS_Store|__MACOSX|.pytest_cache|\.pyc|\.pyo$$)' | xargs rm -rf
