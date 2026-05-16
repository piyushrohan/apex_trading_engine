.PHONY: test test-unit test-risk test-integration test-replay coverage lint format typecheck ci-local setup

# Parallel execution by default for speed, except for strict risk tests
test:
	pytest -m "not slow and not replay" -n auto

test-unit:
	pytest -m unit -n auto

test-risk:
	pytest -m risk

test-integration:
	pytest -m integration

test-replay:
	pytest -m replay

replay-debug:
	@echo "Replaying session $${SESSION:-latest}..."
	pytest -m replay -s -vv --log-cli-level=DEBUG

coverage:
	pytest --cov=src --cov-report=html --cov-report=term-missing

lint:
	ruff check .
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

format:
	black .
	isort .

typecheck:
	mypy src/ --ignore-missing-imports || echo "Mypy warnings found. Run with strictness locally."

ci-local:
	make format
	make lint
	make typecheck
	make test-risk
	make test
	make coverage

setup:
	bash scripts/setup_dev.sh
