.PHONY: test test-unit test-risk test-integration test-replay frontend-test coverage lint format typecheck ci-local setup

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

frontend-test:
	node --check frontend/app.js
	node --test tests/frontend/*.test.mjs

replay-debug:
	@echo "Replaying session $${SESSION:-latest}..."
	pytest -m replay -s -vv --log-cli-level=DEBUG

coverage:
	pytest --cov=src --cov-report=html --cov-report=term-missing

lint:
	ruff check src tests
	flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics

format:
	black src tests
	isort src tests

typecheck:
	mypy src/ --ignore-missing-imports || echo "Mypy warnings found. Run with strictness locally."

ci-local:
	make format
	make lint
	make typecheck
	make frontend-test
	make test-risk
	make test
	make coverage

setup:
	bash scripts/setup_dev.sh
