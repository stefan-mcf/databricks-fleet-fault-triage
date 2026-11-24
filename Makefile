.PHONY: setup check test demo build

setup:
	uv sync --locked --all-extras

check:
	uv run --locked ruff check src tests tools assets
	uv run --locked ruff format --check src tests tools assets
	uv run --locked pytest -q
	uv run --locked python tools/verify_publication.py
	git diff --check

test:
	uv run --locked pytest -q

demo:
	uv run --locked python tools/run_demo.py

build:
	uv build
