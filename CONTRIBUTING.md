# Contributing to NanoIndex

Thanks for your interest in contributing! NanoIndex is open source under Apache 2.0 and we welcome contributions of all kinds.

## Getting started

```bash
git clone https://github.com/NanoNets/nanoindex.git
cd nanoindex
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

## Running tests

```bash
pytest tests/ -x -q
```

Some tests require a `NANONETS_API_KEY` environment variable. Tests that need it are skipped automatically when the key isn't set.

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check nanoindex/
ruff format nanoindex/
```

CI runs both checks on every PR.

## Making changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add tests if you're adding new functionality
4. Run `ruff check` and `pytest` locally
5. Open a PR with a clear description of what you changed and why

## What to work on

- Issues labeled **good first issue** are a great starting point
- Check the roadmap in the README for planned features
- Bug fixes and documentation improvements are always welcome

## Project structure

```
nanoindex/
  __init__.py          # NanoIndex class, main API
  config.py            # Configuration
  models.py            # Pydantic data models
  core/
    extractor.py       # Nanonets API extraction strategies
    tree_builder.py    # Tree construction from extraction results
    enricher.py        # LLM-based summary generation
    agentic.py         # Multi-round agentic retrieval
    client.py          # Nanonets API client
    parsers/           # Document parsers (nanonets, pymupdf)
  utils/               # PDF, markdown, token utilities
```

## Questions?

Open an issue or start a discussion. We're happy to help.
