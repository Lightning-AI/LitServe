# Contributing to LitServe

We welcome all contributions, regardless of your level of experience or hardware. Whether it's a bug fix, a new feature, a new example, or an improvement to the docs — we appreciate your help!

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## What we're looking for

- **Bug fixes** — especially around worker lifecycle, shutdown, streaming, and platform-specific behavior.
- **New features** — batching, streaming, specs (OpenAI/MCP), transports, loggers, and callbacks.
- **Integrations and examples** — serving a new model type or inference engine with LitServe.
- **Documentation** — clarifications, missing docstrings, and better examples.
- **Performance** — LitServe targets a minimum 2x speedup over plain FastAPI; benchmarks that expose regressions are valuable.

## How to contribute

1. **Open an issue** — describe the bug or feature before writing code. This helps us align on scope early.
2. **Fork the repo** and create a branch from `main`.
3. **Make your changes** and add or update relevant tests.
4. **Open a pull request** against `main`. Include a clear description of what changed and why.

New to open source? Start with an issue labeled [good first issue](https://github.com/Lightning-AI/litserve/labels/good%20first%20issue).

## Development setup

```bash
git clone https://github.com/<your-username>/litserve
cd litserve
```

```bash
# using uv (recommended — matches CI)
uv sync --all-extras --dev

# using pip (installing the dev group needs a recent pip)
pip install -e .
pip install --group dev
```

Install pre-commit hooks to catch style issues before pushing:

```bash
# using uvx
uvx pre-commit install          # install hooks
uvx pre-commit run --all-files   # run manually

# using pip
pip install pre-commit
pre-commit install              # install hooks
pre-commit run --all-files       # run manually
```

## Running tests

```bash
# full suite, as CI runs it
pytest --cov=litserve tests/ -v -s --durations=100

# a single file or test
pytest tests/unit/test_batch.py -v
pytest tests/unit/test_batch.py::test_max_batch_size -v
```

Tests live in `tests/unit/`, `tests/integration/`, and `tests/e2e/`. The matching `unit` / `integration` / `e2e` marker is applied automatically based on that directory (see [tests/conftest.py](tests/conftest.py)), so put a new test in the right folder and you're done — no manual marker needed. To run one tier:

```bash
pytest -m unit tests/          # fast, no server startup
pytest -m integration tests/   # spins up real servers
pytest -m e2e tests/           # end-to-end, slowest
```

## Guidelines

- Keep pull requests focused — one logical change per PR.
- Write tests for new functionality, and add a regression test alongside any bug fix.
- Follow the existing code style (enforced via [ruff](https://docs.astral.sh/ruff/) and pre-commit).
- Don't leave `print` calls in library code — use the module logger.
- All code should be your own original work; third-party snippets must be attributed.

## Reviews and merging

Pull requests need approval from at least one code owner (see [.github/CODEOWNERS](.github/CODEOWNERS)) and green CI before merging. See [GOVERNANCE.md](GOVERNANCE.md) for how decisions are made and how maintainers are added.

## Community

- [Request a feature or report a bug](https://github.com/Lightning-AI/litserve/issues)
- [Documentation](https://lightning.ai/docs/litserve/home)
- [Community guide](https://lightning.ai/docs/litserve/community)
- [Join our Discord](https://discord.com/invite/MWAEvnC5fU)
