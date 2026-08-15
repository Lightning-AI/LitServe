# Governance

LitServe is an open source project backed by [Lightning AI](https://lightning.ai/), released under the [Apache-2.0](LICENSE) license.

## Roles and responsibilities

**Maintainers** review and merge contributions, triage issues, guide technical direction, and cut releases. The current maintainers are listed in [.github/CODEOWNERS](.github/CODEOWNERS), where ownership is scoped by path.

**Contributors** are anyone contributing code, tests, docs, triage, or review. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

**Retired committers** are past maintainers, recorded in [.github/CODEOWNERS](.github/CODEOWNERS) and not requested for review.

Maintainers are invited on the strength of sustained contribution, with the agreement of the current maintainers. A maintainer may step down, or be moved to retired status after a long period of inactivity, by a pull request updating [.github/CODEOWNERS](.github/CODEOWNERS).

## Decision making

Changes land through pull requests against `main`; direct pushes are not used. A pull request needs passing CI and approval from a maintainer who owns the affected paths. Larger changes — new public APIs, breaking changes, new dependencies — should start as an issue. Maintainers work by consensus and resolve disagreements in the open.

## Communication

GitHub issues and pull requests are the source of truth for decisions. Discussion also happens on [Discord](https://discord.com/invite/MWAEvnC5fU), but anything affecting the project is recorded on GitHub.

## Releases

LitServe follows `MAJOR.MINOR.PATCH` versioning, with the version in [src/litserve/\_\_about\_\_.py](src/litserve/__about__.py).

- **Cadence:** cut as needed rather than on a fixed calendar, once a meaningful set of fixes or features has landed on `main`. Regressions and security fixes ship as soon as they are ready.
- **Criteria:** `main` green across the CI matrix in [.github/workflows](.github/workflows), plus the multi-GPU pipeline.
- **Process:** a maintainer bumps the version and publishes a GitHub Release; [`release-pypi.yml`](.github/workflows/release-pypi.yml) publishes to PyPI via trusted publishing.
- **Supported versions:** Python and PyTorch versions are declared in [pyproject.toml](pyproject.toml) and the CI matrix. LitServe is validated against the latest two PyTorch releases; torch is not a runtime dependency.

## Code of Conduct

All participants follow the [Code of Conduct](CODE_OF_CONDUCT.md). Reports go to community@lightning.ai.

## Changing this document

By pull request, with approval from at least two maintainers.
