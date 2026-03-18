# Contributing to broom-sm

Thanks for your interest in improving **broom-sm**. This document explains how we collaborate, what we expect from contributors, and how you can get changes merged quickly.

## Code of conduct

We follow the [Python Software Foundation Code of Conduct](https://www.python.org/psf/codeofconduct/). Be kind, constructive, and respectful in every interaction.

## Ways to contribute

- Report bugs or feature requests in [GitHub issues](https://github.com/jcvall/broom-sm/issues).
- Improve documentation (typo fixes, new examples, tutorials).
- Add model coverage, diagnostics, or tidy helpers.
- Expand tests or CI workflows.

When in doubt, open an issue before investing significant effort. We are happy to help shape the solution.

## Development workflow

1. **Fork and clone** the repository.
2. **Create a virtual environment** (via `python -m venv .venv && source .venv/bin/activate` or `conda`).
3. **Install dependencies** in editable mode:

   ```bash
   pip install -U pip setuptools
   pip install -e .[testing,viz,bayes]
   ```

4. **Create a feature branch** (`git checkout -b feature/my-change`).
5. **Add tests** covering your change. Run `pytest` locally before pushing.
6. **Update documentation** (README, docs/, tutorials) whenever user-facing behavior changes.
7. **Open a pull request** that describes the motivation, approach, and any trade-offs.

We use [ruff+black defaults](https://github.com/psf/black) enforced via pre-commit hooks. Install them with:

```bash
pip install pre-commit
pre-commit install
```

## Project internals

- Core tidy verbs live in `src/broom_sm/tidy.py` and rely on the shared `MODEL_CONFIG` registry (`model_registry.py`). You can register new model families via `register_model` to make them available to every tidy verb.
- Diagnostics, plotting helpers, and Bayesian utilities have their own modules to keep imports lightweight. Prefer small, focused functions that return `pd.DataFrame` objects or Matplotlib figures instead of printing to stdout.
- Pandas integration happens through `pandas_flavor`. Whenever you add a new DataFrame method, include a short example in the docstring so it shows up in the API docs.

## Documentation

Docs live in `docs/` and are built with Sphinx + MyST. To preview locally:

```bash
tox -e docs
python -m http.server --directory docs/_build/html
```

When authoring new features, add one of the following:

- An entry in the "How-to" collection (see `docs/howto/`).
- A tutorial style walkthrough in `docs/tutorials/` (MyST Markdown supported).
- API docstring updates so autodoc picks up the latest signatures.

## Testing & CI

- Run `pytest --maxfail=1` before submitting. Tests cover tidy/glance/augment, bootstrapping, diagnostics, and failure paths.
- For documentation-only changes, use the `[skip ci]` marker in your commit message to speed up pipelines.

## Release checklist

1. Update `CHANGELOG.md` with a summary of changes.
2. Bump the version via `pyproject.toml` or `git tag` (depending on release tooling).
3. Build and upload via `python -m build` followed by `twine upload dist/*`.

Please ping maintainers if you need help with any step. We appreciate every improvement—thank you for helping make **broom-sm** better!
