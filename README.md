# broom-sm

Tidy-style helpers for [statsmodels](https://www.statsmodels.org/) that expose the familiar `tidy / glance / augment` verbs plus bootstrapping, diagnostics, and Bayesian add-ons. broom-sm leans heavily on `pandas_flavor`, so every helper is available as a `DataFrame` method:

```python
import pandas as pd
import statsmodels.api as sm
from broom_sm import stats_report

mtcars = sm.datasets.get_rdataset("mtcars").data
report = mtcars.stats_report(formula="mpg ~ wt + hp", stat_type="ols")
print(report["tidy"])
```

## Key features

- 🔁 **Tidy verbs** that work on formulas or pre-fitted statsmodels results.
- 🧱 **Extensible model registry** (`MODEL_CONFIG`) covering OLS, GLMs (Poisson, Gamma, Negative Binomial, Beta), GEE, MixedLM, PHReg/Survival, and Quantile Regression.
- 📦 **Bootstrapping + Bayesian bootstrap** utilities with consistent logging.
- 📊 **Diagnostics + plotting** helpers that return Matplotlib figures instead of printing.
- 🧪 **Robust SEs & weights**: pass `cov_type`, `cov_kwds`, `family`, `link`, or `weights` through tidy/glance/augment.
- 🛠️ **CLI** for quick reports (`broom-sm report --data data.csv --formula 'y ~ x1 + x2' --stat-type ols`).

## Installation

```bash
pip install broom-sm          # core tidy + diagnostics
pip install broom-sm[viz]     # adds seaborn/matplotlib
pip install broom-sm[bayes]   # adds bayesian_bootstrap
```

## CLI usage

```
$ broom-sm report --data data.csv --formula "y ~ x1 + x2" --stat-type ols
$ broom-sm compare --data data.csv --stat-type ols --formulas "y ~ x1" "y ~ x1 + x2"
```

Outputs default to JSON (pass `--format csv` for tabular output).

## Extra namespace (`extra_sm`)

Projects built on broom-sm historically imported `extra_sm.*` helpers for empirical Bayes workflows. broom-sm now ships the empty namespace package so those imports continue to resolve. If you maintain an extension that registers DataFrame methods via `extra_sm`, simply declare `broom-sm` as a dependency—no additional wiring is needed. The namespace acts as a rendezvous point for optional plugins without forcing broom-sm to import them eagerly.

## Changelog

### Version 0.1.2 — 2026-06-20

P0 fixes from the 2026-06-20 code review:

- `stats_augment` now validates index uniqueness for `data` and `new_data`, rejects overlapping indices, and aligns residuals / influence diagnostics position-wise for the in-sample path. The `.in_sample` flag is now a single boolean rather than a `set`-based membership test.
- `prepare_fit` now passes `freq_weights` to GLM-family fitters (Poisson, Gamma, Negative Binomial, Beta, etc.) and keeps `weights` for OLS.
- `boot_tidy`, `boot_glance`, and `boot_augment` raise `RuntimeError` when every bootstrap replication fails, instead of returning an empty DataFrame.
- `stats_residual_plot` validates that the target column `y` is numeric before passing it to plotting / `probplot`.
- `stats_vif` now emits a clear warning that the intercept is omitted, and handles no-intercept formulas consistently without adding a constant.

Selected P1 fixes in the same release:

- `anova_type` is validated to be 1, 2, or 3 in `stats_anova_tidy`.
- `stats_kruskal_tidy` validates that `group_col` and `value_col` exist.
- `stats_correlation_tidy` validates that requested `columns` exist and are numeric.
- `stats_formula` now quotes non-syntactic column names with `Q('...')`.
- `bayes_boot` validates `target_column` / `n_samples` and warns when NaN values are dropped.
- `stats_chisquare_plot` drops NaN categories before building the contingency table.
- Repository URLs in `setup.cfg` updated from `jcvall/broom-sm` to `ezraair555/broom-sm`.
- Removed the unused `src/extra_sm` package.

## Documentation

Full documentation (API, how-to guides, tutorials, and plot gallery) lives in `docs/` and at <https://ezraair555.github.io/broom-sm/>. Contributions are welcome—see `CONTRIBUTING.md` for workflow details.
