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

## Documentation

Full documentation (API, how-to guides, tutorials, and plot gallery) lives in `docs/` and at <https://jcvall.github.io/broom-sm/>. Contributions are welcome—see `CONTRIBUTING.md` for workflow details.
