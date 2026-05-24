# broom-sm

**Tidy-style statistical inference for Python with statsmodels**

broom-sm brings the ergonomic design of [broom](https://broom.tidymodels.org/) and the tidyverse to Python's statsmodels ecosystem. The package centers around three main verbs—`stats_tidy()`, `stats_glance()`, and `stats_augment()`—supplemented by bootstrapping utilities, diagnostic plots, and Bayesian helpers.

```python
import pandas as pd
import statsmodels.api as sm
from broom_sm import stats_report

# Load data
mtcars = sm.datasets.get_rdataset("mtcars").data

# Fit once, get everything
report = mtcars.stats_report(
    formula="mpg ~ wt + hp",
    stat_type="ols"
)

# Tidy coefficient table
print(report["tidy"])
#>       term  estimate  std.error  conf.low  conf.high   statistic    p.value
#> 0  Intercept  37.227270   1.877627  33.343267  41.111273  19.826764  1.27e-17
#> 1         wt  -3.877831   0.714968  -5.355789  -2.399873  -5.423781  1.19e-05
#> 2         hp  -0.031157   0.011436  -0.054812  -0.007501  -2.724389  1.12e-02

# Model-level statistics
print(report["glance"])
#>   stat_type  nobs      llf       aic       bic  df_model  df_resid  rsquared
#> 0       ols    32 -72.54928  153.09856  158.95938       2.0      29.0  0.826783
```

## Installation

```bash
# Core package (tidy verbs + diagnostics)
pip install broom-sm

# With visualization dependencies
pip install broom-sm[viz]

# With Bayesian bootstrap support
pip install broom-sm[bayes]
```

## The Tidy Workflow

broom-sm is built around three core verbs that convert statsmodels objects into tidy DataFrames:

| Verb | Purpose | Output |
|------|---------|--------|
| `stats_tidy()` | Coefficient tables | One row per term |
| `stats_glance()` | Model-level statistics | One row per model |
| `stats_augment()` | Add predictions & residuals | One row per observation |

### Example: Analysis of Variance

Test whether vehicle weight differs by cylinder count:

```python
import pandas as pd
import statsmodels.api as sm

mtcars = sm.datasets.get_rdataset("mtcars").data

# Calculate observed statistic
obs_stat = mtcars.stats_anova_tidy(
    formula="wt ~ factor(cyl)",
    anova_type=2
)

# Bootstrap the null distribution
null_dist = mtcars.boot_tidy(
    formula="wt ~ factor(cyl)",
    stat_type="ols",
    n_boot=1000,
    seed=42
)

# Visualize
from broom_sm import stats_residual_plot
figures = mtcars.stats_residual_plot(["cyl"], y="wt")
figures[0][1].show()

# Calculate p-value
from scipy import stats
f_stat = obs_stat["statistic"].iloc[0]
p_value = 1 - stats.f.cdf(f_stat, obs_stat["df"].iloc[0], obs_stat["df_resid"].iloc[0])
```

## Key Features

### 🔁 Tidy Verbs

All core verbs work with formulas or pre-fitted statsmodels results:

```python
# Formula interface
df.stats_tidy("y ~ x1 + x2", stat_type="ols")

# Pre-fitted model interface
import statsmodels.formula.api as smf
model = smf.ols("y ~ x1 + x2", data=df).fit()
df.stats_tidy(model=model)
```

### 🧱 Extensible Model Registry

Support for OLS, GLMs (Poisson, Gamma, Beta, Negative Binomial), GEE, MixedLM, PHReg/Survival, and Quantile Regression. Register custom models:

```python
from broom_sm.model_registry import ModelSpec, register_model
import statsmodels.formula.api as smf

register_model(
    "tobit",
    ModelSpec(
        fitter=lambda formula, data, **kwargs: smf.tobit(formula, data=data, **kwargs).fit(),
        stat_name="z_stat"
    )
)
```

### 📦 Bootstrapping

Built-in resampling with consistent logging:

```python
boot = mtcars.boot_tidy(
    formula="mpg ~ wt",
    stat_type="ols",
    n_boot=500,
    seed=11
)
boot.groupby("term")["estimate"].agg(["mean", "std"])
```

### 📊 Diagnostics & Visualization

All plot helpers return Matplotlib figures (no implicit `plt.show()`):

```python
# Residual diagnostics
figures = df.stats_residual_plot(["x1", "x2"], y="y")

# Influence plot
fig = df.stats_influence_plot("y ~ x1 + x2", stat_type="ols")

# Coefficient forest plot
tidy = df.stats_tidy("y ~ x1 + x2", stat_type="ols")
fig, ax = stats_coef_forest(tidy)
```

### 🧪 Robust Standard Errors

Pass `cov_type`, `cov_kwds`, `family`, `link`, or `weights` directly:

```python
df.stats_tidy(
    formula="mpg ~ wt",
    stat_type="glm",
    family="binomial",
    weights=df["weights"],
    cov_type="HC3"
)
```

### 🛠️ Command-Line Interface

Quick reports from the terminal:

```bash
# Single model report
broom-sm report --data data.csv --formula 'y ~ x1 + x2' --stat-type ols

# Compare multiple models
broom-sm compare --data data.csv --stat-type ols \
  --formulas "y ~ x1" "y ~ x1 + x2"
```

Output defaults to JSON; pass `--format csv` for tabular output.

## Model Coverage

| Model Type | `stat_type` | Robust SEs | Weights | Family/Link |
|------------|-------------|------------|---------|-------------|
| OLS | `"ols"` | ✅ | ✅ | — |
| GLM (Gaussian) | `"glm"` | ✅ | ✅ | ✅ |
| GLM (Poisson) | `"poisson"` | ✅ | ✅ | ✅ |
| GLM (Gamma) | `"gamma"` | ✅ | ✅ | ✅ |
| GLM (Beta) | `"beta"` | ✅ | ✅ | ✅ |
| Negative Binomial | `"negbin"` | ✅ | ✅ | — |
| Quantile Regression | `"quantreg"` | ✅ | ✅ | — |
| GEE | `"gee"` | ✅ | ✅ | ✅ |
| MixedLM | `"mixedlm"` | ✅ | ✅ | — |
| PHReg (Survival) | `"phreg"` | ✅ | ✅ | — |

## Documentation

- **[Tutorials](docs/tutorials/index.md)** — End-to-end walkthroughs
- **[How-to Guides](docs/howto/index.md)** — Task-oriented recipes
- **[API Reference](docs/api-reference.md)** — Complete function documentation
- **[Quick Start](docs/quickstart.md)** — Get started in 5 minutes

## Contributing

We welcome contributions! Please review our [contributing guidelines](CONTRIBUTING.md) and [code of conduct](.github/CODE_OF_CONDUCT.md).

For questions and discussions, please [post on GitHub Discussions](https://github.com/ezraair555/broom-sm/discussions). If you think you've encountered a bug, please [submit an issue](https://github.com/ezraair555/broom-sm/issues).

## License

MIT License — see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgments

broom-sm draws inspiration from:
- [broom](https://broom.tidymodels.org/) (R) — Tidy model outputs
- [infer](https://infer.tidymodels.org/) (R) — Tidy statistical inference
- [pandas_flavor](https://github.com/Zsailer/pandas_flavor) — DataFrame method registration
- [statsmodels](https://www.statsmodels.org/) — Statistical modeling in Python
