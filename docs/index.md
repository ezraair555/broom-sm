# broom-sm: Tidy Statistical Inference for Python

## Documentation Navigation

**Getting Started**
- [Overview](../README.md) — What is broom-sm?
- [Installation](installation.md) — Setup instructions
- [Quick Start](quickstart.md) — 5-minute guide

**Tutorials**
- [Tutorials Index](tutorials/index.md) — End-to-end walkthroughs
- [Tidy Workflow](tutorials/tidy-workflow.md) — Core verbs example
- [Bootstrap Inference](tutorials/bootstrap-inference.md) — Resampling methods

**How-to Guides**
- [How-to Index](howto/index.md) — Task-oriented recipes
- [Extending MODEL_CONFIG](howto/model_config.md) — Add custom models
- [Robust Standard Errors](howto/robust_ses.md) — Heteroskedasticity-consistent SEs
- [Bootstrapping Workflows](howto/bootstrap.md) — Resampling recipes
- [Plot Gallery](howto/plot_gallery.md) — Visualization examples
- [Comparing Models](howto/compare-models.md) — AIC, BIC, cross-validation

**API Reference**
- [Complete API Docs](api-reference.md) — All functions documented

**About**
- [Contributing](../CONTRIBUTING.md) — How to contribute
- [License](../LICENSE.txt) — MIT License
- [Authors](../AUTHORS.md) — Contributors
- [Changelog](../CHANGELOG.md) — Version history

## Overview

The objective of broom-sm is to perform statistical inference using an expressive grammar that coheres with the tidy design framework. The package is centered around **4 main verbs**, supplemented by many utilities to visualize and extract value from their outputs.

### The Four Verbs

| Verb | Purpose | Returns |
|------|---------|---------|
| [`stats_tidy()`](api/modules.html#broom_sm.tidy.stats_tidy) | Convert model coefficients to tidy DataFrame | One row per term |
| [`stats_glance()`](api/modules.html#broom_sm.tidy.stats_glance) | Extract model-level statistics | One row per model |
| [`stats_augment()`](api/modules.html#broom_sm.tidy.stats_augment) | Add predictions, residuals, and diagnostics | One row per observation |
| [`stats_anova_tidy()`](api/modules.html#broom_sm.tidy.stats_anova_tidy) | ANOVA tables in tidy format | One row per factor |

### Example: Testing Independence

```python
import pandas as pd
import statsmodels.api as sm
from broom_sm import stats_report, boot_tidy

# Load data
gss = pd.read_csv("https://raw.githubusercontent.com/ezraair555/broom-sm/main/data/gss.csv")

# Calculate observed statistic
obs_stat = gss.stats_anova_tidy(
    formula="age ~ partyid",
    anova_type=2
)

# Generate null distribution via bootstrap
null_dist = gss.boot_tidy(
    formula="age ~ partyid",
    stat_type="ols",
    n_boot=1000,
    seed=42
)

# Visualize
from broom_sm import stats_coef_forest
fig, ax = stats_coef_forest(obs_stat)
ax.axvline(obs_stat["statistic"].iloc[0], color="red", linestyle="--")

# Calculate p-value
from scipy import stats
p_value = 1 - stats.f.cdf(
    obs_stat["statistic"].iloc[0],
    obs_stat["df"].iloc[0],
    obs_stat["df_resid"].iloc[0]
)
```

## Installation

### From PyPI

```bash
pip install broom-sm          # Core package
pip install broom-sm[viz]     # With visualization (matplotlib, seaborn)
pip install broom-sm[bayes]   # With Bayesian bootstrap support
```

### From Source

```bash
git clone https://github.com/ezraair555/broom-sm.git
cd broom-sm
pip install -e .
```

## Quick Start

### 1. Load Your Data

```python
import pandas as pd
import statsmodels.api as sm

mtcars = sm.datasets.get_rdataset("mtcars").data
```

### 2. Fit a Model and Get Tidy Output

```python
# One-line report with all three verbs
report = mtcars.stats_report(
    formula="mpg ~ wt + hp",
    stat_type="ols"
)

# Coefficient table
report["tidy"]
#>       term  estimate  std.error  conf.low  conf.high   statistic    p.value
#> 0  Intercept  37.227270   1.877627  33.343267  41.111273  19.826764  1.27e-17
#> 1         wt  -3.877831   0.714968  -5.355789  -2.399873  -5.423781  1.19e-05
#> 2         hp  -0.031157   0.011436  -0.054812  -0.007501  -2.724389  1.12e-02

# Model summary
report["glance"]
#>   stat_type  nobs      llf       aic       bic  df_model  df_resid  rsquared
#> 0       ols    32 -72.54928  153.09856  158.95938       2.0      29.0  0.826783

# Predictions with intervals
report["augment"][[".fitted", ".mean_ci_lower", ".mean_ci_upper"]].head()
```

### 3. Use Robust Standard Errors

```python
mtcars.stats_tidy(
    formula="mpg ~ wt + hp",
    stat_type="ols",
    cov_type="HC3"  # Heteroskedasticity-consistent SEs
)
```

### 4. Bootstrap for Inference

```python
boot = mtcars.boot_tidy(
    formula="mpg ~ wt",
    stat_type="ols",
    n_boot=500,
    seed=11
)

# Summarize bootstrap distribution
boot.groupby("term")["estimate"].agg(["mean", "std", "quantile"])
```

### 5. Visualize Results

```python
# Residual diagnostics
figures = mtcars.stats_residual_plot(["wt", "hp"], y="mpg")

# Influence plot
fig = mtcars.stats_influence_plot("mpg ~ wt + hp", stat_type="ols")

# Coefficient forest plot
tidy = mtcars.stats_tidy("mpg ~ wt + hp", stat_type="ols")
fig, ax = stats_coef_forest(tidy)
```

## Model Coverage

broom-sm supports a wide range of statsmodels fitters through the `MODEL_CONFIG` registry:

- **Linear Models**: OLS, WLS, GLS
- **Generalized Linear Models**: Gaussian, Poisson, Gamma, Beta, Negative Binomial
- **Mixed Effects**: MixedLM, GEE
- **Survival Analysis**: PHReg (Cox proportional hazards)
- **Robust Regression**: Quantile Regression, RLM

See the [Model Configuration guide](howto/model_config.html) for details on extending support to additional models.

## Design Principles

broom-sm follows the tidyverse design philosophy:

1. **Tidy Output**: All functions return DataFrames in long/tidy format
2. **Composability**: Verbs chain naturally with pandas operations
3. **Explicitness**: No hidden state; all parameters are explicit
4. **Extensibility**: Register new models via `MODEL_CONFIG`
5. **Visualization**: Plot helpers return figures for further customization

## Resources

- **[Tutorials](tutorials/index.html)** — Learn by doing with end-to-end examples
- **[How-to Guides](howto/index.html)** — Solve specific problems
- **[API Reference](api/modules.html)** — Complete function documentation
- **[GitHub Repository](https://github.com/ezraair555/broom-sm)** — Source code and issues
- **[broom (R)](https://broom.tidymodels.org/)** — Inspiration for tidy model outputs
- **[infer (R)](https://infer.tidymodels.org/)** — Inspiration for tidy inference

## Contributing

We welcome contributions! See our [Contributing Guide](contributing.html) for how to get started.

For questions and discussions, please [post on GitHub Discussions](https://github.com/ezraair555/broom-sm/discussions).
