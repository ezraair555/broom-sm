# Quick Start Guide

This guide walks you through your first broom-sm analysis in 5 minutes.

## Step 1: Load Data

```python
import pandas as pd
import statsmodels.api as sm
from broom_sm import stats_report

# Use built-in R dataset
mtcars = sm.datasets.get_rdataset("mtcars").data

# Or load your own data
# df = pd.read_csv("your_data.csv")
```

## Step 2: Fit a Model

```python
# One-line report with tidy, glance, and augment
report = mtcars.stats_report(
    formula="mpg ~ wt + hp",
    stat_type="ols"
)
```

## Step 3: Examine Coefficients

```python
# Tidy coefficient table
report["tidy"]
```

Output:
```
       term  estimate  std.error  conf.low  conf.high   statistic    p.value  cov_type stat_type
0  Intercept  37.227270   1.877627  33.343267  41.111273  19.826764  1.27e-17  nonrobust       ols
1         wt  -3.877831   0.714968  -5.355789  -2.399873  -5.423781  1.19e-05  nonrobust       ols
2         hp  -0.031157   0.011436  -0.054812  -0.007501  -2.724389  1.12e-02  nonrobust       ols
```

## Step 4: Check Model Fit

```python
# Model-level statistics
report["glance"]
```

Output:
```
  stat_type  nobs      llf       aic       bic  df_model  df_resid  rsquared  rsquared_adj
0       ols    32 -72.54928  153.09856  158.95938       2.0      29.0  0.826783      0.814767
```

## Step 5: Get Predictions

```python
# Augmented data with predictions and intervals
report["augment"][["mpg", ".fitted", ".resid", ".mean_ci_lower", ".mean_ci_upper"]].head()
```

Output:
```
   mpg  .fitted    .resid  .mean_ci_lower  .mean_ci_upper
0   21  22.4139  -1.413915        20.84534        23.98246
1   21  21.3983  -0.398307        20.12345        22.67315
2   22  23.2695  -1.269503        21.89234        24.64666
3   21  20.1234   0.876597        18.76543        21.48137
4   18  19.8765  -1.876503        18.23456        21.51844
```

## Step 6: Use Robust Standard Errors

```python
# Heteroskedasticity-consistent SEs
robust_report = mtcars.stats_report(
    formula="mpg ~ wt + hp",
    stat_type="ols",
    tidy_kwargs={"cov_type": "HC3"}
)
robust_report["tidy"][["term", "estimate", "std.error", "cov_type"]]
```

## Step 7: Visualize Results

```python
from broom_sm import stats_coef_forest, stats_residual_plot

# Coefficient forest plot
tidy = mtcars.stats_tidy("mpg ~ wt + hp", stat_type="ols")
fig, ax = stats_coef_forest(tidy)
fig.savefig("coefficients.png", dpi=150, bbox_inches="tight")

# Residual diagnostics
figures = mtcars.stats_residual_plot(["wt", "hp"], y="mpg")
for feature, fig in figures:
    fig.savefig(f"residual_{feature}.png", dpi=150)
```

## Step 8: Bootstrap for Inference

```python
# Bootstrap coefficient estimates
boot = mtcars.boot_tidy(
    formula="mpg ~ wt",
    stat_type="ols",
    n_boot=500,
    seed=42
)

# Summarize bootstrap distribution
boot.groupby("term").agg({
    "estimate": ["mean", "std"],
    ".bootstrap_id": "count"
})
```

## Step 9: Compare Models

```python
from broom_sm import stats_compare
import statsmodels.formula.api as smf

# Fit multiple models
models = {
    "intercept_only": smf.ols("mpg ~ 1", data=mtcars).fit(),
    "wt_only": smf.ols("mpg ~ wt", data=mtcars).fit(),
    "full": smf.ols("mpg ~ wt + hp", data=mtcars).fit()
}

# Compare AIC/BIC
stats_compare(models)
```

Output:
```
           model       aic       bic       llf  nobs
0           full  153.09856  158.95938 -72.54928  32.0
1        wt_only  165.43210  169.82345 -79.71605  32.0
2  intercept_only  189.87654  192.80123 -93.93827  32.0
```

## What's Next?

- **[Tutorials](tutorials/index.html)** — End-to-end examples
- **[How-to Guides](howto/index.html)** — Task-oriented recipes
- **[API Reference](api/modules.html)** — Complete function documentation
- **[Model Coverage](howto/model_config.html)** — Supported model types

## Common Patterns

### Formula Interface

```python
df.stats_tidy("y ~ x1 + x2 + x3", stat_type="ols")
```

### Pre-fitted Model Interface

```python
import statsmodels.formula.api as smf
model = smf.ols("y ~ x1 + x2", data=df).fit()
df.stats_tidy(model=model)
```

### DataFrame Method Interface

All verbs are registered as pandas DataFrame methods:

```python
df.stats_tidy("y ~ x1", stat_type="ols")
df.stats_glance("y ~ x1", stat_type="ols")
df.stats_augment("y ~ x1", stat_type="ols")
```

### Functional Interface

You can also use the functional interface:

```python
from broom_sm import stats_tidy, stats_glance, stats_augment

stats_tidy(data=df, formula="y ~ x1", stat_type="ols")
```
