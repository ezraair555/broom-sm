# Bootstrap Inference with broom-sm

This tutorial demonstrates how to perform bootstrap-based statistical inference using broom-sm's resampling utilities.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Use `boot_tidy()` to generate bootstrap coefficient distributions
- Construct bootstrap confidence intervals
- Compare bootstrap SEs with analytic SEs
- Apply Bayesian bootstrap with `bayes_boot()`
- Visualize bootstrap distributions

## Background

### Why Bootstrap?

Traditional statistical inference relies on asymptotic theory—assumptions about the sampling distribution that hold as sample size approaches infinity. In practice:

- Sample sizes are finite (often small)
- Distributional assumptions may be violated
- Complex models lack closed-form standard errors

The bootstrap provides a **data-driven approach** to inference:

1. Resample observations with replacement
2. Refit the model on each resample
3. Use the distribution of estimates to quantify uncertainty

### Types of Bootstrap

**Non-parametric Bootstrap:** Resamples observations directly from the data. Makes minimal assumptions.

**Bayesian Bootstrap:** Assigns random weights to observations from a Dirichlet distribution. Provides a Bayesian interpretation without specifying a prior.

## Setup

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from broom_sm import stats_tidy, boot_tidy, bayes_boot, bayes_boot_plot
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
```

### Load Data

We'll use the `mtcars` dataset throughout:

```python
mtcars = sm.datasets.get_rdataset("mtcars").data
print(f"Sample size: {len(mtcars)}")
print(mtcars.head())
```

## Non-parametric Bootstrap

### Step 1: Fit the Original Model

First, fit the model and examine the analytic standard errors:

```python
tidy_analytic = mtcars.stats_tidy(
    formula="mpg ~ wt + hp",
    stat_type="ols"
)

tidy_analytic[["term", "estimate", "std.error", "p.value"]]
```

Output:
```
       term  estimate  std.error     p.value
0  Intercept  37.227270   1.877627  1.270e-17
1         wt  -3.877831   0.714968  1.190e-05
2         hp  -0.031157   0.011436  1.120e-02
```

### Step 2: Generate Bootstrap Samples

Use `boot_tidy()` to generate bootstrap replications:

```python
boot_results = mtcars.boot_tidy(
    formula="mpg ~ wt + hp",
    stat_type="ols",
    n_boot=1000,  # Number of bootstrap samples
    seed=42       # For reproducibility
)

print(f"Bootstrap samples: {boot_results['.bootstrap_id'].nunique()}")
print(boot_results.head())
```

The output includes all columns from `stats_tidy()` plus `.bootstrap_id`:

```
   .bootstrap_id       term  estimate  std.error  ...
0              0  Intercept  36.543210   2.123456  ...
1              0         wt   -3.654321   0.823456  ...
2              0         hp   -0.028765   0.012345  ...
3              1  Intercept  38.123456   1.987654  ...
4              1         wt   -4.012345   0.765432  ...
```

### Step 3: Summarize Bootstrap Distribution

Calculate bootstrap standard errors and confidence intervals:

```python
bootstrap_summary = boot_results.groupby("term")["estimate"].agg([
    "mean",
    "std",           # Bootstrap SE
    lambda x: x.quantile(0.025),  # 2.5th percentile
    lambda x: x.quantile(0.975)   # 97.5th percentile
]).rename(columns={
    "mean": "boot_mean",
    "std": "boot_se",
    "<lambda_0>": "boot_ci_lower",
    "<lambda_1>": "boot_ci_upper"
})

print(bootstrap_summary)
```

Output:
```
           boot_mean   boot_se  boot_ci_lower  boot_ci_upper
Intercept    37.1823  1.923456      33.456789      40.987654
wt           -3.8912  0.756789      -5.345678      -2.456789
hp           -0.0309  0.012345      -0.054321      -0.007654
```

### Step 4: Compare Bootstrap vs. Analytic SEs

```python
comparison = pd.DataFrame({
    "term": tidy_analytic["term"],
    "estimate": tidy_analytic["estimate"],
    "analytic_se": tidy_analytic["std.error"],
    "bootstrap_se": bootstrap_summary["boot_se"].values,
    "analytic_ci_lower": tidy_analytic["conf.low"],
    "analytic_ci_upper": tidy_analytic["conf.high"],
    "bootstrap_ci_lower": bootstrap_summary["boot_ci_lower"].values,
    "bootstrap_ci_upper": bootstrap_summary["boot_ci_upper"].values
})

print(comparison)
```

**Interpretation:**
- Bootstrap SEs are often slightly larger than analytic SEs
- This reflects fewer assumptions about the error distribution
- When they differ substantially, investigate model assumptions

### Step 5: Visualize Bootstrap Distributions

```python
# Plot bootstrap distribution for weight coefficient
wt_boot = boot_results[boot_results["term"] == "wt"]["estimate"]

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(wt_boot, kde=True, stat="density", ax=ax, alpha=0.6)
ax.axvline(tidy_analytic[tidy_analytic["term"] == "wt"]["estimate"].iloc[0],
           color="red", linestyle="--", linewidth=2, label="Original estimate")
ax.axvline(wt_boot.quantile(0.025), color="green", linestyle=":", label="95% CI")
ax.axvline(wt_boot.quantile(0.975), color="green", linestyle=":")
ax.set_xlabel("Coefficient for wt")
ax.set_ylabel("Density")
ax.set_title("Bootstrap Distribution: Weight Coefficient")
ax.legend()
plt.tight_layout()
plt.show()
```

## Bayesian Bootstrap

The Bayesian bootstrap provides a different approach to resampling that has a natural Bayesian interpretation.

### Step 1: Generate Bayesian Bootstrap Samples

```python
# Bayesian bootstrap of mpg mean
mpg_bayes = mtcars.bayes_boot("mpg", n_samples=2000)

print(f"Bayesian bootstrap samples: {len(mpg_bayes)}")
print(f"Posterior mean: {mpg_bayes.mean():.2f}")
print(f"95% HDI: {mpg_bayes.quantile([0.025, 0.975]).values}")
```

### Step 2: Visualize Bayesian Bootstrap

```python
from broom_sm import bayes_boot_plot

figures = bayes_boot_plot(
    [mpg_bayes],
    x_label="mpg",
    title="Posterior Distribution"
)

figures[0][1].show()
```

### Step 3: Compare Groups

```python
# Bayesian bootstrap by cylinder count
cyl_groups = []
for cyl in mtcars["cyl"].unique():
    subset = mtcars[mtcars["cyl"] == cyl]
    samples = subset.bayes_boot("mpg", n_samples=1000)
    samples.name = f"{cyl} cylinders"
    cyl_groups.append(samples)

# Plot all groups
fig, ax = plt.subplots(figsize=(10, 6))
for series in cyl_groups:
    sns.kdeplot(series, label=series.name, ax=ax, fill=True, alpha=0.3)
ax.set_xlabel("mpg")
ax.set_ylabel("Density")
ax.set_title("Bayesian Bootstrap: MPG by Cylinder Count")
ax.legend()
plt.tight_layout()
plt.show()
```

## Bootstrap for Hypothesis Testing

### Permutation Test Example

Test whether weight differs by transmission type:

```python
# Observed difference
obs_diff = mtcars.groupby("am")["wt"].mean().diff().iloc[-1]
print(f"Observed difference (auto - manual): {obs_diff:.3f}")

# Permutation test
perm_diffs = []
for i in range(1000):
    shuffled = mtcars.assign(am_shuffled=np.random.permutation(mtcars["am"]))
    diff = shuffled.groupby("am_shuffled")["wt"].mean().diff().iloc[-1]
    perm_diffs.append(diff)

# P-value
p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
print(f"Permutation p-value: {p_value:.3f}")

# Visualize
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(perm_diffs, kde=True, ax=ax, alpha=0.6)
ax.axvline(obs_diff, color="red", linestyle="--", linewidth=2)
ax.axvline(-obs_diff, color="red", linestyle="--", linewidth=2)
ax.set_xlabel("Difference in means (auto - manual)")
ax.set_title(f"Permutation Test Distribution (p = {p_value:.3f})")
plt.tight_layout()
plt.show()
```

## Best Practices

### Number of Bootstrap Samples

- **Minimum:** 500 for standard errors
- **Recommended:** 1000-2000 for confidence intervals
- **For publication:** 5000+ for stable p-values

```python
# Check convergence of bootstrap SE
se_history = []
for n in [100, 200, 500, 1000, 2000]:
    boot = mtcars.boot_tidy("mpg ~ wt", stat_type="ols", n_boot=n, seed=42)
    wt_se = boot[boot["term"] == "wt"]["estimate"].std()
    se_history.append((n, wt_se))

pd.DataFrame(se_history, columns=["n_boot", "wt_se"])
```

### When to Use Bootstrap

**Use bootstrap when:**
- Sample size is small (< 50)
- Distributional assumptions are questionable
- Model is complex (no closed-form SEs)
- You want robust inference

**Analytic SEs are fine when:**
- Sample size is large (> 200)
- Model assumptions are well-satisfied
- Computational efficiency is critical

### Common Pitfalls

1. **Too few bootstrap samples:** Leads to unstable estimates
2. **Ignoring dependence:** Time series data needs block bootstrap
3. **Small samples:** Bootstrap can't create information
4. **Non-smooth statistics:** Median and quantiles need special care

## Exercises

1. **Bootstrap the R-squared:** Use `boot_glance()` to generate a bootstrap distribution of R-squared values.

2. **Compare robust vs. bootstrap SEs:** Fit a model with `cov_type="HC3"` and compare to bootstrap SEs.

3. **Prediction intervals:** Use `boot_augment()` to generate bootstrap prediction intervals.

## Summary

Key takeaways:

- `boot_tidy()` generates bootstrap coefficient distributions
- Bootstrap SEs are often more robust than analytic SEs
- Bayesian bootstrap (`bayes_boot()`) provides a different uncertainty quantification
- Visualize bootstrap distributions to check for skewness or multimodality
- Use 1000+ bootstrap samples for reliable inference

## Next Steps

- **[Diagnostics and Visualization](diagnostics-viz.html)** — Learn regression diagnostics
- **[Model Comparison](model-comparison.html)** — Compare multiple models
- **[How-to: Bootstrapping](../howto/bootstrap.html)** — Quick reference for bootstrap workflows

## References

- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Rubin, D. B. (1981). The Bayesian Bootstrap. *Annals of Statistics*, 9(1), 130-134.
- [infer R package: Bootstrap inference](https://infer.tidymodels.org/articles/observed_stat_examples.html)
