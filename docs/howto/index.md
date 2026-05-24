# How-to Guides

How-to guides answer "How do I...?" questions for specific tasks. Each guide is a self-contained recipe you can adapt to your workflow.

## Available Guides

### Model Configuration

- **[Extending MODEL_CONFIG](model_config.md)** — Register custom model types
- **[Working with GLMs](#glm-workflow)** — Generalized linear models *(coming soon)*
- **[Mixed Effects Models](#mixedlm)** — Hierarchical and longitudinal data *(coming soon)*
- **[Survival Analysis](#phreg)** — Time-to-event modeling *(coming soon)*

### Inference & Diagnostics

- **[Robust Standard Errors](robust_ses.md)** — Heteroskedasticity and clustering
- **[Bootstrapping Workflows](bootstrap.md)** — Resampling-based inference
- **[Plot Gallery](plot_gallery.md)** — Visualization recipes

### Model Selection

- **[Comparing Models](compare-models.md)** — AIC, BIC, and cross-validation

## Quick Reference

### Tidy Verbs

```python
# Coefficient table
df.stats_tidy("y ~ x1 + x2", stat_type="ols")

# Model summary
df.stats_glance("y ~ x1 + x2", stat_type="ols")

# Predictions
df.stats_augment("y ~ x1 + x2", stat_type="ols")
```

### Robust SEs

```python
df.stats_tidy("y ~ x1", stat_type="ols", cov_type="HC3")
df.stats_tidy("y ~ x1", stat_type="ols", cov_type="cluster", cov_kwds={"groups": df["cluster_id"]})
```

### Bootstrap

```python
boot = df.boot_tidy("y ~ x1", stat_type="ols", n_boot=1000, seed=42)
boot.groupby("term")["estimate"].agg(["mean", "std"])
```

### Bayesian Bootstrap

```python
samples = df.bayes_boot("y", n_samples=2000)
samples.quantile([0.025, 0.5, 0.975])
```

### Diagnostics

```python
# VIF
df.stats_vif("y ~ x1 + x2 + x3")

# Influence plot
df.stats_influence_plot("y ~ x1 + x2", stat_type="ols")

# Residual plots
figures = df.stats_residual_plot(["x1", "x2"], y="y")
```

### Model Comparison

```python
from broom_sm import stats_compare

models = {
    "simple": smf.ols("y ~ x1", data=df).fit(),
    "full": smf.ols("y ~ x1 + x2", data=df).fit()
}
stats_compare(models)
```

## Related Resources

- **[Tutorials](../docs/tutorials/index.md)** — End-to-end workflows
- **[API Reference](../api-reference.md)** — Complete function documentation
- **[Quick Start](../quickstart.md)** — Get started in 5 minutes
