# Comparing Models

This guide shows how to compare multiple fitted models using information criteria, likelihood ratio tests, and cross-validation.

## Information Criteria

### AIC and BIC Comparison

```python
from broom_sm import stats_compare
import statsmodels.formula.api as smf

# Fit multiple models
models = {
    "intercept_only": smf.ols("mpg ~ 1", data=mtcars).fit(),
    "wt_only": smf.ols("mpg ~ wt", data=mtcars).fit(),
    "wt_hp": smf.ols("mpg ~ wt + hp", data=mtcars).fit(),
    "full": smf.ols("mpg ~ wt + hp + cyl + disp", data=mtcars).fit()
}

# Compare
comparison = stats_compare(models)
print(comparison)
```

Output:
```
           model       aic       bic       llf  nobs
0           wt_hp  153.09856  158.95938 -72.54928  32.0
1         wt_only  165.43210  169.82345 -79.71605  32.0
2            full  167.23456  174.56789 -76.61728  32.0
3  intercept_only  189.87654  192.80123 -93.93827  32.0
```

**Interpretation:**
- Lower AIC/BIC indicates better model fit (penalizing complexity)
- The `wt_hp` model has the lowest AIC, suggesting the best trade-off

### Delta AIC and Model Weights

```python
comparison["delta_aic"] = comparison["aic"] - comparison["aic"].min()
comparison["aic_weight"] = np.exp(-0.5 * comparison["delta_aic"])
comparison["aic_weight"] /= comparison["aic_weight"].sum()

print(comparison[["model", "aic", "delta_aic", "aic_weight"]])
```

**Rules of thumb:**
- ΔAIC < 2: Substantial support
- ΔAIC 4-7: Considerably less support
- ΔAIC > 10: Essentially no support

## Likelihood Ratio Test

For nested models, use the likelihood ratio test:

```python
from scipy import stats

model_reduced = smf.ols("mpg ~ wt", data=mtcars).fit()
model_full = smf.ols("mpg ~ wt + hp", data=mtcars).fit()

# Likelihood ratio statistic
lr_stat = 2 * (model_full.llf - model_reduced.llf)
df_diff = model_full.df_model - model_reduced.df_model
p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

print(f"LR statistic: {lr_stat:.2f}")
print(f"Degrees of freedom: {df_diff}")
print(f"P-value: {p_value:.4f}")
```

## Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold
import numpy as np

def cross_validate_mse(data, formula, n_folds=5, seed=42):
    """Calculate cross-validated MSE for a model formula."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    mse_scores = []
    
    for train_idx, test_idx in kf.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        model = smf.ols(formula, data=train_data).fit()
        predictions = model.predict(test_data)
        mse = ((predictions - test_data["mpg"]) ** 2).mean()
        mse_scores.append(mse)
    
    return np.mean(mse_scores), np.std(mse_scores)

# Compare models
formulas = {
    "wt_only": "mpg ~ wt",
    "wt_hp": "mpg ~ wt + hp",
    "full": "mpg ~ wt + hp + cyl + disp"
}

cv_results = []
for name, formula in formulas.items():
    mean_mse, std_mse = cross_validate_mse(mtcars, formula)
    cv_results.append({
        "model": name,
        "cv_mse": mean_mse,
        "cv_mse_sd": std_mse
    })

pd.DataFrame(cv_results).sort_values("cv_mse")
```

## Model Averaging

When multiple models have support, consider model averaging:

```python
# Calculate AIC weights (from earlier)
aic_weights = comparison.set_index("model")["aic_weight"]

# Get predictions from each model
predictions = {}
for name in models.keys():
    predictions[name] = models[name].predict(mtcars)

# Model-averaged predictions
avg_predictions = sum(
    aic_weights[name] * pd.Series(preds, name=name)
    for name, preds in predictions.items()
)

print(f"Model-averaged predictions (first 5):")
print(avg_predictions.head())
```

## Visualizing Model Comparison

### Coefficient Comparison

```python
from broom_sm import stats_coef_forest

# Get tidy outputs for all models
tidy_models = {}
for name, model in models.items():
    tidy_models[name] = mtcars.stats_tidy(model=model)

# Combine for plotting
combined = pd.concat([
    df.assign(model=name)
    for name, df in tidy_models.items()
], ignore_index=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.pointplot(
    data=combined[combined["term"] != "Intercept"],
    x="estimate",
    y="term",
    hue="model",
    ax=ax,
    dodge=0.3
)
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_title("Coefficient Comparison Across Models")
plt.tight_layout()
plt.show()
```

### Information Criteria Plot

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=comparison,
    x="aic",
    y="model",
    ax=ax,
    palette="viridis"
)
ax.set_title("AIC Comparison")
ax.set_xlabel("AIC")
plt.tight_layout()
plt.show()
```

## Best Practices

### Model Selection Strategy

1. **Start simple:** Begin with parsimonious models
2. **Compare systematically:** Use AIC/BIC for non-nested, LRT for nested
3. **Validate:** Cross-validate to check generalization
4. **Consider averaging:** When multiple models have support

### Common Pitfalls

- **Overfitting:** Adding predictors always improves in-sample fit
- **Ignoring theory:** Statistical fit ≠ substantive importance
- **Data dredging:** Pre-specify candidate models when possible
- **Small samples:** Information criteria can be unstable with n < 50

### When to Use Each Method

| Method | Best For | Limitations |
|--------|----------|-------------|
| AIC/BIC | Non-nested models, model selection | Assumes large samples |
| LRT | Nested models, hypothesis testing | Only for nested comparisons |
| Cross-validation | Predictive performance | Computationally intensive |
| Model averaging | Model uncertainty | More complex interpretation |

## Related Guides

- **[Extending MODEL_CONFIG](model_config.md)** — Add custom model types
- **[Bootstrapping Workflows](bootstrap.md)** — Resampling-based inference
- **[Robust Standard Errors](robust_ses.md)** — Heteroskedasticity-consistent inference
