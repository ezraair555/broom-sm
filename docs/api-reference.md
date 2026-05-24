# API Reference

This page documents all public functions in broom-sm. Functions are organized by module.

## Core Verbs (`broom_sm.tidy`)

### `stats_tidy()`

Convert a fitted model into a tidy coefficient DataFrame.

**Signature:**
```python
df.stats_tidy(
    formula: Optional[str] = None,
    stat_type: Optional[str] = None,
    *,
    model: Any = None,
    family: Any = None,
    link: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
    cov_type: Optional[str] = None,
    cov_kwds: Optional[Dict[str, Any]] = None,
    alpha: float = 0.05,
    **kwargs
) -> pd.DataFrame
```

**Parameters:**
- `formula` (str, optional): Model formula (e.g., `"y ~ x1 + x2"`)
- `stat_type` (str, optional): Model type (e.g., `"ols"`, `"glm"`, `"poisson"`)
- `model` (any, optional): Pre-fitted statsmodels result object
- `family` (any, optional): GLM family specification
- `link` (str, optional): GLM link function
- `weights` (sequence, optional): Observation weights
- `cov_type` (str, optional): Covariance type for robust SEs (e.g., `"HC3"`, `"cluster"`)
- `cov_kwds` (dict, optional): Keyword arguments for covariance estimation
- `alpha` (float, default=0.05): Significance level for confidence intervals
- `**kwargs`: Additional arguments passed to the model fitter

**Returns:**
DataFrame with columns:
- `term`: Predictor name
- `estimate`: Coefficient estimate
- `std.error`: Standard error
- `conf.low`, `conf.high`: Confidence interval bounds
- `statistic`: Test statistic (t, z, etc.)
- `p.value`: P-value
- `cov_type`: Covariance type used
- `family`, `link`: GLM family and link (if applicable)
- `stat_type`: Model type

**Examples:**
```python
# Formula interface
mtcars.stats_tidy("mpg ~ wt + hp", stat_type="ols")

# With robust SEs
mtcars.stats_tidy("mpg ~ wt", stat_type="ols", cov_type="HC3")

# Pre-fitted model
model = smf.ols("mpg ~ wt", data=mtcars).fit()
mtcars.stats_tidy(model=model)
```

---

### `stats_glance()`

Extract model-level statistics into a one-row tidy DataFrame.

**Signature:**
```python
df.stats_glance(
    formula: Optional[str] = None,
    stat_type: Optional[str] = None,
    *,
    model: Any = None,
    family: Any = None,
    link: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
    cov_type: Optional[str] = None,
    cov_kwds: Optional[Dict[str, Any]] = None,
    **kwargs
) -> pd.DataFrame
```

**Parameters:**
Same as `stats_tidy()`.

**Returns:**
DataFrame with one row and columns such as:
- `stat_type`: Model type
- `nobs`: Number of observations
- `llf`: Log-likelihood
- `aic`, `bic`: Information criteria
- `df_model`, `df_resid`: Degrees of freedom
- `rsquared`, `rsquared_adj`: R-squared values (OLS)
- `f_statistic`, `f_pvalue`: F-test results (OLS)
- `deviance`, `pearson_chi2`: Deviance statistics (GLM)
- `family`, `link`: GLM specifications
- `cov_type`: Covariance type

**Examples:**
```python
mtcars.stats_glance("mpg ~ wt + hp", stat_type="ols")
#>   stat_type  nobs      llf       aic       bic  df_model  df_resid  rsquared
#> 0       ols    32 -72.54928  153.09856  158.95938       2.0      29.0  0.826783
```

---

### `stats_augment()`

Add predictions, residuals, and diagnostic columns to the data.

**Signature:**
```python
df.stats_augment(
    formula: Optional[str] = None,
    stat_type: Optional[str] = None,
    *,
    model: Any = None,
    new_data: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
    include_prediction_intervals: bool = True,
    cov_type: Optional[str] = None,
    cov_kwds: Optional[Dict[str, Any]] = None,
    **kwargs
) -> pd.DataFrame
```

**Parameters:**
- `formula`, `stat_type`, `model`, `cov_type`, `cov_kwds`: As in `stats_tidy()`
- `new_data` (DataFrame, optional): New data for out-of-sample predictions
- `alpha` (float, default=0.05): Significance level for prediction intervals
- `include_prediction_intervals` (bool, default=True): Whether to compute intervals

**Returns:**
DataFrame with original columns plus:
- `.fitted`: Fitted/predicted values
- `.resid`: Residuals (in-sample only)
- `.hat`: Leverage values (in-sample)
- `.cooksd`: Cook's distance (in-sample)
- `.std.resid`: Studentized residuals (in-sample)
- `.mean_se`: Standard error of mean prediction
- `.obs_ci_lower`, `.obs_ci_upper`: Observation confidence interval
- `.mean_ci_lower`, `.mean_ci_upper`: Mean confidence interval
- `.in_sample`: Boolean indicating in-sample observations
- `stat_type`: Model type

**Examples:**
```python
# In-sample augmentation
aug = mtcars.stats_augment("mpg ~ wt + hp", stat_type="ols")
aug[["mpg", ".fitted", ".resid"]].head()

# Out-of-sample predictions
new_data = mtcars.sample(5, random_state=42)
mtcars.stats_augment("mpg ~ wt", stat_type="ols", new_data=new_data)
```

---

### `stats_anova_tidy()`

Compute ANOVA table in tidy format.

**Signature:**
```python
df.stats_anova_tidy(
    formula: str,
    anova_type: int = 2,
    **kwargs
) -> pd.DataFrame
```

**Parameters:**
- `formula` (str): Model formula
- `anova_type` (int, default=2): Type of ANOVA (1, 2, or 3)
- `**kwargs`: Passed to `statsmodels.formula.api.ols()`

**Returns:**
DataFrame with columns:
- `term`: Factor name
- `sum_sq`: Sum of squares
- `df`: Degrees of freedom
- `F`: F-statistic
- `p.value`: P-value

**Examples:**
```python
mtcars.stats_anova_tidy("mpg ~ factor(cyl)", anova_type=2)
```

---

### `stats_kruskal_tidy()`

Perform Kruskal-Wallis rank-sum test.

**Signature:**
```python
df.stats_kruskal_tidy(
    value_col: str,
    group_col: str
) -> pd.DataFrame
```

**Parameters:**
- `value_col` (str): Name of the numeric response column
- `group_col` (str): Name of the grouping column

**Returns:**
DataFrame with columns:
- `statistic`: Kruskal-Wallis H statistic
- `p.value`: P-value
- `df`: Degrees of freedom

**Examples:**
```python
mtcars.stats_kruskal_tidy(value_col="mpg", group_col="cyl")
```

---

### `stats_correlation_tidy()`

Compute correlation matrix in tidy format.

**Signature:**
```python
df.stats_correlation_tidy(
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    *,
    method: str = "pearson",
    columns: Optional[Sequence[str]] = None
) -> pd.DataFrame
```

**Parameters:**
- `col1`, `col2` (str, optional): Specific columns to correlate
- `method` (str, default="pearson"): Correlation method ("pearson" or "spearman")
- `columns` (sequence, optional): Columns to include in correlation matrix

**Returns:**
DataFrame with columns:
- `term1`, `term2`: Variable names
- `correlation`: Correlation coefficient
- `p.value`: P-value

**Examples:**
```python
# Correlation between two columns
mtcars.stats_correlation_tidy(col1="mpg", col2="wt")

# Full correlation matrix
mtcars.stats_correlation_tidy(method="spearman")
```

---

### `stats_formula()`

Generate a formula string from a DataFrame.

**Signature:**
```python
df.stats_formula(
    target: str,
    *exclude: str
) -> str
```

**Parameters:**
- `target` (str): Response variable name
- `*exclude`: Predictor names to exclude

**Returns:**
Formula string (e.g., `"mpg ~ wt + hp + cyl"`)

**Examples:**
```python
mtcars.stats_formula("mpg", "qsec")
#> "mpg ~ wt + hp + cyl + drat + disp + carb + gear + am + vs"
```

---

### `stats_partial_dependence()`

Compute partial dependence for a fitted model.

**Signature:**
```python
stats_partial_dependence(
    model: Any,
    feature: str,
    data: pd.DataFrame,
    grid: Optional[Sequence[float]] = None,
    at: Optional[Dict[str, float]] = None
) -> pd.DataFrame
```

**Parameters:**
- `model`: Fitted statsmodels result
- `feature` (str): Feature to compute partial dependence for
- `data` (DataFrame): Data used for marginalization
- `grid` (sequence, optional): Grid of feature values
- `at` (dict, optional): Fixed values for other features

**Returns:**
DataFrame with columns:
- `feature`: Feature name
- `value`: Feature value
- `prediction`: Average prediction at that value

**Examples:**
```python
model = smf.ols("mpg ~ wt + hp", data=mtcars).fit()
stats_partial_dependence(model, "wt", mtcars)
```

---

## Bootstrap Functions (`broom_sm.bootstrap`)

### `boot_tidy()`

Bootstrap coefficient estimates.

**Signature:**
```python
df.boot_tidy(
    formula: str,
    stat_type: str,
    n_boot: int,
    *,
    seed: Optional[int] = None,
    **kwargs
) -> pd.DataFrame
```

**Parameters:**
- `formula`, `stat_type`: As in `stats_tidy()`
- `n_boot` (int): Number of bootstrap samples
- `seed` (int, optional): Random seed for reproducibility
- `**kwargs`: Passed to `stats_tidy()`

**Returns:**
DataFrame with all columns from `stats_tidy()` plus:
- `.bootstrap_id`: Bootstrap sample identifier (0 to n_boot-1)

**Examples:**
```python
boot = mtcars.boot_tidy(
    formula="mpg ~ wt",
    stat_type="ols",
    n_boot=500,
    seed=42
)
boot.groupby("term")["estimate"].agg(["mean", "std"])
```

---

### `boot_glance()`

Bootstrap model-level statistics.

**Signature:**
```python
df.boot_glance(
    formula: str,
    stat_type: str,
    n_boot: int,
    *,
    seed: Optional[int] = None,
    **kwargs
) -> pd.DataFrame
```

**Returns:**
DataFrame with all columns from `stats_glance()` plus `.bootstrap_id`.

---

### `boot_augment()`

Bootstrap augmented data (predictions).

**Signature:**
```python
df.boot_augment(
    formula: str,
    stat_type: str,
    n_boot: int,
    *,
    seed: Optional[int] = None,
    **kwargs
) -> pd.DataFrame
```

**Returns:**
DataFrame with all columns from `stats_augment()` plus `.bootstrap_id`.

---

## Diagnostics (`broom_sm.diagnostics`)

### `stats_power()`

Plot power curves for two-sample t-test.

**Signature:**
```python
stats_power(
    effect_size: float,
    alpha: float,
    *,
    power: Optional[float] = None,
    obs_range: Tuple[int, int] = (5, 200),
    nobs: Optional[int] = None
) -> Tuple[plt.Figure, plt.Axes, Optional[float]]
```

**Returns:**
- `fig`: Matplotlib figure
- `ax`: Matplotlib axes
- `required`: Required sample size per group (if `power` specified)

**Examples:**
```python
fig, ax, required = stats_power(effect_size=0.5, alpha=0.05, power=0.8)
```

---

### `stats_residual_plot()`

Create residual diagnostic plots.

**Signature:**
```python
df.stats_residual_plot(
    x: Sequence[str],
    y: str
) -> List[Tuple[str, plt.Figure]]
```

**Returns:**
List of (feature_name, figure) tuples. Each figure contains scatter and QQ plots.

**Examples:**
```python
figures = mtcars.stats_residual_plot(["wt", "hp"], y="mpg")
for feature, fig in figures:
    fig.savefig(f"residual_{feature}.png")
```

---

### `stats_ols_plot()`

Create OLS scatter plots with regression line.

**Signature:**
```python
df.stats_ols_plot(
    x: Sequence[str],
    y: str
) -> List[Tuple[str, plt.Figure]]
```

**Returns:**
List of (feature_name, figure) tuples with joint plots.

---

### `stats_influence_plot()`

Create influence plot for regression diagnostics.

**Signature:**
```python
df.stats_influence_plot(
    formula: str,
    stat_type: str = "ols",
    alpha: float = 0.05,
    **kwargs
) -> plt.Figure
```

**Examples:**
```python
fig = mtcars.stats_influence_plot("mpg ~ wt + hp", stat_type="ols")
```

---

### `stats_chisquare_plot()`

Perform chi-square test and visualize residuals.

**Signature:**
```python
df.stats_chisquare_plot(
    x: str,
    y: str
) -> Tuple[pd.DataFrame, plt.Figure]
```

**Returns:**
- DataFrame with chi-square statistics
- Heatmap figure of observed - expected residuals

---

### `stats_vif()`

Calculate Variance Inflation Factors.

**Signature:**
```python
df.stats_vif(
    formula: Optional[str] = None
) -> pd.Series
```

**Parameters:**
- `formula` (str, optional): Model formula. If omitted, uses all numeric columns.

**Returns:**
Series with VIF values for each predictor.

**Examples:**
```python
mtcars.stats_vif("mpg ~ wt + hp + cyl")
```

---

### `stats_conprob()`

Compute conditional probabilities from contingency tables.

**Signature:**
```python
df.stats_conprob(
    var_A_name: str,
    var_B_name: str,
    *,
    transpose: bool = False
) -> pd.DataFrame
```

**Returns:**
DataFrame with columns:
- Event variable
- Given variable
- `probability`: Conditional probability
- `marginal`: Marginal probability

---

### `stats_coef_forest()`

Create coefficient forest plot.

**Signature:**
```python
stats_coef_forest(
    tidy_df: pd.DataFrame,
    *,
    reference_line: float = 0.0,
    sort: bool = True
) -> Tuple[plt.Figure, plt.Axes]
```

**Parameters:**
- `tidy_df`: Output from `stats_tidy()`
- `reference_line`: Null value for reference line (default 0)
- `sort`: Whether to sort by estimate

**Examples:**
```python
tidy = mtcars.stats_tidy("mpg ~ wt + hp", stat_type="ols")
fig, ax = stats_coef_forest(tidy)
```

---

## Bayesian Functions (`broom_sm.bayes`)

### `bayes_boot()`

Perform Bayesian bootstrap on a column.

**Signature:**
```python
df.bayes_boot(
    target_column: str,
    n_samples: int
) -> pd.Series
```

**Returns:**
Series of bootstrap samples from the posterior distribution of the mean.

**Examples:**
```python
samples = mtcars.bayes_boot("mpg", n_samples=2000)
samples.quantile([0.025, 0.5, 0.975])
```

---

### `bayes_boot_plot()`

Plot Bayesian bootstrap distributions.

**Signature:**
```python
bayes_boot_plot(
    series_list: Sequence[pd.Series],
    x_label: str,
    title: str
) -> List[Tuple[str, plt.Figure]]
```

**Returns:**
List of (series_name, figure) tuples with histogram and 95% HDI.

---

## Reporting Functions (`broom_sm.reporting`)

### `stats_compare()`

Compare multiple models using information criteria.

**Signature:**
```python
stats_compare(
    models: Mapping[str, Any]
) -> pd.DataFrame
```

**Parameters:**
- `models`: Dict mapping model names to fitted statsmodels results

**Returns:**
DataFrame sorted by AIC with columns:
- `model`: Model name
- `aic`, `bic`: Information criteria
- `llf`: Log-likelihood
- `nobs`: Number of observations

**Examples:**
```python
models = {
    "simple": smf.ols("mpg ~ wt", data=mtcars).fit(),
    "full": smf.ols("mpg ~ wt + hp", data=mtcars).fit()
}
stats_compare(models)
```

---

### `stats_report()`

Generate complete model report with tidy, glance, and augment outputs.

**Signature:**
```python
stats_report(
    data: pd.DataFrame,
    *,
    formula: str,
    stat_type: str,
    tidy_kwargs: Optional[Mapping[str, Any]] = None,
    glance_kwargs: Optional[Mapping[str, Any]] = None,
    augment_kwargs: Optional[Mapping[str, Any]] = None
) -> Dict[str, pd.DataFrame]
```

**Returns:**
Dictionary with keys:
- `"tidy"`: Coefficient table
- `"glance"`: Model summary
- `"augment"`: Predictions and residuals

**Examples:**
```python
report = mtcars.stats_report(
    formula="mpg ~ wt + hp",
    stat_type="ols",
    tidy_kwargs={"cov_type": "HC3"}
)
report["tidy"]
report["glance"]
report["augment"]
```

---

## Model Registry (`broom_sm.model_registry`)

### `register_model()`

Register a new model type in MODEL_CONFIG.

**Signature:**
```python
register_model(
    stat_type: str,
    spec: ModelSpec
) -> None
```

**Examples:**
```python
from broom_sm.model_registry import ModelSpec, register_model
import statsmodels.formula.api as smf

register_model(
    "tobit",
    ModelSpec(
        fitter=lambda formula, data, **kwargs: smf.tobit(formula, data=data, **kwargs).fit(),
        stat_name="z_stat",
        metadata={"aliases": ["censored"]}
    )
)
```

---

### `get_model_spec()`

Retrieve model specification.

**Signature:**
```python
get_model_spec(stat_type: str) -> Optional[ModelSpec]
```

---

### `iter_model_config()`

Iterate over registered model types.

**Signature:**
```python
iter_model_config() -> Iterator[str]
```

**Examples:**
```python
list(iter_model_config())
#> ['ols', 'glm', 'poisson', 'gamma', 'negbin', 'gee', 'mixedlm', 'phreg', 'quantreg']
```
