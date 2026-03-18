# Tidy workflow tutorial

This MyST tutorial walks through an end-to-end regression analysis with broom-sm.

## 1. Load data

```python
import pandas as pd
from broom_sm import stats_report, stats_coef_forest

penguins = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv").dropna()
```

## 2. Fit once, reuse everywhere

```python
report = penguins.stats_report(
    formula="body_mass_g ~ flipper_length_mm + bill_length_mm + sex",
    stat_type="ols",
    tidy_kwargs={"cov_type": "HC3"},
)
report["tidy"].head()
```

## 3. Visualize coefficients

```python
fig, ax = stats_coef_forest(report["tidy"])
fig
```

## 4. Augment for predictions

```python
new_penguins = penguins.sample(10, random_state=21)
preds = penguins.stats_augment(
    formula="body_mass_g ~ flipper_length_mm + bill_length_mm + sex",
    stat_type="ols",
    new_data=new_penguins,
    alpha=0.1,
)
preds[[".fitted", ".mean_ci_lower", ".mean_ci_upper"]]
```

## 5. Compare alternative models

```python
from broom_sm.reporting import stats_compare
import statsmodels.formula.api as smf

models = {
    "basic": smf.ols("body_mass_g ~ flipper_length_mm", data=penguins).fit(),
    "full": smf.ols("body_mass_g ~ flipper_length_mm + bill_length_mm + sex", data=penguins).fit(),
}
stats_compare(models)
```

This workflow produces publishable tables, diagnostic plots, and interval-aware predictions without refitting models repeatedly.
