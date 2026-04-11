# broom-sm

A tidy-minded statistical helper package for Python that blends:

- **infer (R)** – declarative resampling-based inference.
- **DABEST** – estimation statistics and Gardner-Altman style plots.
- **Pingouin** – publication-friendly statistical tables & diagnostics.
- **broom (R)** – tidy/augment/glance helpers for model objects.

## Installation

```bash
pip install -e .
```

Dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `pingouin`, `dabest`, `statsmodels`.

## Modules

### `broom_sm.infer`
```python
from broom_sm import specify, hypothesize, generate, calculate, confidence_interval

pipeline = (
    specify(df, response="outcome", explanatory="group")
    .hypothesize(null="independence")
    .generate(reps=5000, type="permute")
)
null_dist = calculate(pipeline, stat="diff in means")
ci = confidence_interval(pipeline, stat="diff in means")
```

### `broom_sm.estimation`
```python
from broom_sm.estimation import load_dabest, estimation_plot, estimation_summary

est = load_dabest(df, idx=["control", "treated"], x="group", y="score")
fig = estimation_plot(est, contrast="mean_diff")
summary = estimation_summary(est)
```

### `broom_sm.reporting`
```python
from broom_sm.reporting import tidy, glance, augment
model = sm.OLS.from_formula("y ~ x", data=df).fit()
coef_table = tidy(model)
model_stats = glance(model)
augmented = augment(model, df)
```

### `broom_sm.plotting`
```python
from broom_sm.plotting import plot_bootstrap_distribution, pingouin_table
fig = plot_bootstrap_distribution(null_dist["stat"])
anova_table = pingouin_table(df, dv="score", between="group")
```

## Documentation
See docstrings within each module & the examples above for core workflows.
