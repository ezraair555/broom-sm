# Bootstrapping workflows

Use the `boot_*` helpers for quick resampling pipelines. Each helper mirrors the tidy verbs, returning one tidy/glance/augment DataFrame per resample and tagging results with `.bootstrap_id`.

```python
boot = penguins.boot_tidy(
    formula="body_mass_g ~ flipper_length_mm",
    stat_type="ols",
    n_boot=500,
    seed=11,
)
boot.groupby("term")["estimate"].agg(["mean", "std"])
```

Best practices:

- **Log noise, don't print** — bootstrap helpers emit warnings via the `broom_sm` logger. Set `PYTHONWARNINGS` or configure logging to keep notebooks clean.
- **Preserve kwargs** — pass the same `cov_type`, `family`, `weights`, etc. you would feed into `stats_tidy`. They are forwarded to every resample.
- **Plot distributions** — pipe the output into seaborn/kde plots or use the Bayesian helpers (`broom_sm.bayes.bayes_boot`) for highest density intervals.
- **Attach augment data** — `boot_augment` is handy for simulating prediction intervals across refits.
