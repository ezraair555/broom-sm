# Working with robust standard errors

All tidy verbs accept `cov_type` and `cov_kwds` and expose the selection in their outputs.

```python
report = df.stats_report(
    formula="mpg ~ wt + hp",
    stat_type="ols",
    tidy_kwargs={"cov_type": "HC3"},
)
report["tidy"].filter(["term", "estimate", "std.error", "cov_type"])
```

- The `cov_type` string is passed straight to `statsmodels`. Common values: `"HC0"`, `"HC1"`, `"cluster"`, `"hac"`.
- Supply `cov_kwds` for cluster or HAC options, e.g. `cov_kwds={"groups": df["cyl"]}`.
- Robust choices appear in both `stats_tidy` (column) and `stats_glance` (row) so you never lose track of them.

Weights flow through the same plumbing:

```python
df.stats_tidy(
    formula="mpg ~ wt",
    stat_type="glm",
    family="binomial",
    weights=df["weights"],
    cov_type="cluster",
    cov_kwds={"groups": df["manufacturer"]},
)
```

If you attach a pre-fitted statsmodels result (`model=fit`), broom-sm keeps the covariance choice already baked into the result.
