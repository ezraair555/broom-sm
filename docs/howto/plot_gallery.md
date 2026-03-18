# Plot gallery

broom-sm returns Matplotlib `Figure` objects so you decide when to display or save plots. A few quick recipes:

## Residual diagnostics

```python
figures = df.stats_residual_plot(["wt", "hp"], y="mpg")
for feature, fig in figures:
    fig.savefig(f"plots/residual_{feature}.png", dpi=150)
```

## Influence plot

```python
fig = df.stats_influence_plot("mpg ~ wt + hp", stat_type="ols")
fig.savefig("plots/influence.png", dpi=150)
```

## Coefficient forest

```python
tidy = df.stats_tidy("mpg ~ wt + hp", stat_type="ols")
fig, ax = stats_coef_forest(tidy)
fig.savefig("plots/coefficients.png", bbox_inches="tight")
```

## Bayesian bootstrap histogram

```python
series = df.bayes_boot("mpg", n_samples=2000)
figs = bayes_boot_plot([series], x_label="mpg", title="Posterior")
```

Because each helper returns the figure/axes objects, you can chain seaborn styling, add annotations, or embed them in dashboards without relying on implicit `plt.show()` state.
