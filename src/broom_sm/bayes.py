"""Bayesian bootstrap utilities."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import bayesian_bootstrap as bb
import matplotlib.pyplot as plt
import pandas as pd
import pandas_flavor as pf
import seaborn as sns
import warnings


@pf.register_dataframe_method
def bayes_boot(df: pd.DataFrame, target_column: str, n_samples: int) -> pd.Series:
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in data columns: {list(df.columns)}")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer; got {n_samples}")
    target = df[target_column]
    n_missing = target.isna().sum()
    if n_missing:
        warnings.warn(
            f"Dropping {n_missing} NaN value(s) from '{target_column}' before Bayesian bootstrap. "
            f"Effective sample size is {len(target) - n_missing}.",
            stacklevel=2,
        )
    sample = bb.mean(target.dropna().values, n_samples)
    return pd.Series(sample, name=f"{target_column}_bayes_boot")


def bayes_boot_plot(series_list: Sequence[pd.Series], x_label: str, title: str) -> List[Tuple[str, plt.Figure]]:
    figures: List[Tuple[str, plt.Figure]] = []
    for series in series_list:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(series, kde=True, stat="density", ax=ax, label="Bayesian Bootstrap")
        hdi = bb.highest_density_interval(series.values)
        ax.plot(hdi, [0, 0], linewidth=4.0, marker="o", label="95% HDI")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density")
        ax.set_title(f"{title} — {series.name}")
        ax.legend()
        sns.despine(ax=ax)
        figures.append((series.name, fig))
    return figures
