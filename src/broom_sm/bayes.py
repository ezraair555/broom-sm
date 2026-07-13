"""Bayesian bootstrap utilities."""
from __future__ import annotations

from typing import List, Sequence, Tuple

try:
    import bayesian_bootstrap as bb
except ImportError:
    bb = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

import pandas as pd
import pandas_flavor as pf
import warnings


@pf.register_dataframe_method
def bayes_boot(df: pd.DataFrame, target_column: str, n_samples: int) -> pd.Series:
    if bb is None:
        raise ImportError(
            "The 'bayesian_bootstrap' package is required for Bayesian bootstrap helpers. "
            "Please install it using 'pip install broom-sm[bayes]'."
        )
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
    if bb is None or plt is None or sns is None:
        raise ImportError(
            "Visualization and Bayesian bootstrap dependencies are required for bayes_boot_plot. "
            "Please install them using 'pip install broom-sm[viz,bayes]'."
        )
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
