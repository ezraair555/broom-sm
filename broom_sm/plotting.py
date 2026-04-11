"""broom_sm.plotting - Visualization helpers built on matplotlib & pingouin."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

import pingouin as pg


def plot_bootstrap_distribution(dist: pd.Series, figsize: Tuple[int, int] = (8, 4), ci: float = 0.95) -> plt.Figure:
    """Histogram of bootstrap statistics with CI band."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(dist, bins=40, alpha=0.7, color="#2563eb")
    alpha = 1 - ci
    lower, upper = dist.quantile([alpha / 2, 1 - alpha / 2])
    ax.axvline(lower, color="red", linestyle="--", label=f"{ci*100:.1f}% CI")
    ax.axvline(upper, color="red", linestyle="--")
    ax.set_xlabel("Statistic")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap Sampling Distribution")
    ax.legend()
    return fig


def pingouin_table(data: pd.DataFrame, dv: str, between: str) -> pd.DataFrame:
    """Return a Pingouin ANOVA summary formatted for reporting."""
    anova = pg.anova(data=data, dv=dv, between=between, detailed=True)
    # prettify column names
    return anova.rename(
        columns={
            "np2": "eta_sq_partial",
            "p-unc": "p_value",
            "SS": "sum_sq",
            "MS": "mean_sq",
        }
    )


def pingouin_mean_diff_table(data: pd.DataFrame, dv: str, between: str) -> pd.DataFrame:
    """Pairwise comparison table using Pingouin's pairwise_ttests."""
    pw = pg.pairwise_ttests(data=data, dv=dv, between=between, padjust="holm")
    cols = {
        "A": "group1",
        "B": "group2",
        "hedges": "hedges_g",
        "p-corr": "p_adj",
    }
    return pw.rename(columns=cols)[
        ["group1", "group2", "T", "dof", "hedges_g", "p_adj", "CI95%"]
    ]
