"""broom_sm.estimation - High-level wrappers around DABEST's estimation statistics."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional

import dabest


def load_dabest(
    data: pd.DataFrame,
    idx: Union[List[str], List[List[str]]],
    x: str,
    y: str,
    ci: float = 95,
    resamples: int = 5000,
    random_seed: Optional[int] = None,
):
    """Wrapper around dabest.load with sane defaults."""
    return dabest.load(
        data,
        idx=idx,
        x=x,
        y=y,
        ci=ci,
        resamples=resamples,
        random_seed=random_seed,
    )


def estimation_summary(estimator) -> pd.DataFrame:
    """Return tidy effect-size table for the DABEST object."""
    # Works for mean_diff, median_diff, cohen's d, etc.
    frames = []
    for attr in ["mean_diff", "median_diff", "cohens_d", "hedges_g", "cliffs_delta"]:
        if hasattr(estimator, attr):
            effect = getattr(estimator, attr)
            if effect.results is not None:
                df = effect.results.copy()
                df["estimator"] = attr
                frames.append(df)
    if not frames:
        raise ValueError("Estimator does not have summarized results.")
    return pd.concat(frames, ignore_index=True)


def estimation_plot(estimator, contrast: str = "mean_diff", figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """Render estimation plot for the requested contrast."""
    if not hasattr(estimator, contrast):
        raise AttributeError(f"Estimator has no attribute '{contrast}'.")
    fig = getattr(estimator, contrast).plot(figsize=figsize)
    return fig
