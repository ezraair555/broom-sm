"""Bootstrap resampling helpers."""
from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd
import pandas_flavor as pf

LOGGER = logging.getLogger("broom_sm")


def _bootstrap_apply(
    data: pd.DataFrame,
    n_boot: int,
    seed: Optional[int],
    func: Callable[[pd.DataFrame], pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    frames = []
    for i in range(n_boot):
        current_seed = None if seed is None else seed + i
        sample = data.sample(n=len(data), replace=True, random_state=current_seed)
        try:
            chunk = func(sample)
            chunk[".bootstrap_id"] = i
            frames.append(chunk)
        except Exception as exc:  # pragma: no cover - diagnostics only
            LOGGER.warning("Bootstrap sample %s failed for %s: %s", i, label, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).reset_index(drop=True)


@pf.register_dataframe_method
def boot_tidy(
    data: pd.DataFrame,
    formula: str,
    stat_type: str,
    n_boot: int,
    *,
    seed: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    return _bootstrap_apply(
        data,
        n_boot,
        seed,
        lambda df: df.stats_tidy(formula=formula, stat_type=stat_type, **kwargs),
        label="tidy",
    )


@pf.register_dataframe_method
def boot_glance(
    data: pd.DataFrame,
    formula: str,
    stat_type: str,
    n_boot: int,
    *,
    seed: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    return _bootstrap_apply(
        data,
        n_boot,
        seed,
        lambda df: df.stats_glance(formula=formula, stat_type=stat_type, **kwargs),
        label="glance",
    )


@pf.register_dataframe_method
def boot_augment(
    data: pd.DataFrame,
    formula: str,
    stat_type: str,
    n_boot: int,
    *,
    seed: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    return _bootstrap_apply(
        data,
        n_boot,
        seed,
        lambda df: df.stats_augment(formula=formula, stat_type=stat_type, **kwargs),
        label="augment",
    )
