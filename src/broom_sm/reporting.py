"""Reporting utilities built on top of the tidy-glance-augment verbs."""
from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd

from .tidy import stats_augment, stats_glance, stats_tidy


def stats_compare(models: Mapping[str, Any]) -> pd.DataFrame:
    """Create an information criteria table (AIC/BIC) for fitted models."""
    rows = []
    for name, model in models.items():
        rows.append(
            {
                "model": name,
                "aic": getattr(model, "aic", np.nan),
                "bic": getattr(model, "bic", np.nan),
                "llf": getattr(model, "llf", np.nan),
                "nobs": getattr(model, "nobs", np.nan),
            }
        )
    result = pd.DataFrame(rows).sort_values("aic", key=lambda s: s.fillna(np.inf))
    result.reset_index(drop=True, inplace=True)
    return result


def stats_report(
    data: pd.DataFrame,
    *,
    formula: str,
    stat_type: str,
    tidy_kwargs: Optional[Mapping[str, Any]] = None,
    glance_kwargs: Optional[Mapping[str, Any]] = None,
    augment_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """Return a dictionary bundling tidy, glance, and augment outputs."""
    tidy_df = data.stats_tidy(formula=formula, stat_type=stat_type, **(tidy_kwargs or {}))
    glance_df = data.stats_glance(formula=formula, stat_type=stat_type, **(glance_kwargs or {}))
    augment_df = data.stats_augment(formula=formula, stat_type=stat_type, **(augment_kwargs or {}))
    return {"tidy": tidy_df, "glance": glance_df, "augment": augment_df}
