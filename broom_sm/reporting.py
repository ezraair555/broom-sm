"""broom_sm.reporting - broom-style helpers for Python models."""

from __future__ import annotations

import pandas as pd
from typing import Any

try:
    import statsmodels.api as sm  # noqa
except ImportError:  # pragma: no cover
    sm = None


def tidy(model: Any) -> pd.DataFrame:
    """Return coefficient-level summary similar to broom::tidy."""
    if hasattr(model, "summary_frame"):
        df = model.summary_frame()
        df = df.rename(columns={"mean": "estimate"})
        return df.reset_index().rename(columns={"index": "term"})
    if hasattr(model, "params") and hasattr(model, "bse"):
        terms = model.params.index
        return pd.DataFrame(
            {
                "term": terms,
                "estimate": model.params.values,
                "std_error": model.bse.values,
                "z_value": getattr(model, "tvalues", pd.Series([None] * len(terms))).values,
                "p_value": getattr(model, "pvalues", pd.Series([None] * len(terms))).values,
            }
        )
    raise TypeError("Unsupported model type for tidy().")


def glance(model: Any) -> pd.DataFrame:
    """Return model-level metrics (AIC, BIC, R^2, etc.)."""
    metrics = {}
    for attr in ["aic", "bic", "rsquared", "rsquared_adj", "llf", "df_model", "df_resid"]:
        if hasattr(model, attr):
            metrics[attr] = getattr(model, attr)
    if hasattr(model, "nobs"):
        metrics["nobs"] = model.nobs
    return pd.DataFrame([metrics])


def augment(model: Any, data: pd.DataFrame) -> pd.DataFrame:
    """Attach fitted values & residuals to the original data."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    df = data.copy()
    if hasattr(model, "predict"):
        df[".fitted"] = model.predict(data)
    if hasattr(model, "resid"):
        df[".resid"] = model.resid if len(model.resid) == len(df) else model.resid[: len(df)]
    return df
