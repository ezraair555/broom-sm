"""Core tidy-like verbs for statsmodels results."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pandas_flavor as pf
from patsy import dmatrices
from statsmodels.stats.anova import anova_lm
from scipy import stats
import statsmodels.formula.api as smf

from ._utils import LOGGER, dataframe_like
from .model_registry import ModelSpec, get_model_spec, iter_model_config


def _infer_stat_type_from_model(model: Any) -> Optional[str]:
    name_chain = [model.__class__.__name__.lower()]
    if hasattr(model, "model"):
        name_chain.append(model.model.__class__.__name__.lower())
        family = getattr(getattr(model.model, "family", None), "__class__", None)
        if family:
            name_chain.append(family.__name__.lower())
    available = iter_model_config()
    for name in name_chain:
        for key in available:
            if key in name:
                return key
    return None


def prepare_fit(
    data: Optional[pd.DataFrame],
    formula: Optional[str],
    stat_type: Optional[str],
    model: Any,
    kwargs: Dict[str, Any],
) -> tuple[Any, Optional[ModelSpec], str]:
    if model is not None:
        inferred = stat_type or _infer_stat_type_from_model(model)
        spec = get_model_spec(inferred) if inferred else None
        if inferred is None:
            inferred = getattr(model, "model", model).__class__.__name__.lower()
        return model, spec, inferred

    if formula is None:
        raise ValueError("formula must be provided when a model instance is not supplied")
    if stat_type is None:
        raise ValueError("stat_type must be provided when a model instance is not supplied")

    spec = get_model_spec(stat_type)
    call_kwargs = {"formula": formula, "data": dataframe_like(data, None, "data")}
    extra = kwargs.copy()

    family = extra.pop("family", None)
    link = extra.pop("link", None)
    weights = extra.pop("weights", None)

    if spec.accepts_family and family is not None:
        call_kwargs["family"] = family
    if spec.accepts_family and link is not None:
        call_kwargs["link"] = link
    if spec.accepts_weights and weights is not None:
        call_kwargs["weights"] = weights

    call_kwargs.update(extra)
    result = spec.fitter(**call_kwargs)
    return result, spec, stat_type


@pf.register_dataframe_method
def stats_tidy(
    data: pd.DataFrame,
    formula: Optional[str] = None,
    stat_type: Optional[str] = None,
    *,
    model: Any = None,
    family: Any = None,
    link: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
    cov_type: Optional[str] = None,
    cov_kwds: Optional[Dict[str, Any]] = None,
    alpha: float = 0.05,
    **kwargs,
) -> pd.DataFrame:
    """Return a tidy coefficient table, optionally using a pre-fitted model."""

    fit_kwargs = {**kwargs}
    if cov_type:
        fit_kwargs["cov_type"] = cov_type
    if cov_kwds:
        fit_kwargs["cov_kwds"] = cov_kwds
    model_result, spec, resolved_type = prepare_fit(
        data,
        formula,
        stat_type,
        model,
        {**fit_kwargs, "family": family, "link": link, "weights": weights},
    )

    params = model_result.params.rename("estimate").to_frame().reset_index().rename(columns={"index": "term"})
    conf_int = (
        model_result
        .conf_int(alpha=alpha)
        .rename(columns={0: "conf.low", 1: "conf.high"})
        .reset_index()
        .rename(columns={"index": "term"})
    )
    std_err = model_result.bse.rename("std.error").to_frame().reset_index().rename(columns={"index": "term"})

    stat_series = None
    if hasattr(model_result, "tvalues"):
        stat_series = model_result.tvalues
    elif hasattr(model_result, "zvalues"):
        stat_series = model_result.zvalues
    if stat_series is not None:
        stat_df = stat_series.rename("statistic").to_frame().reset_index().rename(columns={"index": "term"})
    else:
        stat_df = pd.DataFrame(columns=["term", "statistic"])

    pvals = model_result.pvalues.rename("p.value").to_frame().reset_index().rename(columns={"index": "term"})

    tidy_df = (
        params
        .merge(conf_int, on="term", how="left")
        .merge(std_err, on="term", how="left")
        .merge(stat_df, on="term", how="left")
        .merge(pvals, on="term", how="left")
    )

    stat_label = spec.stat_name if spec else "statistic"
    tidy_df.rename(columns={"statistic": stat_label}, inplace=True)

    cov_name = getattr(model_result, "cov_type", "nonrobust")
    tidy_df["cov_type"] = cov_name

    family_name = None
    model_family = getattr(getattr(model_result, "model", None), "family", None)
    if model_family is not None:
        family_name = model_family.__class__.__name__
        link_name = getattr(model_family.link, "__class__", None)
        tidy_df["family"] = family_name
        tidy_df["link"] = link_name.__name__ if link_name else None

    if weights is not None:
        tidy_df["weights"] = np.asarray(weights).mean()

    tidy_df["stat_type"] = resolved_type
    return tidy_df


@pf.register_dataframe_method
def stats_glance(
    data: pd.DataFrame,
    formula: Optional[str] = None,
    stat_type: Optional[str] = None,
    *,
    model: Any = None,
    family: Any = None,
    link: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
    cov_type: Optional[str] = None,
    cov_kwds: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Return a one-row summary of model-level statistics."""
    fit_kwargs = {**kwargs}
    if cov_type:
        fit_kwargs["cov_type"] = cov_type
    if cov_kwds:
        fit_kwargs["cov_kwds"] = cov_kwds

    model_result, spec, resolved_type = prepare_fit(
        data,
        formula,
        stat_type,
        model,
        {**fit_kwargs, "family": family, "link": link, "weights": weights},
    )

    glance_dict: Dict[str, Any] = {"stat_type": resolved_type, "cov_type": getattr(model_result, "cov_type", "nonrobust")}
    attrs = ["nobs", "llf", "aic", "bic", "df_model", "df_resid", "deviance", "pearson_chi2", "scale"]
    for attr in attrs:
        value = getattr(model_result, attr, None)
        if value is not None and not isinstance(value, (list, tuple)):
            glance_dict[attr] = float(value)

    if spec and spec.has_rsq:
        for attr in ["rsquared", "rsquared_adj"]:
            if hasattr(model_result, attr):
                glance_dict[attr] = float(getattr(model_result, attr))

    pseudo_attr = spec.pseudo_rsq_attr if spec else None
    if pseudo_attr and hasattr(model_result, pseudo_attr):
        glance_dict["pseudo_rsquared"] = float(getattr(model_result, pseudo_attr))

    if resolved_type == "ols" and hasattr(model_result, "fvalue"):
        glance_dict["f_statistic"] = float(model_result.fvalue)
        glance_dict["f_pvalue"] = float(model_result.f_pvalue)

    family_obj = getattr(getattr(model_result, "model", None), "family", None)
    if family_obj:
        glance_dict["family"] = family_obj.__class__.__name__
        glance_dict["link"] = getattr(family_obj.link, "__class__", family_obj.link).__name__

    return pd.DataFrame([glance_dict])


@pf.register_dataframe_method
def stats_augment(
    data: pd.DataFrame,
    formula: Optional[str] = None,
    stat_type: Optional[str] = None,
    *,
    model: Any = None,
    new_data: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
    include_prediction_intervals: bool = True,
    cov_type: Optional[str] = None,
    cov_kwds: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Augment data with fitted values, residuals, and prediction intervals."""
    prediction_frame = dataframe_like(new_data, data, "new_data")
    fit_kwargs = {**kwargs}
    if cov_type:
        fit_kwargs["cov_type"] = cov_type
    if cov_kwds:
        fit_kwargs["cov_kwds"] = cov_kwds

    model_result, spec, resolved_type = prepare_fit(
        data,
        formula,
        stat_type,
        model,
        fit_kwargs,
    )

    output = prediction_frame.copy()
    output[".fitted"] = model_result.predict(prediction_frame)
    if hasattr(model_result, "resid") and new_data is None:
        resid = model_result.resid
        output.loc[data.index, ".resid"] = resid
    elif new_data is not None:
        output[".resid"] = np.nan

    if hasattr(model_result, "get_influence") and new_data is None:
        influence = model_result.get_influence()
        if hasattr(influence, "hat_matrix_diag"):
            output.loc[data.index, ".hat"] = influence.hat_matrix_diag
        if hasattr(influence, "cooks_distance"):
            cooks, _ = influence.cooks_distance
            output.loc[data.index, ".cooksd"] = cooks
        if hasattr(influence, "resid_studentized_internal"):
            output.loc[data.index, ".std.resid"] = influence.resid_studentized_internal

    if include_prediction_intervals and hasattr(model_result, "get_prediction"):
        try:
            pred = model_result.get_prediction(prediction_frame)
            summary = pred.summary_frame(alpha=alpha)
            for col in ["mean_se", "obs_ci_lower", "obs_ci_upper", "mean_ci_lower", "mean_ci_upper"]:
                if col in summary:
                    output[f".{col}"] = summary[col].values
        except Exception as exc:  # pragma: no cover - statsmodels edge cases
            LOGGER.warning("Prediction intervals unavailable: %s", exc)

    in_sample_idx = set(data.index)
    output[".in_sample"] = [idx in in_sample_idx for idx in output.index]
    output["stat_type"] = resolved_type
    return output


@pf.register_dataframe_method
def stats_anova_tidy(data: pd.DataFrame, formula: str, anova_type: int = 2, **kwargs) -> pd.DataFrame:
    model = smf.ols(formula, data=data, **kwargs).fit()
    table = anova_lm(model, typ=anova_type).reset_index().rename(columns={"index": "term"})
    table.rename(columns={"PR(>F)": "p.value", "F": "statistic"}, inplace=True)
    return table


@pf.register_dataframe_method
def stats_kruskal_tidy(data: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    grouped = [grp[value_col].dropna().values for _, grp in data.groupby(group_col)]
    grouped = [vals for vals in grouped if len(vals)]
    if len(grouped) < 2:
        raise ValueError("Kruskal-Wallis test requires at least two non-empty groups")
    statistic, p_value = stats.kruskal(*grouped)
    return pd.DataFrame({"statistic": [statistic], "p.value": [p_value], "df": [len(grouped) - 1]})


@pf.register_dataframe_method
def stats_correlation_tidy(
    data: pd.DataFrame,
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    *,
    method: str = "pearson",
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr
    if col1 and col2:
        subset = data[[col1, col2]].dropna()
        if len(subset) < 2:
            raise ValueError("Not enough observations for correlation")
        stat, p_val = corr_func(subset[col1], subset[col2])
        return pd.DataFrame({"term1": [col1], "term2": [col2], "correlation": [stat], "p.value": [p_val]})

    cols = columns or data.select_dtypes(include=np.number).columns.tolist()
    records = []
    for idx, col_a in enumerate(cols):
        for col_b in cols[idx + 1 :]:
            subset = data[[col_a, col_b]].dropna()
            if len(subset) < 2:
                stat_val, p_val = np.nan, np.nan
            else:
                stat_val, p_val = corr_func(subset[col_a], subset[col_b])
            records.append({"term1": col_a, "term2": col_b, "correlation": stat_val, "p.value": p_val})
    return pd.DataFrame(records)


@pf.register_dataframe_method
def stats_formula(data: pd.DataFrame, target: str, *exclude: str) -> str:
    predictors = [col for col in data.columns if col != target and col not in exclude]
    return f"{target} ~ {' + '.join(predictors)}"


def stats_partial_dependence(
    model: Any,
    feature: str,
    data: pd.DataFrame,
    grid: Optional[Sequence[float]] = None,
    at: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute a simple partial dependence curve for a fitted statsmodels model."""
    if grid is None:
        series = data[feature]
        grid = np.linspace(series.min(), series.max(), 25)
    baseline = data.copy()
    if at:
        for key, value in at.items():
            baseline[key] = value
    rows = []
    for val in grid:
        temp = baseline.copy()
        temp[feature] = val
        prediction = model.predict(temp)
        rows.append({"feature": feature, "value": val, "prediction": float(np.mean(prediction))})
    return pd.DataFrame(rows)
