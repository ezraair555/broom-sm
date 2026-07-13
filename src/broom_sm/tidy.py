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

from ._utils import LOGGER, dataframe_like, coerce_weights
from .model_registry import ModelSpec, get_model_spec, iter_model_config


class _IndexError(ValueError):
    pass


def _validate_index_unique(data: pd.DataFrame, name: str) -> None:
    if not data.index.is_unique:
        raise _IndexError(
            f"{name} index must be unique for reliable residual/influence alignment. "
            "Reset the index with `.reset_index(drop=True)` or supply unique labels."
        )



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
        if spec.accepts_family:
            call_kwargs["freq_weights"] = weights
        else:
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
        {**fit_kwargs, "family": family, "link": link, "weights": coerce_weights(weights)},
    )

    terms = model_result.params.index
    conf_int = model_result.conf_int(alpha=alpha)
    conf_low = conf_int[0]
    conf_high = conf_int[1]

    stat_series = None
    if hasattr(model_result, "tvalues"):
        stat_series = model_result.tvalues
    elif hasattr(model_result, "zvalues"):
        stat_series = model_result.zvalues

    stat_label = spec.stat_name if spec else "statistic"

    tidy_df = pd.DataFrame({
        "term": terms,
        "estimate": model_result.params.values,
        "conf.low": conf_low.values,
        "conf.high": conf_high.values,
        "std.error": model_result.bse.values,
        stat_label: stat_series.values if stat_series is not None else np.nan,
        "p.value": model_result.pvalues.values
    }).reset_index(drop=True)

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
        {**fit_kwargs, "family": family, "link": link, "weights": coerce_weights(weights)},
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

    _validate_index_unique(data, "data")
    if new_data is not None:
        _validate_index_unique(new_data, "new_data")
        if not data.index.intersection(new_data.index).empty:
            raise _IndexError(
                "data and new_data indices overlap. Use disjoint indices or reset them."
            )

    output = prediction_frame.copy()
    output[".fitted"] = model_result.predict(prediction_frame)

    in_sample = new_data is None
    if in_sample:
        if hasattr(model_result, "resid"):
            output[".resid"] = model_result.resid
        
        if hasattr(model_result, "get_influence"):
            try:
                influence = model_result.get_influence()
                output[".hat"] = np.nan
                output[".cooksd"] = np.nan
                output[".std.resid"] = np.nan
                
                fit_data = getattr(model_result, "model", None)
                fit_idx = getattr(fit_data, "data", None)
                row_labels = getattr(fit_idx, "row_labels", None)
                if row_labels is None and hasattr(model_result, "resid"):
                    row_labels = model_result.resid.index
                
                for src, dst in [
                    ("hat_matrix_diag", ".hat"),
                    ("cooks_distance", ".cooksd"),
                    ("resid_studentized_internal", ".std.resid"),
                ]:
                    if hasattr(influence, src):
                        values = getattr(influence, src)
                        if src == "cooks_distance":
                            values = values[0]
                        if row_labels is not None and len(values) == len(row_labels):
                            output[dst] = pd.Series(values, index=row_labels)
                        else:
                            output[dst] = values
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Influence diagnostics unavailable: %s", exc)
    else:
        output[".resid"] = np.nan

    if include_prediction_intervals and hasattr(model_result, "get_prediction"):
        try:
            if hasattr(model_result, "_transform_predict_exog"):
                exog, exog_index = model_result._transform_predict_exog(prediction_frame, transform=True)
                pred = model_result.get_prediction(exog, transform=False, row_labels=exog_index)
            else:
                pred = model_result.get_prediction(prediction_frame)
            
            summary = pred.summary_frame(alpha=alpha)
            for col in ["mean_se", "obs_ci_lower", "obs_ci_upper", "mean_ci_lower", "mean_ci_upper"]:
                if col in summary:
                    output[f".{col}"] = summary[col]
        except Exception as exc:  # pragma: no cover - statsmodels edge cases
            LOGGER.warning("Prediction intervals unavailable: %s", exc)

    output[".in_sample"] = in_sample
    output["stat_type"] = resolved_type
    return output


@pf.register_dataframe_method
def stats_anova_tidy(data: pd.DataFrame, formula: str, anova_type: int = 2, **kwargs) -> pd.DataFrame:
    if anova_type not in {1, 2, 3}:
        raise ValueError(f"anova_type must be 1, 2, or 3; got {anova_type}")
    model = smf.ols(formula, data=data, **kwargs).fit()
    table = anova_lm(model, typ=anova_type).reset_index().rename(columns={"index": "term"})
    table.rename(columns={"PR(>F)": "p.value", "F": "statistic"}, inplace=True)
    return table


@pf.register_dataframe_method
def stats_kruskal_tidy(data: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    if group_col not in data.columns:
        raise ValueError(f"group_col '{group_col}' not found in data columns: {list(data.columns)}")
    if value_col not in data.columns:
        raise ValueError(f"value_col '{value_col}' not found in data columns: {list(data.columns)}")
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
    if columns is not None:
        missing = [col for col in columns if col not in data.columns]
        if missing:
            raise ValueError(f"columns not found in data: {missing}")
        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric:
            raise ValueError(f"correlation columns must be numeric; non-numeric: {non_numeric}")
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


def _quote_name(name: str) -> str:
    if name.isidentifier():
        return name
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    return f"Q('{escaped}')"


@pf.register_dataframe_method
def stats_formula(data: pd.DataFrame, target: str, *exclude: str) -> str:
    predictors = [col for col in data.columns if col != target and col not in exclude]
    return f"{_quote_name(target)} ~ {' + '.join(_quote_name(col) for col in predictors)}"


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
