from __future__ import annotations
import logging
import warnings
from typing import List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

import numpy as np
import pandas as pd
import pandas_flavor as pf
import statsmodels.api as sm
from patsy import dmatrices
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.power import TTestIndPower

from ._utils import LOGGER


def _assert_viz_available(func_name: str) -> None:
    if plt is None or sns is None:
        raise ImportError(
            f"The visualization libraries (matplotlib/seaborn) are required for function '{func_name}'. "
            "Please install them using 'pip install broom-sm[viz]'."
        )


def stats_power(
    effect_size: float,
    alpha: float,
    *,
    power: Optional[float] = None,
    obs_range: Tuple[int, int] = (5, 200),
    nobs: Optional[int] = None,
) -> Tuple[plt.Figure, plt.Axes, Optional[float]]:
    """Plot power curves for a two-sample t-test and return the figure."""
    _assert_viz_available("stats_power")
    analysis = TTestIndPower()
    required = None
    if power is not None and nobs is None:
        required = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)
        LOGGER.info("Required sample size per group: %.2f", required)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylabel("Power")
    grid = np.arange(obs_range[0], obs_range[1]) if nobs is None else np.array([nobs])
    analysis.plot_power(
        dep_var="nobs",
        nobs=grid,
        effect_size=np.array([effect_size]),
        ax=ax,
        title=fr"alpha = {alpha:0.3f}",
    )
    return fig, ax, required


@pf.register_dataframe_method
def stats_residual_plot(data: pd.DataFrame, x: Sequence[str], y: str) -> List[Tuple[str, plt.Figure]]:
    """Return scatter and QQ plots for each predictor versus target."""
    _assert_viz_available("stats_residual_plot")
    figures: List[Tuple[str, plt.Figure]] = []
    for column in x:
        if data[column].dtype.kind not in "biufc":
            warnings.warn(f"Skipping non-numeric column '{column}' for residual diagnostics", stacklevel=2)
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if data[y].dtype.kind not in "biufc":
            raise ValueError(
                f"Target column '{y}' must be numeric for residual diagnostics; "
                f"got dtype {data[y].dtype}."
            )
        sns.scatterplot(x=data[column], y=data[y], ax=axes[0], color="green", alpha=0.6)
        axes[0].set_title(f"Scatter: {column} vs {y}")
        stats.probplot(data[column].dropna(), plot=axes[1])
        axes[1].set_title(f"Probplot of {column}")
        fig.tight_layout()
        figures.append((column, fig))
    return figures


@pf.register_dataframe_method
def stats_ols_plot(data: pd.DataFrame, x: Sequence[str], y: str) -> List[Tuple[str, plt.Figure]]:
    _assert_viz_available("stats_ols_plot")
    figures: List[Tuple[str, plt.Figure]] = []
    for column in x:
        fig = sns.jointplot(x=column, y=y, data=data, kind="reg")
        figures.append((column, fig.fig))
    return figures


@pf.register_dataframe_method
def stats_influence_plot(
    data: pd.DataFrame,
    formula: str,
    stat_type: str = "ols",
    alpha: float = 0.05,
    **kwargs,
) -> plt.Figure:
    _assert_viz_available("stats_influence_plot")
    from .tidy import prepare_fit

    model_result, _, _ = prepare_fit(data, formula, stat_type, None, kwargs)
    fig = sm.graphics.influence_plot(model_result, alpha=alpha, criterion="cooks")
    fig.tight_layout(pad=1.0)
    return fig


@pf.register_dataframe_method
def stats_chisquare_plot(data: pd.DataFrame, x: str, y: str) -> Tuple[pd.DataFrame, plt.Figure]:
    _assert_viz_available("stats_chisquare_plot")
    x_ser = data[x].astype("category")
    y_ser = data[y].astype("category")
    if any(cat is None or (isinstance(cat, float) and np.isnan(cat)) for cat in x_ser.cat.categories):
        x_ser = x_ser.cat.remove_categories([None, np.nan])
    if any(cat is None or (isinstance(cat, float) and np.isnan(cat)) for cat in y_ser.cat.categories):
        y_ser = y_ser.cat.remove_categories([None, np.nan])
    contingency = pd.crosstab(y_ser, x_ser)
    if contingency.empty:
        raise ValueError("Contingency table is empty after removing missing categories.")
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency, correction=True)
    residuals = contingency - expected
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(residuals, annot=True, fmt=".1f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"Observed - Expected (p = {p_value:0.4f})")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    result = pd.DataFrame({
        "chi2": [chi2],
        "p_value": [p_value],
        "dof": [dof],
    })
    return result, fig


@pf.register_dataframe_method
def stats_vif(data: pd.DataFrame, formula: Optional[str] = None) -> pd.Series:
    if formula:
        _, X_df = dmatrices(formula, data=data, return_type="dataframe")
        has_user_intercept = "Intercept" in X_df.columns
    else:
        numeric = data.select_dtypes(include=np.number)
        non_numeric_cols = [col for col in data.columns if col not in numeric.columns]
        if non_numeric_cols:
            raise ValueError(
                "stats_vif requires numeric predictors when no formula is supplied. "
                f"Non-numeric columns: {non_numeric_cols}"
            )
        X_df = numeric
        has_user_intercept = "Intercept" in X_df.columns
        if not has_user_intercept:
            X_df = sm.add_constant(X_df, has_constant="add")
            has_user_intercept = True

    vif_values = {}
    dropped_intercept = False
    for idx, column in enumerate(X_df.columns):
        if column == "Intercept":
            dropped_intercept = True
            continue
        try:
            vif_values[column] = variance_inflation_factor(X_df.values, idx)
        except np.linalg.LinAlgError:
            warnings.warn(f"Perfect multicollinearity detected for '{column}'. Setting VIF to infinity.", stacklevel=2)
            vif_values[column] = np.inf
    if dropped_intercept:
        warnings.warn(
            "VIF is undefined for the intercept; it has been omitted from the output. "
            "Set include_constant=False in statsmodels or remove the intercept explicitly "
            "if you do not want it in the design matrix.",
            stacklevel=2,
        )
    return pd.Series(vif_values, name="VIF")


@pf.register_dataframe_method
def stats_conprob(
    data: pd.DataFrame,
    var_A_name: str,
    var_B_name: str,
    *,
    transpose: bool = False,
) -> pd.DataFrame:
    base = pd.crosstab(data[var_A_name], data[var_B_name])
    total = base.to_numpy().sum()
    if total == 0:
        raise ValueError("Contingency table is empty. Cannot compute probabilities.")
    if transpose:
        base = base.T
        event_label, given_label = var_B_name, var_A_name
    else:
        event_label, given_label = var_A_name, var_B_name

    conditional = base.div(base.sum(axis=0), axis=1)
    index_name = conditional.index.name or event_label
    tidy = conditional.reset_index().melt(id_vars=index_name, var_name=given_label, value_name="probability")
    tidy.rename(columns={index_name: event_label}, inplace=True)
    marginals = base.sum(axis=1) / total
    tidy = tidy.merge(marginals.rename("marginal"), left_on=event_label, right_index=True)
    return tidy[[event_label, given_label, "probability", "marginal"]]


def stats_coef_forest(
    tidy_df: pd.DataFrame,
    *,
    reference_line: float = 0.0,
    sort: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    _assert_viz_available("stats_coef_forest")
    if sort:
        tidy_df = tidy_df.sort_values("estimate")
    fig, ax = plt.subplots(figsize=(6, max(3, len(tidy_df) * 0.4)))
    ax.errorbar(
        tidy_df["estimate"],
        tidy_df["term"],
        xerr=[tidy_df["estimate"] - tidy_df["conf.low"], tidy_df["conf.high"] - tidy_df["estimate"]],
        fmt="o",
        color="black",
        ecolor="#0072B2",
        capsize=4,
    )
    ax.axvline(reference_line, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Term")
    fig.tight_layout()
    return fig, ax
