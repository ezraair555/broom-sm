"""Regression tests for P0 and selected P1 fixes from the 2026-06-20 review."""
from __future__ import annotations

import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from broom_sm import (
    boot_augment,
    boot_glance,
    boot_tidy,
    stats_anova_tidy,
    stats_augment,
    stats_correlation_tidy,
    stats_formula,
    stats_kruskal_tidy,
    stats_residual_plot,
    stats_vif,
)
from broom_sm.bayes import bayes_boot
from broom_sm.bootstrap import _bootstrap_apply
from broom_sm.diagnostics import stats_chisquare_plot


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    y = 2 + 3 * x1 - 1.5 * x2 + rng.normal(scale=0.5, size=200)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# --- P0: stats_augment alignment and in-sample flag ----------------------------


def test_stats_augment_in_sample_alignment_non_default_index(sample_df):
    df = sample_df.copy()
    df.index = [f"row_{i}" for i in range(len(df))]
    aug = df.stats_augment(formula="y ~ x1 + x2", stat_type="ols")
    assert aug[".in_sample"].all()
    assert aug[".resid"].notna().all()
    assert aug[".hat"].notna().all()
    assert aug[".cooksd"].notna().all()
    assert aug[".std.resid"].notna().all()
    # Spot check that values are in the right rows, not accidentally broadcast.
    fitted = smf.ols("y ~ x1 + x2", data=df).fit()
    assert np.allclose(aug[".resid"].values, fitted.resid.values)
    assert np.allclose(aug[".hat"].values, fitted.get_influence().hat_matrix_diag)


def test_stats_augment_raises_on_duplicate_data_index(sample_df):
    df = sample_df.copy()
    df.index = [0] * len(df)
    with pytest.raises(ValueError, match="index must be unique"):
        df.stats_augment(formula="y ~ x1 + x2", stat_type="ols")


def test_stats_augment_new_data_disjoint_index(sample_df):
    df = sample_df.copy()
    new_data = sample_df.iloc[:20].copy()
    new_data.index = [f"new_{i}" for i in range(20)]
    aug = df.stats_augment(formula="y ~ x1 + x2", stat_type="ols", new_data=new_data)
    assert not aug[".in_sample"].any()
    assert aug[".resid"].isna().all()
    assert aug[".fitted"].notna().all()
    # Influence diagnostics are not computed for out-of-sample rows.
    assert ".hat" not in aug.columns


def test_stats_augment_raises_on_overlapping_new_data_index(sample_df):
    df = sample_df.copy()
    new_data = sample_df.iloc[:20].copy()
    with pytest.raises(ValueError, match="indices overlap"):
        df.stats_augment(formula="y ~ x1 + x2", stat_type="ols", new_data=new_data)


# --- P0: prepare_fit weights key for GLM vs OLS --------------------------------


def test_glm_with_weights_uses_freq_weights_and_succeeds(sample_df):
    # Force a positive response for Gamma so the fit is well defined.
    df = sample_df.assign(y_pos=lambda d: np.abs(d["y"]) + 0.1)
    weights = np.ones(len(df))
    weights[:50] = 2.0
    tidy = df.stats_tidy(
        formula="y_pos ~ x1 + x2",
        stat_type="glm",
        family="gamma",
        weights=weights,
    )
    assert not tidy.empty
    assert {"estimate", "std.error", "z_stat", "p.value"}.issubset(tidy.columns)


def test_ols_with_weights_uses_weights_and_succeeds(sample_df):
    weights = np.ones(len(sample_df))
    weights[:50] = 2.0
    tidy = sample_df.stats_tidy(
        formula="y ~ x1 + x2",
        stat_type="ols",
        weights=weights,
    )
    assert not tidy.empty
    assert {"estimate", "std.error", "t_stat", "p.value"}.issubset(tidy.columns)


# --- P0: bootstrap helpers raise on total failure ------------------------------


def test_bootstrap_apply_raises_when_all_fail():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(RuntimeError, match="All .* bootstrap replications failed"):
        _bootstrap_apply(df, n_boot=3, seed=1, func=lambda d: (_ for _ in ()).throw(ValueError("boom")), label="test")


def test_boot_tidy_raises_on_total_failure(sample_df):
    with mock.patch.object(pd.DataFrame, "stats_tidy", side_effect=ValueError("boom")):
        with pytest.raises(RuntimeError, match="All .* bootstrap replications failed"):
            sample_df.boot_tidy(formula="y ~ x1", stat_type="ols", n_boot=2, seed=1)


def test_boot_glance_raises_on_total_failure(sample_df):
    with mock.patch.object(pd.DataFrame, "stats_glance", side_effect=ValueError("boom")):
        with pytest.raises(RuntimeError, match="All .* bootstrap replications failed"):
            sample_df.boot_glance(formula="y ~ x1", stat_type="ols", n_boot=2, seed=1)


def test_boot_augment_raises_on_total_failure(sample_df):
    with mock.patch.object(pd.DataFrame, "stats_augment", side_effect=ValueError("boom")):
        with pytest.raises(RuntimeError, match="All .* bootstrap replications failed"):
            sample_df.boot_augment(formula="y ~ x1", stat_type="ols", n_boot=2, seed=1)


# --- P0: stats_residual_plot validates y -------------------------------------


def test_stats_residual_plot_raises_on_non_numeric_y(sample_df):
    df = sample_df.assign(y_str=lambda d: d["y"].astype(str))
    with pytest.raises(ValueError, match="Target column 'y_str' must be numeric"):
        df.stats_residual_plot(x=["x1"], y="y_str")


# --- P0: stats_vif intercept documentation/consistency -------------------------


def test_stats_vif_warns_that_intercept_is_omitted(sample_df):
    with pytest.warns(UserWarning, match="VIF is undefined for the intercept"):
        vif = sample_df.stats_vif(formula="y ~ x1 + x2")
    assert "Intercept" not in vif.index
    assert {"x1", "x2"}.issubset(vif.index)


def test_stats_vif_no_intercept_formula_does_not_add_constant(sample_df):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vif = sample_df.stats_vif(formula="y ~ x1 + x2 - 1")
    # No intercept warning, no constant added.
    assert "Intercept" not in vif.index
    assert set(vif.index) == {"x1", "x2"}
    assert not any("intercept" in str(warn.message).lower() for warn in w)


# --- P1: anova_type validation -------------------------------------------------


def test_stats_anova_tidy_validates_anova_type(sample_df):
    with pytest.raises(ValueError, match="anova_type must be 1, 2, or 3"):
        sample_df.stats_anova_tidy(formula="y ~ x1 + x2", anova_type=4)


def test_stats_anova_tidy_accepts_valid_types(sample_df):
    for typ in (1, 2, 3):
        out = sample_df.stats_anova_tidy(formula="y ~ x1 + x2", anova_type=typ)
        assert not out.empty
        assert {"term", "statistic", "p.value"}.issubset(out.columns)


# --- P1: kruskal group_col validation -----------------------------------------


def test_stats_kruskal_tidy_validates_group_col(sample_df):
    with pytest.raises(ValueError, match="group_col 'missing' not found"):
        sample_df.stats_kruskal_tidy(value_col="y", group_col="missing")


# --- P1: correlation columns validation ---------------------------------------


def test_stats_correlation_tidy_validates_missing_columns(sample_df):
    with pytest.raises(ValueError, match="columns not found in data"):
        sample_df.stats_correlation_tidy(columns=["x1", "missing"])


def test_stats_correlation_tidy_validates_non_numeric_columns():
    df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    with pytest.raises(ValueError, match="correlation columns must be numeric"):
        df.stats_correlation_tidy(columns=["a", "b"])


# --- P1: stats_formula quotes non-syntactic names ----------------------------


def test_stats_formula_quotes_non_syntactic_columns():
    df = pd.DataFrame({"target col": [1, 2], "1st pred": [3, 4], "good": [5, 6]})
    formula = df.stats_formula("target col", "good")
    assert formula == "Q('target col') ~ Q('1st pred')"
    # The quoted formula must be parseable by patsy.
    out = df.rename(columns={"target col": "y", "1st pred": "x"}).stats_tidy(
        formula="Q('y') ~ Q('x')", stat_type="ols"
    )
    assert not out.empty


# --- P1: bayes_boot validation and NaN warning -------------------------------


def test_bayes_boot_validates_target_column_and_n_samples():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="target_column 'missing' not found"):
        bayes_boot(df, "missing", 10)
    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        bayes_boot(df, "a", 0)
    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        bayes_boot(df, "a", -5)


def test_bayes_boot_warns_on_nan():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    with pytest.warns(UserWarning, match="Dropping 1 NaN"):
        series = bayes_boot(df, "a", 20)
    assert len(series) == 20


# --- P1: chisquare plot handles NaN category ----------------------------------


def test_stats_chisquare_plot_drops_nan_category():
    df = pd.DataFrame({
        "x": ["a", "a", "b", "b", np.nan, "a"],
        "y": ["c", "d", "c", "d", "c", np.nan],
    })
    result, fig = stats_chisquare_plot(df, x="x", y="y")
    assert not result.empty
    assert "nan" not in result.columns.str.lower().tolist()
    assert fig is not None


# --- P1: extra_sm package removed --------------------------------------------


def test_extra_sm_package_removed():
    with pytest.raises(ModuleNotFoundError):
        import extra_sm  # noqa: F401
