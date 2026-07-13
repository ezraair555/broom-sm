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
    tidy_unweighted = sample_df.stats_tidy(
        formula="y ~ x1 + x2",
        stat_type="ols",
    )
    weights = np.ones(len(sample_df))
    weights[:50] = 10.0
    tidy_weighted = sample_df.stats_tidy(
        formula="y ~ x1 + x2",
        stat_type="ols",
        weights=weights,
    )
    assert not tidy_weighted.empty
    assert {"estimate", "std.error", "t_stat", "p.value"}.issubset(tidy_weighted.columns)
    
    # Assert that estimates differ, confirming weights were actually used (and not placebo)
    est_unweighted = tidy_unweighted.set_index("term")["estimate"]
    est_weighted = tidy_weighted.set_index("term")["estimate"]
    assert not np.allclose(est_unweighted.values, est_weighted.values, rtol=1e-4)


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


def test_stats_augment_handles_nans_in_input():
    df = pd.DataFrame({
        "y": [1.0, 2.0, np.nan, 4.0],
        "x": [2.0, 3.0, 4.0, 5.0]
    }, index=["a", "b", "c", "d"])
    
    res = df.stats_augment(formula="y ~ x", stat_type="ols")
    
    assert len(res) == 4
    # .resid, .hat, .cooksd, .std.resid should have NaN at index 'c'
    assert pd.isna(res.loc["c", ".resid"])
    assert pd.isna(res.loc["c", ".hat"])
    assert pd.isna(res.loc["c", ".cooksd"])
    assert pd.isna(res.loc["c", ".std.resid"])
    
    # Check that valid rows are correctly predicted and not NaN
    assert not pd.isna(res.loc["a", ".resid"])
    assert not pd.isna(res.loc["a", ".hat"])


def test_stats_augment_handles_nan_predictors():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "x": [2.0, 3.0, np.nan, 5.0]
    }, index=["a", "b", "c", "d"])
    
    # This should not raise ValueError anymore!
    res = df.stats_augment(formula="y ~ x", stat_type="ols")
    assert len(res) == 4
    assert pd.isna(res.loc["c", ".fitted"])
    assert pd.isna(res.loc["c", ".mean_ci_lower"])
    assert not pd.isna(res.loc["a", ".fitted"])
    assert not pd.isna(res.loc["a", ".mean_ci_lower"])


def test_broom_func_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="broom_sm.broom_func is deprecated"):
        import broom_sm.broom_func  # noqa: F401


def test_utils_fallbacks():
    from broom_sm._utils import prepare_family, coerce_weights, dataframe_like, ensure_iterable
    
    # Nonexistent family fallback
    with pytest.warns(UserWarning, match="family 'nonexistent' is not available"):
        fam = prepare_family("nonexistent", default="gaussian")
        assert fam.__class__.__name__ == "Gaussian"
    
    # Nonexistent link fallback
    with pytest.warns(UserWarning, match="link 'nonexistent' is not available"):
        fam = prepare_family("gaussian", default="gaussian", link="nonexistent")
        # default link is identity
        assert fam.link.__class__.__name__ == "Identity"
        
    # Coerce weights ValueError
    with pytest.raises(ValueError, match="weights must be a 1-D sequence"):
        coerce_weights([[1.0, 2.0], [3.0, 4.0]])
        
    # dataframe_like ValueError
    with pytest.raises(ValueError, match="'data' must be provided"):
        dataframe_like(None, None, "data")
        
    # ensure_iterable scalar fallback
    assert list(ensure_iterable(123)) == [123]
    assert list(ensure_iterable(None)) == []


def test_optional_dependencies_import_guards():
    from unittest import mock
    # Mock the dependencies to raise ImportError on import
    with mock.patch("broom_sm.bayes.bb", None):
        with pytest.raises(ImportError, match="The 'bayesian_bootstrap' package is required"):
            from broom_sm.bayes import bayes_boot
            bayes_boot(pd.DataFrame({"x": [1.0]}), "x", 10)
            
    with mock.patch("broom_sm.bayes.bb", None):
        with pytest.raises(ImportError, match="Visualization and Bayesian bootstrap dependencies"):
            from broom_sm.bayes import bayes_boot_plot
            bayes_boot_plot([pd.Series([1])], "x", "title")
            
    with mock.patch("broom_sm.diagnostics.plt", None):
        from broom_sm.diagnostics import stats_power, stats_coef_forest
        with pytest.raises(ImportError, match="required for function 'stats_power'"):
            stats_power(0.5, 0.05)
        with pytest.raises(ImportError, match="required for function 'stats_coef_forest'"):
            stats_coef_forest(pd.DataFrame({"estimate": [1], "term": ["x"], "conf.low": [0.5], "conf.high": [1.5]}))


def test_additional_tidy_helpers(sample_df):
    # Partial dependence
    from broom_sm.tidy import stats_partial_dependence, stats_correlation_tidy, stats_kruskal_tidy, prepare_fit
    import statsmodels.formula.api as smf
    
    model = smf.ols("y ~ x1 + x2", data=sample_df).fit()
    pdep = stats_partial_dependence(model, "x1", sample_df)
    assert not pdep.empty
    assert {"feature", "value", "prediction"}.issubset(pdep.columns)
    
    # stats_correlation_tidy col1, col2
    corr = sample_df.stats_correlation_tidy(col1="x1", col2="x2")
    assert len(corr) == 1
    assert corr.loc[0, "term1"] == "x1"
    assert corr.loc[0, "term2"] == "x2"
    
    # stats_kruskal_tidy raising ValueError
    df_empty = pd.DataFrame({"val": [], "grp": []})
    with pytest.raises(ValueError, match="Kruskal-Wallis test requires at least two non-empty groups"):
        df_empty.stats_kruskal_tidy(value_col="val", group_col="grp")
        
    # prepare_fit error paths
    with pytest.raises(ValueError, match="formula must be provided"):
        prepare_fit(sample_df, None, "ols", None, {})
    with pytest.raises(ValueError, match="stat_type must be provided"):
        prepare_fit(sample_df, "y ~ x1", None, None, {})
        
    # inferred type fallback path (unknown wrapper object)
    class DummyModel:
        pass
    res, spec, resolved = prepare_fit(sample_df, "y ~ x1", None, DummyModel(), {})
    assert resolved == "dummymodel"


def test_reload_modules():
    import importlib
    import broom_sm
    import broom_sm.tidy
    import broom_sm._utils
    import broom_sm.model_registry
    import broom_sm.bayes
    import broom_sm.diagnostics
    import broom_sm.bootstrap
    import broom_sm.reporting
    
    importlib.reload(broom_sm._utils)
    importlib.reload(broom_sm.model_registry)
    importlib.reload(broom_sm.tidy)
    importlib.reload(broom_sm.bayes)
    importlib.reload(broom_sm.diagnostics)
    importlib.reload(broom_sm.bootstrap)
    importlib.reload(broom_sm.reporting)
    importlib.reload(broom_sm)


def test_tidy_with_zvalues():
    from unittest import mock
    from broom_sm.tidy import stats_tidy
    
    mock_result = mock.MagicMock()
    del mock_result.tvalues  # Ensure it doesn't have tvalues
    mock_result.zvalues = pd.Series([1.96], index=["x"])
    mock_result.params = pd.Series([2.0], index=["x"])
    mock_result.bse = pd.Series([1.0], index=["x"])
    mock_result.pvalues = pd.Series([0.05], index=["x"])
    mock_result.conf_int.return_value = pd.DataFrame([[0.04, 3.96]], index=["x"])
    mock_result.cov_type = "nonrobust"
    
    # Call stats_tidy
    tidy = stats_tidy(pd.DataFrame(), model=mock_result)
    assert "statistic" in tidy.columns
    assert tidy.loc[0, "statistic"] == 1.96


def test_diagnostics_coverage_extension(sample_df):
    from unittest import mock
    from broom_sm.diagnostics import stats_power, stats_coef_forest, stats_conprob
    
    # stats_power with nobs
    stats_power(effect_size=0.5, alpha=0.05, nobs=100)
    
    # stats_coef_forest with sort=False
    tidy = sample_df.stats_tidy(formula="y ~ x1 + x2", stat_type="ols")
    stats_coef_forest(tidy, sort=False)
    
    # stats_conprob empty contingency table
    with pytest.raises(ValueError, match="Contingency table is empty"):
        pd.DataFrame({"a": [], "b": []}).stats_conprob("a", "b")
        
    # stats_conprob transpose
    sample_df.stats_conprob("x1", "x2", transpose=True)
    
    # VIF perfect multicollinearity simulation
    with mock.patch("broom_sm.diagnostics.variance_inflation_factor", side_effect=np.linalg.LinAlgError("singular")):
        with pytest.warns(UserWarning, match="Perfect multicollinearity detected"):
            vif = sample_df.stats_vif(formula="y ~ x1 + x2")
            assert np.isinf(vif["x1"])


def test_model_registry_glm_cov(sample_df):
    # Fit GLM with robust cov_type and cov_kwds
    tidy_glm = sample_df.stats_tidy(
        formula="y ~ x1 + x2",
        stat_type="glm",
        cov_type="HAC",
        cov_kwds={"maxlags": 1}
    )
    assert not tidy_glm.empty
    
    # Fit OLS with robust cov_type
    tidy_ols = sample_df.stats_tidy(
        formula="y ~ x1 + x2",
        stat_type="ols",
        cov_type="HC3"
    )
    assert not tidy_ols.empty


def test_stats_augment_exog_fallback():
    # Fit OLS without formula to trigger get_prediction exog fallback
    import statsmodels.api as sm
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 5.0]])
    y = np.array([1.0, 2.0, 4.0])
    model = sm.OLS(y, X).fit()
    
    df = pd.DataFrame(X, columns=["const", "x"])
    res = df.stats_augment(model=model)
    assert not res.empty
    assert ".fitted" in res.columns


def test_infer_stat_type_glm(sample_df):
    # Fit GLM model and pass to stats_tidy to test _infer_stat_type_from_model family logic
    import statsmodels.formula.api as smf
    fitted = smf.glm("y ~ x1", data=sample_df).fit()
    tidy = sample_df.stats_tidy(model=fitted)
    assert not tidy.empty
    assert tidy.loc[0, "stat_type"] == "glm"


def test_stats_glance_pseudo_rsq(sample_df):
    # Fit Logit and run stats_glance with robust cov to cover rsq & robust glance blocks
    cats = sample_df.copy()
    cats["y_bin"] = (cats["y"] > cats["y"].median()).astype(int)
    
    glance = cats.stats_glance(
        formula="y_bin ~ x1",
        stat_type="logit",
        cov_type="HAC",
        cov_kwds={"maxlags": 1}
    )
    assert not glance.empty
    assert "cov_type" in glance.columns
    assert glance.loc[0, "cov_type"] == "HAC"


def test_stats_augment_cov_type(sample_df):
    res = sample_df.stats_augment(
        formula="y ~ x1",
        stat_type="ols",
        cov_type="HC3"
    )
    assert not res.empty


def test_stats_augment_out_of_sample_nan(sample_df):
    new_df = pd.DataFrame({"x1": [1.0, np.nan]}, index=["n1", "n2"])
    res = sample_df.stats_augment(
        formula="y ~ x1",
        stat_type="ols",
        new_data=new_df
    )
    assert len(res) == 2
    assert pd.isna(res.loc["n2", ".fitted"])
    assert pd.isna(res.loc["n2", ".resid"])


def test_utils_family_link_instances(sample_df):
    import statsmodels.api as sm
    from broom_sm._utils import prepare_family
    
    # 1. Pass family instance (line 77)
    fam_inst = sm.families.Poisson()
    assert prepare_family(fam_inst, "gaussian") is fam_inst
    
    # 2. Pass custom class/fallback (line 85)
    class CustomFamily:
        pass
    cfam = CustomFamily()
    assert prepare_family(cfam, "gaussian") is cfam
    
    # 3. Pass valid link string (line 67 and 89-90)
    res3 = sample_df.stats_tidy(
        formula="y ~ x1",
        stat_type="glm",
        family="poisson",
        link="log"
    )
    assert not res3.empty
    
    # 4. ensure_iterable list/set/tuple input (line 124)
    from broom_sm._utils import ensure_iterable
    assert ensure_iterable([1, 2]) == [1, 2]


def test_tidy_coverage_edge_cases(sample_df):
    # 1. stats_partial_dependence with grid and at
    import statsmodels.formula.api as smf
    from broom_sm.tidy import stats_partial_dependence, _quote_name
    model = smf.ols("y ~ x1 + x2", data=sample_df).fit()
    pdep = stats_partial_dependence(model, "x1", sample_df, grid=[1.0, 2.0], at={"x2": 0.5})
    assert len(pdep) == 2
    
    # 2. stats_correlation_tidy error cases & 1-row grid fallback
    with pytest.raises(ValueError, match="method must be 'pearson' or 'spearman'"):
        sample_df.stats_correlation_tidy(method="invalid")
        
    df_small = pd.DataFrame({"x1": [1.0], "x2": [2.0]})
    with pytest.raises(ValueError, match="Not enough observations for correlation"):
        df_small.stats_correlation_tidy(col1="x1", col2="x2")
        
    corr_grid_small = df_small.stats_correlation_tidy()
    assert pd.isna(corr_grid_small.loc[0, "correlation"])
    
    # 3. _quote_name valid/invalid (lines 378-380)
    assert _quote_name("abc") == "abc"
    assert _quote_name("a b") == "Q('a b')"
    
    # 4. stats_kruskal_tidy column validation (lines 325-328)
    with pytest.raises(ValueError, match="group_col 'missing_grp' not found"):
        sample_df.stats_kruskal_tidy(value_col="y", group_col="missing_grp")
    with pytest.raises(ValueError, match="value_col 'missing_val' not found"):
        sample_df.stats_kruskal_tidy(value_col="missing_val", group_col="x1")
        
    # 5. stats_augment on a mock model without resid
    from unittest import mock
    mock_model = mock.MagicMock()
    mock_model.cov_type = "nonrobust"
    del mock_model.resid
    # Since it is in-sample and lacks resid, it should gracefully skip it
    from broom_sm.tidy import stats_augment
    res_aug = stats_augment(sample_df, formula="y ~ x1", model=mock_model)
    assert ".resid" not in res_aug.columns


def test_model_registry_standard_fitter_cov(sample_df):
    cats = sample_df.copy()
    cats["y_bin"] = (cats["y"] > cats["y"].median()).astype(int)
    
    # logit fit wrapped by _standard_formula_fitter with robust cov (lines 66-70 of model_registry.py)
    tidy_logit = cats.stats_tidy(
        formula="y_bin ~ x1",
        stat_type="logit",
        cov_type="HC3"
    )
    assert not tidy_logit.empty


def test_diagnostics_edge_cases(sample_df):
    from broom_sm.diagnostics import stats_residual_plot, stats_chisquare_plot, stats_vif
    
    # 1. stats_residual_plot non-numeric warning (lines 69-70 of diagnostics.py)
    df_str = pd.DataFrame({"x1": ["a", "b"], "y": [1.0, 2.0]})
    with pytest.warns(UserWarning, match="Skipping non-numeric column"):
        df_str.stats_residual_plot(["x1"], y="y")
        
    # 2. stats_chisquare_plot with nan drop warning and empty result (lines 118, 120, 123)
    df_nan = pd.DataFrame({"x": ["a", np.nan], "y": ["b", np.nan]})
    # Let's verify it drops and computes
    res, fig = stats_chisquare_plot(df_nan, "x", "y")
    assert not res.empty
    
    df_all_nan = pd.DataFrame({"x": [np.nan], "y": [np.nan]})
    with pytest.raises(ValueError, match="Contingency table is empty after removing missing categories"):
        stats_chisquare_plot(df_all_nan, "x", "y")
        
    # 3. stats_vif without formula (lines 145-156)
    vif = sample_df[["x1", "x2"]].stats_vif()
    assert "x1" in vif.index
    assert "x2" in vif.index


def test_sys_modules_import_fallbacks():
    import sys
    import importlib
    from unittest import mock
    
    # 1. diagnostics import fallback (lines 8-10 of diagnostics.py)
    with mock.patch.dict(sys.modules, {"matplotlib.pyplot": None, "seaborn": None}):
        import broom_sm.diagnostics
        importlib.reload(broom_sm.diagnostics)
        assert broom_sm.diagnostics.plt is None
        assert broom_sm.diagnostics.sns is None
        
    # 2. bayes import fallback (lines 8-9, 14-16 of bayes.py)
    with mock.patch.dict(sys.modules, {"bayesian_bootstrap": None, "matplotlib.pyplot": None}):
        import broom_sm.bayes
        importlib.reload(broom_sm.bayes)
        assert broom_sm.bayes.bb is None
        
    # Restore original imports
    importlib.reload(broom_sm.diagnostics)
    importlib.reload(broom_sm.bayes)
