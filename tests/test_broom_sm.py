import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from broom_sm import (
    stats_report,
    stats_compare,
)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    y = 2 + 3 * x1 - 1.5 * x2 + rng.normal(scale=0.5, size=200)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_stats_tidy_returns_expected_columns(sample_df):
    tidy = sample_df.stats_tidy(formula="y ~ x1 + x2", stat_type="ols")
    required = {"term", "estimate", "std.error", "conf.low", "conf.high", "t_stat", "p.value", "cov_type"}
    assert required.issubset(set(tidy.columns))


def test_stats_glance_with_prefitted_model(sample_df):
    fitted = smf.ols("y ~ x1 + x2", data=sample_df).fit()
    glance = sample_df.stats_glance(model=fitted)
    assert glance.loc[0, "nobs"] == pytest.approx(200)
    assert glance.loc[0, "cov_type"].lower() == fitted.cov_type


def test_stats_augment_retains_columns(sample_df):
    new_data = sample_df.copy()
    new_data["x1"] += 0.1
    new_data.index = [f"new_{i}" for i in range(len(new_data))]
    augmented = sample_df.stats_augment(formula="y ~ x1 + x2", stat_type="ols", new_data=new_data)
    for column in new_data.columns:
        assert column in augmented.columns
    assert ".in_sample" in augmented.columns
    assert not augmented[".in_sample"].any()


def test_boot_tidy_smoke(sample_df):
    boot = sample_df.boot_tidy(formula="y ~ x1 + x2", stat_type="ols", n_boot=3, seed=13)
    assert boot[".bootstrap_id"].nunique() == 3


def test_stats_vif_multicollinearity_warns(sample_df):
    duplicated = sample_df.assign(x_dup=sample_df["x1"])
    vif = duplicated.stats_vif(formula="y ~ x1 + x_dup + x2")
    assert np.isinf(vif.loc["x_dup"])


def test_stats_conprob_returns_tidy(sample_df):
    cats = sample_df.copy()
    cats["y_cat"] = (cats["y"] > cats["y"].median()).astype(int)
    cats["x_cat"] = (cats["x1"] > 0).astype(int)
    tidy = cats.stats_conprob("y_cat", "x_cat")
    assert {"y_cat", "x_cat", "probability", "marginal"}.issubset(tidy.columns)


def test_stats_compare_and_report(sample_df):
    model_a = smf.ols("y ~ x1", data=sample_df).fit()
    model_b = smf.ols("y ~ x1 + x2", data=sample_df).fit()
    table = stats_compare({"x1": model_a, "x1_x2": model_b})
    assert set(table["model"]) == {"x1", "x1_x2"}

    report = stats_report(sample_df, formula="y ~ x1 + x2", stat_type="ols")
    assert set(report.keys()) == {"tidy", "glance", "augment"}
    assert not report["tidy"].empty


def test_invalid_stat_type_raises(sample_df):
    with pytest.raises(ValueError):
        sample_df.stats_tidy(formula="y ~ x1", stat_type="unknown")
