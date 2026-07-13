import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from broom_sm import (
    stats_power,
    stats_residual_plot,
    stats_ols_plot,
    stats_influence_plot,
    stats_chisquare_plot,
    stats_coef_forest,
)
from broom_sm.bayes import bayes_boot_plot


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=50)
    x2 = rng.normal(size=50)
    y = 2 + 3 * x1 - 1.5 * x2 + rng.normal(scale=0.5, size=50)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_stats_power():
    fig, ax, req = stats_power(effect_size=0.5, alpha=0.05, power=0.8)
    assert fig is not None
    assert ax is not None
    assert req > 0


def test_stats_residual_plot(sample_df):
    figs = sample_df.stats_residual_plot(["x1", "x2"], y="y")
    assert len(figs) == 2
    assert figs[0][0] == "x1"
    assert isinstance(figs[0][1], plt.Figure)


def test_stats_ols_plot(sample_df):
    figs = sample_df.stats_ols_plot(["x1"], y="y")
    assert len(figs) == 1
    assert isinstance(figs[0][1], plt.Figure)


def test_stats_influence_plot(sample_df):
    fig = sample_df.stats_influence_plot(formula="y ~ x1 + x2")
    assert isinstance(fig, plt.Figure)


def test_stats_chisquare_plot():
    df = pd.DataFrame({
        "x": ["a", "a", "b", "b", "a"],
        "y": ["c", "d", "c", "d", "c"]
    })
    res, fig = stats_chisquare_plot(df, "x", "y")
    assert not res.empty
    assert isinstance(fig, plt.Figure)


def test_stats_coef_forest(sample_df):
    tidy = sample_df.stats_tidy(formula="y ~ x1 + x2", stat_type="ols")
    fig, ax = stats_coef_forest(tidy)
    assert isinstance(fig, plt.Figure)


def test_bayes_boot_plot():
    series = [pd.Series([1.0, 2.0, 3.0], name="metric")]
    figs = bayes_boot_plot(series, "value", "title")
    assert len(figs) == 1
    assert figs[0][0] == "metric"
    assert isinstance(figs[0][1], plt.Figure)
