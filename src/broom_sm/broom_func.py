"""Backward-compatible shim for the old monolithic module."""
from __future__ import annotations

import warnings

from . import (  # noqa: F401
    stats_tidy,
    stats_glance,
    stats_augment,
    boot_tidy,
    boot_glance,
    boot_augment,
    stats_power,
    stats_residual_plot,
    stats_ols_plot,
    stats_influence_plot,
    stats_chisquare_plot,
    stats_vif,
    stats_conprob,
    bayes_boot,
    bayes_boot_plot,
    stats_anova_tidy,
    stats_kruskal_tidy,
    stats_correlation_tidy,
    stats_formula,
    stats_partial_dependence,
)

warnings.warn(
    "broom_sm.broom_func is deprecated. Import from 'broom_sm' directly.",
    DeprecationWarning,
    stacklevel=2,
)
