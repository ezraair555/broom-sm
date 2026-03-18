"""Public API for broom-sm."""
from __future__ import annotations

import logging
from importlib import metadata

from .tidy import (
    stats_tidy,
    stats_glance,
    stats_augment,
    stats_anova_tidy,
    stats_kruskal_tidy,
    stats_correlation_tidy,
    stats_formula,
    stats_partial_dependence,
)
from .bootstrap import boot_tidy, boot_glance, boot_augment
from .diagnostics import (
    stats_power,
    stats_residual_plot,
    stats_ols_plot,
    stats_influence_plot,
    stats_chisquare_plot,
    stats_vif,
    stats_conprob,
    stats_coef_forest,
)
from .bayes import bayes_boot, bayes_boot_plot
from .reporting import stats_compare, stats_report

LOGGER = logging.getLogger("broom_sm")

try:  # pragma: no cover
    dist_name = "broom-sm"
    __version__ = metadata.version(dist_name)
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "stats_tidy",
    "stats_glance",
    "stats_augment",
    "stats_anova_tidy",
    "stats_kruskal_tidy",
    "stats_correlation_tidy",
    "stats_formula",
    "stats_partial_dependence",
    "boot_tidy",
    "boot_glance",
    "boot_augment",
    "stats_power",
    "stats_residual_plot",
    "stats_ols_plot",
    "stats_influence_plot",
    "stats_chisquare_plot",
    "stats_vif",
    "stats_conprob",
    "stats_coef_forest",
    "bayes_boot",
    "bayes_boot_plot",
    "stats_compare",
    "stats_report",
]
