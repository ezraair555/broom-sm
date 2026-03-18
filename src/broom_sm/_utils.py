"""Utility helpers shared across broom-sm modules."""
from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

LOGGER = logging.getLogger("broom_sm")

_FAMILY_ALIASES: Dict[str, str] = {
    "gaussian": "Gaussian",
    "normal": "Gaussian",
    "binomial": "Binomial",
    "poisson": "Poisson",
    "gamma": "Gamma",
    "inversegaussian": "InverseGaussian",
    "negativebinomial": "NegativeBinomial",
    "nbinom": "NegativeBinomial",
    "beta": "Beta",
    "tweedie": "Tweedie",
}

_LINK_ALIASES: Dict[str, str] = {
    "identity": "identity",
    "log": "log",
    "logit": "logit",
    "probit": "probit",
    "cloglog": "cloglog",
    "inverse_squared": "inverse_squared",
    "inverse": "inverse",
    "power": "power",
}


def _get_family_class(name: str) -> Optional[Callable[[], sm.families.family.Family]]:
    """Resolve a statsmodels family class name, guarding for availability."""
    attr_name = _FAMILY_ALIASES.get(name.lower(), name)
    try:
        return getattr(sm.families, attr_name)
    except AttributeError:
        warnings.warn(
            f"statsmodels family '{name}' is not available in this environment; "
            "falling back to Gaussian.",
            stacklevel=2,
        )
        return sm.families.Gaussian


def _get_link_object(link: Optional[str], family: sm.families.family.Family) -> Optional[Any]:
    if link is None:
        return None
    attr_name = _LINK_ALIASES.get(link.lower(), link)
    try:
        link_module = sm.families.links
        link_cls = getattr(link_module, attr_name)
    except AttributeError:
        warnings.warn(
            f"statsmodels link '{link}' is not available; using the default link "
            f"for {family.__class__.__name__}.",
            stacklevel=2,
        )
        return None
    return link_cls()


def prepare_family(
    family: Optional[Any],
    default: str,
    link: Optional[str] = None,
) -> sm.families.family.Family:
    """Return a family instance honoring optional link overrides."""
    if isinstance(family, sm.families.family.Family):
        fam = family
    elif isinstance(family, str):
        fam_cls = _get_family_class(family)
        fam = fam_cls()
    elif family is None:
        fam_cls = _get_family_class(default)
        fam = fam_cls()
    else:
        fam = family

    link_obj = _get_link_object(link, fam)
    if link_obj is not None:
        try:
            fam.link = link_obj
        except Exception:  # pragma: no cover - statsmodels internals
            warnings.warn(
                f"Unable to set link '{link}' on family {fam}; using default link.",
                stacklevel=2,
            )
    return fam


def coerce_weights(weights: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if weights is None:
        return None
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1:
        raise ValueError("weights must be a 1-D sequence")
    return arr


def dataframe_like(
    data: Optional[pd.DataFrame],
    fallback: Optional[pd.DataFrame],
    name: str,
) -> pd.DataFrame:
    if data is not None:
        return data
    if fallback is not None:
        return fallback
    raise ValueError(f"'{name}' must be provided when a pre-fitted model is supplied")


def ensure_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return value
    return [value]
