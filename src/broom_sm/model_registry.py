"""Statsmodels model registry used by tidy/glance/augment helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

import statsmodels.api as sm
import statsmodels.formula.api as smf

from ._utils import LOGGER, prepare_family


@dataclass
class ModelSpec:
    fitter: Callable[..., Any]
    stat_name: str
    has_rsq: bool = False
    pseudo_rsq_attr: Optional[str] = None
    accepts_family: bool = False
    accepts_weights: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


_MODEL_CONFIG: MutableMapping[str, ModelSpec] = {}


def register_model(name: str, spec: ModelSpec) -> None:
    normalized = name.lower()
    _MODEL_CONFIG[normalized] = spec
    LOGGER.debug("Registered model '%s' with metadata %s", normalized, spec.metadata)


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _MODEL_CONFIG[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            f"Unsupported stat_type: {name}. Available models: {list(_MODEL_CONFIG)}"
        ) from exc


def iter_model_config() -> Mapping[str, ModelSpec]:
    return dict(_MODEL_CONFIG)


# --- Default registry -----------------------------------------------------

def _glm_factory(default_family: str) -> Callable[..., Any]:
    def _wrapper(formula: str, data, family=None, link=None, weights=None, **kwargs):
        fam = prepare_family(family, default=default_family, link=link)
        fit_kwargs = {}
        if kwargs.get("cov_type"):
            fit_kwargs["cov_type"] = kwargs.pop("cov_type")
            if kwargs.get("cov_kwds"):
                fit_kwargs["cov_kwds"] = kwargs.pop("cov_kwds")
        if weights is not None:
            kwargs.setdefault("freq_weights", weights)
        model = smf.glm(formula, data=data, family=fam, **kwargs)
        return model.fit(**fit_kwargs) if fit_kwargs else model.fit()

    return _wrapper


def _standard_formula_fitter(fitter: Callable[..., Any]) -> Callable[..., Any]:
    def _wrapper(formula: str, data, **kwargs):
        fit_kwargs = {
            key: kwargs.pop(key)
            for key in ["cov_type", "cov_kwds"]
            if key in kwargs
        }
        model = fitter(formula, data=data, **kwargs)
        return model.fit(**fit_kwargs) if fit_kwargs else model.fit()

    return _wrapper


register_model(
    "ols",
    ModelSpec(
        fitter=_standard_formula_fitter(smf.ols),
        stat_name="t_stat",
        has_rsq=True,
        accepts_weights=True,
    ),
)

register_model(
    "glm",
    ModelSpec(
        fitter=_glm_factory("gaussian"),
        stat_name="z_stat",
        accepts_family=True,
        accepts_weights=True,
    ),
)

register_model(
    "logit",
    ModelSpec(
        fitter=_glm_factory("binomial"),
        stat_name="z_stat",
        pseudo_rsq_attr="prsquared",
        accepts_family=True,
        accepts_weights=True,
    ),
)

register_model(
    "poisson",
    ModelSpec(
        fitter=_glm_factory("poisson"),
        stat_name="z_stat",
        accepts_family=True,
        accepts_weights=True,
    ),
)

register_model(
    "gamma",
    ModelSpec(
        fitter=_glm_factory("gamma"),
        stat_name="z_stat",
        accepts_family=True,
        accepts_weights=True,
    ),
)

register_model(
    "beta",
    ModelSpec(
        fitter=_glm_factory("beta"),
        stat_name="z_stat",
        accepts_family=True,
        accepts_weights=True,
    ),
)

register_model(
    "nbinom",
    ModelSpec(
        fitter=_glm_factory("negativebinomial"),
        stat_name="z_stat",
        accepts_family=True,
        accepts_weights=True,
        metadata={"aliases": ["negativebinomial"]},
    ),
)

register_model(
    "quantreg",
    ModelSpec(
        fitter=_standard_formula_fitter(smf.quantreg),
        stat_name="t_stat",
    ),
)


register_model(
    "gee",
    ModelSpec(
        fitter=_standard_formula_fitter(smf.gee),
        stat_name="z_stat",
        accepts_weights=True,
        metadata={"requires": ["groups"]},
    ),
)

register_model(
    "mixedlm",
    ModelSpec(
        fitter=_standard_formula_fitter(smf.mixedlm),
        stat_name="t_stat",
        accepts_weights=True,
        metadata={"requires": ["groups"]},
    ),
)

register_model(
    "phreg",
    ModelSpec(
        fitter=_standard_formula_fitter(smf.phreg),
        stat_name="z_stat",
        metadata={"aliases": ["survival"]},
    ),
)


# Maintain backward-compatible aliases
_ALIAS_MAP: Dict[str, str] = {}
for key, spec in list(iter_model_config().items()):
    for alias in spec.metadata.get("aliases", []):
        _ALIAS_MAP[alias] = key


if _ALIAS_MAP:
    for alias, target in _ALIAS_MAP.items():
        _MODEL_CONFIG.setdefault(alias, _MODEL_CONFIG[target])
