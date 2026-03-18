# Extending `MODEL_CONFIG`

`broom_sm.model_registry` keeps a central registry of statsmodels fitters. Every tidy verb (`stats_tidy`, `stats_glance`, `stats_augment`) looks up behavior there, so adding a new model is a matter of one function call.

```python
from broom_sm.model_registry import ModelSpec, register_model
import statsmodels.formula.api as smf

register_model(
    "tobit",
    ModelSpec(
        fitter=lambda formula, data, **kwargs: smf.tobit(formula, data=data, **kwargs).fit(),
        stat_name="z_stat",
        metadata={"aliases": ["censored"]},
    ),
)
```

A few tips:

- **Aliases** — add `metadata={"aliases": ["censored"]}` to reuse the same spec under multiple `stat_type` strings.
- **Families & weights** — set `accepts_family=True` or `accepts_weights=True` on the spec so tidy verbs expose `family=`, `link=`, or `weights=` arguments.
- **Robust SEs** — tidy verbs automatically forward `cov_type` and `cov_kwds` to the fitter. No extra work is required.
- **Documentation** — mention new keys in the README and docs so users discover them.

After registration, the new `stat_type` is available immediately:

```python
penguins.stats_tidy("body_mass_g ~ bill_length_mm", stat_type="tobit")
```
