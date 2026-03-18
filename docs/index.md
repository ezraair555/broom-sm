# broom-sm documentation

broom-sm brings [tidy-models](https://broom.tidymodels.org/) ergonomics to statsmodels. Use pandas-native verbs (`stats_tidy`, `stats_glance`, `stats_augment`) plus bootstrapping, diagnostics, Bayesian helpers, and a CLI.

- **Model coverage**: OLS, GLMs (Poisson, Gamma, Beta, Negative Binomial), Quantile Regression, GEE, MixedLM, and PHReg.
- **Robust stats**: pass `cov_type`, `cov_kwds`, `family`, `link`, and `weights` directly.
- **Extensible**: register new fitters via `MODEL_CONFIG` and reuse tidy verbs instantly.

```{toctree}
:maxdepth: 2
:caption: Guides

Overview <readme>
Tutorials <tutorials/index>
How-to guides <howto/index>
Contributions & Help <contributing>
License <license>
Authors <authors>
Changelog <changelog>
Module Reference <api/modules>
```

## API reference

The documentation is auto-generated with `sphinx.ext.autodoc` and refreshed each time you run `tox -e docs` or `sphinx-build -b html docs docs/_build/html`. All public functions live under `broom_sm.*` — explore them in the "Module Reference" section above.
