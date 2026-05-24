# Installation

## Requirements

broom-sm requires Python 3.8+ and the following core dependencies:

- `pandas` ≥ 1.3
- `statsmodels` ≥ 0.12
- `pandas_flavor` ≥ 0.2
- `patsy` ≥ 0.5
- `numpy` ≥ 1.20
- `scipy` ≥ 1.7

### Optional Dependencies

**For visualization** (`broom-sm[viz]`):
- `matplotlib` ≥ 3.4
- `seaborn` ≥ 0.11

**For Bayesian bootstrap** (`broom-sm[bayes]`):
- `bayesian_bootstrap` ≥ 1.0

**For development** (`broom-sm[dev]`):
- `pytest` ≥ 6.0
- `pytest-cov` ≥ 2.0
- `black` ≥ 22.0
- `flake8` ≥ 4.0
- `mypy` ≥ 0.900
- `sphinx` ≥ 4.0
- `myst-parser` ≥ 0.15

## Installation Methods

### From PyPI (Recommended)

Install the core package:

```bash
pip install broom-sm
```

Install with visualization support:

```bash
pip install broom-sm[viz]
```

Install with Bayesian bootstrap support:

```bash
pip install broom-sm[bayes]
```

Install all extras:

```bash
pip install broom-sm[all]
```

### From Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ezraair555/broom-sm.git
cd broom-sm
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

### From Conda

broom-sm is available on conda-forge:

```bash
conda install -c conda-forge broom-sm
```

## Verifying Installation

After installation, verify that broom-sm is working:

```python
import broom_sm
print(broom_sm.__version__)

import pandas as pd
import statsmodels.api as sm

mtcars = sm.datasets.get_rdataset("mtcars").data
report = mtcars.stats_report(
    formula="mpg ~ wt + hp",
    stat_type="ols"
)
print(report["tidy"])
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install --upgrade pandas statsmodels pandas_flavor patsy numpy scipy
```

### Visualization Issues

If plot functions fail, install the visualization extras:

```bash
pip install broom-sm[viz]
```

### Bayesian Bootstrap Issues

If Bayesian helpers fail, install the bayesian_bootstrap package:

```bash
pip install bayesian_bootstrap
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade broom-sm
```

To install a specific version:

```bash
pip install broom-sm==0.5.0
```

## Uninstalling

To remove broom-sm:

```bash
pip uninstall broom-sm
```

## Platform Support

broom-sm is tested on:
- Linux (Ubuntu 20.04+, CentOS 7+)
- macOS (10.15+)
- Windows (10+)

Python versions tested: 3.8, 3.9, 3.10, 3.11, 3.12
