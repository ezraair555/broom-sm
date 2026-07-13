# Changelog

## Version 0.1.3 — 2026-07-13

### Quality Fixes and Enhancements
- **Fixed OLS Weights:** Changed the Ordinary Least Squares fitter to use WLS when weights are supplied, making weights functional rather than silent placebos.
- **Fixed `stats_augment` NaN alignment:** Rewrote alignment logic to assign pandas Series directly, avoiding length mismatches when rows are dropped. Used pre-transformed exog values for predictions.
- **Optimized `stats_tidy` merges:** Replaced consecutive DataFrame merges on the `"term"` column with direct coefficient construction.
- **Dependency cleanup:** Moved `seaborn`, `matplotlib`, and `bayesian_bootstrap` to optional package extras, adding guarded imports and descriptive import errors.

### Tests
- Created extensive tests for visual diagnostics, CLI parameters (`--index-col`), fallback paths, and mocked import environments.
- Raised codebase line coverage to **96%** (62 passed).

## Version 0.1.2 — 2026-06-20

### P0 fixes

- `stats_augment` now validates index uniqueness for `data` and `new_data`, rejects overlapping indices, and aligns residuals / influence diagnostics position-wise for the in-sample path. The `.in_sample` flag is now a single boolean rather than a `set`-based membership test.
- `prepare_fit` passes `freq_weights` to GLM-family fitters and `weights` to OLS.
- `boot_tidy`, `boot_glance`, and `boot_augment` raise `RuntimeError` when all bootstrap replications fail.
- `stats_residual_plot` validates that `y` is numeric.
- `stats_vif` warns that the intercept is omitted and handles no-intercept formulas consistently.

### P1 fixes

- Validate `anova_type` is 1/2/3 in `stats_anova_tidy`.
- Validate `group_col` and `value_col` in `stats_kruskal_tidy`.
- Validate `columns` existence and numeric dtype in `stats_correlation_tidy`.
- Quote non-syntactic column names in `stats_formula`.
- `bayes_boot` validates inputs and warns on NaN.
- `stats_chisquare_plot` drops NaN categories.
- Updated `setup.cfg` URLs to `ezraair555/broom-sm`.
- Removed the unused `extra_sm` package.

### Tests

- Added 23 regression tests covering all P0 fixes and selected P1 fixes.
- Full test suite: 31 passed.
