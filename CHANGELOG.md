# Changelog

## Version 0.1.1 — 2026-06-20

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
