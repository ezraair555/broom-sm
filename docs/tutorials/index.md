# Tutorials

Tutorials provide end-to-end walkthroughs of statistical workflows using broom-sm. Each tutorial combines multiple verbs and utilities to solve real-world analysis problems.

```{toctree}
:maxdepth: 1

Tidy Workflow with broom-sm <tidy-workflow>
Bootstrap Inference <bootstrap-inference>
Model Comparison and Selection <model-comparison>
Diagnostics and Visualization <diagnostics-viz>
```

## Tutorial Structure

Each tutorial follows a consistent structure:

1. **Learning Objectives** — What you'll learn
2. **Background** — Statistical context and principles
3. **Setup** — Data loading and package imports
4. **Step-by-Step Analysis** — Guided workflow with explanations
5. **Interpretation** — How to understand the results
6. **Next Steps** — Related topics and extensions

## Available Tutorials

### [Tidy Workflow with broom-sm](tidy-workflow.md)

Learn the core tidy verbs (`stats_tidy`, `stats_glance`, `stats_augment`) through a complete regression analysis. Covers:
- Formula specification
- Model fitting with statsmodels
- Extracting tidy coefficient tables
- Generating predictions with intervals
- Visualizing results

**Prerequisites:** Basic pandas and statsmodels knowledge

**Time:** 15 minutes

---

### [Bootstrap Inference](bootstrap-inference.md)

Master resampling-based inference using broom-sm's bootstrap helpers. Covers:
- Non-parametric bootstrap with `boot_tidy()`
- Confidence interval construction
- Comparing bootstrap vs. analytic SEs
- Bayesian bootstrap with `bayes_boot()`

**Prerequisites:** Completion of Tidy Workflow tutorial

**Time:** 20 minutes

---

### [Model Comparison and Selection](../howto/compare-models.md)

Learn to compare multiple models and select the best fit. Covers:
- Information criteria (AIC, BIC)
- Nested model comparison
- Cross-validation basics
- Model averaging concepts

**Prerequisites:** Tidy Workflow tutorial

**Time:** 25 minutes

---

### [Diagnostics and Visualization](../howto/plot_gallery.md)

Comprehensive guide to regression diagnostics and visualization. Covers:
- Residual analysis
- Influence diagnostics
- Multicollinearity detection (VIF)
- Creating publication-quality plots

**Prerequisites:** Tidy Workflow tutorial

**Time:** 30 minutes

---

## Learning Path

For beginners, we recommend following this order:

```
Tidy Workflow → Bootstrap Inference → Model Comparison → Diagnostics
```

Each tutorial builds on concepts from the previous ones.

## Contributing

Have a tutorial idea? See our [Contributing Guide](../CONTRIBUTING.md) for how to submit new tutorials.

## Related Resources

- **[How-to Guides](../howto/index.md)** — Task-oriented recipes
- **[API Reference](../api-reference.md)** — Function documentation
- **[infer (R) tutorials](https://infer.tidymodels.org/articles/)** — Inspiration for statistical inference workflows
- **[broom (R) vignettes](https://broom.tidymodels.org/articles/)** — Tidy model output examples
