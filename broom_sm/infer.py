"""broom_sm.infer - Declarative inference workflows inspired by R's infer."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Literal, Tuple

StatType = Literal[
    "mean",
    "median",
    "proportion",
    "diff in means",
    "diff in medians",
    "diff in proportions",
]


class InferPipeline:
    """Pipeline that mimics infer::specify() %>% hypothesize() %>% generate() %>% calculate()."""

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.data = data.copy()
        self.response: Optional[str] = None
        self.explanatory: Optional[str] = None
        self.null_type: Optional[str] = None
        self.generated_data: Optional[pd.DataFrame] = None

    # ----- Core verbs -----------------------------------------------------
    def specify(self, response: str, explanatory: Optional[str] = None) -> "InferPipeline":
        if response not in self.data.columns:
            raise KeyError(f"Response column '{response}' not found in data.")
        if explanatory and explanatory not in self.data.columns:
            raise KeyError(f"Explanatory column '{explanatory}' not found in data.")
        self.response = response
        self.explanatory = explanatory
        return self

    def hypothesize(self, null: Literal["independence", "point"] = "independence") -> "InferPipeline":
        self.null_type = null
        return self

    def generate(self, reps: int = 1000, type: Literal["bootstrap", "permute"] = "bootstrap") -> "InferPipeline":
        if self.response is None:
            raise RuntimeError("Call specify() before generate().")
        results = []
        for i in range(reps):
            if type == "bootstrap":
                sample = self.data.sample(frac=1.0, replace=True)
            elif type == "permute":
                sample = self.data.copy()
                sample[self.response] = np.random.permutation(sample[self.response].values)
            else:
                raise ValueError("Type must be 'bootstrap' or 'permute'.")
            sample = sample.assign(replicate=i + 1)
            results.append(sample)
        self.generated_data = pd.concat(results, ignore_index=True)
        return self

    def calculate(self, stat: StatType = "mean") -> pd.DataFrame:
        if self.response is None:
            raise RuntimeError("Call specify() before calculate().")
        df = self.generated_data if self.generated_data is not None else self.data.assign(replicate=1)

        if stat == "mean":
            values = df.groupby("replicate")[self.response].mean()
        elif stat == "median":
            values = df.groupby("replicate")[self.response].median()
        elif stat == "proportion":
            values = df.groupby("replicate")[self.response].mean()
        elif stat in {"diff in means", "diff in medians", "diff in proportions"}:
            if not self.explanatory:
                raise RuntimeError("Explanatory variable required for difference statistics.")
            if df[self.explanatory].nunique() != 2:
                raise RuntimeError("Explanatory variable must have exactly two levels.")
            agg = {
                "diff in means": "mean",
                "diff in medians": "median",
                "diff in proportions": "mean",
            }[stat]

            def diff_func(group: pd.DataFrame) -> float:
                pivot = group.groupby(self.explanatory)[self.response].agg(agg)
                return pivot.iloc[0] - pivot.iloc[1]

            values = df.groupby("replicate").apply(diff_func)
        else:
            raise NotImplementedError(f"Statistic '{stat}' not yet implemented.")

        return values.reset_index(name="stat")

    # ----- Convenience helpers -------------------------------------------
    def confidence_interval(
        self,
        stat: StatType = "mean",
        level: float = 0.95,
    ) -> Tuple[float, float]:
        """Return percentile-based confidence interval for the chosen statistic."""
        dist = self.calculate(stat)["stat"].values
        alpha = 1 - level
        return (np.quantile(dist, alpha / 2), np.quantile(dist, 1 - alpha / 2))


# Functional interface ------------------------------------------------------

def specify(data: pd.DataFrame, response: str, explanatory: Optional[str] = None) -> InferPipeline:
    return InferPipeline(data).specify(response, explanatory)


def hypothesize(pipeline: InferPipeline, null: Literal["independence", "point"] = "independence") -> InferPipeline:
    return pipeline.hypothesize(null)


def generate(pipeline: InferPipeline, reps: int = 1000, type: Literal["bootstrap", "permute"] = "bootstrap") -> InferPipeline:
    return pipeline.generate(reps=reps, type=type)


def calculate(pipeline: InferPipeline, stat: StatType = "mean") -> pd.DataFrame:
    return pipeline.calculate(stat)


def confidence_interval(pipeline: InferPipeline, stat: StatType = "mean", level: float = 0.95) -> Tuple[float, float]:
    return pipeline.confidence_interval(stat=stat, level=level)
