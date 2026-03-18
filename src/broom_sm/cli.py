"""Lightweight CLI entry point for broom-sm."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

import pandas as pd

from .reporting import stats_compare, stats_report
from .tidy import prepare_fit


def _load_frame(path: str, delimiter: str = ",", index_col: str | None = None) -> pd.DataFrame:
    kwargs = {"sep": delimiter}
    if index_col:
        kwargs["index_col"] = index_col
    return pd.read_csv(path, **kwargs)


def cmd_report(args: argparse.Namespace) -> None:
    df = _load_frame(args.data, delimiter=args.delimiter, index_col=args.index_col)
    report = stats_report(
        df,
        formula=args.formula,
        stat_type=args.stat_type,
    )
    if args.format == "json":
        payload = {key: value.to_dict(orient="records") for key, value in report.items()}
        print(json.dumps(payload, indent=2))
    else:
        for key, value in report.items():
            print(f"## {key}\n{value.to_csv(index=False)}")


def cmd_compare(args: argparse.Namespace) -> None:
    df = _load_frame(args.data, delimiter=args.delimiter, index_col=args.index_col)
    models = {}
    for formula in args.formulas:
        result, _, _ = prepare_fit(df, formula, args.stat_type, None, {})
        models[formula] = result
    table = stats_compare(models)
    if args.format == "json":
        print(table.to_json(orient="records"))
    else:
        print(table.to_csv(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="broom-sm CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="Generate tidy/glance/augment report")
    report.add_argument("--data", required=True, help="Path to CSV data file")
    report.add_argument("--formula", required=True)
    report.add_argument("--stat-type", required=True)
    report.add_argument("--format", choices=["json", "csv"], default="json")
    report.add_argument("--delimiter", default=",")
    report.add_argument("--index-col")
    report.set_defaults(func=cmd_report)

    compare = sub.add_parser("compare", help="Compare formulas via AIC/BIC")
    compare.add_argument("--data", required=True)
    compare.add_argument("--stat-type", required=True)
    compare.add_argument("--formulas", nargs="+", required=True)
    compare.add_argument("--format", choices=["json", "csv"], default="json")
    compare.add_argument("--delimiter", default=",")
    compare.add_argument("--index-col")
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
