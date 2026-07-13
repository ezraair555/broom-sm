import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from broom_sm.cli import main


@pytest.fixture
def temp_csv() -> str:
    rng = np.random.default_rng(42)
    x = rng.normal(size=50)
    y = 2 + 1.5 * x + rng.normal(scale=0.1, size=50)
    df = pd.DataFrame({"y": y, "x": x, "idx_col": np.arange(50)})
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        path = f.name
    yield path
    os.remove(path)


def test_cli_report_json(temp_csv, capsys):
    main(["report", "--data", temp_csv, "--formula", "y ~ x", "--stat-type", "ols", "--format", "json", "--index-col", "idx_col"])
    captured = capsys.readouterr()
    
    # Verify valid JSON output
    data = json.loads(captured.out)
    assert "tidy" in data
    assert "glance" in data
    assert "augment" in data
    assert len(data["tidy"]) == 2  # Intercept + x


def test_cli_report_csv(temp_csv, capsys):
    main(["report", "--data", temp_csv, "--formula", "y ~ x", "--stat-type", "ols", "--format", "csv"])
    captured = capsys.readouterr()
    
    assert "## tidy" in captured.out
    assert "## glance" in captured.out
    assert "## augment" in captured.out


def test_cli_compare_json(temp_csv, capsys):
    main(["compare", "--data", temp_csv, "--stat-type", "ols", "--formulas", "y ~ x", "y ~ 1", "--format", "json"])
    captured = capsys.readouterr()
    
    data = json.loads(captured.out)
    assert len(data) == 2
    assert "model" in data[0]
    assert "aic" in data[0]


def test_cli_compare_csv(temp_csv, capsys):
    main(["compare", "--data", temp_csv, "--stat-type", "ols", "--formulas", "y ~ x", "y ~ 1", "--format", "csv"])
    captured = capsys.readouterr()
    
    assert "model,aic,bic,llf,nobs" in captured.out
