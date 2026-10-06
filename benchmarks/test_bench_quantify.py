"""Benchmarks for ``DataFrame.pint.quantify`` and ``DataFrame.pint.dequantify``
on wide DataFrames.

See https://github.com/hgrecco/pint-pandas/issues/289

The ``float64`` DataFrame mimics an OpenFAST ``.out`` file of a floating
offshore wind turbine simulation: many float channels with a mix of units, a
few of which are unitless (``NO_UNIT``) and some are dimensionless. The
``mixed`` DataFrame has the same shape but cycles through several column
dtypes, including missing values and unitless string columns.

Run with::

    pixi run -e bench bench

or, with ``pytest-benchmark`` installed::

    pytest benchmarks --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pytest_benchmark")

from pint_pandas import PintType  # noqa: E402
from pint_pandas.pint_array import NO_UNIT  # noqa: E402

UNITS = [
    "s",
    "m/s",
    "deg",
    "deg/s",
    "deg/s^2",
    "m",
    "m/s^2",
    "kN",
    "kN*m",
    "kW",
    "rpm",
    "dimensionless",
    NO_UNIT,
]

N_ROWS = 10_000
N_COLS = 200


def _mixed_column(rng, i, unit, n_rows):
    if unit == NO_UNIT and i % 2:
        return pd.Series(rng.choice(["a", "b", "c"], n_rows), dtype="str")
    values = rng.standard_normal(n_rows)
    values[:: 100 + i] = np.nan
    kind = i % 4
    if kind == 0:
        return pd.Series(values)
    if kind == 1:
        return pd.Series(values, dtype="Float64")
    if kind == 2:
        return pd.Series(values, dtype="float32")
    return pd.Series(rng.integers(-100, 100, n_rows))


def make_dataframe(kind="float64", n_rows=N_ROWS, n_cols=N_COLS, seed=0):
    rng = np.random.default_rng(seed)
    columns = pd.MultiIndex.from_tuples(
        [(f"Channel{i}", UNITS[i % len(UNITS)]) for i in range(n_cols)]
    )
    if kind == "float64":
        return pd.DataFrame(rng.standard_normal((n_rows, n_cols)), columns=columns)
    df = pd.concat(
        [_mixed_column(rng, i, col[-1], n_rows) for i, col in enumerate(columns)],
        axis=1,
    )
    df.columns = columns
    return df


@pytest.fixture(scope="module", params=["float64", "mixed"])
def df(request):
    return make_dataframe(request.param)


def test_quantify(benchmark, df):
    df_new = benchmark(df.pint.quantify, level=-1)
    n_units = sum(u != NO_UNIT for u in df.columns.get_level_values(-1))
    assert df_new.shape == df.shape
    assert sum(isinstance(dt, PintType) for dt in df_new.dtypes) == n_units


def test_dequantify(benchmark, df):
    df_q = df.pint.quantify(level=-1)
    df_new = benchmark(df_q.pint.dequantify)
    assert df_new.shape == df.shape
