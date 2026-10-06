"""Benchmarks for ``DataFrame.pint.quantify`` on wide DataFrames.

See https://github.com/hgrecco/pint-pandas/issues/289

The ``float64`` DataFrame mimics an OpenFAST ``.out`` file of a floating
offshore wind turbine simulation: many float channels with a mix of units, a
few of which are unitless (``NO_UNIT``) and some are dimensionless. The
``mixed`` DataFrame has the same shape but cycles through several column
dtypes, including missing values and unitless string columns.

Besides the public ``quantify`` / ``dequantify`` API, candidate
implementations are benchmarked against the current one. Each candidate must
give exactly the same result as the current implementation, in particular the
subdtype of each ``PintArray`` must match the dtype of the input column.

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

from pint_pandas import PintArray, PintType  # noqa: E402
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
def kind(request):
    return request.param


@pytest.fixture(scope="module")
def df(kind):
    return make_dataframe(kind)


@pytest.fixture(scope="module")
def units(df):
    return df.columns.get_level_values(-1)


@pytest.fixture(scope="module")
def expected(df, units):
    return _iloc(df, units)


def _check(df_new, expected):
    # assert_frame_equal compares PintArrays element by element, which is very
    # slow, so compare dtypes and the underlying magnitude arrays instead
    assert df_new.shape == expected.shape
    pd.testing.assert_index_equal(df_new.index, expected.index)
    assert df_new.dtypes.tolist() == expected.dtypes.tolist()
    for i, dtype in enumerate(expected.dtypes):
        result, exp = df_new.iloc[:, i].array, expected.iloc[:, i].array
        if isinstance(dtype, PintType):
            result, exp = result._data, exp._data
        pd.testing.assert_extension_array_equal(result, exp)


def test_quantify(benchmark, df, units, expected):
    df_new = benchmark(lambda: df.pint.quantify(level=-1))
    _check(df_new, expected)


def test_dequantify(benchmark, df):
    df_q = df.pint.quantify(level=-1)
    df_new = benchmark(df_q.pint.dequantify)
    assert df_new.shape == df.shape


# Candidate implementations, benchmarked for comparison with the current
# ``iloc`` based implementation of ``quantify``.
#
# Profiling ``quantify`` shows three hotspots: parsing/formatting the unit
# string for every column in ``PintArray.__init__``, ``df.iloc`` column
# access, and ``DataFrame(dict)`` copying every array.


def _iloc(df, units):
    # current implementation
    return pd.DataFrame(
        {
            i: PintArray(df.iloc[:, i], unit) if unit != NO_UNIT else df.iloc[:, i]
            for i, unit in enumerate(units)
        }
    )


class _DtypeCache(dict):
    # parse each distinct unit once, and build one PintType per (unit, subdtype)
    def __missing__(self, key):
        unit, subdtype = key
        units = self.get(unit) or PintType(unit).units
        self[unit] = units
        self[key] = PintType(units, subdtype)
        return self[key]


def _columns(df):
    """Each column's array, keeping its dtype.

    Columns sharing a numpy dtype are taken from a single 2D block, extension
    arrays (e.g. ``Float64``, ``str``) are taken as they are. The arrays are
    copies, so the result does not share memory with ``df``.
    """
    columns = [None] * df.shape[1]
    numpy_positions = {}
    for i, dtype in enumerate(df.dtypes):
        if isinstance(dtype, np.dtype):
            numpy_positions.setdefault(dtype, []).append(i)
        else:
            columns[i] = df.iloc[:, i].array.copy()
    for positions in numpy_positions.values():
        # .T of the (F-ordered) block is already contiguous, so copy explicitly
        block = df.iloc[:, positions].to_numpy().T.copy()
        for i, values in zip(positions, block):
            columns[i] = pd.arrays.NumpyExtensionArray(values)
    return columns


def _to_numpy(df, units):
    # the issue's to_numpy approach, made to work per dtype block
    return pd.DataFrame(
        {
            i: PintArray(values, unit) if unit != NO_UNIT else values
            for i, (unit, values) in enumerate(zip(units, _columns(df)))
        }
    )


def _apply(df, units):
    return df.apply(
        lambda col: PintArray(col, col.name[-1]) if col.name[-1] != NO_UNIT else col,
        axis=0,
    )


def _astype(df, units):
    cache = _DtypeCache()
    return df.astype(
        {
            col: cache[col[-1], pd.array(df[col]).dtype]
            for col in df.columns
            if col[-1] != NO_UNIT
        }
    )


def _items(df, units):
    # public API only: df.items, cached dtypes, no copy
    cache = _DtypeCache()
    return pd.DataFrame(
        {
            i: PintArray(col.array.copy(), cache[unit, col.array.dtype])
            if unit != NO_UNIT
            else col
            for i, (unit, (_, col)) in enumerate(zip(units, df.items()))
        },
        copy=False,
    )


def _cached_dtype(df, units):
    cache = _DtypeCache()
    return pd.DataFrame(
        {
            i: PintArray(values, cache[unit, values.dtype])
            if unit != NO_UNIT
            else values
            for i, (unit, values) in enumerate(zip(units, _columns(df)))
        }
    )


def _no_copy(df, units):
    cache = _DtypeCache()
    return pd.DataFrame(
        {
            i: PintArray(values, cache[unit, values.dtype])
            if unit != NO_UNIT
            else values
            for i, (unit, values) in enumerate(zip(units, _columns(df)))
        },
        copy=False,
    )


def _pint_array_fast(data, dtype):
    # bypass PintArray.__init__ validation, as an internal _simple_new would
    pa = object.__new__(PintArray)
    pa._dtype = dtype
    pa._data = data
    pa._Q = dtype.ureg.Quantity
    return pa


def _simple_new(df, units):
    cache = _DtypeCache()
    return pd.DataFrame(
        {
            i: _pint_array_fast(values, cache[unit, values.dtype])
            if unit != NO_UNIT
            else values
            for i, (unit, values) in enumerate(zip(units, _columns(df)))
        },
        copy=False,
    )


def _concat(df, units):
    cache = _DtypeCache()
    return pd.concat(
        [
            pd.Series(
                _pint_array_fast(values, cache[unit, values.dtype])
                if unit != NO_UNIT
                else values,
                copy=False,
            )
            for unit, values in zip(units, _columns(df))
        ],
        axis=1,
        ignore_index=True,
    )


@pytest.mark.parametrize(
    "impl",
    [
        _iloc,
        _to_numpy,
        _apply,
        _astype,
        _items,
        _cached_dtype,
        _no_copy,
        _simple_new,
        _concat,
    ],
    ids=lambda f: f.__name__.lstrip("_"),
)
def test_quantify_candidates(benchmark, kind, df, units, expected, impl):
    benchmark.group = f"quantify candidates ({kind})"
    df_new = benchmark(impl, df, units)
    _check(df_new, expected)


@pytest.mark.parametrize("dtype", ["float64", "Float64", "int64"])
@pytest.mark.parametrize(
    "impl",
    [
        _iloc,
        _to_numpy,
        _apply,
        _astype,
        _items,
        _cached_dtype,
        _no_copy,
        _simple_new,
        _concat,
    ],
    ids=lambda f: f.__name__.lstrip("_"),
)
def test_quantify_candidates_copy(impl, dtype):
    # the result must be writable and must not share memory with the input
    df = pd.DataFrame(
        {
            ("a", "m"): pd.Series([1, 2], dtype=dtype),
            ("b", "s"): pd.Series([3, 4], dtype=dtype),
        }
    )
    df_new = impl(df, df.columns.get_level_values(-1))
    df_new.iloc[0, 0] = PintType.ureg.Quantity(99, "m")
    assert df.iloc[0, 0] == 1
