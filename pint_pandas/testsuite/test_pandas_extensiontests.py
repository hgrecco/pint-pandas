"""
This file contains the tests required by pandas for an ExtensionArray and ExtensionType.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from pandas.core import ops
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
    PandasDtype,
    PeriodDtype,
)
from pandas.tests.extension import base
from pandas.tests.extension.conftest import (  # noqa: F401,F811
    as_array,
    as_frame,
    as_series,
    fillna_method,
    groupby_apply_op,
    use_numpy,
)
from pint.errors import DimensionalityError

from pint_pandas import PintArray, PintType

ureg = PintType.ureg


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture
def dtype():
    return PintType("pint[meter]")


@pytest.fixture
def data():
    return PintArray.from_1darray_quantity(np.arange(start=1.0, stop=101.0) * ureg.nm)


@pytest.fixture
def data_missing():
    return PintArray.from_1darray_quantity([np.nan, 1.0] * ureg.meter)


@pytest.fixture
def data_for_twos():
    x = [
        2.0,
    ] * 100
    return PintArray.from_1darray_quantity(x * ureg.meter)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data):
    """Return different versions of data for count times"""
    # no idea what I'm meant to put here, try just copying from https://github.com/pandas-dev/pandas/blob/master/pandas/tests/extension/integer/test_integer.py
    def gen(count):
        for _ in range(count):
            yield data

    yield gen


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


@pytest.fixture
def data_for_sorting():
    return PintArray.from_1darray_quantity([0.3, 10.0, -50.0] * ureg.centimeter)
    # should probably get more sophisticated and do something like
    # [1 * ureg.meter, 3 * ureg.meter, 10 * ureg.centimeter]


@pytest.fixture
def data_missing_for_sorting():
    return PintArray.from_1darray_quantity([4.0, np.nan, -5.0] * ureg.centimeter)
    # should probably get more sophisticated and do something like
    # [4 * ureg.meter, np.nan, 10 * ureg.centimeter]


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values."""
    return lambda x, y: bool(np.isnan(x.magnitude)) & bool(np.isnan(y.magnitude))


@pytest.fixture
def na_value():
    return PintType("meter").na_value


@pytest.fixture
def data_for_grouping():
    # should probably get more sophisticated here and use units on all these
    # quantities
    a = 1.0
    b = 2.0**32 + 1
    c = 2.0**32 + 10
    return PintArray.from_1darray_quantity([b, b, np.nan, np.nan, a, a, b, c] * ureg.m)


# === missing from pandas extension docs about what has to be included in tests ===
# copied from pandas/pandas/conftest.py
_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations
    """
    return request.param


@pytest.fixture(params=["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations

    * >=
    * >
    * ==
    * !=
    * <
    * <=
    """
    return request.param


# commented functions aren't implemented
_all_numeric_reductions = [
    "sum",
    "max",
    "min",
    "mean",
    # "prod",
    "std",
    "var",
    "median",
    "sem",
    "kurt",
    "skew",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    return request.param


_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.
    """
    return request.param


@pytest.fixture
def invalid_scalar(data):
    """
    A scalar that *cannot* be held by this ExtensionArray.
    The default should work for most subclasses, but is not guaranteed.
    If the array can hold any item (i.e. object dtype), then use pytest.skip.
    """
    return object.__new__(object)


# =================================================================


class TestCasting(base.BaseCastingTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestDtype(base.BaseDtypeTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    pass


class TestGroupby(base.BaseGroupbyTests):
    # @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
    def test_groupby_apply_identity(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        result = df.groupby("A").B.apply(lambda x: x.array)
        expected = pd.Series(
            [
                df.B.iloc[[0, 1, 6]].array,
                df.B.iloc[[2, 3]].array,
                df.B.iloc[[4, 5]].array,
                df.B.iloc[[7]].array,
            ],
            index=pd.Index([1, 2, 3, 4], name="A"),
            name="B",
        )
        self.assert_series_equal(result, expected)

    # @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        result = df.groupby("B", as_index=as_index).A.mean()
        _, uniques = pd.factorize(data_for_grouping, sort=True)

        if as_index:
            index = pd.Index._with_infer(uniques, name="B")
            expected = pd.Series([3.0, 1.0, 4.0], index=index, name="A")
            self.assert_series_equal(result, expected)
        else:
            expected = pd.DataFrame({"B": uniques, "A": [3.0, 1.0, 4.0]})
            self.assert_frame_equal(result, expected)

    def test_in_numeric_groupby(self, data_for_grouping):
        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3, 3, 1, 4],
                "B": data_for_grouping,
                "C": [1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        result = df.groupby("A").sum().columns

        if data_for_grouping.dtype._is_numeric:
            expected = pd.Index(["B", "C"])
        else:
            expected = pd.Index(["C"])

        tm.assert_index_equal(result, expected)

    # @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
    def test_groupby_extension_no_sort(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        result = df.groupby("B", sort=False).A.mean()
        _, index = pd.factorize(data_for_grouping, sort=False)

        index = pd.Index._with_infer(index, name="B")
        expected = pd.Series([1.0, 3.0, 4.0], index=index, name="A")
        self.assert_series_equal(result, expected)


class TestInterface(base.BaseInterfaceTests):
    pass


class TestMethods(base.BaseMethodsTests):
    # @pytest.mark.xfail(
    #     run=True, reason="TypeError: 'float' object is not subscriptable"
    # )
    def test_where_series(self, data, na_value, as_frame):
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]

        orig = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))
        ser = orig.copy()
        cond = np.array([True, True, False, False])

        if as_frame:
            ser = ser.to_frame(name="a")
            cond = cond.reshape(-1, 1)

        result = ser.where(cond)
        expected = pd.Series(
            cls._from_sequence([a, a, na_value, na_value], dtype=data.dtype)
        )

        if as_frame:
            expected = expected.to_frame(name="a")
        self.assert_equal(result, expected)

        ser.mask(~cond, inplace=True)
        self.assert_equal(ser, expected)

        # array other
        ser = orig.copy()
        if as_frame:
            ser = ser.to_frame(name="a")
        cond = np.array([True, False, True, True])
        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        if as_frame:
            other = pd.DataFrame({"a": other})
            cond = pd.DataFrame({"a": cond})
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype))
        if as_frame:
            expected = expected.to_frame(name="a")
        self.assert_equal(result, expected)

        ser.mask(~cond, other, inplace=True)
        self.assert_equal(ser, expected)


class TestArithmeticOps(base.BaseArithmeticOpsTests):
    def _check_divmod_op(self, s, op, other, exc=None):
        # divmod has multiple return values, so check separately
        if exc is None:
            result_div, result_mod = op(s, other)
            if op is divmod:
                expected_div, expected_mod = s // other, s % other
            else:
                expected_div, expected_mod = other // s, other % s
            self.assert_series_equal(result_div, expected_div)
            self.assert_series_equal(result_mod, expected_mod)
        else:
            with pytest.raises(exc):
                divmod(s, other)

    def _get_exception(self, data, op_name):
        if op_name in ["__pow__", "__rpow__"]:
            return op_name, DimensionalityError
        else:
            return op_name, None

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        # series & scalar
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        s = pd.Series(data)
        self.check_opname(s, op_name, s.iloc[0], exc=exc)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # ndarray & other series
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        ser = pd.Series(data)
        self.check_opname(ser, op_name, pd.Series([ser.iloc[0]] * len(ser)), exc)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        # frame & scalar
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        df = pd.DataFrame({"A": data})
        self.check_opname(df, op_name, data[0], exc=exc)

    # parameterise this to try divisor not equal to 1
    def test_divmod(self, data):
        s = pd.Series(data)
        self._check_divmod_op(s, divmod, 1 * ureg.Mm)
        self._check_divmod_op(1 * ureg.Mm, ops.rdivmod, s)

    @pytest.mark.parametrize("box", [pd.Series, pd.DataFrame])
    def test_direct_arith_with_ndframe_returns_not_implemented(self, data, box):
        # EAs should return NotImplemented for ops with Series/DataFrame
        # Pandas takes care of unboxing the series and calling the EA's op.
        other = pd.Series(data)
        if box is pd.DataFrame:
            other = other.to_frame()
        if hasattr(data, "__add__"):
            result = data.__add__(other)
            assert result is NotImplemented
        else:
            raise pytest.skip(f"{type(data).__name__} does not implement add")

    def test_assignment_add_empty(self, data):
        # GH 68
        result = pd.Series(data)
        result[[]] += data[0]
        expected = pd.Series(data)
        self.assert_series_equal(result, expected)


class TestComparisonOps(base.BaseComparisonOpsTests):
    def _compare_other(self, s, data, op_name, other):
        op = self.get_op_from_name(op_name)

        result = op(s, other)
        expected = op(s.values.quantity, other)
        assert (result == expected).all()

    def test_compare_scalar(self, data, all_compare_operators):
        op_name = all_compare_operators
        s = pd.Series(data)
        other = data[0]
        self._compare_other(s, data, op_name, other)

    def test_compare_array(self, data, all_compare_operators):
        # nb this compares an quantity containing array
        # eg Q_([1,2],"m")
        op_name = all_compare_operators
        s = pd.Series(data)
        other = data
        self._compare_other(s, data, op_name, other)

    @pytest.mark.parametrize("box", [pd.Series, pd.DataFrame])
    def test_direct_arith_with_ndframe_returns_not_implemented(self, data, box):
        # EAs should return NotImplemented for ops with Series/DataFrame
        # Pandas takes care of unboxing the series and calling the EA's op.
        other = pd.Series(data)
        if box is pd.DataFrame:
            other = other.to_frame()

        if hasattr(data, "__eq__"):
            result = data.__eq__(other)
            assert result is NotImplemented
        else:
            raise pytest.skip(f"{type(data).__name__} does not implement __eq__")

        if hasattr(data, "__ne__"):
            result = data.__ne__(other)
            assert result is NotImplemented
        else:
            raise pytest.skip(f"{type(data).__name__} does not implement __ne__")


class TestOpsUtil(base.BaseOpsUtil):
    pass


class TestParsing(base.BaseParsingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestMissing(base.BaseMissingTests):
    pass


class TestNumericReduce(base.BaseNumericReduceTests):
    def check_reduce(self, s, op_name, skipna):
        result = getattr(s, op_name)(skipna=skipna)
        expected_m = getattr(pd.Series(s.values.quantity._magnitude), op_name)(
            skipna=skipna
        )
        if op_name in {"kurt", "skew"}:
            expected_u = None
        elif op_name in {"var"}:
            expected_u = s.values.quantity.units**2
        else:
            expected_u = s.values.quantity.units
        if expected_u is not None:
            expected = ureg.Quantity(expected_m, expected_u)
        else:
            expected = expected_m
        assert result == expected

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_scaling(self, data, all_numeric_reductions, skipna):
        """Make sure that the reductions give the same physical result independent of the unit representation.

        This verifies that the result units are sensible.
        """
        op_name = all_numeric_reductions
        s_nm = pd.Series(data)
        # Attention: `mm` is fine here, but with `m`, the magnitudes become so small
        # that pandas discards them in the kurtosis calculation, leading to different results.
        s_mm = pd.Series(PintArray.from_1darray_quantity(data.quantity.to(ureg.mm)))

        # min/max with empty produce numpy warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_nm = getattr(s_nm, op_name)(skipna=skipna)
            r_mm = getattr(s_mm, op_name)(skipna=skipna)
            if isinstance(r_nm, ureg.Quantity):
                # convert both results to the same units, then take the magnitude
                v_nm = r_nm.m_as(r_mm.units)
                v_mm = r_mm.m
            else:
                v_nm = r_nm
                v_mm = r_mm
            assert np.isclose(v_nm, v_mm, rtol=1e-3), f"{r_nm} == {r_mm}"


class TestBooleanReduce(base.BaseBooleanReduceTests):
    def check_reduce(self, s, op_name, skipna):
        result = getattr(s, op_name)(skipna=skipna)
        expected = getattr(pd.Series(s.values.quantity._magnitude), op_name)(
            skipna=skipna
        )
        assert result == expected


class TestReshaping(base.BaseReshapingTests):
    # @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
    @pytest.mark.parametrize(
        "index",
        [
            # Two levels, uniform.
            pd.MultiIndex.from_product(([["A", "B"], ["a", "b"]]), names=["a", "b"]),
            # non-uniform
            pd.MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "b")]),
            # three levels, non-uniform
            pd.MultiIndex.from_product([("A", "B"), ("a", "b", "c"), (0, 1, 2)]),
            pd.MultiIndex.from_tuples(
                [
                    ("A", "a", 1),
                    ("A", "b", 0),
                    ("A", "a", 0),
                    ("B", "a", 0),
                    ("B", "c", 1),
                ]
            ),
        ],
    )
    @pytest.mark.parametrize("obj", ["series", "frame"])
    def test_unstack(self, data, index, obj):
        data = data[: len(index)]
        if obj == "series":
            ser = pd.Series(data, index=index)
        else:
            ser = pd.DataFrame({"A": data, "B": data}, index=index)

        n = index.nlevels
        levels = list(range(n))
        # [0, 1, 2]
        # [(0,), (1,), (2,), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        combinations = itertools.chain.from_iterable(
            itertools.permutations(levels, i) for i in range(1, n)
        )

        for level in combinations:
            result = ser.unstack(level=level)
            assert all(
                isinstance(result[col].array, type(data)) for col in result.columns
            )

            if obj == "series":
                # We should get the same result with to_frame+unstack+droplevel
                df = ser.to_frame()

                alt = df.unstack(level=level).droplevel(0, axis=1)
                self.assert_frame_equal(result, alt)

            expected = ser.astype(object).unstack(level=level)
            result = result.astype(object)

            self.assert_frame_equal(result, expected)


class TestSetitem(base.BaseSetitemTests):
    # @pytest.mark.xfail(run=True, reason="excess warnings, needs debugging")
    def test_setitem_frame_2d_values(self, data):
        # GH#44514
        df = pd.DataFrame({"A": data})

        # These dtypes have non-broken implementations of _can_hold_element
        has_can_hold_element = isinstance(
            data.dtype, (PandasDtype, PeriodDtype, IntervalDtype, DatetimeTZDtype)
        )

        # Avoiding using_array_manager fixture
        #  https://github.com/pandas-dev/pandas/pull/44514#discussion_r754002410
        using_array_manager = isinstance(df._mgr, pd.core.internals.ArrayManager)

        blk_data = df._mgr.arrays[0]

        orig = df.copy()

        msg = "will attempt to set the values inplace instead"
        warn = None
        if has_can_hold_element and not isinstance(data.dtype, PandasDtype):
            # PandasDtype excluded because it isn't *really* supported.
            warn = FutureWarning

        with tm.assert_produces_warning(warn, match=msg):
            df.iloc[:] = df
        self.assert_frame_equal(df, orig)

        df.iloc[:-1] = df.iloc[:-1]
        self.assert_frame_equal(df, orig)

        if isinstance(data.dtype, DatetimeTZDtype):
            # no warning bc df.values casts to object dtype
            warn = None
        with tm.assert_produces_warning(warn, match=msg):
            df.iloc[:] = df.values
        self.assert_frame_equal(df, orig)
        if not using_array_manager:
            # GH#33457 Check that this setting occurred in-place
            # FIXME(ArrayManager): this should work there too
            assert df._mgr.arrays[0] is blk_data

        df.iloc[:-1] = df.values[:-1]
        self.assert_frame_equal(df, orig)
