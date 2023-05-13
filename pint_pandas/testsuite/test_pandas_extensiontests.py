"""
This file contains the tests required by pandas for an ExtensionArray and ExtensionType.
"""
import warnings

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from pandas.core import ops
from pandas.tests.extension import base
from pandas.tests.extension.conftest import (  # noqa: F401
    as_array,
    as_frame,
    as_series,
    fillna_method,
    groupby_apply_op,
    use_numpy,
)
from pint.errors import DimensionalityError

from pint_pandas import PintArray, PintType
from pint_pandas.pint_array import dtypemap

ureg = PintType.ureg


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture
def dtype():
    return PintType("pint[meter]")


_base_numeric_dtypes = [float, int]
_all_numeric_dtypes = _base_numeric_dtypes + [np.complex128]


@pytest.fixture(params=_all_numeric_dtypes)
def numeric_dtype(request):
    return request.param


@pytest.fixture
def data(request, numeric_dtype):
    return PintArray.from_1darray_quantity(
        np.arange(start=1.0, stop=101.0, dtype=numeric_dtype) * ureg.nm
    )


@pytest.fixture
def data_missing(numeric_dtype):
    numeric_dtype = dtypemap.get(numeric_dtype, numeric_dtype)
    return PintArray.from_1darray_quantity(
        ureg.Quantity(pd.array([np.nan, 1], dtype=numeric_dtype), ureg.meter)
    )


@pytest.fixture
def data_for_twos(numeric_dtype):
    x = [
        2.0,
    ] * 100
    return PintArray.from_1darray_quantity(
        pd.array(x, dtype=numeric_dtype) * ureg.meter
    )


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data):
    """Return different versions of data for count times"""

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
def data_for_sorting(numeric_dtype):
    return PintArray.from_1darray_quantity(
        pd.array([0.3, 10.0, -50.0], numeric_dtype) * ureg.centimeter
    )


@pytest.fixture
def data_missing_for_sorting(numeric_dtype):
    numeric_dtype = dtypemap.get(numeric_dtype, numeric_dtype)
    return PintArray.from_1darray_quantity(
        ureg.Quantity(
            pd.array([4.0, np.nan, -5.0], dtype=numeric_dtype), ureg.centimeter
        )
    )


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values."""
    return lambda x, y: bool(pd.isna(x.magnitude)) & bool(pd.isna(y.magnitude))


@pytest.fixture
def na_value(numeric_dtype):
    return PintType("meter").na_value


@pytest.fixture
def data_for_grouping(numeric_dtype):
    a = 1.0
    b = 2.0**32 + 1
    c = 2.0**32 + 10

    numeric_dtype = dtypemap.get(numeric_dtype, numeric_dtype)
    return PintArray.from_1darray_quantity(
        ureg.Quantity(
            pd.array([b, b, np.nan, np.nan, a, a, b, c], dtype=numeric_dtype), ureg.m
        )
    )


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
    @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
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

    @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
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

    @pytest.mark.xfail(run=True, reason="fails with pandas > 1.5.2 and pint > 0.20.1")
    def test_in_numeric_groupby(self, data_for_grouping):
        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3, 3, 1, 4],
                "B": data_for_grouping,
                "C": [1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        result = df.groupby("A").sum().columns

        # FIXME: Why dies C get included for e.g. PandasDtype('complex128') but not for Float64Dtype()? This seems buggy,
        # but very hard for us to fix...
        if df.B.isna().sum() == 0 or isinstance(
            df.B.values.data.dtype, pd.core.dtypes.dtypes.PandasDtype
        ):
            expected = pd.Index(["B", "C"])
        else:
            expected = pd.Index(["C"])

        tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
    def test_groupby_extension_no_sort(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        result = df.groupby("B", sort=False).A.mean()
        _, index = pd.factorize(data_for_grouping, sort=False)

        index = pd.Index._with_infer(index, name="B")
        expected = pd.Series([1.0, 3.0, 4.0], index=index, name="A")
        self.assert_series_equal(result, expected)


class TestInterface(base.BaseInterfaceTests):
    @pytest.mark.xfail(run=True, reason="incompatible with Pint 0.21")
    def test_contains(self, data, data_missing):
        base.BaseInterfaceTests.test_contains(self, data, data_missing)


class TestMethods(base.BaseMethodsTests):
    @pytest.mark.skip("All values are valid as magnitudes")
    def test_insert_invalid(self):
        pass


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
        if data.data.dtype == pd.core.dtypes.dtypes.PandasDtype("complex128"):
            if op_name in ["__floordiv__", "__rfloordiv__", "__mod__", "__rmod__"]:
                return op_name, TypeError
        if op_name in ["__pow__", "__rpow__"]:
            return op_name, DimensionalityError

        return op_name, None

    @pytest.mark.parametrize("numeric_dtype", _base_numeric_dtypes, indirect=True)
    def test_divmod_series_array(self, data, data_for_twos):
        base.BaseArithmeticOpsTests.test_divmod_series_array(self, data, data_for_twos)

    @pytest.mark.xfail(run=True, reason="incompatible with Pint 0.21")
    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        # With Pint 0.21, series and scalar need to have compatible units for
        # the arithmetic to work
        # series & scalar
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        s = pd.Series(data)
        self.check_opname(s, op_name, s.iloc[0], exc=exc)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # ndarray & other series
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        ser = pd.Series(data)
        self.check_opname(ser, op_name, pd.Series([ser.iloc[0]] * len(ser)), exc)

    @pytest.mark.xfail(run=True, reason="incompatible with Pint 0.21")
    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        # frame & scalar
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        df = pd.DataFrame({"A": data})
        self.check_opname(df, op_name, data[0], exc=exc)

    # parameterise this to try divisor not equal to 1
    @pytest.mark.parametrize("numeric_dtype", _base_numeric_dtypes, indirect=True)
    def test_divmod(self, data):
        s = pd.Series(data)
        self._check_divmod_op(s, divmod, 1 * ureg.Mm)
        self._check_divmod_op(1 * ureg.Mm, ops.rdivmod, s)


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

    @pytest.mark.xfail(run=True, reason="incompatible with Pint 0.21")
    def test_compare_array(self, data, all_compare_operators):
        # nb this compares an quantity containing array
        # eg Q_([1,2],"m")
        op_name = all_compare_operators
        s = pd.Series(data)
        other = data
        self._compare_other(s, data, op_name, other)


class TestOpsUtil(base.BaseOpsUtil):
    pass


@pytest.mark.parametrize("numeric_dtype", _base_numeric_dtypes, indirect=True)
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
    @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
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
        base.TestReshaping.test_unstack(self, data, index, obj)


class TestSetitem(base.BaseSetitemTests):
    @pytest.mark.parametrize("numeric_dtype", _base_numeric_dtypes, indirect=True)
    def test_setitem_scalar_key_sequence_raise(self, data):
        base.BaseSetitemTests.test_setitem_scalar_key_sequence_raise(self, data)
