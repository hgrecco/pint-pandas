"""
This file contains the tests required by pandas for an ExtensionArray and ExtensionType.
"""
import warnings

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest

try:
    import uncertainties.unumpy as unp
    from uncertainties import ufloat, UFloat
    from uncertainties.core import AffineScalarFunc  # noqa: F401

    def AffineScalarFunc__hash__(self):
        if not self._linear_part.expanded():
            self._linear_part.expand()
        combo = tuple(iter(self._linear_part.linear_combo.items()))
        if len(combo) > 1 or combo[0][1] != 1.0:
            return hash(combo)
        # The unique value that comes from a unique variable (which it also hashes to)
        return id(combo[0][0])

    AffineScalarFunc.__hash__ = AffineScalarFunc__hash__

    _ufloat_nan = ufloat(np.nan, 0)
    HAS_UNCERTAINTIES = True
except ImportError:
    unp = np
    HAS_UNCERTAINTIES = False

from pandas.core import ops
from pandas.tests.extension import base
from pandas.tests.extension.conftest import (
    as_frame,  # noqa: F401
    as_array,  # noqa: F401
    as_series,  # noqa: F401
    fillna_method,  # noqa: F401
    groupby_apply_op,  # noqa: F401
    use_numpy,  # noqa: F401
)


from pint.errors import DimensionalityError

from pint_pandas import PintArray, PintType
from pint_pandas.pint_array import dtypemap, pandas_version_info

ureg = PintType.ureg

from pandas import (
    Categorical,  # noqa: F401
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,  # noqa: F401
    MultiIndex,  # noqa: F401
    PeriodIndex,  # noqa: F401
    RangeIndex,  # noqa: F401
    Series,
    TimedeltaIndex,
)
from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    IntervalArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas._testing.asserters import (
    assert_equal,
    assert_index_equal,
    assert_interval_array_equal,
    assert_period_array_equal,
    assert_datetime_array_equal,
    assert_timedelta_array_equal,
    assert_almost_equal,
    assert_extension_array_equal,  # noqa: F401
    assert_numpy_array_equal,  # noqa: F401
)


def uassert_equal(left, right, **kwargs) -> None:
    """
    Wrapper for tm.assert_*_equal to dispatch to the appropriate test function.
    Parameters
    ----------
    left, right : Index, Series, DataFrame, ExtensionArray, or np.ndarray
        The two items to be compared.
    **kwargs
        All keyword arguments are passed through to the underlying assert method.
    """
    __tracebackhide__ = True

    if isinstance(left, Index):
        assert_index_equal(left, right, **kwargs)
        if isinstance(left, (DatetimeIndex, TimedeltaIndex)):
            assert left.freq == right.freq, (left.freq, right.freq)
    elif isinstance(left, Series):
        uassert_series_equal(left, right, **kwargs)
    elif isinstance(left, DataFrame):
        uassert_frame_equal(left, right, **kwargs)
    elif isinstance(left, IntervalArray):
        assert_interval_array_equal(left, right, **kwargs)
    elif isinstance(left, PeriodArray):
        assert_period_array_equal(left, right, **kwargs)
    elif isinstance(left, DatetimeArray):
        assert_datetime_array_equal(left, right, **kwargs)
    elif isinstance(left, TimedeltaArray):
        assert_timedelta_array_equal(left, right, **kwargs)
    elif isinstance(left, ExtensionArray):
        uassert_extension_array_equal(left, right, **kwargs)
    elif isinstance(left, np.ndarray):
        uassert_numpy_array_equal(left, right, **kwargs)
    elif isinstance(left, str):
        assert kwargs == {}
        assert left == right
    else:
        assert kwargs == {}
        uassert_almost_equal(left, right)


def uassert_series_equal(left, right, **kwargs):
    assert left.shape == right.shape
    if getattr(left, "dtype", False):
        assert left.dtype == right.dtype
    assert_equal(left.index, right.index)
    uassert_equal(left.values, right.values)


def uassert_frame_equal(left, right, **kwargs):
    assert left.shape == right.shape
    if getattr(left, "dtype", False):
        assert left.dtype == right.dtype
    assert_equal(left.index, right.index)
    uassert_equal(left.values, right.values)


def uassert_extension_array_equal(left, right, **kwargs):
    assert left.shape == right.shape
    if getattr(left, "dtype", False):
        assert left.dtype == right.dtype
    assert all([str(l) == str(r) for l, r in zip(left, right)])  # noqa: E741


def uassert_numpy_array_equal(left, right, **kwargs):
    if getattr(left, "dtype", False):
        assert left.dtype == right.dtype
    assert all([str(l) == str(r) for l, r in zip(left, right)])  # noqa: E741


def uassert_almost_equal(left, right, **kwargs):
    assert_almost_equal(left, right, **kwargs)


_use_uncertainties = [True, False] if HAS_UNCERTAINTIES else [False]
_use_ufloat_nan = [True, False] if HAS_UNCERTAINTIES else [False]


@pytest.fixture(params=_use_uncertainties)
def USE_UNCERTAINTIES(request):
    """Whether to use uncertainties in Pint-Pandas"""
    return request.param


@pytest.fixture(params=_use_ufloat_nan)
def USE_UFLOAT_NAN(request):
    """Whether to uncertainties using np.nan or ufloat(np.nan,0) in Pint-Pandas"""
    return request.param


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture
def dtype():
    return PintType("pint[meter]")


_base_numeric_dtypes = [float, int]
_all_numeric_dtypes = _base_numeric_dtypes + (
    [] if HAS_UNCERTAINTIES else [np.complex128]
)


@pytest.fixture(params=_all_numeric_dtypes)
def numeric_dtype(request):
    return request.param


@pytest.fixture
def data(numeric_dtype, USE_UNCERTAINTIES):
    if USE_UNCERTAINTIES:
        d = (np.arange(start=1.0, stop=101.0, dtype=None) + ufloat(0, 0)) * ureg.nm
    else:
        d = (
            np.arange(
                start=1.0,
                stop=101.0,
                dtype=numeric_dtype,
            )
            * ureg.nm
        )
    return PintArray.from_1darray_quantity(d)


@pytest.fixture
def data_missing(numeric_dtype, USE_UNCERTAINTIES, USE_UFLOAT_NAN):
    numeric_dtype = dtypemap.get(numeric_dtype, numeric_dtype)
    if USE_UNCERTAINTIES:
        numeric_dtype = None
        if USE_UFLOAT_NAN:
            dm = [_ufloat_nan, ufloat(1, 0)]
        else:
            dm = [np.nan, ufloat(1, 0)]
    else:
        dm = [numeric_dtype.na_value, 1]
    return PintArray.from_1darray_quantity(
        ureg.Quantity(pd.array(dm, dtype=numeric_dtype), ureg.meter)
    )


@pytest.fixture
def data_for_twos(numeric_dtype, USE_UNCERTAINTIES):
    if USE_UNCERTAINTIES:
        numeric_dtype = None
        x = [ufloat(2.0, 0)] * 100
    else:
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
def data_for_sorting(numeric_dtype, USE_UNCERTAINTIES):
    if USE_UNCERTAINTIES:
        numeric_dtype = None
        ds = [ufloat(0.3, 0), ufloat(10, 0), ufloat(-50, 0)]
    else:
        ds = [0.3, 10, -50]
    return PintArray.from_1darray_quantity(
        pd.array(ds, numeric_dtype) * ureg.centimeter
    )


@pytest.fixture
def data_missing_for_sorting(numeric_dtype, USE_UNCERTAINTIES):
    numeric_dtype = dtypemap.get(numeric_dtype, numeric_dtype)
    if USE_UNCERTAINTIES:
        numeric_dtype = None
        if USE_UFLOAT_NAN:
            dms = [ufloat(4, 0), _ufloat_nan, ufloat(-5, 0)]
        else:
            dms = [ufloat(4, 0), np.nan, ufloat(-5, 0)]
    else:
        dms = [4, numeric_dtype.na_value, -5]
    return PintArray.from_1darray_quantity(
        ureg.Quantity(pd.array(dms, dtype=numeric_dtype), ureg.centimeter)
    )


@pytest.fixture
def na_cmp(USE_UNCERTAINTIES):
    """Binary operator for comparing NA values."""
    if USE_UNCERTAINTIES:
        return lambda x, y: (
            bool(pd.isna(x.m))
            or (isinstance(x.m, UFloat) and unp.isnan(x.m)) & bool(pd.isna(y.m))
            or (isinstance(y.m, UFloat) and unp.isnan(y.m))
        )
    return lambda x, y: bool(pd.isna(x.magnitude)) & bool(pd.isna(y.magnitude))


@pytest.fixture
def na_value(numeric_dtype):
    return PintType("meter").na_value


@pytest.fixture
def data_for_grouping(numeric_dtype, USE_UNCERTAINTIES, USE_UFLOAT_NAN):
    a = 1.0
    b = 2.0**32 + 1
    c = 2.0**32 + 10
    if USE_UNCERTAINTIES:
        a = a + ufloat(0, 0)
        b = b + ufloat(0, 0)
        c = c + ufloat(0, 0)
        if USE_UFLOAT_NAN:
            _n = _ufloat_nan
        else:
            _n = np.nan
        numeric_dtype = None
    elif numeric_dtype:
        numeric_dtype = dtypemap.get(numeric_dtype, numeric_dtype)
        _n = np.nan
    else:
        _n = pd.NA
    return PintArray.from_1darray_quantity(
        ureg.Quantity(pd.array([b, b, _n, _n, a, a, b, c], dtype=numeric_dtype), ureg.m)
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


# commented functions aren't implemented in uncertainties
_uncertain_numeric_reductions = [
    "sum",
    "max",
    "min",
    # "mean",
    # "prod",
    # "std",
    # "var",
    # "median",
    # "sem",
    # "kurt",
    # "skew",
]

# commented functions aren't implemented in numpy/pandas
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


_all_numeric_accumulations = ["cumsum", "cumprod", "cummin", "cummax"]


@pytest.fixture(params=_all_numeric_accumulations)
def all_numeric_accumulations(request):
    """
    Fixture for numeric accumulation names
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
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        result = df.groupby("B", as_index=as_index).A.mean()
        _, uniques = pd.factorize(data_for_grouping, sort=True)

        if as_index:
            index = pd.Index._with_infer(uniques, name="B")
            expected = pd.Series([3.0, 1.0, 4.0], index=index, name="A")
            tm.assert_series_equal(result, expected)
        else:
            expected = pd.DataFrame({"B": uniques, "A": [3.0, 1.0, 4.0]})
            tm.assert_frame_equal(result, expected)

    def test_in_numeric_groupby(self, data_for_grouping):
        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3, 3, 1, 4],
                "B": data_for_grouping,
                "C": [1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        result = df.groupby("A").sum().columns

        expected = pd.Index(["B", "C"])

        tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
    def test_groupby_extension_no_sort(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        result = df.groupby("B", sort=False).A.mean()
        _, index = pd.factorize(data_for_grouping, sort=False)

        index = pd.Index._with_infer(index, name="B")
        expected = pd.Series([1.0, 3.0, 4.0], index=index, name="A")
        tm.assert_series_equal(result, expected)


class TestInterface(base.BaseInterfaceTests):
    def test_contains(self, data, data_missing, USE_UFLOAT_NAN):
        if USE_UFLOAT_NAN:
            pytest.skip(
                "any NaN-like other than data.dtype.na_value should fail (see GH-37867); also see BaseInterfaceTests in pandas/tests/extension/base/interface.py"
            )
        super().test_contains(data, data_missing)


class TestMethods(base.BaseMethodsTests):
    def test_apply_simple_series(self, data):
        result = pd.Series(data).apply(lambda x: x * 2 + ureg.Quantity(1, x.u))
        assert isinstance(result, pd.Series)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data_missing, na_action):
        s = pd.Series(data_missing)
        if pandas_version_info < (2, 1) and na_action is not None:
            pytest.skip(
                "Pandas EA map function only accepts None as na_action parameter"
            )
        result = s.map(lambda x: x, na_action=na_action)
        expected = s
        tm.assert_series_equal(result, expected)

    @pytest.mark.skip("All values are valid as magnitudes")
    def test_insert_invalid(self):
        pass


class TestArithmeticOps(base.BaseArithmeticOpsTests):
    divmod_exc = None
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None

    def _get_expected_exception(
        self, op_name: str, obj, other
    ):  #  -> type[Exception] | None, but Union types not understood by Python 3.9
        if op_name in ["__pow__", "__rpow__"]:
            return DimensionalityError
        if op_name in [
            "__divmod__",
            "__rdivmod__",
            "floor_divide",
            "remainder",
            "__floordiv__",
            "__rfloordiv__",
            "__mod__",
            "__rmod__",
        ]:
            exc = None
            if isinstance(obj, complex):
                pytest.skip(f"{type(obj).__name__} does not support {op_name}")
                return TypeError
            if isinstance(other, complex):
                pytest.skip(f"{type(other).__name__} does not support {op_name}")
                return TypeError
            if isinstance(obj, ureg.Quantity):
                pytest.skip(
                    f"{type(obj.m).__name__} Quantity does not support {op_name}"
                )
                return TypeError
            if isinstance(other, ureg.Quantity):
                pytest.skip(
                    f"{type(other.m).__name__} Quantity does not support {op_name}"
                )
                return TypeError
            if isinstance(obj, pd.Series):
                try:
                    if obj.pint.m.dtype.kind == "c":
                        pytest.skip(
                            f"{obj.pint.m.dtype.name} {obj.dtype} does not support {op_name}"
                        )
                        return TypeError
                except AttributeError:
                    exc = super()._get_expected_exception(op_name, obj, other)
                    if exc:
                        return exc
            if isinstance(other, pd.Series):
                try:
                    if other.pint.m.dtype.kind == "c":
                        pytest.skip(
                            f"{other.pint.m.dtype.name} {other.dtype} does not support {op_name}"
                        )
                        return TypeError
                except AttributeError:
                    exc = super()._get_expected_exception(op_name, obj, other)
                    if exc:
                        return exc
            if isinstance(obj, pd.DataFrame):
                try:
                    df = obj.pint.dequantify()
                    for i, col in enumerate(df.columns):
                        if df.iloc[:, i].dtype.kind == "c":
                            pytest.skip(
                                f"{df.iloc[:, i].dtype.name} {df.dtypes[i]} does not support {op_name}"
                            )
                            return TypeError
                except AttributeError:
                    exc = super()._get_expected_exception(op_name, obj, other)
                    if exc:
                        return exc
            if isinstance(other, pd.DataFrame):
                try:
                    df = other.pint.dequantify()
                    for i, col in enumerate(df.columns):
                        if df.iloc[:, i].dtype.kind == "c":
                            pytest.skip(
                                f"{df.iloc[:, i].dtype.name} {df.dtypes[i]} does not support {op_name}"
                            )
                            return TypeError
                except AttributeError:
                    exc = super()._get_expected_exception(op_name, obj, other)
                    # Fall through...
            return exc

    # The following methods are needed to work with Pandas < 2.1
    def _check_divmod_op(self, s, op, other, exc=None):
        # divmod has multiple return values, so check separately
        if exc is None:
            result_div, result_mod = op(s, other)
            if op is divmod:
                expected_div, expected_mod = s // other, s % other
            else:
                expected_div, expected_mod = other // s, other % s
            tm.assert_series_equal(result_div, expected_div)
            tm.assert_series_equal(result_mod, expected_mod)
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

    # With Pint 0.21, series and scalar need to have compatible units for
    # the arithmetic to work
    # series & scalar

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        # With Pint 0.21, series and scalar need to have compatible units for
        # the arithmetic to work
        # series & scalar
        if pandas_version_info < (2, 1):
            op_name, exc = self._get_exception(data, all_arithmetic_operators)
            s = pd.Series(data)
            self.check_opname(s, op_name, s.iloc[0], exc=exc)
        else:
            op_name = all_arithmetic_operators
            ser = pd.Series(data)
            self.check_opname(ser, op_name, ser.iloc[0])

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # ndarray & other series
        if pandas_version_info < (2, 1):
            op_name, exc = self._get_exception(data, all_arithmetic_operators)
            ser = pd.Series(data)
            self.check_opname(ser, op_name, pd.Series([ser.iloc[0]] * len(ser)), exc)
        else:
            op_name = all_arithmetic_operators
            ser = pd.Series(data)
            self.check_opname(ser, op_name, pd.Series([ser.iloc[0]] * len(ser)))

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        # frame & scalar
        if pandas_version_info < (2, 1):
            op_name, exc = self._get_exception(data, all_arithmetic_operators)
            df = pd.DataFrame({"A": data})
            self.check_opname(df, op_name, data[0], exc=exc)
        else:
            op_name = all_arithmetic_operators
            df = pd.DataFrame({"A": data})
            self.check_opname(df, op_name, data[0])

    # parameterise this to try divisor not equal to 1 Mm
    @pytest.mark.parametrize("numeric_dtype", _base_numeric_dtypes, indirect=True)
    def test_divmod(self, data, USE_UNCERTAINTIES):
        if USE_UNCERTAINTIES:
            pytest.skip(reason="uncertainties does not implement divmod")
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, 1 * ureg.Mm)
        self._check_divmod_op(1 * ureg.Mm, ops.rdivmod, ser)

    @pytest.mark.parametrize("numeric_dtype", _base_numeric_dtypes, indirect=True)
    def test_divmod_series_array(self, data, data_for_twos, USE_UNCERTAINTIES):
        if USE_UNCERTAINTIES:
            pytest.skip(reason="uncertainties does not implement divmod")
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, data)

        other = data_for_twos
        self._check_divmod_op(other, ops.rdivmod, ser)

        other = pd.Series(other)
        self._check_divmod_op(other, ops.rdivmod, ser)


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
    def _supports_reduction(self, obj, op_name: str) -> bool:
        # Specify if we expect this reduction to succeed.
        if (
            HAS_UNCERTAINTIES
            and op_name in _all_numeric_reductions
            and op_name not in _uncertain_numeric_reductions
        ):
            if any([isinstance(v, UFloat) for v in obj.values.quantity._magnitude]):
                pytest.skip(f"reduction {op_name} not implemented in uncertainties")
        return super()._supports_reduction(obj, op_name)

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

    @pytest.mark.skip("tests not written yet")
    def check_reduce_frame(self, ser: pd.Series, op_name: str, skipna: bool):
        pass

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_scaling(
        self, data, all_numeric_reductions, skipna, USE_UNCERTAINTIES
    ):
        """Make sure that the reductions give the same physical result independent of the unit representation.

        This verifies that the result units are sensible.
        """
        op_name = all_numeric_reductions
        if (
            USE_UNCERTAINTIES
            and op_name in _all_numeric_reductions
            and op_name not in _uncertain_numeric_reductions
        ):
            if any([isinstance(v, UFloat) for v in data.quantity._magnitude]):
                pytest.skip(f"reduction {op_name} not implemented in uncertainties")
        s_nm = pd.Series(data)
        # Attention: `mm` is fine here, but with `m`, the magnitudes become so small
        # that pandas discards them in the kurtosis calculation, leading to different results.
        s_mm = pd.Series(PintArray.from_1darray_quantity(data.quantity.to(ureg.mm)))

        # min/max with empty produce numpy warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                r_nm = getattr(s_nm, op_name)(skipna=skipna)
            except AttributeError:
                pytest.skip("bye!")
            r_mm = getattr(s_mm, op_name)(skipna=skipna)
            if isinstance(r_nm, ureg.Quantity):
                # convert both results to the same units, then take the magnitude
                v_nm = r_nm.m_as(r_mm.units)
                v_mm = r_mm.m
            else:
                v_nm = r_nm
                v_mm = r_mm
            if (
                USE_UNCERTAINTIES
                and isinstance(v_nm, UFloat)
                and isinstance(v_mm, UFloat)
            ):
                assert np.isclose(v_nm.n, v_mm.n, rtol=1e-3), f"{r_nm} == {r_mm}"
            else:
                assert np.isclose(v_nm, v_mm, rtol=1e-3), f"{r_nm} == {r_mm}"

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series(
        self, data, all_numeric_reductions, skipna, USE_UNCERTAINTIES
    ):
        op_name = all_numeric_reductions
        if (
            USE_UNCERTAINTIES
            and op_name in _all_numeric_reductions
            and op_name not in _uncertain_numeric_reductions
        ):
            if any([isinstance(v, UFloat) for v in data.quantity._magnitude]):
                pytest.skip(f"reduction {op_name} not implemented in uncertainties")
        s = pd.Series(data)

        # min/max with empty produce numpy warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.check_reduce(s, op_name, skipna)


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
        # This can be removed when https://github.com/pandas-dev/pandas/pull/54441 is accepted
        arr = data[:5].copy()
        with pytest.raises((ValueError, TypeError)):
            arr[0] = arr[[0, 1]]

    def test_setitem_invalid(self, data, invalid_scalar):
        # This can be removed when https://github.com/pandas-dev/pandas/pull/54441 is accepted
        msg = ""  # messages vary by subclass, so we do not test it
        with pytest.raises((ValueError, TypeError), match=msg):
            data[0] = invalid_scalar

        with pytest.raises((ValueError, TypeError), match=msg):
            data[:] = invalid_scalar

    @pytest.mark.parametrize("numeric_dtype", _base_numeric_dtypes, indirect=True)
    def test_setitem_2d_values(self, data):
        # GH50085
        original = data.copy()
        df = pd.DataFrame({"a": data, "b": data})
        df.loc[[0, 1], :] = df.loc[[1, 0], :].values
        assert (df.loc[0, :] == original[1]).all()
        assert (df.loc[1, :] == original[0]).all()


class TestAccumulate(base.BaseAccumulateTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_accumulate_series_raises(self, data, all_numeric_accumulations, skipna):
        if pandas_version_info < (2, 1):
            # Should this be skip?  Historic code simply used pass.
            pass

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        return True

    def check_accumulate(self, s, op_name, skipna):
        if op_name == "cumprod":
            with pytest.raises(TypeError):
                getattr(s, op_name)(skipna=skipna)
        else:
            result = getattr(s, op_name)(skipna=skipna)
            s_unitless = pd.Series(s.values.data)
            expected = getattr(s_unitless, op_name)(skipna=skipna)
            expected = pd.Series(expected, dtype=s.dtype)
            tm.assert_series_equal(result, expected, check_dtype=False)
