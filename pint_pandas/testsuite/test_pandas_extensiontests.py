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


class TestPintArray(base.ExtensionTests):
    # Groupby
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

    @pytest.mark.xfail(run=True, reason="assert_frame_equal issue")
    def test_groupby_extension_no_sort(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        result = df.groupby("B", sort=False).A.mean()
        _, index = pd.factorize(data_for_grouping, sort=False)

        index = pd.Index._with_infer(index, name="B")
        expected = pd.Series([1.0, 3.0, 4.0], index=index, name="A")
        tm.assert_series_equal(result, expected)

    # Methods
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

    # ArithmeticOps
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
                    # PintSeriesAccessor is dynamically constructed; need stubs to make it mypy-compatible
                    if obj.pint.m.dtype.kind == "c":  # type: ignore
                        pytest.skip(
                            f"{obj.pint.m.dtype.name} {obj.dtype} does not support {op_name}"  # type: ignore
                        )
                        return TypeError
                except AttributeError:
                    exc = super()._get_expected_exception(op_name, obj, other)
                    if exc:
                        return exc
            if isinstance(other, pd.Series):
                try:
                    if other.pint.m.dtype.kind == "c":  # type: ignore
                        pytest.skip(
                            f"{other.pint.m.dtype.name} {other.dtype} does not support {op_name}"  # type: ignore
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
    def test_divmod(self, data):
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, 1 * ureg.Mm)
        self._check_divmod_op(1 * ureg.Mm, ops.rdivmod, ser)

    @pytest.mark.parametrize("numeric_dtype", _base_numeric_dtypes, indirect=True)
    def test_divmod_series_array(self, data, data_for_twos):
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, data)

        other = data_for_twos
        self._check_divmod_op(other, ops.rdivmod, ser)

        other = pd.Series(other)
        self._check_divmod_op(other, ops.rdivmod, ser)

    # ComparisonOps
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

    # NumericReduce and BooleanReduce
    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return True

    def check_reduce(self, s, op_name, skipna):
        result = getattr(s, op_name)(skipna=skipna)
        expected_m = getattr(pd.Series(s.values.quantity._magnitude), op_name)(
            skipna=skipna
        )
        if op_name in {"kurt", "skew", "all", "any"}:
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

    # Reshaping
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

    # UnaryOps
    @pytest.mark.xfail(run=True, reason="invert not implemented")
    def test_invert(self, data):
        base.BaseUnaryOpsTests.test_invert(self, data)

    # Accumulate
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

    # Parsing
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        if request.getfixturevalue("numeric_dtype") == np.complex128:
            mark = pytest.mark.xfail(
                reason="can't parse complex numbers",
            )
            request.node.add_marker(mark)
        base.BaseParsingTests.test_EA_types(self, engine, data, request)
