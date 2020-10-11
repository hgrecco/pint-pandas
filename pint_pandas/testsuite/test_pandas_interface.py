import itertools
import operator
from os.path import dirname, join

import numpy as np
import pandas as pd
import pint
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
from pint.testsuite.test_quantity import QuantityTestCase

import pint_pandas as ppi
from pint_pandas import PintArray

ureg = pint.UnitRegistry()


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture
def dtype():
    return ppi.PintType("pint[meter]")


@pytest.fixture
def data():
    return ppi.PintArray.from_1darray_quantity(
        np.arange(start=1.0, stop=101.0) * ureg.nm
    )


@pytest.fixture
def data_missing():
    return ppi.PintArray.from_1darray_quantity([np.nan, 1] * ureg.meter)


@pytest.fixture
def data_for_twos():
    x = [
        2,
    ] * 100
    return ppi.PintArray.from_1darray_quantity(x * ureg.meter)


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


@pytest.fixture
def data_for_sorting():
    return ppi.PintArray.from_1darray_quantity([0.3, 10, -50] * ureg.centimeter)
    # should probably get more sophisticated and do something like
    # [1 * ureg.meter, 3 * ureg.meter, 10 * ureg.centimeter]


@pytest.fixture
def data_missing_for_sorting():
    return ppi.PintArray.from_1darray_quantity([4, np.nan, -5] * ureg.centimeter)
    # should probably get more sophisticated and do something like
    # [4 * ureg.meter, np.nan, 10 * ureg.centimeter]


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values."""
    return lambda x, y: bool(np.isnan(x.magnitude)) & bool(np.isnan(y.magnitude))


@pytest.fixture
def na_value():
    return ppi.PintType("meter").na_value


@pytest.fixture
def data_for_grouping():
    # should probably get more sophisticated here and use units on all these
    # quantities
    a = 1
    b = 2 ** 32 + 1
    c = 2 ** 32 + 10
    return ppi.PintArray.from_1darray_quantity(
        [b, b, np.nan, np.nan, a, a, b, c] * ureg.m
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
    @pytest.mark.xfail(
        run=True, reason="pintarrays seem not to be numeric in one version of pd"
    )
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

        self.assert_index_equal(result, expected)

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
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


class TestInterface(base.BaseInterfaceTests):
    pass


class TestMethods(base.BaseMethodsTests):
    @pytest.mark.filterwarnings("ignore::pint.UnitStrippedWarning")
    # See test_setitem_mask_broadcast note
    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna):
        all_data = all_data[:10]
        if dropna:
            other = all_data[~all_data.isna()]
        else:
            other = all_data

        result = pd.Series(all_data).value_counts(dropna=dropna).sort_index()
        expected = pd.Series(other).value_counts(dropna=dropna).sort_index()

        self.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings("ignore::pint.UnitStrippedWarning")
    # See test_setitem_mask_broadcast note
    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method):
        duplicated = box(data._from_sequence([data[0], data[0]]))

        result = method(duplicated)

        assert len(result) == 1
        assert isinstance(result, type(data))
        assert result[0] == duplicated[0]

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_fillna_copy_frame(self, data_missing):
        arr = data_missing.take([1, 1])
        df = pd.DataFrame({"A": arr})

        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)

        assert df.A.values is not result.A.values

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_fillna_copy_series(self, data_missing):
        arr = data_missing.take([1, 1])
        ser = pd.Series(arr)

        filled_val = ser[0]
        result = ser.fillna(filled_val)

        assert ser._values is not result._values
        assert ser._values is arr

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_searchsorted(self, data_for_sorting, as_series):  # noqa: F811
        b, c, a = data_for_sorting
        arr = type(data_for_sorting)._from_sequence([a, b, c])

        if as_series:
            arr = pd.Series(arr)
        assert arr.searchsorted(a) == 0
        assert arr.searchsorted(a, side="right") == 1

        assert arr.searchsorted(b) == 1
        assert arr.searchsorted(b, side="right") == 2

        assert arr.searchsorted(c) == 2
        assert arr.searchsorted(c, side="right") == 3

        result = arr.searchsorted(arr.take([0, 2]))
        expected = np.array([0, 2], dtype=np.intp)

        self.assert_numpy_array_equal(result, expected)

        # sorter
        sorter = np.array([1, 2, 0])
        assert data_for_sorting.searchsorted(a, sorter=sorter) == 0

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_where_series(self, data, na_value, as_frame):  # noqa: F811
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]

        ser = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))
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

        # array other
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


class TestArithmeticOps(base.BaseArithmeticOpsTests):
    def check_opname(self, s, op_name, other, exc=None):
        op = self.get_op_from_name(op_name)

        self._check_op(s, op, other, exc)

    def _check_op(self, s, op, other, exc=None):
        if exc is None:
            result = op(s, other)
            expected = s.combine(other, op)
            self.assert_series_equal(result, expected)
        else:
            with pytest.raises(exc):
                op(s, other)

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

    @pytest.mark.xfail(run=True, reason="_reduce needs implementation")
    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        # frame & scalar
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        df = pd.DataFrame({"A": data})
        self.check_opname(df, op_name, data[0], exc=exc)

    @pytest.mark.xfail(run=True, reason="s.combine does not accept arrays")
    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # ndarray & other series
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        s = pd.Series(data)
        self.check_opname(s, op_name, data, exc=exc)

    # parameterise this to try divisor not equal to 1
    def test_divmod(self, data):
        s = pd.Series(data)
        self._check_divmod_op(s, divmod, 1 * ureg.Mm)
        self._check_divmod_op(1 * ureg.Mm, ops.rdivmod, s)

    def test_error(self, data, all_arithmetic_operators):
        # invalid ops

        op = all_arithmetic_operators
        s = pd.Series(data)
        ops = getattr(s, op)
        opa = getattr(data, op)

        # invalid scalars
        # TODO: work out how to make this more specific/test for the two
        #       different possible errors here
        with pytest.raises(Exception):
            ops("foo")

        # TODO: work out how to make this more specific/test for the two
        #       different possible errors here
        with pytest.raises(Exception):
            ops(pd.Timestamp("20180101"))

        # invalid array-likes
        # TODO: work out how to make this more specific/test for the two
        #       different possible errors here
        #
        # This won't always raise exception, eg for foo % 3 m
        if "mod" not in op:
            with pytest.raises(Exception):
                ops(pd.Series("foo", index=s.index))

        # 2d
        with pytest.raises(KeyError):
            opa(pd.DataFrame({"A": s}))

        with pytest.raises(ValueError):
            opa(np.arange(len(s)).reshape(-1, len(s)))


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


class TestMissing(base.BaseMissingTests):
    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_fillna_scalar(self, data_missing):
        valid = data_missing[1]
        result = data_missing.fillna(valid)
        expected = data_missing.fillna(valid)
        self.assert_extension_array_equal(result, expected)

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_fillna_series(self, data_missing):
        fill_value = data_missing[1]
        ser = pd.Series(data_missing)

        result = ser.fillna(fill_value)
        expected = pd.Series(
            data_missing._from_sequence(
                [fill_value, fill_value], dtype=data_missing.dtype
            )
        )
        self.assert_series_equal(result, expected)

        # Fill with a series
        result = ser.fillna(expected)
        self.assert_series_equal(result, expected)

        # Fill with a series not affecting the missing values
        result = ser.fillna(ser)
        self.assert_series_equal(result, ser)

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_fillna_frame(self, data_missing):
        fill_value = data_missing[1]

        result = pd.DataFrame({"A": data_missing, "B": [1, 2]}).fillna(fill_value)

        expected = pd.DataFrame(
            {
                "A": data_missing._from_sequence(
                    [fill_value, fill_value], dtype=data_missing.dtype
                ),
                "B": [1, 2],
            }
        )
        self.assert_series_equal(result, expected)


class TestReshaping(base.BaseReshapingTests):
    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
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
    @pytest.mark.parametrize("setter", ["loc", None])
    @pytest.mark.filterwarnings("ignore::pint.UnitStrippedWarning")
    # Pandas performs a hasattr(__array__), which triggers the warning
    # Debugging it does not pass through a PintArray, so
    # I think this needs changing in pint quantity
    # eg s[[True]*len(s)]=Q_(1,"m")
    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_setitem_mask_broadcast(self, data, setter):
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True

        if setter:  # loc
            target = getattr(ser, setter)
        else:  # __setitem__
            target = ser

        operator.setitem(target, mask, data[10])
        assert ser[0] == data[10]
        assert ser[1] == data[10]

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        data[[0, 1]] = data[2]
        assert data[0] == data[2]
        assert data[1] == data[2]

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series):
        arr = data[:5].copy()
        expected = data.take([0, 0, 0, 3, 4])

        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)

        arr[idx] = arr[0]
        self.assert_equal(arr, expected)

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_setitem_slice(self, data, box_in_series):
        arr = data[:5].copy()
        expected = data.take([0, 0, 0, 3, 4])
        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)

        arr[:3] = data[0]
        self.assert_equal(arr, expected)

    @pytest.mark.xfail(run=True, reason="__iter__ / __len__ issue")
    def test_setitem_loc_iloc_slice(self, data):
        arr = data[:5].copy()
        s = pd.Series(arr, index=["a", "b", "c", "d", "e"])
        expected = pd.Series(data.take([0, 0, 0, 3, 4]), index=s.index)

        result = s.copy()
        result.iloc[:3] = data[0]
        self.assert_equal(result, expected)

        result = s.copy()
        result.loc[:"c"] = data[0]
        self.assert_equal(result, expected)


# would be ideal to just test all of this by running the example notebook
# but this isn't a discussion we've had yet


class TestUserInterface(object):
    def test_get_underlying_data(self, data):
        ser = pd.Series(data)
        # this first test creates an array of bool (which is desired, eg for indexing)
        assert all(ser.values == data)
        assert ser.values[23] == data[23]

    def test_arithmetic(self, data):
        ser = pd.Series(data)
        ser2 = ser + ser
        assert all(ser2.values == 2 * data)

    def test_initialisation(self, data):
        # fails with plain array
        # works with PintArray
        df = pd.DataFrame(
            {
                "length": pd.Series([2, 3], dtype="pint[m]"),
                "width": PintArray([2, 3], dtype="pint[m]"),
                "distance": PintArray([2, 3], dtype="m"),
                "height": PintArray([2, 3], dtype=ureg.m),
                "depth": PintArray.from_1darray_quantity(ureg.Quantity([2, 3], ureg.m)),
            }
        )

        for col in df.columns:
            assert all(df[col] == df.length)

    def test_df_operations(self):
        # simply a copy of what's in the notebook
        df = pd.DataFrame(
            {
                "torque": pd.Series([1, 2, 2, 3], dtype="pint[lbf ft]"),
                "angular_velocity": pd.Series([1, 2, 2, 3], dtype="pint[rpm]"),
            }
        )

        df["power"] = df["torque"] * df["angular_velocity"]

        df.power.values
        df.power.values.quantity
        df.angular_velocity.values.data

        df.power.pint.units

        df.power.pint.to("kW").values

        test_csv = join(dirname(__file__), "pandas_test.csv")

        df = pd.read_csv(test_csv, header=[0, 1])
        df_ = df.pint.quantify(level=-1)

        df_["mech power"] = df_.speed * df_.torque
        df_["fluid power"] = df_["fuel flow rate"] * df_["rail pressure"]

        df_.pint.dequantify()

        df_["fluid power"] = df_["fluid power"].pint.to("kW")
        df_["mech power"] = df_["mech power"].pint.to("kW")
        df_.pint.dequantify()

        df_.pint.to_base_units().pint.dequantify()


class TestDataFrameAccessor(object):
    def test_index_maintained(self):
        test_csv = join(dirname(__file__), "pandas_test.csv")

        df = pd.read_csv(test_csv, header=[0, 1])
        df.columns = pd.MultiIndex.from_arrays(
            [
                ["Holden", "Holden", "Holden", "Ford", "Ford", "Ford"],
                [
                    "speed",
                    "mech power",
                    "torque",
                    "rail pressure",
                    "fuel flow rate",
                    "fluid power",
                ],
                ["rpm", "kW", "N m", "bar", "l/min", "kW"],
            ],
            names=["Car type", "metric", "unit"],
        )
        df.index = pd.MultiIndex.from_arrays(
            [
                [1, 12, 32, 48],
                ["Tim", "Tim", "Jane", "Steve"],
            ],  # noqa E231
            names=["Measurement number", "Measurer"],
        )

        expected = df.copy()

        # we expect the result to come back with pint names, not input
        # names
        def get_pint_value(in_str):
            return str(ureg.Quantity(1, in_str).units)

        units_level = [i for i, name in enumerate(df.columns.names) if name == "unit"][
            0
        ]

        expected.columns = df.columns.set_levels(
            df.columns.levels[units_level].map(get_pint_value), level="unit"
        )

        result = df.pint.quantify(level=-1).pint.dequantify()

        pd.testing.assert_frame_equal(result, expected)


class TestSeriesAccessors(object):
    @pytest.mark.parametrize(
        "attr",
        [
            "debug_used",
            "default_format",
            "dimensionality",
            "dimensionless",
            "force_ndarray",
            "shape",
            "u",
            "unitless",
            "units",
        ],
    )
    def test_series_scalar_property_accessors(self, data, attr):
        s = pd.Series(data)
        assert getattr(s.pint, attr) == getattr(data.quantity, attr)

    @pytest.mark.parametrize(
        "attr",
        [
            "m",
            "magnitude",
            # 'imag', # failing, not sure why
            # 'real', # failing, not sure why
        ],
    )
    def test_series_property_accessors(self, data, attr):
        s = pd.Series(data)
        assert all(getattr(s.pint, attr) == pd.Series(getattr(data.quantity, attr)))

    @pytest.mark.parametrize(
        "attr_args",
        [
            ("check", ({"[length]": 1})),
            ("compatible_units", ()),
            # ('format_babel', ()), Needs babel installed?
            # ('plus_minus', ()), Needs uncertanties
            # ('to_tuple', ()),
            ("tolist", ()),
        ],
    )
    def test_series_scalar_method_accessors(self, data, attr_args):
        attr = attr_args[0]
        args = attr_args[1]
        s = pd.Series(data)
        assert getattr(s.pint, attr)(*args) == getattr(data.quantity, attr)(*args)

    @pytest.mark.parametrize(
        "attr_args",
        [
            ("ito", ("mi",)),
            ("ito_base_units", ()),
            ("ito_reduced_units", ()),
            ("ito_root_units", ()),
            ("put", (1, 1 * ureg.nm)),
        ],
    )
    def test_series_inplace_method_accessors(self, data, attr_args):
        attr = attr_args[0]
        args = attr_args[1]
        from copy import deepcopy

        s = pd.Series(deepcopy(data))
        getattr(s.pint, attr)(*args)
        getattr(data.quantity, attr)(*args)
        assert all(s.values == data)

    @pytest.mark.parametrize(
        "attr_args",
        [
            ("clip", (10 * ureg.nm, 20 * ureg.nm)),
            (
                "from_tuple",
                (PintArray(np.arange(1, 101), dtype=ureg.m).quantity.to_tuple(),),
            ),
            ("m_as", ("mi",)),
            ("searchsorted", (10 * ureg.nm,)),
            ("to", ("m")),
            ("to_base_units", ()),
            ("to_compact", ()),
            ("to_reduced_units", ()),
            ("to_root_units", ()),
            # ('to_timedelta', ()),
        ],
    )
    def test_series_method_accessors(self, data, attr_args):
        attr = attr_args[0]
        args = attr_args[1]
        s = pd.Series(data)
        assert all(getattr(s.pint, attr)(*args) == getattr(data.quantity, attr)(*args))


arithmetic_ops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
]

comparative_ops = [
    operator.eq,
    operator.le,
    operator.lt,
    operator.ge,
    operator.gt,
]


class TestPintArrayQuantity(QuantityTestCase):
    FORCE_NDARRAY = True

    def test_pintarray_creation(self):
        x = ureg.Quantity([1, 2, 3], "m")
        ys = [
            PintArray.from_1darray_quantity(x),
            PintArray._from_sequence([item for item in x]),
        ]
        for y in ys:
            self.assertQuantityAlmostEqual(x, y.quantity)

    @pytest.mark.filterwarnings("ignore::pint.UnitStrippedWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_pintarray_operations(self):
        # Perform operations with Quantities and PintArrays
        # The resulting Quantity and PintArray.Data should be the same
        # a op b == c
        # warnings ignored here as it these tests are to ensure
        # pint array behaviour is the same as quantity
        def test_op(a_pint, a_pint_array, b_, coerce=True):
            try:
                result_pint = op(a_pint, b_)
                if coerce:
                    # a PintArray is returned from arithmetics, so need the data
                    c_pint_array = op(a_pint_array, b_).quantity
                else:
                    # a boolean array is returned from comparatives
                    c_pint_array = op(a_pint_array, b_)

                self.assertQuantityAlmostEqual(result_pint, c_pint_array)

            except Exception as caught_exception:
                self.assertRaises(type(caught_exception), op, a_pint_array, b_)

        a_pints = [
            ureg.Quantity([3, 4], "m"),
            ureg.Quantity([3, 4], ""),
        ]

        a_pint_arrays = [PintArray.from_1darray_quantity(q) for q in a_pints]

        bs = [
            2,
            ureg.Quantity(3, "m"),
            [1.0, 3.0],
            [3.3, 4.4],
            ureg.Quantity([6, 6], "m"),
            ureg.Quantity([7.0, np.nan]),
        ]

        for a_pint, a_pint_array in zip(a_pints, a_pint_arrays):
            for b in bs:
                for op in arithmetic_ops:
                    test_op(a_pint, a_pint_array, b)
                for op in comparative_ops:
                    test_op(a_pint, a_pint_array, b, coerce=False)

    def test_mismatched_dimensions(self):
        x_and_ys = [
            (PintArray.from_1darray_quantity(ureg.Quantity([5], "m")), [1, 1]),
            (PintArray.from_1darray_quantity(ureg.Quantity([5, 5, 5], "m")), [1, 1]),
            (PintArray.from_1darray_quantity(self.Q_([5, 5], "m")), [1]),
        ]
        for x, y in x_and_ys:
            for op in comparative_ops + arithmetic_ops:
                self.assertRaises(ValueError, op, x, y)
