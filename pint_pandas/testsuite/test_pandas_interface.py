import operator
from os.path import dirname, join

import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension.conftest import (
    as_array,  # noqa: F401
    as_frame,  # noqa: F401
    as_series,  # noqa: F401
    fillna_method,  # noqa: F401
    groupby_apply_op,  # noqa: F401
    use_numpy,  # noqa: F401
)
from pint.testsuite import QuantityTestCase, helpers

from pint_pandas import PintArray, PintType
from pint_pandas.pint_array import NO_UNIT
import pint_pandas

ureg = PintType.ureg


@pytest.fixture
def data():
    return PintArray.from_1darray_quantity(np.arange(start=1.0, stop=101.0) * ureg.nm)


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
                "length": pd.Series([2.0, 3.0], dtype="pint[m]"),
                "width": PintArray([2.0, 3.0], dtype="pint[m]"),
                "distance": PintArray([2.0, 3.0], dtype="m"),
                "height": PintArray([2.0, 3.0], dtype=ureg.m),
                "depth": PintArray.from_1darray_quantity(
                    ureg.Quantity([2.0, 3.0], ureg.m)
                ),
            }
        )

        for col in df.columns:
            assert all(df[col] == df.length)

    def test_df_operations(self):
        # simply a copy of what's in the notebook
        df = pd.DataFrame(
            {
                "torque": pd.Series([1.0, 2.0, 2.0, 3.0], dtype="pint[lbf ft]"),
                "angular_velocity": pd.Series([1.0, 2.0, 2.0, 3.0], dtype="pint[rpm]"),
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

    @pytest.mark.parametrize(
        "in_unit,out_unit",
        [
            pytest.param(" [lbf ft]", " [foot * force_pound]", id=" []"),
            pytest.param(" (lbf ft)", " (foot * force_pound)", id=" ()"),
            pytest.param(" / lbf ft", " / foot * force_pound", id=" / "),
        ],
    )
    def test_roundtrip_singlerow(self, in_unit, out_unit):
        expected = pd.DataFrame(
            {
                "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                "torque": pd.Series([1.0, 2.0, 2.0, 3.0], dtype="pint[lbf ft]"),
            }
        )
        df = pd.DataFrame(
            {
                "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                "torque" + in_unit: pd.Series([1.0, 2.0, 2.0, 3.0], dtype="Float64"),
            }
        )

        result = df.pint.quantify()
        pd.testing.assert_frame_equal(result, expected)

        expected = pd.DataFrame(
            {
                "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                "torque" + out_unit: pd.Series([1.0, 2.0, 2.0, 3.0], dtype="Float64"),
            }
        )
        result = result.pint.dequantify()
        pd.testing.assert_frame_equal(result, expected)
        pint_pandas.pint_array.SINGLE_ROW_HEADER_SEPARATOR = None
        pint_pandas.pint_array.SINGLE_ROW_HEADER_SUFFIX = None

    def test_quantify_singlerow(self):
        expected = pd.DataFrame(
            {
                "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                "torque": pd.Series([1.0, 2.0, 2.0, 3.0], dtype="pint[lbf ft]"),
                "angular_velocity": pd.Series([1.0, 2.0, 2.0, 3.0], dtype="pint[rpm]"),
            }
        )
        df = pd.DataFrame(
            {
                "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                "torque [lbf ft]": pd.Series([1.0, 2.0, 2.0, 3.0], dtype="Float64"),
                "angular_velocity [rpm]": pd.Series(
                    [1.0, 2.0, 2.0, 3.0], dtype="Float64"
                ),
            }
        )

        def parsing_function(column_name):
            if "[" in column_name:
                return column_name.split("]")[0].split(" [")
            return column_name, NO_UNIT

        result = df.pint.quantify(parsing_function=parsing_function)
        pd.testing.assert_frame_equal(result, expected)

    def test_dequantify_singlerow(self):
        df = pd.DataFrame(
            {
                "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                "torque": pd.Series([1.0, 2.0, 2.0, 3.0], dtype="pint[lbf ft]"),
                "angular_velocity": pd.Series([1.0, 2.0, 2.0, 3.0], dtype="pint[rpm]"),
            }
        )
        expected = pd.DataFrame(
            {
                "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                "torque [foot * force_pound]": pd.Series(
                    [1.0, 2.0, 2.0, 3.0], dtype="Float64"
                ),
                "angular_velocity [revolutions_per_minute]": pd.Series(
                    [1.0, 2.0, 2.0, 3.0], dtype="Float64"
                ),
            }
        )

        def writing_function(column_name, unit):
            if unit == NO_UNIT:
                return column_name
            return column_name + " [" + str(unit) + "]"

        result = df.pint.dequantify(writing_function=writing_function)
        pd.testing.assert_frame_equal(result, expected)

    def test_dequantify(self):
        df = pd.DataFrame(
            {
                "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                "pintarray_column": pd.Series(
                    [1.0, 2.0, 2.0, 3.0], dtype="pint[lbf ft]"
                ),
            }
        )
        expected = pd.DataFrame(
            {
                ("no_unit_column", "No Unit"): pd.Series(
                    {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}, dtype="Float64"
                ),
                ("pintarray_column", "foot * force_pound"): pd.Series(
                    {
                        0: 1.0,
                        1: 2.0,
                        2: 2.0,
                        3: 3.0,
                    },
                    dtype="Float64",
                ),
            }
        )
        expected.columns.names = [None, "unit"]

        result = df.pint.dequantify()
        pd.testing.assert_frame_equal(result, expected)

    def test_quantify(self):
        df = pd.DataFrame(
            {
                ("no_unit_column", "No Unit"): {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0},
                ("pintarray_column", "foot * force_pound"): {
                    0: 1.0,
                    1: 2.0,
                    2: 2.0,
                    3: 3.0,
                },
            },
            dtype="Float64",
        )
        df.columns.names = [None, "unit"]
        expected = pd.DataFrame(
            {
                "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                "pintarray_column": pd.Series(
                    [1.0, 2.0, 2.0, 3.0], dtype="pint[lbf ft]"
                ),
            }
        )

        result = df.pint.quantify()
        pd.testing.assert_frame_equal(result, expected)

        def test_to_base_units(self):
            df = pd.DataFrame(
                {
                    "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                    "pintarray_column": pd.Series(
                        [1.0, 2.0, 2.0, 3.0], dtype="pint[lbf ft]"
                    ),
                }
            )
            result = df.pint.to_base_units()

            expected = pd.DataFrame(
                {
                    "no_unit_column": pd.Series([i for i in range(4)], dtype="Float64"),
                    "pintarray_column": pd.Series(
                        [
                            1.3558179483314006,
                            2.711635896662801,
                            2.711635896662801,
                            4.067453844994201,
                        ],
                        dtype="pint[kilogram * meter ** 2 / second ** 2]",
                    ),
                }
            )
            pd.testing.assert_frame_equal(result, expected)


class TestDataFrameAccessor(object):
    def test_index_maintained(self):
        test_csv = join(dirname(__file__), "pandas_test.csv")

        df = pd.read_csv(test_csv, header=[0, 1], dtype="Float64")
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


class TestQuantify:
    # Cases a faster quantify implementation must handle, see
    # https://github.com/hgrecco/pint-pandas/issues/289

    @pytest.mark.parametrize(
        "values",
        [
            pd.array([1.0, np.nan, 3.0], dtype="float64"),
            pd.array([1.0, 2.0, 3.0], dtype="float32"),
            pd.array([1, 2, 3], dtype="int64"),
            pd.array([1.0, None, 3.0], dtype="Float64"),
            pd.array([1, None, 3], dtype="Int64"),
        ],
        ids=lambda v: str(v.dtype),
    )
    def test_subdtype_matches_input(self, values):
        df = pd.DataFrame({("a", "m"): values, ("b", NO_UNIT): values})
        result = df.pint.quantify(level=-1)
        expected = pd.DataFrame(
            {
                "a": pd.Series(values, dtype=f"pint[m][{values.dtype}]"),
                "b": pd.Series(values),
            }
        )
        assert result["a"].dtype.subdtype == values.dtype
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["float64", "Float64", "int64"])
    def test_result_independent_of_input(self, dtype):
        df = pd.DataFrame(
            {
                ("a", "m"): pd.Series([1, 2], dtype=dtype),
                ("b", "s"): pd.Series([3, 4], dtype=dtype),
                ("c", NO_UNIT): pd.Series([5, 6], dtype=dtype),
            }
        )
        result = df.pint.quantify(level=-1)
        result.iloc[0, 0] = ureg.Quantity(99, "m")
        result.iloc[0, 2] = 99
        assert df.iloc[0, 0] == 1
        assert df.iloc[0, 2] == 5

    def test_non_numeric_unitless_columns_unchanged(self):
        index = pd.Index([10, 20], name="idx")
        unitless = {
            "str": pd.Series(["x", "y"], index=index),
            "datetime": pd.Series(pd.to_datetime(["2020", "2021"]), index=index),
            "object": pd.Series([1, "x"], dtype=object, index=index),
            "category": pd.Series(pd.Categorical(["p", "q"]), index=index),
        }
        df = pd.DataFrame(
            {("a", "m"): pd.Series([1.0, 2.0], dtype="Float64", index=index)}
            | {(name, NO_UNIT): s for name, s in unitless.items()}
        )
        result = df.pint.quantify(level=-1)

        pd.testing.assert_index_equal(result.index, index)
        assert result["a"].dtype == PintType("m", "Float64")
        for name, s in unitless.items():
            pd.testing.assert_series_equal(result[name], s.rename(name))

    def test_duplicate_column_names(self):
        df = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            columns=pd.MultiIndex.from_tuples([("a", "m"), ("a", "s")]),
        )
        result = df.pint.quantify(level=-1)
        assert result.columns.tolist() == ["a", "a"]
        assert [dt.units for dt in result.dtypes] == [ureg.m, ureg.s]
        np.testing.assert_array_equal(
            result.iloc[:, 1].pint.magnitude.to_numpy(dtype="float64"), [2.0, 4.0]
        )

    def test_unit_level_not_last(self):
        df = pd.DataFrame(
            [[1.0, 2.0]],
            columns=pd.MultiIndex.from_tuples([("x", "m", "a"), ("x", "s", "b")]),
        )
        result = df.pint.quantify(level=1)
        assert result.columns.tolist() == [("x", "a"), ("x", "b")]
        assert [dt.units for dt in result.dtypes] == [ureg.m, ureg.s]

    def test_empty(self):
        df = pd.DataFrame(
            np.empty((0, 2)),
            columns=pd.MultiIndex.from_tuples([("a", "m"), ("b", NO_UNIT)]),
        )
        result = df.pint.quantify(level=-1)
        assert result.shape == (0, 2)
        assert result["a"].pint.units == ureg.m
        assert not isinstance(result["b"].dtype, PintType)


class TestSeriesAccessors(object):
    @pytest.mark.parametrize(
        "attr",
        [
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
        q = data.quantity
        getattr(q, attr)(*args)
        assert all(s.values == q)

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

    def test_convert_object_dtype(self, data):
        ser = pd.Series(data)
        ser_obj = pd.Series(ser.values, dtype="object")
        assert ser_obj.pint.convert_object_dtype().dtype == ser.dtype

        df = pd.DataFrame({"A": ser, "B": ser})
        df2 = pd.DataFrame({"A": ser, "B": ser_obj})

        assert all(df2.pint.convert_object_dtype().dtypes == df.dtypes)


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

unit_ops = [
    operator.mul,
    operator.truediv,
]


class TestPintArrayQuantity(QuantityTestCase):
    FORCE_NDARRAY = True

    def test_pintarray_creation(self):
        x = ureg.Quantity([1.0, 2.0, 3.0], "m")
        ys = [
            PintArray.from_1darray_quantity(x),
            PintArray._from_sequence([item for item in x]),
        ]
        for y in ys:
            helpers.assert_quantity_almost_equal(x, y.quantity)

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

                helpers.assert_quantity_almost_equal(result_pint, c_pint_array)

            except Exception as caught_exception:
                with pytest.raises(type(caught_exception)):
                    op(a_pint_array, b)

        a_pints = [
            ureg.Quantity([3.0, 4.0], "m"),
            ureg.Quantity([3.0, 4.0], ""),
        ]

        a_pint_arrays = [PintArray.from_1darray_quantity(q) for q in a_pints]

        bs = [
            2,
            ureg.Quantity(3, "m"),
            [1.0, 3.0],
            [3.3, 4.4],
            ureg.Quantity([6.0, 6.0], "m"),
            ureg.Quantity([7.0, np.nan]),
        ]

        us = [ureg.m]

        for a_pint, a_pint_array in zip(a_pints, a_pint_arrays):
            for b in bs:
                for op in arithmetic_ops:
                    test_op(a_pint, a_pint_array, b)
                for op in comparative_ops:
                    test_op(a_pint, a_pint_array, b, coerce=False)
            # also test for operations involving units
            for b in us:
                for op in unit_ops:
                    test_op(a_pint, a_pint_array, b)

    def test_mismatched_dimensions(self):
        x_and_ys = [
            (PintArray.from_1darray_quantity(ureg.Quantity([5.0], "m")), [1, 1]),
            (
                PintArray.from_1darray_quantity(ureg.Quantity([5.0, 5.0, 5.0], "m")),
                [1, 1],
            ),
            (PintArray.from_1darray_quantity(self.Q_([5.0, 5.0], "m")), [1]),
        ]
        for x, y in x_and_ys:
            for op in comparative_ops + arithmetic_ops:
                with pytest.raises(ValueError):
                    op(x, y)

    def test_numpy_data(self):
        foo = PintArray([1, 2, 3], dtype="pint[m]")
        result = foo.numpy_data
        expected = np.array([1, 2, 3], dtype="int64")
        np.testing.assert_array_equal(result, expected, strict=True)
