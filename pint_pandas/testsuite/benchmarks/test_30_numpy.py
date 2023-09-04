from typing import Generator, Any
import itertools as it
import operator

import pytest

import pint
from pint.compat import np
from pint_pandas.pint_array import pd
from pint.testsuite.helpers import requires_numpy

from pint_pandas import PintArray


SMALL_VEC_LEN = 3
MID_VEC_LEN = 1_000
LARGE_VEC_LEN = 1_000_000

LENGTHS = ("short", "mid")
ALL_VALUES = tuple(
    f"{a}_{b}" for a, b in it.product(LENGTHS, ("list", "tuple", "array"))
)
ALL_NDARRAYS = ("short_ndarray", "mid_ndarray")
ALL_QARRAYS = ("short_Qarray", "mid_Qarray")
ALL_PINTARRAYS = ("short_PintArray", "mid_PintArray")
ALL_SERIES = ("short_Series", "mid_Series")
ALL_SERIES_PINTARRAYS = ("short_Series_PA", "mid_Series_PA")
UNITS = ("meter", "kilometer")

OP1 = (operator.neg,)  # operator.truth,
OP2_CMP = (operator.eq, operator.lt)
OP2_MATH = (operator.add, operator.mul, operator.truediv)

if np is None:
    NUMPY_OP1_MATH = NUMPY_OP2_CMP = NUMPY_OP2_MATH = ()
else:
    NUMPY_OP1_MATH = (np.sqrt, np.square)
    NUMPY_OP2_CMP = (np.equal, np.less)
    NUMPY_OP2_MATH = (np.add, np.multiply, np.true_divide)


if pd is None:
    PANDAS_OP1_MATH = PANDAS_OP2_CMP = PANDAS_OP2_MATH = ()
else:
    PANDAS_OP1_MATH = ()
    PANDAS_OP2_CMP = (pd.Series.eq, pd.Series.lt)
    PANDAS_OP2_MATH = (pd.Series.add, pd.Series.multiply, pd.Series.truediv)


def float_range(n: int) -> Generator[float, None, None]:
    return (float(x) for x in range(1, n + 1))


@pytest.fixture
def setup(registry_tiny) -> tuple[pint.UnitRegistry, dict[str, Any]]:
    data = {}
    short = list(float_range(3))
    mid = list(float_range(1_000))

    data["short_list"] = short
    data["short_tuple"] = tuple(short)
    data["short_array"] = np.asarray(short)
    data["mid_list"] = mid
    data["mid_tuple"] = tuple(mid)
    data["mid_array"] = np.asarray(mid)

    ureg = registry_tiny

    data["short_ndarray"] = data["short_array"]
    data["mid_ndarray"] = data["mid_array"]
    for key in ALL_NDARRAYS:
        length, _ = key.split("_", 1)
        data[key + "_meter"] = np.array([x * ureg.meter for x in data[f"{length}_list"]], dtype="object")
        data[key + "_kilometer"] = np.array([x * ureg.kilometer for x in data[f"{length}_list"]], dtype="object")

    data["short_Qarray"] = data["short_array"]
    data["mid_Qarray"] = data["mid_array"]
    for key in ALL_QARRAYS:
        length, _ = key.split("_", 1)
        data[key + "_meter"] = data[f"{length}_array"] * ureg.meter
        data[key + "_kilometer"] = data[f"{length}_array"] * ureg.kilometer

    data["short_PintArray"] = data["short_array"]
    data["mid_PintArray"] = data["mid_array"]
    for key in ALL_PINTARRAYS:
        length, _ = key.split("_", 1)
        data[key + "_meter"] = PintArray(data[f"{length}_array"], ureg.meter)
        data[key + "_kilometer"] = PintArray(data[f"{length}_array"], ureg.kilometer)

    data["short_Series"] = data["short_array"]
    data["mid_Series"] = data["mid_array"]
    for key in ALL_SERIES:
        length, _ = key.split("_", 1)
        data[key + "_meter"] = pd.Series([x * ureg.meter for x in data[f"{length}_list"]])
        data[key + "_kilometer"] = pd.Series([x * ureg.kilometer for x in data[f"{length}_list"]])

    data["short_Series_PA"] = data["short_array"]
    data["mid_Series_PA"] = data["mid_array"]
    for key in ALL_SERIES_PINTARRAYS:
        length, _ = key.split("_", 1)
        data[key + "_meter"] = pd.Series(data[f"{length}_PintArray_meter"])
        data[key + "_kilometer"] = pd.Series(data[f"{length}_PintArray_kilometer"])

    return ureg, data


@requires_numpy
def test_finding_meter_getattr(benchmark, setup):
    ureg, _ = setup
    benchmark(getattr, ureg, "meter")


# @requires_numpy
# def test_finding_meter_getitem(benchmark, setup):
#     ureg, _ = setup
#     benchmark(operator.getitem, ureg, "meter")


# @requires_numpy
# @pytest.mark.parametrize(
#     "unit", ["meter", "angstrom", "meter/second", "angstrom/minute"]
# )
# def test_base_units(benchmark, setup, unit):
#     ureg, _ = setup
#     benchmark(ureg.get_base_units, unit)


@requires_numpy
@pytest.mark.parametrize("key", ALL_NDARRAYS)
def test_build_array_by_mul(benchmark, setup, key):
    ureg, data = setup
    benchmark(operator.mul, data[key], ureg.meter)


@requires_numpy
@pytest.mark.parametrize("key", ALL_QARRAYS)
def test_build_array_by_mul(benchmark, setup, key):
    ureg, data = setup
    benchmark(operator.mul, data[key], ureg.meter)


@requires_numpy
@pytest.mark.parametrize("key", ALL_PINTARRAYS)
def test_build_PintArray_by_mul(benchmark, setup, key):
    ureg, data = setup
    benchmark(operator.mul, data[key], ureg.meter)


@requires_numpy
@pytest.mark.parametrize("key", ALL_SERIES)
def test_build_PintArray_by_mul(benchmark, setup, key):
    ureg, data = setup
    benchmark(operator.mul, data[key], ureg.meter)


@requires_numpy
@pytest.mark.parametrize(
    "keys",
    (
        ("short_Qarray_meter", "short_Qarray_meter"),
        ("short_Qarray_meter", "short_Qarray_kilometer"),
        ("short_Qarray_kilometer", "short_Qarray_meter"),
        ("short_Qarray_kilometer", "short_Qarray_kilometer"),
        ("mid_Qarray_meter", "mid_Qarray_meter"),
        ("mid_Qarray_meter", "mid_Qarray_kilometer"),
        ("mid_Qarray_kilometer", "mid_Qarray_meter"),
        ("mid_Qarray_kilometer", "mid_Qarray_kilometer"),
    ),
)
@pytest.mark.parametrize("op", OP2_MATH + OP2_CMP)
def test_op2_pint(benchmark, setup, keys, op):
    _, data = setup
    key1, key2 = keys
    benchmark(op, data[key1], data[key2])


@requires_numpy
@pytest.mark.parametrize(
    "pa_keys",
    (
        ("short_PintArray_meter", "short_PintArray_meter"),
        ("short_PintArray_meter", "short_PintArray_kilometer"),
        ("short_PintArray_kilometer", "short_PintArray_meter"),
        ("short_PintArray_kilometer", "short_PintArray_kilometer"),
        ("mid_PintArray_meter", "mid_PintArray_meter"),
        ("mid_PintArray_meter", "mid_PintArray_kilometer"),
        ("mid_PintArray_kilometer", "mid_PintArray_meter"),
        ("mid_PintArray_kilometer", "mid_PintArray_kilometer"),
    ),
)
@pytest.mark.parametrize("op", OP2_MATH + OP2_CMP)
def test_op2_PintArray(benchmark, setup, pa_keys, op):
    _, data = setup
    key1, key2 = pa_keys
    benchmark(op, data[key1], data[key2])


@requires_numpy
@pytest.mark.parametrize(
    "np_keys",
    (
        ("short_ndarray_meter", "short_ndarray_meter"),
        ("short_ndarray_meter", "short_ndarray_kilometer"),
        ("short_ndarray_kilometer", "short_ndarray_meter"),
        ("short_ndarray_kilometer", "short_ndarray_kilometer"),
        ("mid_ndarray_meter", "mid_ndarray_meter"),
        ("mid_ndarray_meter", "mid_ndarray_kilometer"),
        ("mid_ndarray_kilometer", "mid_ndarray_meter"),
        ("mid_ndarray_kilometer", "mid_ndarray_kilometer"),
    ),
)
@pytest.mark.parametrize("op", NUMPY_OP2_MATH + NUMPY_OP2_CMP)
def test_op2_numpy(benchmark, setup, np_keys, op):
    _, data = setup
    key1, key2 = np_keys
    benchmark(op, data[key1], data[key2])


@requires_numpy
@pytest.mark.parametrize(
    "pd_keys",
    (
        ("short_Series_meter", "short_Series_meter"),
        ("short_Series_meter", "short_Series_kilometer"),
        ("short_Series_kilometer", "short_Series_meter"),
        ("short_Series_kilometer", "short_Series_kilometer"),
        ("mid_Series_meter", "mid_Series_meter"),
        ("mid_Series_meter", "mid_Series_kilometer"),
        ("mid_Series_kilometer", "mid_Series_meter"),
        ("mid_Series_kilometer", "mid_Series_kilometer"),
    ),
)
@pytest.mark.parametrize("op", PANDAS_OP2_MATH + PANDAS_OP2_CMP)
def test_op2_pandas(benchmark, setup, pd_keys, op):
    _, data = setup
    key1, key2 = pd_keys
    benchmark(op, data[key1], data[key2])

@requires_numpy
@pytest.mark.parametrize(
    "pd_pa_keys",
    (
        ("short_Series_PA_meter", "short_Series_PA_meter"),
        ("short_Series_PA_meter", "short_Series_PA_kilometer"),
        ("short_Series_PA_kilometer", "short_Series_PA_meter"),
        ("short_Series_PA_kilometer", "short_Series_PA_kilometer"),
        ("mid_Series_PA_meter", "mid_Series_PA_meter"),
        ("mid_Series_PA_meter", "mid_Series_PA_kilometer"),
        ("mid_Series_PA_kilometer", "mid_Series_PA_meter"),
        ("mid_Series_PA_kilometer", "mid_Series_PA_kilometer"),
    ),
)
@pytest.mark.parametrize("op", PANDAS_OP2_MATH + PANDAS_OP2_CMP)
def test_op2_pint_pandas(benchmark, setup, pd_pa_keys, op):
    _, data = setup
    key1, key2 = pd_pa_keys
    benchmark(op, data[key1], data[key2])

