import pickle
import time

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
import pint
from pandas.tests.extension.base.base import BaseExtensionTests
from pint.testsuite import helpers

from pint_pandas import PintArray, PintType
from pint_pandas.pint_array import pandas_version_info

ureg = PintType.ureg


class TestIssue165(BaseExtensionTests):
    def test_force_ndarray_like(self):
        # store previous registries to undo our changes
        prev_PintType_ureg = PintType.ureg
        prev_appreg = pint.get_application_registry().get()
        prev_cache = PintType._cache
        try:
            # create a temporary registry with force_ndarray_like = True (`pint_xarray` insists on that)
            test_ureg = pint.UnitRegistry()
            test_ureg.force_ndarray_like = True
            # register
            pint.set_application_registry(test_ureg)
            PintType.ureg = test_ureg
            # clear units cache
            PintType._cache = {}

            # run TestIssue21.test_offset_concat with our test-registry (one of many that currently fails with force_ndarray_like=True)
            q_a = ureg.Quantity(np.arange(5), test_ureg.Unit("degC"))
            q_b = ureg.Quantity(np.arange(6), test_ureg.Unit("degC"))
            q_a_ = np.append(q_a, np.nan)

            a = pd.Series(PintArray(q_a))
            b = pd.Series(PintArray(q_b))

            result = pd.concat([a, b], axis=1)
            expected = pd.DataFrame(
                {0: PintArray(q_a_), 1: PintArray(q_b)}, dtype="pint[degC]"
            )
            tm.assert_equal(result, expected)

        finally:
            # restore registry
            PintType.ureg = prev_PintType_ureg
            PintType._cache = prev_cache
            pint.set_application_registry(prev_appreg)


class TestIssue21(BaseExtensionTests):
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_offset_concat(self):
        q_a = ureg.Quantity(np.arange(5), ureg.Unit("degC"))
        q_b = ureg.Quantity(np.arange(6), ureg.Unit("degC"))
        q_a_ = np.append(q_a, np.nan)

        a = pd.Series(PintArray(q_a))
        b = pd.Series(PintArray(q_b))

        result = pd.concat([a, b], axis=1)
        expected = pd.DataFrame(
            {0: PintArray(q_a_), 1: PintArray(q_b)}, dtype="pint[degC]"
        )
        tm.assert_equal(result, expected)

        # issue #141
        print(PintArray(q_a))


class TestIssue68(BaseExtensionTests):
    def test_assignment_add_empty(self):
        # GH 68
        data = PintArray.from_1darray_quantity(
            np.arange(start=1.0, stop=101.0, dtype=float) * ureg.nm
        )

        result = pd.Series(data)
        result[[]] += data[0]
        expected = pd.Series(data)
        tm.assert_series_equal(result, expected)


class TestIssue80:
    @staticmethod
    def _timeit(fun, n_runs=5):
        run_time = []
        for k in range(n_runs):
            t_start = time.monotonic_ns()
            fun()
            t_end = time.monotonic_ns()
            run_time.append(t_end - t_start)
        return np.median(run_time) * ureg.ns

    @staticmethod
    def _make_df(size, pint_units=True, dtype=float):
        if pint_units:
            dist_unit = "pint[m]"
            time_unit = "pint[s]"
        else:
            dist_unit = dtype
            time_unit = dtype
        return pd.DataFrame(
            {
                "distance": pd.Series(
                    np.arange(1, size + 1, dtype=dtype), dtype=dist_unit
                ),
                "time": pd.Series(np.arange(1, size + 1, dtype=dtype), dtype=time_unit),
            }
        )

    def test_div(self):
        n = 1_000_000
        df_pint = self._make_df(n)
        df = self._make_df(n, pint_units=False)

        tp = self._timeit(lambda: df_pint["distance"] / df_pint["time"]).to("ms")
        t = self._timeit(lambda: df["distance"] / df["time"]).to("ms")

        assert tp <= 5 * t

    @pytest.mark.parametrize(
        "reduction",
        ["min", "max", "sum", "mean", "median"],
    )
    def test_reductions(self, reduction):
        # before the fix, those reductions could be *very* slow. Fail early.
        for n in [10_000, 1_000_000]:
            s_pint = self._make_df(n)["time"]
            s = self._make_df(n, pint_units=False)["time"]

            tp = self._timeit(getattr(s_pint, reduction)).to("ms")
            t = self._timeit(getattr(s, reduction)).to("ms")

            assert tp <= 5 * t


def test_issue_86():
    a = PintArray([1, 2], ureg.m)
    b_listlike = [1 * ureg.km, 1 * ureg.m]
    units = b_listlike[0].units
    b_pa = PintArray([v.m_as(units) for v in b_listlike], units)

    assert np.all(a + b_listlike == a + b_pa)


def test_issue_71():
    a = PintArray([1, 2], ureg.m)
    s = pickle.dumps(a)
    b = pickle.loads(s)
    assert np.all(a == b)


def test_issue_88():
    q_m = ureg.Quantity([1, 2], "m")
    a = PintArray(q_m)
    helpers.assert_quantity_almost_equal(q_m, a.quantity)

    q_mm = ureg.Quantity([1000, 2000], "mm")
    b = PintArray(q_mm, "m")
    helpers.assert_quantity_almost_equal(q_m, b.quantity)


def test_issue_127():
    a = PintType.construct_from_string("pint[dimensionless]")
    b = PintType.construct_from_string("pint[]")
    assert a == b


class TestIssue174(BaseExtensionTests):
    def test_sum(self):
        if pandas_version_info < (2, 1):
            pytest.skip("Pandas reduce functions strip units prior to version 2.1.0")
        a = pd.DataFrame([[0, 1, 2], [3, 4, 5]]).astype("pint[m]")
        row_sum = a.sum(axis=0)
        expected_1 = pd.Series([3, 5, 7], dtype="pint[m]")

        tm.assert_series_equal(row_sum, expected_1)

        col_sum = a.sum(axis=1)
        expected_2 = pd.Series([3, 12], dtype="pint[m]")

        tm.assert_series_equal(col_sum, expected_2)
