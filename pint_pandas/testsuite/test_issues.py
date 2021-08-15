import time

import numpy as np
import pandas as pd
import pytest

import pint_pandas as ppi
from pint_pandas import PintArray

ureg = ppi.PintType.ureg


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

        assert tp < 5 * t

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

            assert tp < 5 * t


def test_issue_86():
    a = PintArray([1, 2], ureg.m)
    b_listlike = [1 * ureg.km, 1 * ureg.m]
    units = b_listlike[0].units
    b_pa = PintArray([v.m_as(units) for v in b_listlike], units)

    assert np.all(a + b_listlike == a + b_pa)
