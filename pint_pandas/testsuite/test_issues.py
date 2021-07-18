import time

import numpy as np
import pandas as pd
import pint_pandas as ppi
from pint_pandas import PintArray

ureg = ppi.PintType.ureg


def test_issue_80():
    def timeit(fun, n_runs=5):
        run_time = []
        for k in range(n_runs):
            t_start = time.monotonic_ns()
            fun()
            t_end = time.monotonic_ns()
            run_time.append(t_end - t_start)
        return np.median(run_time) * ureg.ns

    def make_df(size, pint_units=True, dtype=float):
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

    n = 1_000_000
    df_pint = make_df(n)
    df = make_df(n, pint_units=False)

    tp = timeit(lambda: df_pint["distance"] / df_pint["time"]).to("ms")
    t = timeit(lambda: df["distance"] / df["time"]).to("ms")

    assert tp < 5 * t


def test_issue_86():
    a = PintArray([1, 2] * ureg.m, ureg.m)
    b_listlike = [1 * ureg.km, 1 * ureg.m]
    units = b_listlike[0].units
    b_pa = PintArray([v.m_as(units) for v in b_listlike], units)

    assert np.all(a + b_listlike == a + b_pa)
