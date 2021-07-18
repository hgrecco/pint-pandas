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
