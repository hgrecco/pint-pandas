.. _initializing:

**************************
Initializing data
**************************

There are several ways to initialize PintArrays in a DataFrame. Here's the most common methods. We'll use `PA_` and `Q_` as shorthand for PintArray and Quantity.



.. ipython:: python

    import pandas as pd
    import pint
    import pint_pandas
    import io

    PA_ = pint_pandas.PintArray
    ureg = pint_pandas.PintType.ureg
    Q_ = ureg.Quantity

    df = pd.DataFrame(
        {
            "A": pd.Series([1.0, 2.0], dtype="pint[m]"),
            "B": pd.Series([1.0, 2.0]).astype("pint[m]"),
            "C": PA_([2.0, 3.0], dtype="pint[m]"),
            "D": PA_([2.0, 3.0], dtype="m"),
            "E": PA_([2.0, 3.0], dtype=ureg.m),
            "F": PA_.from_1darray_quantity(Q_([2, 3], ureg.m)),
            "G": PA_(Q_([2.0, 3.0], ureg.m)),
        }
    )
    df


.. note::

   "pint[unit]" must be used for the Series or DataFrame constuctor.
