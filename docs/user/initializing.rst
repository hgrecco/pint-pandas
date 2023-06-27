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
            "length": pd.Series([1.0, 2.0], dtype="pint[m]"),
            "width": PA_([2.0, 3.0], dtype="pint[m]"),
            "distance": PA_([2.0, 3.0], dtype="m"),
            "height": PA_([2.0, 3.0], dtype=ureg.m),
            "depth": PA_.from_1darray_quantity(Q_([2, 3], ureg.m)),
            "displacement": PA_(Q_([2.0, 3.0], ureg.m)),
        }
    )
    df


.. note::

   "pint[unit]" must be used for the Series or DataFrame constuctor.
