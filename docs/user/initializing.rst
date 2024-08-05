.. _initializing:

**************************
Initializing data
**************************

There are several ways to initialize a `PintArray`s` in a `DataFrame`. Here's the most common methods. We'll use `PA_` and `Q_` as shorthand for `PintArray` and `Quantity`.



.. ipython:: python

    import pandas as pd
    import pint
    import pint_pandas

    PA_ = pint_pandas.PintArray
    ureg = pint_pandas.PintType.ureg
    Q_ = ureg.Quantity

    df = pd.DataFrame(
        {
            "Ser1": pd.Series([1, 2], dtype="pint[m]"),
            "Ser2": pd.Series([1, 2]).astype("pint[m]"),
            "Ser3": pd.Series([1, 2], dtype="pint[m][Int64]"),
            "Ser4": pd.Series([1, 2]).astype("pint[m][Int64]"),
            "PArr1": PA_([1, 2], dtype="pint[m]"),
            "PArr2": PA_([1, 2], dtype="pint[m][Int64]"),
            "PArr3": PA_([1, 2], dtype="m"),
            "PArr4": PA_([1, 2], dtype=ureg.m),
            "PArr5": PA_(Q_([1, 2], ureg.m)),
            "PArr6": PA_([1, 2],"m"),
        }
    )
    df


In the first two Series examples above, the data was converted to Float64. 

.. ipython:: python

    df.dtypes
    

To avoid this conversion, specify the subdtype (dtype of the magnitudes) in the dtype `"pint[m][Int64]"` when constructing using a `Series`. The default data dtype that pint-pandas converts to can be changed by modifying `pint_pandas.DEFAULT_SUBDTYPE`.

`PintArray` infers the subdtype from the data passed into it when there is no subdtype specified in the dtype. It also accepts a pint `Unit`` or unit string as the dtype.


.. note::

   `"pint[unit]"` or `"pint[unit][subdtype]"` must be used for the Series or DataFrame constuctor.
