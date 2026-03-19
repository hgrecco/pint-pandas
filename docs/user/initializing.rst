.. _initializing:

**************************
Initializing data
**************************

There are several ways to initialize a ``PintArray`` in a ``DataFrame``. Here's the most common methods.

.. ipython:: python
    :suppress:

    import pandas as pd
    import pint
    import pint_pandas
    import numpy as np

    PintArray = pint_pandas.PintArray
    ureg = pint_pandas.PintType.ureg
    Quantity = ureg.Quantity


.. ipython:: python
    :okwarning:

    df = pd.DataFrame(
        {
            "Ser1": pd.Series([1, 2], dtype="pint[m]"),
            "Ser2": pd.Series([1, 2]).astype("pint[m]"),
            "Ser3": pd.Series([1, 2], dtype="pint[m][Int64]"),
            "Ser4": pd.Series([1, 2]).astype("pint[m][Int64]"),
            "PArr1": PintArray([1, 2], dtype="pint[m]"),
            "PArr2": PintArray([1, 2], dtype="pint[m][Int64]"),
            "PArr3": PintArray([1, 2], dtype="m"),
            "PArr4": PintArray([1, 2], dtype=ureg.m),
            "PArr5": PintArray(Quantity([1, 2], ureg.m)),
            "PArr6": PintArray([1, 2],"m"),
        }
    )
    df


In the first two Series examples above, the data was converted to Float64.

.. ipython:: python

    df.dtypes


To avoid this conversion, specify the subdtype (dtype of the magnitudes) in the dtype ``"pint[m][Int64]"`` when constructing using a ``Series``. The default data dtype that pint-pandas converts to can be changed by modifying ``pint_pandas.pint_array.DEFAULT_SUBDTYPE``.

``PintArray`` infers the subdtype from the data passed into it when there is no subdtype specified in the dtype. It also accepts a pint ``Unit`` or unit string as the dtype.


.. note::

   ``"pint[unit]"`` or ``"pint[unit][subdtype]"`` must be used for the Series or DataFrame constuctor.

Non-native pandas dtypes
-------------------------

``PintArray`` uses an ``ExtensionArray`` to hold its data inclluding those from other libraries that extend pandas.
For example, an ``UncertaintyArray`` can be used.

.. ipython:: python

    from uncertainties_pandas import UncertaintyArray, UncertaintyDtype
    from uncertainties import ufloat, umath, unumpy

    ufloats = [ufloat(i, abs(i) / 100) for i in [4.0, np.nan, -5.0]]
    uarr = UncertaintyArray(ufloats)
    uarr
    PintArray(uarr,"m")
    pd.Series(PintArray(uarr,"m")*2)
