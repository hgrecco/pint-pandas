.. _numpy:

**************************
Numpy support
**************************

Numpy functions that work on pint ``Quantity`` ``ndarray`` objects also work on ``PintArray``.


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

    pa = PintArray([1, 2, np.nan, 4, 10], dtype="pint[m]")
    np.clip(pa, 3 * ureg.m, 5 * ureg.m)

Note that this function errors when applied to a ``Series``.

.. ipython:: python
    :okexcept:

    df = pd.DataFrame({"A": pa})
    np.clip(df['A'], 3 * ureg.m, 5 * ureg.m)

Apply the function to the ``PintArray`` instead of the ``Series`` using ``Series.values``.

.. ipython:: python
    :okexcept:

    np.clip(df['A'].values, 3 * ureg.m, 5 * ureg.m)
