.. _unitsincells:

**************************
Units in Cells
**************************

The most common issue pint-pandas users encouter is that they have a DataFrame with column that aren't PintArrays. 
An obvious indicator is unit strings showing in cells when viewing the DataFrame.


.. ipython:: python
    :suppress:

    import pandas as pd
    import pint
    import pint_pandas

    PA_ = pint_pandas.PintArray
    ureg = pint_pandas.PintType.ureg
    Q_ = ureg.Quantity

.. ipython:: python
    :okwarning:

    df = pd.DataFrame(
        {
            "length": pd.Series(np.array([Q_(2.0, ureg.m), Q_(3.0, ureg.m)],dtype="object")),
        }
    )
    df


To confirm the DataFrame does not contain PintArrays, check the dtypes.

.. ipython:: python

    df.dtypes


Pint-pandas provides an accessor to fix this issue by converting the non PintArray columns to PintArrays.

.. ipython:: python

    df.pint.convert_object_dtype() 