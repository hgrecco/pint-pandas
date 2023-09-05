.. _common:

**************************
Common Issues
**************************

Pandas support for ``ExtensionArray`` is still in development. As a result, there are some common issues that pint-pandas users may encounter.
This page provides some guidance on how to resolve these issues.

Units in Cells (Object dtype columns)
-------------------------------------

The most common issue pint-pandas users encouter is that they have a DataFrame with column that aren't PintArrays.
An obvious indicator is unit strings showing in cells when viewing the DataFrame.
Several pandas operations return numpy arrays of ``Quantity`` objects, which can cause this.


.. ipython:: python
    :suppress:
    :okwarning:

    import pandas as pd
    import pint
    import pint_pandas

    PA_ = pint_pandas.PintArray
    ureg = pint_pandas.PintType.ureg
    Q_ = ureg.Quantity

    df = pd.DataFrame(
        {
            "length": pd.Series(np.array([Q_(2.0, ureg.m), Q_(3.0, ureg.m)], dtype="object")),
        }
    )

.. ipython:: python

    df


To confirm the DataFrame does not contain PintArrays, check the dtypes.

.. ipython:: python

    df.dtypes


Pint-pandas provides an accessor to fix this issue by converting the non PintArray columns to PintArrays.

.. ipython:: python

    df.pint.convert_object_dtype()


Creating DataFrames from Series
---------------------------------

The default operation of Pandas `pd.concat` function is to perform row-wise concatenation.  When given a list of Series, each of which is backed by a PintArray, this will inefficiently convert all the PintArrays to arrays of `object` type, concatenate the several series into a DataFrame with that many rows, and then leave it up to you to convert that DataFrame back into column-wise PintArrays.  A much more efficient approach is to concatenate Series in a column-wise fashion:

.. ipython:: python
    :suppress:
    :okwarning:
        df = pd.concat(list_of_series, axis=1)


This will preserve all the PintArrays in each of the Series.
