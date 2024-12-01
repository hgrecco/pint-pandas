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


Using a Shared Unit Registry
----------------------------

As described `in the documentation of the main pint package: <https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html#using-pint-in-your-projects>`_:

    If you use Pint in multiple modules within your Python package, you normally want to avoid creating multiple instances of the unit registry. The best way to do this is by instantiating the registry in a single place. For example, you can add the following code to your package ``__init__.py``

When using `pint_pandas`, this extends to using the same unit registry that was created by the main `pint` package. This is done by using the 
:func:`pint.get_application_registry() <pint:get_application_registry>` function.

In a sample project structure of this kind:

.. code-block:: text

    .
    └── mypackage/
        ├── __init__.py
        ├── main.py
        └── mysubmodule/
            ├── __init__.py
            └── calculations.py

After defining the registry in the ``mypackage.__init__`` module:

.. code-block:: python

    import pint
    ureg = pint.get_application_registry()

In the ``mypackage.mysubmodule.calculations`` module, you should *get* the shared registry like so:

.. code-block:: python

    import pint
    ureg = pint.get_application_registry()

    @ureg.check(
        '[length]',
    )
    def multiply_value(distance):
        return distance * 2

Failure to do this will result in a ``DimensionalityError`` of the kind:

    Cannot convert from '<VALUE> <UNIT>' ([<DIMENSION>]) to 'a quantity of' ([<DIMENSION>])".

For example:

.. code-block:: text

    DimensionalityError: Cannot convert from '200 metric_ton' ([mass]) to 'a quantity of' ([mass])"
