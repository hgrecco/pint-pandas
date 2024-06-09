.. _tutorial:

**************************
Tutorial
**************************

This example will show the simplest way to use pandas with pint and the underlying objects.
It's slightly fiddly to set up units compared to reading data and units from a file.
A more typical use case is given in :doc:`Reading from csv <../user/reading>`.


Imports
-----------------------
First some imports

.. ipython:: python

   import pandas as pd
   import pint
   import pint_pandas

   pint_pandas.show_versions()


Create a DataFrame
-----------------------
Next, we create a DataFrame with PintArrays as columns.

.. ipython:: python

   df = pd.DataFrame(
      {
         "torque": pd.Series([1.0, 2.0, 2.0, 3.0], dtype="pint[lbf ft]"),
         "angular_velocity": pd.Series([1.0, 2.0, 2.0, 3.0], dtype="pint[rpm]"),
      }
   )
   df


DataFrame Operations
-----------------------
Operations with columns are units aware so behave as we would intuitively expect.

.. ipython:: python

   df["power"] = df["torque"] * df["angular_velocity"]
   df


.. note::

   Notice that the units are not displayed in the cells of the DataFrame.
   If you ever see units in the cells of the DataFrame, something isn't right.
   See :ref:`units_in_cells` for more information.

We can see the columns' units in the dtypes attribute

.. ipython:: python

   df.dtypes

Each column can be accessed as a Pandas Series

.. ipython:: python

   df.power

Which contains a PintArray

.. ipython:: python

   df.power.values

The PintArray contains a Quantity

.. ipython:: python

   df.power.values.quantity

Pandas Series Accessors
-----------------------
Pandas Series accessors are provided for most Quantity properties and methods.
Methods that return arrays will be converted to Series.

.. ipython:: python

   df.power.pint.units
   df.power.pint.to("kW")
