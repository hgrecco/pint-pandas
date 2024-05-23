.. _reading:

**************************
Reading data and plotting
**************************

Reading from files is the far more standard way to use pandas.
To facilitate this, :py:class:`DataFrame` accessors are provided to make it easy to get to :py:class:`PintArray` objects.


Read data from csv
-----------------------
First some imports

.. ipython:: python

    import pandas as pd
    import pint
    import pint_pandas
    import io


Here's the contents of the csv file.

.. ipython:: python

    test_data = """ShaftSpeedIndex,rpm,1200,1200,1200,1600,1600,1600,2300,2300,2300
    pump,,A,B,C,A,B,C,A,B,C
    TestDate,No Unit,01/01,01/01,01/01,01/01,01/01,01/01,01/02,01/02,01/02
    ShaftSpeed,rpm,1200,1200,1200,1600,1600,1600,2300,2300,2300
    FlowRate,m^3 h^-1,8.72,9.28,9.31,11.61,12.78,13.51,18.32,17.90,19.23
    DifferentialPressure,kPa,162.03,144.16,136.47,286.86,241.41,204.21,533.17,526.74,440.76
    ShaftPower,kW,1.32,1.23,1.18,3.09,2.78,2.50,8.59,8.51,7.61
    Efficiency,dimensionless,30.60,31.16,30.70,30.72,31.83,31.81,32.52,31.67,32.05"""

Let's read that into a DataFrame. Here io.StringIO is used in place of reading a file from disk, whereas a csv file path would typically be used and is shown commented.

.. ipython:: python

    df = pd.read_csv(io.StringIO(test_data), header=[0, 1], index_col=[0, 1]).T
    # df = pd.read_csv("/path/to/test_data.csv", header=[0, 1])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    df.dtypes


Pandas DataFrame Accessors
---------------------------
Then use the :py:class:`DataFrame`'s pint accessor's quantify method to convert the columns from :py:class:`ndarray` to :py:class:`PintArray`, with units from the bottom column level.

Using 'No Unit' as the unit will prevent quantify converting a column to a :py:class:`PintArray`. This can be changed by changing :py:attr:`pint_pandas.pint_array.NO_UNIT`.

.. ipython:: python

    df_ = df.pint.quantify(level=-1)
    df_

Let's confirm the units have been parsed correctly by looking at the dtypes.

.. ipython:: python

    df_.dtypes

Here the Efficiency has been parsed as dimensionless. Let's change it to percent.

.. ipython:: python

    df_["Efficiency"] = pint_pandas.PintArray(
        df_["Efficiency"].values.quantity.m, dtype="pint[percent]"
    )
    df_.dtypes

As previously, operations between DataFrame columns are unit aware

.. ipython:: python

    df_.ShaftPower / df_.ShaftSpeed
    df_["ShaftTorque"] = df_.ShaftPower / df_.ShaftSpeed
    df_["FluidPower"] = df_["FlowRate"] * df_["DifferentialPressure"]
    df_


The DataFrame's pint.dequantify method then allows us to retrieve the units information as a header row once again.

.. ipython:: python

    df_.pint.dequantify()

This allows for some rather powerful abilities. For example, to change a column's units

.. ipython:: python

    df_["FluidPower"] = df_["FluidPower"].pint.to("kW")
    df_["FlowRate"] = df_["FlowRate"].pint.to("L/s")
    df_["ShaftTorque"] = df_["ShaftTorque"].pint.to("N m")
    df_.pint.dequantify()

The units are harder to read than they need be, so lets change pint's `default format for displaying units <https://pint.readthedocs.io/en/stable/user/formatting.html>`_.

.. ipython:: python

    pint_pandas.PintType.ureg.formatter.default_format = "P~"
    df_.pint.dequantify()

or the entire table's units

.. ipython:: python

    df_.pint.to_base_units().pint.dequantify()


Plotting
-----------------------

Pint's `matplotlib support <https://pint.readthedocs.io/en/stable/user/plotting.html>`_ allows columns with the same dimensionality to be plotted.
First, set up matplotlib to use pint's units.


.. ipython:: python

    import matplotlib.pyplot as plt
    pint_pandas.PintType.ureg.setup_matplotlib()

Let's convert a column to a different unit and plot two columns with different units. Pint's matplotlib support will automatically convert the units to the first units and add the units to the axis labels.

.. ipython:: python

    df_['FluidPower'] = df_['FluidPower'].pint.to('W')
    df_[["ShaftPower", "FluidPower"]].dtypes

    fig, ax = plt.subplots()

    @savefig plot_simple.png
    ax = df_[["ShaftPower", "FluidPower"]].unstack("pump").plot(ax=ax)


.. ipython:: python

    ax.yaxis.units
    ax.yaxis.label

.. TODO add index with units example
