
Pandas support
==============

It is convenient to use the Pandas package when dealing with numerical
data, so Pint provides PintArray. A PintArray is a Pandas Extension
Array, which allows Pandas to recognise the Quantity and store it in
Pandas DataFrames and Series.

Basic example
-------------

This example will show the simplist way to use pandas with pint and the
underlying objects. It's slightly fiddly as you are not reading from a
file. A more normal use case is given in Reading a csv.

First some imports

.. code:: python

    import pandas as pd 
    import pint

Next, we create a DataFrame with PintArrays as columns.

.. code:: python

    df = pd.DataFrame({
        "torque": pd.Series([1, 2, 2, 3], dtype="pint[lbf ft]"),
        "angular_velocity": pd.Series([1, 2, 2, 3], dtype="pint[rpm]"),
    })
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>torque</th>
          <th>angular_velocity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>2</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>3</td>
        </tr>
      </tbody>
    </table>
    </div>



Operations with columns are units aware so behave as we would
intuitively expect.

.. code:: python

    df['power'] = df['torque'] * df['angular_velocity']
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>torque</th>
          <th>angular_velocity</th>
          <th>power</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>2</td>
          <td>4</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>3</td>
          <td>9</td>
        </tr>
      </tbody>
    </table>
    </div>



We can see the columns' units in the dtypes attribute

.. code:: python

    df.dtypes




.. parsed-literal::

    torque                                       pint[foot * force_pound]
    angular_velocity                         pint[revolutions_per_minute]
    power               pint[foot * force_pound * revolutions_per_minute]
    dtype: object



Each column can be accessed as a Pandas Series

.. code:: python

    df.power




.. parsed-literal::

    0    1
    1    4
    2    4
    3    9
    Name: power, dtype: pint[foot * force_pound * revolutions_per_minute]



Which contains a PintArray

.. code:: python

    df.power.values




.. parsed-literal::

    PintArray([1 foot * force_pound * revolutions_per_minute,
               4 foot * force_pound * revolutions_per_minute,
               4 foot * force_pound * revolutions_per_minute,
               9 foot * force_pound * revolutions_per_minute],
              dtype='pint[foot * force_pound * revolutions_per_minute]')



The PintArray contains a Quantity

.. code:: python

    df.power.values.quantity




.. raw:: html

    \[\begin{pmatrix}1 & 4 & 4 & 9\end{pmatrix} foot force_pound revolutions_per_minute\]



Pandas Series accessors are provided for most Quantity properties and
methods, which will convert the result to a Series where possible.

.. code:: python

    df.power.pint.units




.. raw:: html

    foot force_pound revolutions_per_minute



.. code:: python

    df.power.pint.to("kW").values




.. parsed-literal::

    PintArray([0.00014198092353610376 kilowatt, 0.000567923694144415 kilowatt,
               0.000567923694144415 kilowatt, 0.0012778283118249339 kilowatt],
              dtype='pint[kilowatt]')



Reading from csv
----------------

Reading from files is the far more standard way to use pandas. To
facilitate this, DataFrame accessors are provided to make it easy to get
to PintArrays.

Setup
~~~~~

Here we create the DateFrame and save it to file, next we will show you
how to load and read it.

We start with a DateFrame with column headers only.

.. code:: python

    import pandas as pd 
    import pint
    import numpy as np

.. code:: python

    df_init = pd.DataFrame({
        "speed": [1000, 1100, 1200, 1200],
        "mech power": [np.nan, np.nan, np.nan, np.nan],
        "torque": [10, 10, 10, 10],
        "rail pressure": [1000, 1000000000000, 1000, 1000],
        "fuel flow rate": [10, 10, 10, 10],
        "fluid power": [np.nan, np.nan, np.nan, np.nan],
    })
    df_init




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>speed</th>
          <th>mech power</th>
          <th>torque</th>
          <th>rail pressure</th>
          <th>fuel flow rate</th>
          <th>fluid power</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1000</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1100</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000000000000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1200</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



Then we add a column header which contains units information

.. code:: python

    units = ["rpm", "kW", "N m", "bar", "l/min", "kW"]
    df_to_save = df_init.copy()
    df_to_save.columns = pd.MultiIndex.from_arrays([df_init.columns, units])
    df_to_save




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>speed</th>
          <th>mech power</th>
          <th>torque</th>
          <th>rail pressure</th>
          <th>fuel flow rate</th>
          <th>fluid power</th>
        </tr>
        <tr>
          <th></th>
          <th>rpm</th>
          <th>kW</th>
          <th>N m</th>
          <th>bar</th>
          <th>l/min</th>
          <th>kW</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1000</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1100</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000000000000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1200</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



Now we save this to disk as a csv to give us our starting point.

.. code:: python

    test_csv_name = "pandas_test.csv"
    df_to_save.to_csv(test_csv_name, index=False)

Now we are in a position to read the csv we just saved. Let's start by
reading the file with units as a level in a multiindex column.

.. code:: python

    df = pd.read_csv(test_csv_name, header=[0,1])
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>speed</th>
          <th>mech power</th>
          <th>torque</th>
          <th>rail pressure</th>
          <th>fuel flow rate</th>
          <th>fluid power</th>
        </tr>
        <tr>
          <th></th>
          <th>rpm</th>
          <th>kW</th>
          <th>N m</th>
          <th>bar</th>
          <th>l/min</th>
          <th>kW</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1000</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1100</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000000000000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1200</td>
          <td>NaN</td>
          <td>10</td>
          <td>1000</td>
          <td>10</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



Then use the DataFrame's pint accessor's quantify method to convert the
columns from ``np.ndarray``\ s to PintArrays, with units from the bottom
column level.

.. code:: python

    df.dtypes




.. parsed-literal::

    speed           rpm        int64
    mech power      kW       float64
    torque          N m        int64
    rail pressure   bar        int64
    fuel flow rate  l/min      int64
    fluid power     kW       float64
    dtype: object



.. code:: python

    df_ = df.pint.quantify(level=-1)
    df_




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>speed</th>
          <th>mech power</th>
          <th>torque</th>
          <th>rail pressure</th>
          <th>fuel flow rate</th>
          <th>fluid power</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1000.0</td>
          <td>nan</td>
          <td>10.0</td>
          <td>1000.0</td>
          <td>10.0</td>
          <td>nan</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1100.0</td>
          <td>nan</td>
          <td>10.0</td>
          <td>1000000000000.0</td>
          <td>10.0</td>
          <td>nan</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200.0</td>
          <td>nan</td>
          <td>10.0</td>
          <td>1000.0</td>
          <td>10.0</td>
          <td>nan</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1200.0</td>
          <td>nan</td>
          <td>10.0</td>
          <td>1000.0</td>
          <td>10.0</td>
          <td>nan</td>
        </tr>
      </tbody>
    </table>
    </div>



As previously, operations between DataFrame columns are unit aware

.. code:: python

    df_.speed*df_.torque




.. parsed-literal::

    0    10000.0
    1    11000.0
    2    12000.0
    3    12000.0
    dtype: pint[meter * newton * revolutions_per_minute]



.. code:: python

    df_




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>speed</th>
          <th>mech power</th>
          <th>torque</th>
          <th>rail pressure</th>
          <th>fuel flow rate</th>
          <th>fluid power</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1000.0</td>
          <td>nan</td>
          <td>10.0</td>
          <td>1000.0</td>
          <td>10.0</td>
          <td>nan</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1100.0</td>
          <td>nan</td>
          <td>10.0</td>
          <td>1000000000000.0</td>
          <td>10.0</td>
          <td>nan</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200.0</td>
          <td>nan</td>
          <td>10.0</td>
          <td>1000.0</td>
          <td>10.0</td>
          <td>nan</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1200.0</td>
          <td>nan</td>
          <td>10.0</td>
          <td>1000.0</td>
          <td>10.0</td>
          <td>nan</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    df_['mech power'] = df_.speed*df_.torque
    df_['fluid power'] = df_['fuel flow rate'] * df_['rail pressure']
    df_




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>speed</th>
          <th>mech power</th>
          <th>torque</th>
          <th>rail pressure</th>
          <th>fuel flow rate</th>
          <th>fluid power</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1000.0</td>
          <td>10000.0</td>
          <td>10.0</td>
          <td>1000.0</td>
          <td>10.0</td>
          <td>10000.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1100.0</td>
          <td>11000.0</td>
          <td>10.0</td>
          <td>1000000000000.0</td>
          <td>10.0</td>
          <td>10000000000000.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200.0</td>
          <td>12000.0</td>
          <td>10.0</td>
          <td>1000.0</td>
          <td>10.0</td>
          <td>10000.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1200.0</td>
          <td>12000.0</td>
          <td>10.0</td>
          <td>1000.0</td>
          <td>10.0</td>
          <td>10000.0</td>
        </tr>
      </tbody>
    </table>
    </div>



The DataFrame's ``pint.dequantify`` method then allows us to retrieve
the units information as a header row once again.

.. code:: python

    df_.pint.dequantify()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>speed</th>
          <th>mech power</th>
          <th>torque</th>
          <th>rail pressure</th>
          <th>fuel flow rate</th>
          <th>fluid power</th>
        </tr>
        <tr>
          <th>unit</th>
          <th>revolutions_per_minute</th>
          <th>meter * newton * revolutions_per_minute</th>
          <th>meter * newton</th>
          <th>bar</th>
          <th>liter / minute</th>
          <th>bar * liter / minute</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1000.0</td>
          <td>10000.0</td>
          <td>10.0</td>
          <td>1.000000e+03</td>
          <td>10.0</td>
          <td>1.000000e+04</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1100.0</td>
          <td>11000.0</td>
          <td>10.0</td>
          <td>1.000000e+12</td>
          <td>10.0</td>
          <td>1.000000e+13</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200.0</td>
          <td>12000.0</td>
          <td>10.0</td>
          <td>1.000000e+03</td>
          <td>10.0</td>
          <td>1.000000e+04</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1200.0</td>
          <td>12000.0</td>
          <td>10.0</td>
          <td>1.000000e+03</td>
          <td>10.0</td>
          <td>1.000000e+04</td>
        </tr>
      </tbody>
    </table>
    </div>



This allows for some rather powerful abilities. For example, to change
single column units

.. code:: python

    df_['fluid power'] = df_['fluid power'].pint.to("kW")
    df_['mech power'] = df_['mech power'].pint.to("kW")
    df_.pint.dequantify()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>speed</th>
          <th>mech power</th>
          <th>torque</th>
          <th>rail pressure</th>
          <th>fuel flow rate</th>
          <th>fluid power</th>
        </tr>
        <tr>
          <th>unit</th>
          <th>revolutions_per_minute</th>
          <th>kilowatt</th>
          <th>meter * newton</th>
          <th>bar</th>
          <th>liter / minute</th>
          <th>kilowatt</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1000.0</td>
          <td>1.047198</td>
          <td>10.0</td>
          <td>1.000000e+03</td>
          <td>10.0</td>
          <td>1.666667e+01</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1100.0</td>
          <td>1.151917</td>
          <td>10.0</td>
          <td>1.000000e+12</td>
          <td>10.0</td>
          <td>1.666667e+10</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200.0</td>
          <td>1.256637</td>
          <td>10.0</td>
          <td>1.000000e+03</td>
          <td>10.0</td>
          <td>1.666667e+01</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1200.0</td>
          <td>1.256637</td>
          <td>10.0</td>
          <td>1.000000e+03</td>
          <td>10.0</td>
          <td>1.666667e+01</td>
        </tr>
      </tbody>
    </table>
    </div>



The units are harder to read than they need be, so lets change pints
default format for displaying units.

.. code:: python

    pint.pintpandas.PintType.ureg.default_format = "~P"
    df_.pint.dequantify()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>speed</th>
          <th>mech power</th>
          <th>torque</th>
          <th>rail pressure</th>
          <th>fuel flow rate</th>
          <th>fluid power</th>
        </tr>
        <tr>
          <th>unit</th>
          <th>rpm</th>
          <th>kW</th>
          <th>N·m</th>
          <th>bar</th>
          <th>l/min</th>
          <th>kW</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1000.0</td>
          <td>1.047198</td>
          <td>10.0</td>
          <td>1.000000e+03</td>
          <td>10.0</td>
          <td>1.666667e+01</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1100.0</td>
          <td>1.151917</td>
          <td>10.0</td>
          <td>1.000000e+12</td>
          <td>10.0</td>
          <td>1.666667e+10</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200.0</td>
          <td>1.256637</td>
          <td>10.0</td>
          <td>1.000000e+03</td>
          <td>10.0</td>
          <td>1.666667e+01</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1200.0</td>
          <td>1.256637</td>
          <td>10.0</td>
          <td>1.000000e+03</td>
          <td>10.0</td>
          <td>1.666667e+01</td>
        </tr>
      </tbody>
    </table>
    </div>



or the entire table's units

.. code:: python

    df_.pint.to_base_units().pint.dequantify()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>speed</th>
          <th>mech power</th>
          <th>torque</th>
          <th>rail pressure</th>
          <th>fuel flow rate</th>
          <th>fluid power</th>
        </tr>
        <tr>
          <th>unit</th>
          <th>rad/s</th>
          <th>kg·m²/s³</th>
          <th>kg·m²/s²</th>
          <th>kg/m/s²</th>
          <th>m³/s</th>
          <th>kg·m²/s³</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>104.719755</td>
          <td>1047.197551</td>
          <td>10.0</td>
          <td>1.000000e+08</td>
          <td>0.000167</td>
          <td>1.666667e+04</td>
        </tr>
        <tr>
          <th>1</th>
          <td>115.191731</td>
          <td>1151.917306</td>
          <td>10.0</td>
          <td>1.000000e+17</td>
          <td>0.000167</td>
          <td>1.666667e+13</td>
        </tr>
        <tr>
          <th>2</th>
          <td>125.663706</td>
          <td>1256.637061</td>
          <td>10.0</td>
          <td>1.000000e+08</td>
          <td>0.000167</td>
          <td>1.666667e+04</td>
        </tr>
        <tr>
          <th>3</th>
          <td>125.663706</td>
          <td>1256.637061</td>
          <td>10.0</td>
          <td>1.000000e+08</td>
          <td>0.000167</td>
          <td>1.666667e+04</td>
        </tr>
      </tbody>
    </table>
    </div>



Advanced example
----------------

This example shows alternative ways to use pint with pandas and other
features.

Start with the same imports.

.. code:: python

    import pandas as pd 
    import pint

We'll be use a shorthand for PintArray

.. code:: python

    PA_ = pint.pintpandas.PintArray

And set up a unit registry and quantity shorthand.

.. code:: python

    ureg=pint.UnitRegistry()
    Q_=ureg.Quantity

Operations between PintArrays of different unit registry will not work.
We can change the unit registry that will be used in creating new
PintArrays to prevent this issue.

.. code:: python

    pint.pintpandas.PintType.ureg = ureg

These are the possible ways to create a PintArray.

Note that pint[unit] must be used for the Series constuctor, whereas the
PintArray constructor allows the unit string or object.

.. code:: python

    df = pd.DataFrame({
            "length" : pd.Series([1,2], dtype="pint[m]"),
            "width" : PA_([2,3], dtype="pint[m]"),
            "distance" : PA_([2,3], dtype="m"),
            "height" : PA_([2,3], dtype=ureg.m),
            "depth" : PA_.from_1darray_quantity(Q_([2,3],ureg.m)),
        })
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>length</th>
          <th>width</th>
          <th>distance</th>
          <th>height</th>
          <th>depth</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>2</td>
          <td>2</td>
          <td>2</td>
          <td>2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>3</td>
          <td>3</td>
          <td>3</td>
          <td>3</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    df.length.values.units




.. raw:: html

    meter


