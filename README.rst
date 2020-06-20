.. image:: https://img.shields.io/pypi/v/pint-pandas.svg
    :target: https://pypi.python.org/pypi/pint-pandas
    :alt: Latest Version

.. image:: https://readthedocs.org/projects/pip/badge/
    :target: http://pint-pandas.readthedocs.org/
    :alt: Documentation

.. image:: https://img.shields.io/pypi/l/pint-pandas.svg
    :target: https://pypi.python.org/pypi/pint-pandas
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/pint-pandas.svg
    :target: https://pypi.python.org/pypi/pint-pandas
    :alt: Python Versions

.. image:: https://travis-ci.org/hgrecco/pint-pandas.svg?branch=master
    :target: https://travis-ci.org/hgrecco/pint-pandas
    :alt: CI

.. image:: https://coveralls.io/repos/github/hgrecco/pint-pandas/badge.svg?branch=master
    :target: https://coveralls.io/github/hgrecco/pint-pandas?branch=master
    :alt: Coverage

.. image:: https://readthedocs.org/projects/pint-pandas/badge/
    :target: http://pint-pandas.readthedocs.org/
    :alt: Docs


Pint-Pandas
===========

Pandas support for pint

.. code-block:: python

    >>> import pandas as pd
    >>> import pint


.. code-block:: python

    >>> df = pd.DataFrame({
    ...     "torque": pd.Series([1, 2, 2, 3], dtype="pint[lbf ft]"),
    ...     "angular_velocity": pd.Series([1, 2, 2, 3], dtype="pint[rpm]"),
    ... })
    >>> df['power'] = df['torque'] * df['angular_velocity']
    >>> df.dtypes
    torque                                       pint[foot * force_pound]
    angular_velocity                         pint[revolutions_per_minute]
    power               pint[foot * force_pound * revolutions_per_minute]
    dtype: object


Quick Installation
------------------

To install Pint-Pandas, simply:

.. code-block:: bash

    $ pip install pint-pandas

or utilizing conda, with the conda-forge channel:

.. code-block:: bash

    $ conda install -c conda-forge pint-pandas

and then simply enjoy it!

