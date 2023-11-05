Getting Started
===============

The getting started guide aims to get you using pint-pandas productively as quickly as possible.


What is Pint-pandas?
--------------------

The Pandas package provides powerful DataFrame and Series abstractions for dealing with numerical, temporal, categorical, string-based, and even user-defined data (using its ExtensionArray feature). The Pint package provides a rich and extensible vocabulary of units for constructing Quantities and an equally rich and extensible range of unit conversions to make it easy to perform unit-safe calculations using Quantities. Pint-pandas provides PintArray, a Pandas ExtensionArray that efficiently implements Pandas DataFrame and Series functionality as unit-aware operations where appropriate.

Those who have used Pint know well that good units discipline often catches not only simple mistakes, but sometimes more fundamental errors as well. Pint-pandas can reveal similar errors when it comes to slicing and dicing Pandas data.


Installation
------------

Pint-pandas requires pint and pandas.

.. grid:: 2

    .. grid-item-card::  Prefer pip?

        **pint-pandas** can be installed via pip from `PyPI <https://pypi.org/project/pint-pandas>`__.

        ++++++++++++++++++++++

        .. code-block:: bash

            pip install pint-pandas

    .. grid-item-card::  Working with conda?

        **pint-pandas** is part of the `Conda-Forge <https://conda-forge.org//>`__
        channel and can be installed with Anaconda or Miniconda:

        ++++++++++++++++++++++

        .. code-block:: bash

            conda install -c conda-forge pint-pandas


That's all! You can check that Pint is correctly installed by starting up python, and importing Pint:

.. code-block:: python

    >>> import pint_pandas
    >>> pint_pandas.__version__  # doctest: +SKIP

.. toctree::
    :maxdepth: 2
    :hidden:

    tutorial
    faq
