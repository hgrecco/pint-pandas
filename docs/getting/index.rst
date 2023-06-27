Getting Started
===============

The getting started guide aims to get you using pint-pandas productively as quickly as possible.


What is Pint-pandas?
--------------------

It is convenient to use the Pandas package when dealing with numerical data, so Pint-pandas provides PintArray. A PintArray is a Pandas ExtensionArray, which allows Pandas to recognise the Quantity and store it in Pandas DataFrames and Series.


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
