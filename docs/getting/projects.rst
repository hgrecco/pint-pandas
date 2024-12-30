*****************************
Pint-Pandas in your projects
*****************************

Using a Shared Unit Registry
----------------------------

As described `in the documentation of the main pint package: <https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html#using-pint-in-your-projects>`_:

    If you use Pint in multiple modules within your Python package, you normally want to avoid creating multiple instances of the unit registry. The best way to do this is by instantiating the registry in a single place. For example, you can add the following code to your package ``__init__.py``

When using `pint_pandas`, this extends to using the same unit registry that was created by the main `pint` package. This is done by using the :func:`pint.get_application_registry() <pint:get_application_registry>` function.

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

    from pint import UnitRegistry, set_application_registry
    ureg = UnitRegistry()
    ureg.formatter.default_format = "P"

    set_application_registry(ureg)

In the ``mypackage.mysubmodule.calculations`` module, you should *get* the shared registry like so:

.. code-block:: python

    import pint
    ureg = pint.get_application_registry()

    @ureg.check(
        '[length]',
    )
    def multiply_value(distance):
        return distance * 2

Failure to use the application registry will result in a ``DimensionalityError`` of the kind:

    Cannot convert from '<VALUE> <UNIT>' ([<DIMENSION>]) to 'a quantity of' ([<DIMENSION>])".

For example:

.. code-block:: text

    DimensionalityError: Cannot convert from '200 metric_ton' ([mass]) to 'a quantity of' ([mass])"
