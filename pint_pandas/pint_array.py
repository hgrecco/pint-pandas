import copy
import re

import numpy as np
import pint
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_dataframe_accessor,
    register_extension_dtype,
    register_series_accessor,
)
from pandas.api.types import is_integer, is_list_like, is_scalar
from pandas.compat import set_function_name
from pandas.core import ops
from pandas.arrays import BooleanArray, IntegerArray
from pandas.core.arrays.base import ExtensionOpsMixin
from pint import errors, compat
from pint.quantity import _Quantity
from pint.unit import _Unit


def convert_indexing_key(key):
    if isinstance(key, BooleanArray):
        return key.to_numpy(dtype=np.bool, na_value=False)

    elif isinstance(key, IntegerArray):
        if pd.isna(key).any():
            raise ValueError("Cannot index with an integer indexer containing NA values")
        return key.to_numpy(dtype=np.int)

    elif isinstance(key, (list, tuple)):
        if all(isinstance(val, bool) for val in key if val is not pd.NA):
            return np.asarray([(False if val is pd.NA else val) for val in key], dtype=np.bool)
        if all(isinstance(val, int) for val in key if val is not pd.NA):
            if any(val is pd.NA for val in key):
                raise ValueError("Cannot index with an integer indexer containing NA values")
            return np.asarray([val for val in key if val is not pd.NA], dtype=np.int)

    return key


class PintType(ExtensionDtype):
    """
    A Pint duck-typed class, suitable for holding a quantity (with unit specified) dtype.
    """

    type = _Quantity
    # kind = 'O'
    # str = '|O08'
    # base = np.dtype('O')
    # num = 102
    _metadata = ("units",)
    _match = re.compile(r"(P|p)int\[(?P<units>.+)\]")
    _cache = {}
    ureg = pint.UnitRegistry()

    @property
    def _is_numeric(self):
        # type: () -> bool
        return True

    def __new__(cls, units=None):
        """
        Parameters
        ----------
        units : Pint units or string
        """

        if isinstance(units, PintType):
            return units

        elif units is None:
            # empty constructor for pickle compat
            # import pdb
            # pdb.set_trace()
            return object.__new__(cls)

        if not isinstance(units, _Unit):
            units = cls._parse_dtype_strict(units)
            # ureg.unit returns a quantity with a magnitude of 1
            # eg 1 mm. Initialising a quantity and taking it's unit
            # TODO: Seperate units from quantities in pint
            # to simplify this bit
            units = cls.ureg.Quantity(1, units).units

        try:
            return cls._cache["{:P}".format(units)]
        except KeyError:
            u = object.__new__(cls)
            u.units = units
            cls._cache["{:P}".format(units)] = u
            return u

    @classmethod
    def _parse_dtype_strict(cls, units):
        if isinstance(units, str):
            if units.startswith("pint[") or units.startswith("Pint["):
                if not units[-1] == "]":
                    raise ValueError("could not construct PintType")
                m = cls._match.search(units)
                if m is not None:
                    units = m.group("units")
            if units is not None:
                return units

        raise ValueError("could not construct PintType")

    @classmethod
    def construct_from_string(cls, string):
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
        if isinstance(string, str) and (
            string.startswith("pint[") or string.startswith("Pint[")
        ):
            # do not parse string like U as pint[U]
            # avoid tuple to be regarded as unit
            try:
                return cls(units=string)
            except ValueError:
                pass
        # This else block may allow pd.Series([1,2],dtype="m")
        #         else:
        #             try:
        #                 return cls(units=string)
        #             except ValueError:
        #                 pass

        raise TypeError("Cannot construct a 'PintType' from '{}'".format(string))

    # def __unicode__(self):
    # return compat.text_type(self.name)

    @property
    def name(self):
        return str("pint[{units}]".format(units=self.units))

    @property
    def na_value(self):
        return np.nan * self.units

    def __hash__(self):
        # make myself hashable
        return hash(str(self))

    def __eq__(self, other):
        try:
            other = PintType(other)
        except (ValueError, errors.UndefinedUnitError):
            return False
        return self.units == other.units

    @classmethod
    def is_dtype(cls, dtype):
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            if dtype.startswith("pint[") or dtype.startswith("Pint["):
                try:
                    if cls._parse_dtype_strict(dtype) is not None:
                        return True
                    else:
                        return False
                except ValueError:
                    return False
            else:
                return False
        return super(PintType, cls).is_dtype(dtype)

    @classmethod
    def construct_array_type(cls):
        return PintArray

    def __repr__(self):
        """
        Return a string representation for this object.

        Invoked by unicode(df) in py2 only. Yields a Unicode String in both
        py2/py3.
        """

        return self.name


class PintArray(ExtensionArray, ExtensionOpsMixin):
    _data = np.array([])
    context_name = None
    context_units = None

    def __init__(self, values, dtype=None, copy=False, data_dtype=None):
        if dtype is None:
            raise NotImplementedError

        if not isinstance(dtype, PintType):
            dtype = PintType(dtype)
        self._dtype = dtype
        if len(values) == 0:
            data_dtype = "float"
        else:
            data_dtype = type(values[0])
        self._data = np.array(values, data_dtype, copy=copy)

    @property
    def dtype(self):
        # type: () -> ExtensionDtype
        """An instance of 'ExtensionDtype'."""
        return self._dtype

    def __len__(self):
        # type: () -> int
        """Length of this array

        Returns
        -------
        length : int
        """
        return len(self._data)

    def __getitem__(self, item):
        # type (Any) -> Any
        """Select a subset of self.
        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'
        Returns
        -------
        item : scalar or PintArray
        """
        if is_integer(item):
            return self._data[item] * self.units

        item = convert_indexing_key(item)

        return self.__class__(self._data[item], self.dtype)

    def __setitem__(self, key, value):

        # need to not use `not value` on numpy arrays
        if isinstance(value, (list, tuple)) and (not value):
            # doing nothing here seems to be ok
            return

        if isinstance(value, _Quantity):
            value = value.to(self.units).magnitude
        elif is_list_like(value) and isinstance(value[0], _Quantity):
            value = [item.to(self.units).magnitude for item in value]
        _is_scalar = is_scalar(value)
        if _is_scalar:
            value = [value]

        if _is_scalar:
            value = value[0]

        key = convert_indexing_key(key)
        self._data[key] = value

    def _formatter(self, boxed=False):
        """Formatting function for scalar values.
        This is used in the default '__repr__'. The returned formatting
        function receives scalar Quantities.

        # type: (bool) -> Callable[[Any], Optional[str]]

        Parameters
        ----------
        boxed: bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """
        float_format = pint.formatting.remove_custom_flags(
            self.dtype.ureg.default_format
        )

        def formatting_function(quantity):
            return "{:{float_format}}".format(
                quantity.magnitude, float_format=float_format
            )

        return formatting_function

    def isna(self):
        # type: () -> np.ndarray
        """Return a Boolean NumPy array indicating if each value is missing.

        Returns
        -------
        missing : np.array
        """
        return np.isnan(self._data)

    def astype(self, dtype, copy=True):
        """Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        if isinstance(dtype, str) and (
            dtype.startswith("Pint[") or dtype.startswith("pint[")
        ):
            dtype = PintType(dtype)
        if isinstance(dtype, PintType):
            if dtype == self._dtype:
                return self
            else:
                return PintArray(self.quantity.to(dtype.units), dtype)
        return self.__array__(dtype, copy)

    @property
    def units(self):
        return self._dtype.units

    @property
    def quantity(self):
        return self.data * self._dtype.units

    def take(self, indices, allow_fill=False, fill_value=None):
        """Take elements from an array.

        # type: (Sequence[int], bool, Optional[Any]) -> PintArray

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.
            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.
            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.
        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

        Returns
        -------
        PintArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        Notes
        -----
        PintArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignemnt, with a `fill_value`.
        See Also
        --------
        numpy.take
        pandas.api.extensions.take
        Examples
        --------
        """
        from pandas.core.algorithms import take

        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        if isinstance(fill_value, _Quantity):
            fill_value = fill_value.to(self.units).magnitude

        result = take(data, indices, fill_value=fill_value, allow_fill=allow_fill)

        return PintArray(result, dtype=self.dtype)

    def copy(self, deep=False):
        data = self._data
        if deep:
            data = copy.deepcopy(data)
        else:
            data = data.copy()

        return type(self)(data, dtype=self.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        output_units = to_concat[0].units

        data = []
        for a in to_concat:
            converted_values = a.quantity.to(output_units).magnitude
            data.append(np.atleast_1d(converted_values))

        return cls(np.concatenate(data), output_units)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Initialises a PintArray from a list like of quantity scalars or a list like of floats and dtype
        -----
        Usage
        PintArray._from_sequence([Q_(1,"m"),Q_(2,"m")])
        """
        master_scalar = None
        try:
            master_scalar = next(i for i in scalars if hasattr(i, "units"))
        except StopIteration:
            if isinstance(scalars, PintArray):
                dtype = scalars._dtype
            if dtype is None:
                raise ValueError(
                    "Cannot infer dtype. No dtype specified and empty array"
                )
        if dtype is None and not isinstance(master_scalar, _Quantity):
            raise ValueError("No dtype specified and not a sequence of quantities")
        if dtype is None and isinstance(master_scalar, _Quantity):
            dtype = PintType(master_scalar.units)

        def quantify_nan(item):
            if type(item) is float:
                return item * dtype.units
            return item

        if isinstance(master_scalar, _Quantity):
            scalars = [quantify_nan(item) for item in scalars]
            scalars = [item.to(dtype.units).magnitude for item in scalars]
        # import pdb
        # pdb.set_trace()
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, dtype=original.dtype)

    def _values_for_factorize(self):
        arr = self._data
        return arr, np.NaN

    def value_counts(self, dropna=True):
        """
        Returns a Series containing counts of each category.

        Every category will have an entry, even those with a count of 0.

        Parameters
        ----------
        dropna : boolean, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """

        from pandas import Series

        # compute counts on the data with no nans
        data = self._data
        if dropna:
            data = data[~np.isnan(data)]

        data_list = data.tolist()
        index = list(set(data))
        array = [data_list.count(item) for item in index]

        return Series(array, index=index)

    def unique(self):
        """Compute the PintArray of unique values.

        Returns
        -------
        uniques : PintArray
        """
        from pandas import unique

        return self._from_sequence(unique(self._data), dtype=self.dtype)

    @property
    def data(self):
        return self._data

    @property
    def nbytes(self):
        return self._data.nbytes

    # The _can_hold_na attribute is set to True so that pandas internals
    # will use the ExtensionDtype.na_value as the NA value in operations
    # such as take(), reindex(), shift(), etc.  In addition, those results
    # will then be of the ExtensionArray subclass rather than an array
    # of objects
    _can_hold_na = True

    @property
    def _ndarray_values(self):
        # type: () -> np.ndarray
        """Internal pandas method for lossy conversion to a NumPy ndarray.
        This method is not part of the pandas interface.
        The expectation is that this is cheap to compute, and is primarily
        used for interacting with our indexers.
        """
        return np.array(self)

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True):
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.
        Parameters
        ----------
        op : function
            An operator that takes arguments op(a, b)
        coerce_to_dtype :  bool
            boolean indicating whether to attempt to convert
            the result to the underlying ExtensionArray dtype
            (default True)
        Returns
        -------
        A method that can be bound to a method of a class
        Example
        -------
        Given an ExtensionArray subclass called MyExtensionArray, use
        >>> __add__ = cls._create_method(operator.add)
        in the class definition of MyExtensionArray to create the operator
        for addition, that will be based on the operator implementation
        of the underlying elements of the ExtensionArray
        """

        def _binop(self, other):
            def validate_length(obj1, obj2):
                # validates length and converts to listlike
                try:
                    if len(obj1) == len(obj2):
                        return obj2
                    else:
                        raise ValueError("Lengths must match")
                except TypeError:
                    return [obj2] * len(obj1)

            def convert_values(param):
                # convert to a quantity or listlike
                if isinstance(param, cls):
                    return param.quantity
                elif isinstance(param, _Quantity):
                    return param
                elif is_list_like(param) and isinstance(param[0], _Quantity):
                    return type(param[0])([p.magnitude for p in param], param[0].units)
                else:
                    return param

            if isinstance(other, Series):
                return NotImplemented
            lvalues = self.quantity
            other = validate_length(lvalues, other)
            rvalues = convert_values(other)
            # Pint quantities may only be exponented by single values, not arrays.
            # Reduce single value arrays to single value to allow power ops
            if isinstance(rvalues, _Quantity):
                if len(set(np.array(rvalues.data))) == 1:
                    rvalues = rvalues[0]
            elif len(set(np.array(rvalues))) == 1:
                rvalues = rvalues[0]
            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            res = op(lvalues, rvalues)

            if op.__name__ == "divmod":
                return (
                    cls.from_1darray_quantity(res[0]),
                    cls.from_1darray_quantity(res[1]),
                )

            if coerce_to_dtype:
                try:
                    res = cls.from_1darray_quantity(res)
                except TypeError:
                    pass

            return res

        op_name = ops._get_op_name(op, True)
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op):
        return cls._create_method(op)

    @classmethod
    def _create_comparison_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False)

    @classmethod
    def from_1darray_quantity(cls, quantity):
        if not is_list_like(quantity.magnitude):
            raise TypeError("quantity's magnitude is not list like")
        return cls(quantity.magnitude, quantity.units)

    def __array__(self, dtype=None, copy=False):
        if dtype in [None, "object"]:
            return self._to_array_of_quantity(copy=copy)
        return np.array(self._data, dtype=dtype, copy=copy)

    def _to_array_of_quantity(self, copy=False):
        qtys = [item * self.units for item in self._data]
        return np.array(qtys, dtype="object", copy=copy)

    def searchsorted(self, value, side="left", sorter=None):
        """
        Find indices where elements should be inserted to maintain order.

        .. versionadded:: 0.24.0

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `v` were inserted before the indices, the
        order of `self` would be preserved.

        Assuming that `a` is sorted:

        ======  ============================
        `side`  returned index `i` satisfies
        ======  ============================
        left    ``self[i-1] < v <= self[i]``
        right   ``self[i-1] <= v < self[i]``
        ======  ============================

        Parameters
        ----------
        value : array_like
            Values to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array_like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        indices : array of ints
            Array of insertion points with the same shape as `value`.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.
        """
        # Note: the base tests provided by pandas only test the basics.
        # We do not test
        # 1. Values outside the range of the `data_for_sorting` fixture
        # 2. Values between the values in the `data_for_sorting` fixture
        # 3. Missing values.
        arr = self._data
        if isinstance(value, _Quantity):
            value = value.to(self.units).magnitude
        elif is_list_like(value) and isinstance(value[0], _Quantity):
            value = [item.to(self.units).magnitude for item in value]
        return arr.searchsorted(value, side=side, sorter=sorter)


PintArray._add_arithmetic_ops()
PintArray._add_comparison_ops()
register_extension_dtype(PintType)


@register_dataframe_accessor("pint")
class PintDataFrameAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def quantify(self, level=-1):
        df = self._obj
        df_columns = df.columns.to_frame()
        unit_col_name = df_columns.columns[level]
        units = df_columns[unit_col_name]
        df_columns = df_columns.drop(columns=unit_col_name)

        df_new = DataFrame(
            {i: PintArray(df.values[:, i], unit) for i, unit in enumerate(units.values)}
        )

        df_new.columns = df_columns.index.droplevel(unit_col_name)
        df_new.index = df.index

        return df_new

    def dequantify(self):
        def formatter_func(units):
            formatter = "{:" + units._REGISTRY.default_format + "}"
            return formatter.format(units)

        df = self._obj

        df_columns = df.columns.to_frame()
        df_columns["units"] = [
            formatter_func(df[col].values.units) for col in df.columns
        ]
        from collections import OrderedDict

        data_for_df = OrderedDict()
        for i, col in enumerate(df.columns):
            data_for_df[tuple(df_columns.iloc[i])] = df[col].values.data
        df_new = DataFrame(data_for_df, columns=data_for_df.keys())

        df_new.columns.names = df.columns.names + ["unit"]
        df_new.index = df.index

        return df_new

    def to_base_units(self):
        obj = self._obj
        df = self._obj
        index = object.__getattribute__(obj, "index")
        # name = object.__getattribute__(obj, '_name')
        return DataFrame(
            {col: df[col].pint.to_base_units() for col in df.columns}, index=index
        )


@register_series_accessor("pint")
class PintSeriesAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self.pandas_obj = pandas_obj
        self.quantity = pandas_obj.values.quantity
        self._index = pandas_obj.index
        self._name = pandas_obj.name

    @staticmethod
    def _validate(obj):
        if not is_pint_type(obj):
            raise AttributeError(
                "Cannot use 'pint' accessor on objects of "
                "dtype '{}'.".format(obj.dtype)
            )


class Delegated:
    # Descriptor for delegating attribute access to from
    # a Series to an underlying array
    to_series = True

    def __init__(self, name):
        self.name = name


class DelegatedProperty(Delegated):
    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, "_index")
        name = object.__getattribute__(obj, "_name")
        result = getattr(object.__getattribute__(obj, "quantity"), self.name)
        if self.to_series:
            if isinstance(result, _Quantity):
                result = PintArray(result)
            return Series(result, index, name=name)
        else:
            return result


class DelegatedScalarProperty(DelegatedProperty):
    to_series = False


class DelegatedMethod(Delegated):
    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, "_index")
        name = object.__getattribute__(obj, "_name")
        method = getattr(object.__getattribute__(obj, "quantity"), self.name)

        def delegated_method(*args, **kwargs):
            result = method(*args, **kwargs)
            if self.to_series:
                if isinstance(result, _Quantity):
                    result = PintArray.from_1darray_quantity(result)
                result = Series(result, index, name=name)
            return result

        return delegated_method


class DelegatedScalarMethod(DelegatedMethod):
    to_series = False


for attr in [
    "debug_used",
    "default_format",
    "dimensionality",
    "dimensionless",
    "force_ndarray",
    "shape",
    "u",
    "unitless",
    "units",
]:
    setattr(PintSeriesAccessor, attr, DelegatedScalarProperty(attr))
for attr in ["imag", "m", "magnitude", "real"]:
    setattr(PintSeriesAccessor, attr, DelegatedProperty(attr))

for attr in [
    "check",
    "compatible_units",
    "format_babel",
    "ito",
    "ito_base_units",
    "ito_reduced_units",
    "ito_root_units",
    "plus_minus",
    "put",
    "to_tuple",
    "tolist",
]:
    setattr(PintSeriesAccessor, attr, DelegatedScalarMethod(attr))
for attr in [
    "clip",
    "from_tuple",
    "m_as",
    "searchsorted",
    "to",
    "to_base_units",
    "to_compact",
    "to_reduced_units",
    "to_root_units",
    "to_timedelta",
]:
    setattr(PintSeriesAccessor, attr, DelegatedMethod(attr))


def is_pint_type(obj):
    t = getattr(obj, "dtype", obj)
    try:
        return isinstance(t, PintType) or issubclass(t, PintType)
    except Exception:
        return False


compat.upcast_types.append(PintArray)