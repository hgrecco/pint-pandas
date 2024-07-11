import copy
import numbers
import re
import warnings
from importlib.metadata import version
from typing import Any, Callable, Dict, Optional, Union, cast
from packaging.version import parse as version_parse

import numpy as np
import pandas as pd
import pint
from pandas import DataFrame, Series, Index
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    ExtensionScalarOpsMixin,
    register_dataframe_accessor,
    register_extension_dtype,
    register_series_accessor,
)
from pandas.api.indexers import check_array_indexer
from pandas.api.types import is_integer, is_list_like, is_object_dtype, is_string_dtype
from pandas.compat import set_function_name
from pandas.core import nanops  # type: ignore
from pint import Quantity as _Quantity
from pint import Unit as _Unit
from pint import compat, errors

# Magic 'unit' flagging columns with no unit support, used in
# quantify/dequantify
NO_UNIT = "No Unit"

pandas_version = version("pandas")
pandas_version_info = tuple(
    int(x) if x.isdigit() else x for x in pandas_version.split(".")
)


class PintType(ExtensionDtype):
    """
    A Pint duck-typed class, suitable for holding a quantity (with unit specified) dtype.
    """

    type = _Quantity
    # kind = 'O'
    # str = '|O08'
    # base = np.dtype('O')
    # num = 102
    units: Optional[_Unit] = None  # Filled in by `construct_from_..._string`
    _metadata = ("units",)
    _match = re.compile(r"(P|p)int\[(?P<units>.+)\]")
    _cache = {}  # type: ignore
    ureg = pint.get_application_registry()

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
            return object.__new__(cls)

        if not isinstance(units, _Unit):
            units = cls._parse_dtype_strict(units)
            # ureg.unit returns a quantity with a magnitude of 1
            # eg 1 mm. Initialising a quantity and taking its unit
            # TODO: Seperate units from quantities in pint
            # to simplify this bit
            units = cls.ureg.Quantity(1, units).units

        try:
            # TODO: fix when Pint implements Callable typing
            # TODO: wrap string into PintFormatStr class
            return cls._cache["{:P}".format(units)]  # type: ignore
        except KeyError:
            u = object.__new__(cls)
            u.units = units
            cls._cache["{:P}".format(units)] = u  # type: ignore
            return u

    @classmethod
    def _parse_dtype_strict(cls, units):
        if isinstance(units, str):
            if units.lower() == "pint[]":
                units = "pint[dimensionless]"
            if units.lower().startswith("pint["):
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
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if isinstance(string, str) and (
            string.startswith("pint[") or string.startswith("Pint[")
        ):
            # do not parse string like U as pint[U]
            # avoid tuple to be regarded as unit
            try:
                return cls(units=string)
            except ValueError:
                pass
        raise TypeError(f"Cannot construct a 'PintType' from '{string}'")

    @classmethod
    def construct_from_quantity_string(cls, string):
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_quantity_string' expects a string, got {type(string)}"
            )

        quantity = cls.ureg.Quantity(string)
        return cls(units=quantity.units)

    # def __unicode__(self):
    # return compat.text_type(self.name)

    @property
    def name(self):
        return str("pint[{units}]".format(units=self.units))

    @property
    def na_value(self):
        return self.ureg.Quantity(np.nan, self.units)

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

    def _get_common_dtype(self, dtypes):
        """return the common dtype from list provided.

        If this function is called this means at least on of the ``dtypes``
        list is a ``PintType``

        In order to be able to be able to perform operation on ``PintType``
        with scalars, mix of ``PintType`` and numeric values are allowed.


        Parameters
        ----------
        dtypes (list): list of dtypes in which common is requested

        Returns
        -------
        returns self for acceptable cases or None otherwise
        """
        if all(
            isinstance(x, PintType) or pd.api.types.is_numeric_dtype(x) for x in dtypes
        ):
            return self
        else:
            return None


_NumpyEADtype = (
    pd.core.dtypes.dtypes.PandasDtype  # type: ignore
    if pandas_version_info < (2, 1)
    else pd.core.dtypes.dtypes.NumpyEADtype  # type: ignore
)

dtypemap = {
    int: pd.Int64Dtype(),
    np.int64: pd.Int64Dtype(),
    np.int32: pd.Int32Dtype(),
    np.int16: pd.Int16Dtype(),
    np.int8: pd.Int8Dtype(),
    # np.float128: pd.Float128Dtype(),
    float: pd.Float64Dtype(),
    np.float64: pd.Float64Dtype(),
    np.float32: pd.Float32Dtype(),
    np.complex128: _NumpyEADtype("complex128"),
    np.complex64: _NumpyEADtype("complex64"),
    # np.float16: pd.Float16Dtype(),
}
ddtypemap: dict[np.dtype, object] = {np.dtype(k): v for k, v in dtypemap.items()}
dtypeunmap = {v: k for k, v in ddtypemap.items()}


def convert_np_inputs(inputs):
    if isinstance(inputs, tuple):
        return tuple(x.quantity if isinstance(x, PintArray) else x for x in inputs)
    if isinstance(inputs, dict):
        return {
            item: (x.quantity if isinstance(x, PintArray) else x) for item, x in inputs
        }


class PintArray(ExtensionArray, ExtensionScalarOpsMixin):
    """Implements a class to describe an array of physical quantities:
    the product of an array of numerical values and a unit of measurement.

    Parameters
    ----------
    values : pint.Quantity or array-like
        Array of physical quantity values to be created.
    dtype : PintType, str, or pint.Unit
        Units of the physical quantity to be created. (Default value = None)
        When values is a pint.Quantity, passing None as the dtype will use
        the units from the pint.Quantity.
    copy: bool
        Whether to copy the values.
    Returns
    -------

    """

    _data: ExtensionArray = cast(ExtensionArray, np.array([]))
    context_name = None
    context_units = None
    _HANDLED_TYPES = (np.ndarray, numbers.Number, _Quantity)

    def __init__(self, values, dtype=None, copy=False):
        if dtype is None:
            if isinstance(values, _Quantity):
                dtype = values.units
            elif isinstance(values, PintArray):
                dtype = values._dtype
        if dtype is None:
            raise NotImplementedError

        if not isinstance(dtype, PintType):
            dtype = PintType(dtype)
        self._dtype = dtype

        if isinstance(values, _Quantity):
            values = values.to(dtype.units).magnitude
        elif isinstance(values, PintArray):
            values = values._data
        if isinstance(values, np.ndarray):
            dtype = values.dtype
            if dtype in ddtypemap:
                dtype = ddtypemap[dtype]
            values = pd.array(values, copy=copy, dtype=dtype)
            copy = False
        elif not isinstance(values, pd.core.arrays.numeric.NumericArray):
            values = pd.array(values, copy=copy)
        if copy:
            values = values.copy()
        self._data = values
        self._Q = self.dtype.ureg.Quantity

    def __getstate__(self):
        # we need to discard the cached _Q, which is not pickleable
        ret = dict(self.__dict__)
        ret.pop("_Q")
        return ret

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        self._Q = self.dtype.ureg.Quantity

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (PintArray,)):
                return NotImplemented

        # Defer to pint's implementation of the ufunc.
        inputs = convert_np_inputs(inputs)
        if out:
            kwargs["out"] = convert_np_inputs(out)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        return self._convert_np_result(result)

    def _convert_np_result(self, result):
        if isinstance(result, _Quantity) and is_list_like(result.m):
            return PintArray.from_1darray_quantity(result)
        elif isinstance(result, _Quantity):
            return result
        elif type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif isinstance(result, np.ndarray) and all(
            isinstance(item, _Quantity) for item in result
        ):
            return PintArray._from_sequence(result)
        elif result is None:
            # no return value
            return result
        elif pd.api.types.is_bool_dtype(result):
            return result
        else:
            # one return value
            return type(self)(result)

    def __pos__(self):
        return 1 * self

    def __neg__(self):
        return -1 * self

    def __abs__(self):
        return type(self)(self._Q(np.abs(self._data), self._dtype.units))

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
            return self._Q(self._data[item], self.units)

        item = check_array_indexer(self, item)

        return self.__class__(self._data[item], self.dtype)

    def __setitem__(self, key, value):
        # need to not use `not value` on numpy arrays
        if isinstance(value, (list, tuple)) and (not value):
            # doing nothing here seems to be ok
            return

        if isinstance(value, _Quantity):
            value = value.to(self.units).magnitude
        elif is_list_like(value) and len(value) > 0:
            if isinstance(value[0], _Quantity):
                value = [item.to(self.units).magnitude for item in value]
            if len(value) == 1:
                value = value[0]

        key = check_array_indexer(self, key)
        # Filter out invalid values for our array type(s)
        try:
            self._data[key] = value
        except IndexError as e:
            msg = "Mask is wrong length. {}".format(e)
            raise IndexError(msg)
        except TypeError as e:
            raise ValueError(e)

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
        # TODO: remove this once 0.24 is min pint version
        if version_parse(pint.__version__).base_version < "0.24":
            float_format = pint.formatting.remove_custom_flags(
                self.dtype.ureg.default_format
            )
        else:
            float_format = pint.formatting.remove_custom_flags(
                self.dtype.ureg.formatter.default_format
            )

        def formatting_function(quantity):
            if isinstance(quantity.magnitude, float):
                return "{:{float_format}}".format(
                    quantity.magnitude, float_format=float_format
                )
            else:
                return str(quantity.magnitude)

        return formatting_function

    def isna(self):
        # type: () -> np.ndarray
        """Return a Boolean NumPy array indicating if each value is missing.

        Returns
        -------
        missing : np.array
        """
        return cast(np.ndarray, self._data.isna())

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
            if dtype == self._dtype and not copy:
                return self
            else:
                return PintArray(self.quantity.to(dtype.units).magnitude, dtype)
        # do *not* delegate to __array__ -> is required to return a numpy array,
        # but somebody may be requesting another pandas array
        # examples are e.g. PyArrow arrays as requested by "string[pyarrow]"
        if is_object_dtype(dtype):
            return self._to_array_of_quantity(copy=copy)
        if is_string_dtype(dtype):
            return pd.array([str(x) for x in self.quantity], dtype=dtype)
        if isinstance(self._data, ExtensionArray):
            return self._data.astype(dtype, copy=copy)
        return pd.array(self.quantity.m, dtype, copy)

    @property
    def units(self):
        return self._dtype.units

    @property
    def quantity(self):
        return self._Q(self.numpy_data, self._dtype.units)

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
        from pandas.api.types import is_scalar

        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        if isinstance(fill_value, _Quantity):
            fill_value = fill_value.to(self.units).magnitude
            if not is_scalar(fill_value) and not fill_value.ndim:
                # deal with Issue #165; for unit registries with force_ndarray_like = True,
                # magnitude is in fact an array scalar, which will get rejected by pandas.
                fill_value = fill_value[()]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Turn off warning that PandasArray is deprecated for ``take``
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
        if dtype is None:
            if not isinstance(master_scalar, _Quantity):
                raise ValueError("No dtype specified and not a sequence of quantities")
            dtype = PintType(master_scalar.units)

        if isinstance(master_scalar, _Quantity):
            scalars = [
                (item.to(dtype.units).magnitude if hasattr(item, "to") else item)
                for item in scalars
            ]
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_of_strings(cls, scalars, dtype=None, copy=False):
        if not dtype:
            dtype = PintType.construct_from_quantity_string(scalars[0])
        return cls._from_sequence([dtype.ureg.Quantity(x) for x in scalars])

    @classmethod
    def _from_factorized(cls, values, original):
        from pandas.api.types import infer_dtype

        if infer_dtype(values) != "object":
            values = pd.array(values, copy=False)
        return cls(values, dtype=original.dtype)

    def _values_for_factorize(self):
        # factorize can now handle differentiating various types of null values.
        # These can only occur when the array has object dtype.
        # However, for backwards compatibility we only use the null for the
        # provided dtype. This may be revisited in the future, see GH#48476.
        arr = self._data
        if arr.dtype.kind == "O":
            return np.array(arr, copy=False), self.dtype.na_value
        return arr._values_for_factorize()

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
        nafilt = pd.isna(data)
        na_value = pd.NA  # NA value for index, not data, so not quantified
        data = data[~nafilt]
        index = list(set(data))

        data_list = data.tolist()
        array = [data_list.count(item) for item in index]

        if not dropna:
            index.append(na_value)
            array.append(nafilt.sum())

        return Series(np.asarray(array), index=index)

    def unique(self):
        """Compute the PintArray of unique values.

        Returns
        -------
        uniques : PintArray
        """
        from pandas import unique

        data = self._data
        return self._from_sequence(unique(data), dtype=self.dtype)

    def __contains__(self, item) -> Union[bool, np.bool_]:
        if not isinstance(item, _Quantity):
            return False
        elif pd.isna(item.magnitude):
            return cast(np.ndarray, self.isna()).any()
        else:
            return super().__contains__(item)

    @property
    def data(self):
        return self._data

    @property
    def numpy_data(self):
        data = self.data
        if data.dtype in dtypeunmap:
            try:
                data = data.astype(dtypeunmap[data.dtype])
            except Exception:
                # We might get here for integer arrays with <NA> values
                # In that case, the returned quantity will have dtype=O, which is less useful.
                pass
        if hasattr(data, "to_numpy"):
            data = data.to_numpy()
        return data

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
                # validates length
                # CHANGED: do not convert to listlike (why should we? pint.Quantity is perfecty able to handle that...)
                try:
                    if len(obj1) != len(obj2):
                        raise ValueError("Lengths must match")
                except TypeError:
                    pass

            def convert_values(param):
                # convert to a quantity or listlike
                if isinstance(param, cls):
                    return param.quantity
                elif isinstance(param, (_Quantity, _Unit)):
                    return param
                elif (
                    is_list_like(param)
                    and len(param) > 0
                    and isinstance(param[0], _Quantity)
                ):
                    units = param[0].units
                    return type(param[0])([p.m_as(units) for p in param], units)
                else:
                    return param

            if isinstance(other, (Series, DataFrame, Index)):
                return NotImplemented
            lvalues = self.quantity
            validate_length(lvalues, other)
            rvalues = convert_values(other)
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

        op_name = f"__{op}__"
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
        if dtype is None or is_object_dtype(dtype):
            return self._to_array_of_quantity(copy=copy)
        if is_string_dtype(dtype):
            return np.array([str(x) for x in self.quantity], dtype=str)
        return np.array(self._data, dtype=dtype)

    def _to_array_of_quantity(self, copy=False):
        qtys = [
            self._Q(item, self._dtype.units)
            if not pd.isna(item)
            else self.dtype.na_value
            for item in self._data
        ]
        with warnings.catch_warnings(record=True):
            return np.array(qtys, dtype="object")

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
        elif is_list_like(value) and len(value) > 0 and isinstance(value[0], _Quantity):
            value = [item.to(self.units).magnitude for item in value]
        return arr.searchsorted(value, side=side, sorter=sorter)

    def map(self, mapper, na_action=None):
        """
        Map values using an input mapping or function.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence. If 'ignore' is not supported, a
            ``NotImplementedError`` should be raised.

        Returns
        -------
        If mapper is a function, operate on the magnitudes of the array and

        """
        if pandas_version_info < (2, 1):
            ser = pd.Series(self._to_array_of_quantity())
            arr = ser.map(mapper, na_action).values
        else:
            from pandas.core.algorithms import map_array

            arr = map_array(self, mapper, na_action)

        master_scalar = None
        try:
            master_scalar = next(i for i in arr if hasattr(i, "units"))
        except StopIteration:
            # JSON mapper formatting Qs as str don't create PintArrays
            # ...and that's OK.  Caller will get array of values
            return arr
        return PintArray._from_sequence(arr, PintType(master_scalar.units))

    def _reduce(self, name, *, skipna: bool = True, keepdims: bool = False, **kwds):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """

        functions = {
            "any": nanops.nanany,
            "all": nanops.nanall,
            "min": nanops.nanmin,
            "max": nanops.nanmax,
            "sum": nanops.nansum,
            "mean": nanops.nanmean,
            "median": nanops.nanmedian,
            "std": nanops.nanstd,
            "var": nanops.nanvar,
            "sem": nanops.nansem,
            "kurt": nanops.nankurt,
            "skew": nanops.nanskew,
        }
        if name not in functions:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        if isinstance(self._data, ExtensionArray):
            try:
                # TODO: https://github.com/pandas-dev/pandas-stubs/issues/850
                result = self._data._reduce(  # type: ignore
                    name, skipna=skipna, keepdims=keepdims, **kwds
                )
            except NotImplementedError:
                result = cast(_Quantity, functions[name](self.numpy_data, **kwds))

        if name in {"all", "any", "kurt", "skew"}:
            return result
        if name == "var":
            if keepdims:
                return PintArray(result, f"pint[({self.units})**2]")
            return self._Q(result, self.units**2)
        if keepdims:
            return PintArray(result, self.dtype)
        return self._Q(result, self.units)

    def _accumulate(self, name: str, *, skipna: bool = True, **kwds):
        if name == "cumprod":
            raise TypeError("cumprod not supported for pint arrays")
        functions: Dict[
            str, Callable[[np._typing._SupportsArray[np.dtype[Any]]], Any]
        ] = {
            "cummin": np.minimum.accumulate,
            "cummax": np.maximum.accumulate,
            "cumsum": np.cumsum,
        }

        if isinstance(self._data, ExtensionArray):
            try:
                # TODO: https://github.com/pandas-dev/pandas-stubs/issues/850
                result = self._data._accumulate(name, **kwds)  # type: ignore
            except NotImplementedError:
                result = functions[name](self.numpy_data, **kwds)

        return self._from_sequence(result, self.units)


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
            {
                i: PintArray(df.iloc[:, i], unit) if unit != NO_UNIT else df.iloc[:, i]
                for i, unit in enumerate(units.values)
            }
        )

        df_new.columns = df_columns.index.droplevel(unit_col_name)
        df_new.index = df.index

        return df_new

    def dequantify(self):
        def formatter_func(dtype):
            # TODO: remove once pint 0.24 is min version supported
            if version_parse(pint.__version__).base_version < "0.24":
                formatter = "{:" + dtype.ureg.default_format + "}"
            else:
                formatter = "{:" + dtype.ureg.formatter.default_format + "}"
            return formatter.format(dtype.units)

        df = self._obj

        df_columns = df.columns.to_frame()
        df_columns["units"] = [
            formatter_func(df.dtypes.iloc[i])
            if isinstance(df.dtypes.iloc[i], PintType)
            else NO_UNIT
            for i, col in enumerate(df.columns)
        ]

        data_for_df = []
        for i, col in enumerate(df.columns):
            if isinstance(df.dtypes.iloc[i], PintType):
                data_for_df.append(
                    pd.Series(
                        data=df.iloc[:, i].values.data,
                        name=tuple(df_columns.iloc[i]),
                        index=df.index,
                        copy=False,
                    )
                )
            else:
                data_for_df.append(
                    pd.Series(
                        data=df.iloc[:, i].values,
                        name=tuple(df_columns.iloc[i]),
                        index=df.index,
                        copy=False,
                    )
                )

        df_new = pd.concat(data_for_df, axis=1, copy=False)
        df_new.columns.names = df.columns.names + ["unit"]

        return df_new

    def to_base_units(self):
        obj = self._obj
        df = self._obj
        index = object.__getattribute__(obj, "index")
        # name = object.__getattribute__(obj, '_name')
        return DataFrame(
            {
                col: (
                    df[col].pint.to_base_units()
                    if isinstance(df[col].dtype, PintType)
                    else df[col]
                )
                for col in df.columns
            },
            index=index,
        )

    def convert_object_dtype(self):
        df = self._obj
        df_new = pd.DataFrame()
        for col in df.columns:
            s = df[col]
            if s.dtype == "object":
                try:
                    df_new[col] = s.pint.convert_object_dtype()
                except AttributeError:
                    df_new[col] = s
            else:
                df_new[col] = s
        return df_new


@register_series_accessor("pint")
class PintSeriesAccessor(object):
    def __init__(self, pandas_obj):
        if self._is_object_dtype_and_quantity(pandas_obj):
            self.pandas_obj = pandas_obj
        else:
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

    @staticmethod
    def _is_object_dtype_and_quantity(obj):
        return obj.dtype == "object" and all(
            [(isinstance(item, _Quantity) or pd.isna(item)) for item in obj.values]
        )

    def convert_object_dtype(self):
        return pd.Series(
            data=PintArray._from_sequence(self.pandas_obj.values),
            index=self.pandas_obj.index,
            name=self.pandas_obj.name,
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


try:
    # for pint < 0.21 we need to explicitly register
    # TODO: fix when Pint is properly typed for mypy
    compat.upcast_types.append(PintArray)  # type: ignore
except AttributeError:
    # for pint = 0.21 we need to add the full names of PintArray and DataFrame,
    # which is to be added in pint > 0.21
    compat.upcast_type_map.setdefault("pint_pandas.pint_array.PintArray", PintArray)  # type: ignore
    compat.upcast_type_map.setdefault("pandas.core.frame.DataFrame", DataFrame)  # type: ignore
