import pprint

from .pint_array import PintArray, PintType

try:
    from importlib.metadata import version
except ImportError:
    # Backport for Python < 3.8
    from importlib_metadata import version  # type: ignore

try:  # pragma: no cover
    __version__ = version("pint_pandas")
except Exception:  # pragma: no cover
    # we seem to have a local copy not installed without setuptools
    # so the reported version will be unknown
    __version__ = "unknown"

__all__ = ["PintArray", "PintType", "__version__"]


def show_versions():
    deps = [
        "pint_pandas",
        "pint",
        "pandas",
        "numpy",
    ]

    versions = {dep: version(dep) for dep in deps}
    pprint.pprint(versions)
