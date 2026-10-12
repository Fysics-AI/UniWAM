"""Compatibility import path for the renamed :mod:`uniwam` package.

New code should import ``uniwam``. The package search path is redirected to
the canonical implementation so historical Hydra targets such as
``fastwam.runtime.create_fastwam_mixed_stream`` continue to load without a
second copy of the implementation.
"""

from importlib import import_module as _import_module

_uniwam = _import_module("uniwam")
for _name, _value in vars(_uniwam).items():
    if _name not in {"__name__", "__package__", "__path__"}:
        globals()[_name] = _value

# Let ``fastwam.<submodule>`` resolve against the canonical package tree.
__path__ = list(_uniwam.__path__)
__all__ = getattr(_uniwam, "__all__", [])
