"""Compatibility package forwarding to :mod:`uniwam_piper_common`."""

from importlib import import_module as _import_module

_uniwam_common = _import_module("uniwam_piper_common")
__path__ = list(_uniwam_common.__path__)

