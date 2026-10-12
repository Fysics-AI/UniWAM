"""Compatibility package forwarding to :mod:`uniwam_piper_cloud`."""

from importlib import import_module as _import_module

_uniwam_cloud = _import_module("uniwam_piper_cloud")
__path__ = list(_uniwam_cloud.__path__)

