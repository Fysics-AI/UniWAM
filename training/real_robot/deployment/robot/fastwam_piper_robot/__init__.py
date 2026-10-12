"""Compatibility package forwarding to :mod:`uniwam_piper_robot`."""

from importlib import import_module as _import_module

_uniwam_robot = _import_module("uniwam_piper_robot")
__path__ = list(_uniwam_robot.__path__)

