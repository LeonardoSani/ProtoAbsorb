# Compatibility shim: robustbench 1.1.x imports 'autoattack' but the PyPI package
# is named 'pyautoattack'. This stub redirects all autoattack.* imports to
# pyautoattack.* so robustbench works without modification.
import sys
import importlib
import pyautoattack  # noqa: F401

# Register pyautoattack as the authoritative autoattack module
sys.modules[__name__] = pyautoattack

# Pre-register known submodules used by robustbench
for _submod in ['state', 'autoattack', 'autopgd_base', 'checks', 'fab_base',
                 'fab_projections', 'fab_pt', 'other_utils', 'square']:
    try:
        _m = importlib.import_module(f'pyautoattack.{_submod}')
        sys.modules[f'autoattack.{_submod}'] = _m
    except ImportError:
        pass
