"""Model modules. Call load() to import them and populate nnlib.core.REGISTRY."""

import importlib
import pkgutil


def names():
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith('_'))


def load(only=None):
    """Import one module (only='diffusion') or all of them (only=None)."""
    for name in ([only] if only else names()):
        importlib.import_module(f'{__name__}.{name}')
