"""Model loader module.

This is the real model-loading implementation. New code should import from
this package directly.
"""

from .base_loader import BaseModelLoader as BaseModelLoader
from .base_loader import LoadConfig as LoadConfig
from .base_loader import __all__ as _base_all
from .default_loader import MODEL_REGISTRY as MODEL_REGISTRY
from .default_loader import DefaultModelLoader as DefaultModelLoader
from .default_loader import __all__ as _default_all
from .default_loader import get_model as get_model
from .default_loader import get_model_loader as get_model_loader

__all__ = [*_base_all, *_default_all]
