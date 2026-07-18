"""Inkling Quant Lab public package.

Importing the package intentionally does not import PyTorch, Transformers, or any
optional quantization backend. Components are loaded through lazy registries.
"""

from inkling_quant_lab.version import __version__

__all__ = ["__version__"]
