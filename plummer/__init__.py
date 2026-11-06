"""plummer-gen: condições iniciais de esferas de Plummer para simulações de N-corpos."""
from .generate import PlummerModel, generate
from . import checks, io, units

__version__ = "1.0.0"
__all__ = ["PlummerModel", "generate", "checks", "io", "units"]
