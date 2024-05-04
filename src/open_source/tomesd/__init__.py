# Token Merging for Stable Diffusion, source code is derived from: https://github.com/dbolya/tomesd

from . import merge, patch
from .patch import apply_patch, remove_patch

__all__ = ["merge", "patch", "apply_patch", "remove_patch"]