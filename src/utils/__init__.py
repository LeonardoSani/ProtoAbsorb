"""Public surface for the `utils` package.

Callers that previously did `from proto_absorb.utils import …` should now do
`from utils import …`. CLI entry points in `cli.py` are NOT re-exported here
(they are referenced from `[project.scripts]` and remain out of scope of this
refactor).
"""

from .utils import ensure_dir, get_device, get_logger, set_seed

__all__ = ["ensure_dir", "get_device", "get_logger", "set_seed"]
