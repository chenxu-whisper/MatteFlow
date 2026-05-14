"""Vendored MatAnyone2 runtime package from EZ-CorridorKey."""

from __future__ import annotations

import sys

from . import matanyone2 as _vendored_matanyone2

# Preserve the original upstream package name so absolute imports inside the
# vendored MatAnyone2 source continue to resolve without editing every file.
sys.modules.setdefault("matanyone2", _vendored_matanyone2)
