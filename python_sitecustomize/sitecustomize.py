"""Process-wide FlashInfer source overlay hook.

Put this directory on PYTHONPATH and set SPARK_FLASHINFER_SOURCE_ROOT to make
server processes apply the same FlashInfer JIT source-path rewrite used by the
standalone probes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


if os.environ.get("SPARK_FLASHINFER_SOURCE_ROOT"):
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    scripts_path = str(scripts_dir)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    import flashinfer_source_sitecustomize  # noqa: F401
