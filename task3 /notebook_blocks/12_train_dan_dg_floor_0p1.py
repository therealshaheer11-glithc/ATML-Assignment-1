"""Block 12: run only the approved bandwidth-floor lambda-0.1 condition."""

import sys
from pathlib import Path


CODE_ROOT = Path("/content/atml_pa1_task3_source")
sys.path.insert(0, str(CODE_ROOT))

from task3.research_variants.notebook_runner import run_variant_block  # noqa: E402


run_variant_block("dan_dg_floor_0p1")
