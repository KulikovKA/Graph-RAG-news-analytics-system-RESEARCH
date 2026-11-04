"""Run from a checkout without importing/loading GLiNER, torch, or transformers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgraph.entity_typing_v2.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
