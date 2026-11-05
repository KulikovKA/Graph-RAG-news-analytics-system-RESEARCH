"""Original E3 -> RelEx Base, threshold 0.35, isolated relex_base_t035 graph."""
from pathlib import Path
import sys

from relex_experiment import run

if __name__ == "__main__":
    run("e3", "base", [
        "--relation-threshold", "0.35",
        "--namespace", "relex_base_t035",
        "--output", str(Path(__file__).resolve().parents[1] / "reports/re_v2/relex_comparison/threshold035/e3_base"),
        *sys.argv[1:],
    ])
