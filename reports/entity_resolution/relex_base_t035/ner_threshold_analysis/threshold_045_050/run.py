"""Refine NER thresholds .45 through .50 without changing the original sweep.

Uses the exact original 2F.0 analysis implementation and frozen source inputs.
Only this directory is written. No models, network or database access.
"""
from pathlib import Path
import importlib.util
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ner_threshold_analysis_2f0', OUT.parent / 'analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
analysis.OUT = OUT
analysis.THRESHOLDS = [t / 100 for t in range(45, 51)]


if __name__ == '__main__':
    analysis.main()
