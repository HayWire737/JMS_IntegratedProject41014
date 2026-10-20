"""Shared helpers for the entry-point scripts."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))      # lets `python scripts/x.py` import src/ and evaluation/


def load_config(path):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def results_dir():
    d = ROOT / "results"
    d.mkdir(exist_ok=True)
    return d
