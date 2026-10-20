"""Enforce the ground-truth policy: nothing under src/ may read simulator truth.

The controller must use the estimated state, never ground truth (automatic
fail in the marking guide), so this test makes the separation visible and
checkable. Run with:  python -m pytest
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# MuJoCo state that reveals the true pose or velocity
FORBIDDEN = [r"\bxpos\b", r"\bxquat\b", r"\bxmat\b", r"\bqvel\b", r"\bcvel\b",
             r"\bsubtree_com\b", r"\bsensor\(\s*['\"]\w*(pos|quat)"]
IMPORTS = [r"^\s*from\s+evaluation", r"^\s*import\s+evaluation", r"true_pose"]


def _src_files():
    return [p for p in SRC.rglob("*.py")]


def test_src_never_imports_evaluation():
    for path in _src_files():
        for line in path.read_text().splitlines():
            for pat in IMPORTS:
                assert not re.search(pat, line), f"{path.relative_to(SRC.parent)} uses ground truth: {line.strip()}"


def test_src_never_reads_true_state():
    for path in _src_files():
        text = path.read_text()
        for pat in FORBIDDEN:
            assert not re.search(pat, text), f"{path.relative_to(SRC.parent)} reads simulator state matching {pat}"


def test_qpos_only_written_by_sim_reset():
    """qpos holds the true pose. Only the simulator may touch it, to set the start pose."""
    for path in _src_files():
        if path.parent.name == "sim":
            continue
        assert "qpos" not in path.read_text(), f"{path.relative_to(SRC.parent)} touches qpos"
