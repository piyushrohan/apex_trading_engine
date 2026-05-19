from pathlib import Path

import pytest

FORBIDDEN = "USDT"
SCAN_DIRS = ("src", "configs", "tests")


@pytest.mark.unit
def test_no_usdt_symbols_in_project_sources():
    """CI gate: USDC-only mandate — no USDT in src, configs, or tests."""
    root = Path(__file__).resolve().parents[2]
    violations = []
    for scan_dir in SCAN_DIRS:
        base = root / scan_dir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml", ".md"}:
                continue
            if path.name == "test_usdt_forbidden.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if FORBIDDEN in text:
                violations.append(str(path.relative_to(root)))
    assert not violations, f"USDT references found: {violations}"
