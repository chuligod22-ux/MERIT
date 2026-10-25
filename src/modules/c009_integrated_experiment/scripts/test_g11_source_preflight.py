from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("g11_source_preflight.py")
SPEC = importlib.util.spec_from_file_location("g11_source_preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_declared_classes_reads_python_ast(tmp_path: Path) -> None:
    source = tmp_path / "arch.py"
    source.write_text(
        "class First:\n    pass\n\nclass Second(First):\n    pass\n",
        encoding="utf-8",
    )
    assert MODULE.declared_classes(source) == ["First", "Second"]


def test_tree_hash_is_stable_and_ignores_git_and_cache(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("beta", encoding="utf-8")
    first = MODULE.source_tree_sha256(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "index").write_text("ignored", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"ignored")
    assert MODULE.source_tree_sha256(tmp_path) == first
    (tmp_path / "sub" / "b.txt").write_text("changed", encoding="utf-8")
    assert MODULE.source_tree_sha256(tmp_path)[0] != first[0]


def test_missing_snapshot_is_blocked(tmp_path: Path) -> None:
    candidate = {
        "repository": "https://example.invalid/repository",
        "commit": "a" * 40,
        "configuration_path": "config.yml",
    }
    spec = {
        "directory": "missing",
        "architecture_path": "arch.py",
        "expected_classes": ["Model"],
        "license_candidates": ["LICENSE"],
    }
    result = MODULE.inspect_candidate(tmp_path, "fixture", candidate, spec)
    assert result["status"] == "BLOCKED"
    assert result["errors"] == ["LOCAL_SNAPSHOT_MISSING"]
