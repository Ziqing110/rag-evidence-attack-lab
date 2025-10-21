def test_package_import():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import rag_eval_lab  # noqa: F401

