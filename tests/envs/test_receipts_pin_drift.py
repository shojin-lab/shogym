"""The code pin moves when the code it covers moves, shown by editing that code.

These tests write into the package's own source while they run. One appends a line to every
pinned module in turn and reads the pin again, one does the same to a package initializer, and one
adds a package beside the others to ask what its initializer imports. Every other receipts test
reads those same files whenever it builds, verifies or opens a bundle, because that is when the pin
is computed. A test that computed it while one of these had a file rewritten would see a pin that
moved and refuse a bundle that is sound.

So they are apart from the attacks module they came from, which runs with parallel workers, and in
a selection that runs in one process, where nothing else is reading the source while they edit it.
"""

from __future__ import annotations

from shogym.envs.receipts.generators.ledger import GENERATOR


def test_every_pinned_module_moves_the_pin_when_it_drifts() -> None:
    import importlib.util
    from pathlib import Path as _Path

    from shogym.envs.receipts import bank as bank_mod

    base = bank_mod.current_code_digest(GENERATOR)
    for name in bank_mod.pinned_modules(GENERATOR):
        spec = importlib.util.find_spec(name)
        assert spec and spec.origin
        path = _Path(spec.origin)
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# drift\n")
            assert bank_mod.current_code_digest(GENERATOR) != base, name
        finally:
            path.write_bytes(original)
    assert bank_mod.current_code_digest(GENERATOR) == base


def test_a_package_initializer_change_moves_the_pin() -> None:
    import importlib.util
    from pathlib import Path as _Path

    from shogym.envs.receipts import bank as bank_mod

    base = bank_mod.current_code_digest(GENERATOR)
    spec = importlib.util.find_spec("shogym.envs.receipts.generators")
    assert spec and spec.origin
    path = _Path(spec.origin)
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# drift\n")
        assert bank_mod.current_code_digest(GENERATOR) != base
    finally:
        path.write_bytes(original)
    assert bank_mod.current_code_digest(GENERATOR) == base


def test_a_package_initializer_resolves_its_relative_imports_as_a_package(tmp_path):
    """Counting dots from the module name is one level too high inside every
    `__init__.py`, so a child a package initializer imports relatively went unseen."""
    import shutil as _shutil
    from pathlib import Path as _Path

    from shogym.envs.receipts import bank as bank_mod

    probe = _Path(str(_Path(bank_mod.__file__).parent)) / "_pin_probe"
    probe.mkdir()
    try:
        (probe / "__init__.py").write_text(
            "from . import decider\n\n__all__ = [\"decider\"]\n", encoding="utf-8"
        )
        (probe / "decider.py").write_text("VALUE = 1\n", encoding="utf-8")
        seen = bank_mod._imported("shogym.envs.receipts._pin_probe")
        assert "shogym.envs.receipts._pin_probe.decider" in seen
        assert bank_mod._ancestors("shogym.envs.receipts._pin_probe.decider") == {
            "shogym.envs.receipts", "shogym.envs.receipts._pin_probe"
        }
    finally:
        _shutil.rmtree(probe)
