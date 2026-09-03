"""Test-wide fixtures.

Both of the ones here are about a directory rather than about any test. An episode opened without
a trace path resolves its finalization store to ``~/.cache/shogym/sessions``, which is shared by
every session ever run on the machine and never pruned, so a suite run writes its records into the
developer's own store and every later run reads them back. A suite that grows it is a suite that
slowly poisons the machine it runs on. Wordle's sealed plays are the same kind of directory and
get the same treatment.
"""

from __future__ import annotations

import pytest

from shogym.envs.wordle import protocol_v2 as wordle_protocol_v2
from shogym.serve import lifecycle


@pytest.fixture(autouse=True)
def _sessions_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Point the no-trace finalization store at a temporary directory for every test.

    Per test, so one test's records are never another's to recover, and so the once-per-process
    recovery cache is asked about a directory this test owns. Redirected at the module function
    rather than by moving ``HOME``, because that is the seam the store itself resolves through
    and moving ``HOME`` would take every other cache with it.

    Through ``monkeypatch``, so a test whose subject *is* the real root can undo it: the
    durability test in ``test_terminal_lifecycle`` calls ``monkeypatch.undo()`` and then moves
    ``HOME`` itself, and that only works if this patch was made on the same object."""
    root = tmp_path_factory.mktemp("sessions")
    monkeypatch.setattr(lifecycle, "_sessions_cache_root", lambda: root)
    # And in the environment, because a patched function reaches only this interpreter. Several
    # tests here open an episode in a *fresh* one (a killed-stream reconciliation, the fork
    # refusals), and those children inherit nothing.
    monkeypatch.setenv(lifecycle.SESSIONS_ROOT_ENV_VAR, str(root))


@pytest.fixture(autouse=True)
def _wordle_seals(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Point wordle's sealed plays at a temporary directory for every test.

    A sealed play is the target and the words played against it, kept so a Worker that replaced
    the one which sealed can still grade. The suite seals plenty of them and none is worth
    keeping, so they go where the run that made them goes. Through the environment as well as the
    module, because ``shogym serve`` is spawned by some of these tests and a patched function
    reaches only this interpreter.
    """
    root = tmp_path_factory.mktemp("wordle-seals")
    monkeypatch.setattr(wordle_protocol_v2, "seal_store_root", lambda: root)
    monkeypatch.setenv(wordle_protocol_v2.SEALS_ROOT_ENV_VAR, str(root))
