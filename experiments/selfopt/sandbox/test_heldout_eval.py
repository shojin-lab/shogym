"""Offline unit tests for the two tricky pieces of the standalone held-out evaluator:

  (a) ``build_filtered_home`` — mounting a checkpoint's ``~/.claude`` must carry ONLY the durable
      self-surface (memory / skills / CLAUDE.md / settings) and STRIP the raw session transcript
      (the training CONTEXT), sessions/backups/telemetry, and machine-state files.
  (b) ``resolve_checkpoints`` — each checkpoint's (workdir self, memory home) pair must come from
      the SAME task boundary; ``end`` is the LIVE self + home; the empty-seed home resolves to none.
  (c) ``_heldout_pass`` completeness — a pass must report a mean only when EVERY requested unit
      holds its one sealed score for its own task, and must stay resumable when it does not.

No Docker / agent / OAuth: these exercise pure filesystem resolution + filtering only (the pass
tests stub the docker plumbing with a fake agent that writes broker-style provenance rows).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

from experiments.selfopt.sandbox.study import (
    HeldoutIncomplete,
    _heldout_pass,
    build_filtered_home,
    resolve_checkpoints,
)
from experiments.selfopt.snapshot import content_hash, home_skip


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _fake_home(root: Path) -> Path:
    """A realistic ~/.claude: durable self-surface + transient noise that must be stripped."""
    _write(root / "projects" / "-work" / "memory" / "MEMORY.md", "# index")
    _write(root / "projects" / "-work" / "memory" / "note.md", "a durable lesson")
    _write(root / "projects" / "-work" / "0680d5b2.jsonl", "raw transcript line\n" * 100)
    _write(root / "skills" / "foo" / "SKILL.md", "a skill")
    _write(root / "CLAUDE.md", "# self")
    _write(root / "settings.json", "{}")
    _write(root / "sessions" / "s.json", "{}")
    _write(root / "backups" / ".claude.json.backup.1", "{}")
    _write(root / "telemetry" / "events.jsonl", "evt\n")
    _write(root / "policy-limits.json", "{}")
    _write(root / "remote-settings.json", "{}")
    _write(root / ".last-cleanup", "0")
    return root


def _rel_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_build_filtered_home_keeps_surface_strips_transcript(tmp_path: Path) -> None:
    src = _fake_home(tmp_path / "src")
    dst = build_filtered_home(src, tmp_path / "dst")
    got = _rel_files(dst)

    # KEEP: the memory knowledge-base, skills, CLAUDE.md, and config settings.
    assert "projects/-work/memory/MEMORY.md" in got
    assert "projects/-work/memory/note.md" in got
    assert "skills/foo/SKILL.md" in got
    assert "CLAUDE.md" in got
    assert "settings.json" in got

    # STRIP: the raw session transcript (training context) and all transient/machine state.
    assert "projects/-work/0680d5b2.jsonl" not in got
    assert not any(f.endswith(".jsonl") for f in got), got
    for stripped in ("sessions", "backups", "telemetry", "policy-limits.json",
                     "remote-settings.json", ".last-cleanup"):
        assert not any(stripped in f for f in got), (stripped, got)


def test_build_filtered_home_empty_seed(tmp_path: Path) -> None:
    dst = build_filtered_home(None, tmp_path / "seed")  # the empty-seed checkpoint
    assert dst.exists()
    assert _rel_files(dst) == set()


def _fake_prov(root: Path) -> Path:
    """A provenance dir with three train rows and matching snapshot dirs.

    row0 (start) references a MISSING workdir-self snapshot (so start.self must fall back to the
    live self_dir) and the empty-seed home; row1 (mid) has both snapshots present; row2 (end) is
    the last row (superseded by the live self/home at the 'end' checkpoint)."""
    prov = root / "prov"
    snaps = prov / "snapshots"
    # A present workdir-self snapshot (used by mid) and a present mid home snapshot.
    _write(snaps / "selfAAAA" / "CLAUDE.md", "# self")
    _write(snaps / "homeMID01" / "projects" / "-work" / "memory" / "MEMORY.md", "# mid")
    # The empty-seed home hash exists as a dir but is empty -> must normalize to None.
    (snaps / "emptyHHHH").mkdir(parents=True, exist_ok=True)
    rows = [
        {"seq": 1, "split": "train", "self_hash_before": "missingSELF",
         "home_hash_before": "emptyHHHH"},
        {"seq": 2, "split": "train", "self_hash_before": "selfAAAA",
         "home_hash_before": "homeMID01"},
        {"seq": 3, "split": "train", "self_hash_before": "selfAAAA",
         "home_hash_before": "homeEND99"},
    ]
    (prov / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return prov


def test_resolve_checkpoints_pairs_self_and_home(tmp_path: Path) -> None:
    prov = _fake_prov(tmp_path)
    self_dir = tmp_path / "self"
    _write(self_dir / "CLAUDE.md", "# self")
    home_dir = _fake_home(tmp_path / "self_home")  # live home (with a transcript)

    cps = resolve_checkpoints(prov, self_dir, home_dir)

    # start: row0 — missing self snapshot falls back to the live self_dir; empty-seed home -> None.
    assert cps["start"]["self"] == self_dir
    assert cps["start"]["home"] is None
    assert cps["start"]["self_hash"] == "missingSELF"
    assert cps["start"]["home_hash"] == "emptyHHHH"

    # mid: row1 (index len//2 == 1) — both snapshots present, paired at THAT boundary.
    assert cps["mid"]["self"] == prov / "snapshots" / "selfAAAA"
    assert cps["mid"]["home"] == prov / "snapshots" / "homeMID01"
    assert cps["mid"]["self_hash"] == "selfAAAA"
    assert cps["mid"]["home_hash"] == "homeMID01"

    # end: the LIVE self + live home (never the archived row2), home hash via the durable-surface
    # filter so it matches what the broker archives.
    assert cps["end"]["self"] == self_dir
    assert cps["end"]["home"] == home_dir
    assert cps["end"]["home_hash"] == content_hash(home_dir, skip=home_skip)


def test_resolve_checkpoints_no_rows_is_seed(tmp_path: Path) -> None:
    prov = tmp_path / "prov"
    prov.mkdir()
    self_dir = tmp_path / "self"
    _write(self_dir / "CLAUDE.md", "# self")
    home_dir = tmp_path / "self_home"  # does not exist -> no live home

    cps = resolve_checkpoints(prov, self_dir, home_dir)
    for cp in ("start", "mid"):
        assert cps[cp]["self"] == self_dir
        assert cps[cp]["home"] is None
    assert cps["end"]["self"] == self_dir
    assert cps["end"]["home"] is None


# --- held-out pass completeness ---------------------------------------------------------

_INDICES = [11, 22, 33]  # a 3-task held-out pass; unit i plays _INDICES[i]


def _stub_pass(monkeypatch: pytest.MonkeyPatch, agent: Callable[[Path, int, int], int]) -> List[int]:
    """Replace the docker plumbing with a fake agent so a pass runs offline.

    ``agent(prov, unit, task_idx) -> rc`` stands in for the (broker, container) pair: whatever it
    writes to ``prov/results.jsonl`` is what the broker is deemed to have sealed. Returns the list
    of unit indices the fake agent was actually invoked for (empty on a fully-resumed pass)."""
    import experiments.selfopt.sandbox.study as study

    ran: List[int] = []
    started: Dict[str, Path] = {}

    def fake_start_broker(name: str, *, prov: Path, indices: Optional[List[int]] = None,
                          **kw: object) -> None:
        prov.mkdir(parents=True, exist_ok=True)
        assert indices and len(indices) == 1, "each unit's broker dispenses exactly one task"
        started[name] = prov

    def fake_run_agent(*, work: Path, stream_path: Path, broker_name: str, **kw: object) -> int:
        unit = int(stream_path.stem.split("-")[-1])
        prov = started[broker_name]
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        ran.append(unit)
        return agent(prov, unit, _INDICES[unit])

    monkeypatch.setattr(study, "start_broker", fake_start_broker)
    monkeypatch.setattr(study, "run_agent", fake_run_agent)
    monkeypatch.setattr(study, "_rm", lambda *names: None)
    monkeypatch.setattr(study, "build_filtered_home", lambda src, dst, **kw: dst)
    return ran


def _seal(prov: Path, idx: int, reward: float, success: bool = True) -> None:
    """Append one broker-style sealed held-out row to a unit's provenance."""
    prov.mkdir(parents=True, exist_ok=True)
    with (prov / "results.jsonl").open("a") as fh:
        fh.write(json.dumps({"seq": 1, "split": "heldout", "task_idx": idx,
                             "reward": reward, "success": success}) + "\n")


def _run(stream_dir: Path) -> dict:
    return _heldout_pass(run_id="r", arm="treatment", cp="end", indices=_INDICES, src=None,
                         home_src=None, stream_dir=stream_dir, disallowed=None, system="s",
                         oauth="tok", wandb_key=None, project="p", concurrency=2)


_REWARDS = {0: 1.0, 1: 0.5, 2: 0.0}


def test_pass_complete_arm_aggregates_over_every_unit(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """The unchanged happy path: all units seal their own task, mean is over all of them."""
    ran = _stub_pass(monkeypatch, lambda prov, unit, idx: (_seal(prov, idx, _REWARDS[unit]), 0)[1])
    agg = _run(tmp_path / "end")
    assert sorted(ran) == [0, 1, 2]
    assert agg == {"n": 3, "mean_reward": pytest.approx(0.5), "success_rate": pytest.approx(1.0)}


def test_pass_refuses_to_average_survivors_when_a_unit_dies(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A unit that exits non-zero with no sealed row must fail the pass — not vanish from the mean
    (2 survivors averaging 0.75 is exactly the confidently-wrong number this guards)."""
    def agent(prov: Path, unit: int, idx: int) -> int:
        if unit == 2:
            (prov.parent / f"stream-{unit:03d}.err.txt").write_text(
                "boot\nClaude configuration file not found\n")
            return 1
        _seal(prov, idx, _REWARDS[unit])
        return 0

    _stub_pass(monkeypatch, agent)
    with pytest.raises(HeldoutIncomplete) as ei:
        _run(tmp_path / "end")
    exc = ei.value
    assert (exc.requested, exc.scored) == (3, 2)
    assert [f["unit"] for f in exc.failures] == [2]
    assert exc.failures[0]["rc"] == 1 and "no sealed score row" in exc.failures[0]["reason"]
    assert "Claude configuration file not found" in exc.failures[0]["stderr"]
    assert "0.75" not in str(exc) and "INCOMPLETE" in str(exc)


def test_pass_catches_row_for_the_wrong_task(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """A sealed row is only this unit's score if it is for the task this unit was handed."""
    def agent(prov: Path, unit: int, idx: int) -> int:
        _seal(prov, 999 if unit == 1 else idx, _REWARDS[unit])
        return 0

    _stub_pass(monkeypatch, agent)
    with pytest.raises(HeldoutIncomplete) as ei:
        _run(tmp_path / "end")
    assert [f["unit"] for f in ei.value.failures] == [1]
    assert "task_idx 999" in ei.value.failures[0]["reason"]


def test_pass_resumes_only_the_failed_units(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """Failing loudly must leave the scored units on disk: re-running the SAME pass re-runs only the
    units that never scored, and then completes the arm."""
    stream_dir = tmp_path / "end"
    dead = {2}

    def agent(prov: Path, unit: int, idx: int) -> int:
        if unit in dead:
            return 1
        _seal(prov, idx, _REWARDS[unit])
        return 0

    first = _stub_pass(monkeypatch, agent)
    with pytest.raises(HeldoutIncomplete):
        _run(stream_dir)
    assert sorted(first) == [0, 1, 2]

    dead.clear()  # the cause (usage cap / bad config) is fixed; re-run the same command
    second = _stub_pass(monkeypatch, agent)
    agg = _run(stream_dir)
    assert second == [2], "already-scored units must be skipped, not replayed"
    assert agg == {"n": 3, "mean_reward": pytest.approx(0.5), "success_rate": pytest.approx(1.0)}
