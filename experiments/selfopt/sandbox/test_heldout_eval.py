"""Offline unit tests for the two tricky pieces of the standalone held-out evaluator:

  (a) ``build_filtered_home`` — mounting a checkpoint's ``~/.claude`` must carry ONLY the durable
      self-surface (memory / skills / CLAUDE.md / settings) and STRIP the raw session transcript
      (the training CONTEXT), sessions/backups/telemetry, and machine-state files.
  (b) ``resolve_checkpoints`` — each checkpoint's (workdir self, memory home) pair must come from
      the SAME task boundary; ``end`` is the LIVE self + home; the empty-seed home resolves to none.
  (c) ``_heldout_pass`` completeness — a pass must report a mean only when EVERY requested unit
      holds its one sealed score for its own task, and must stay resumable when it does not.
  (d) ``audit_rows`` — an arm that ran before its rows carried the context verdict must be able to
      acquire it from its own persisted streams, without replaying (or paying for) a single unit.

No Docker / agent / OAuth: these exercise pure filesystem resolution + filtering only (the pass
tests stub the docker plumbing with a fake agent that writes broker-style provenance rows).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pytest

from experiments.selfopt.sandbox.heldout_eval import CHECKPOINTS, audit_rows, detect_server_name
from experiments.selfopt.sandbox.study import (
    HeldoutIncomplete,
    _dns_safe,
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


def _stub_pass(monkeypatch: pytest.MonkeyPatch, agent: Callable[[Path, int, int], int],
               indices: Sequence[int] = _INDICES) -> List[int]:
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
        return agent(prov, unit, indices[unit])

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


def _run(stream_dir: Path, *, indices: Sequence[int] = _INDICES, concurrency: int = 2) -> dict:
    return _heldout_pass(run_id="r", arm="treatment", cp="end", indices=list(indices), src=None,
                         home_src=None, stream_dir=stream_dir, disallowed=None, system="s",
                         oauth="tok", wandb_key=None, project="p", concurrency=concurrency)


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


# --- warm start: one cold prefix per pass, not one per worker ------------------------------
#
# Every unit of an arm boots the SAME cached conversation prefix. The first unit to send it pays to
# WRITE it and everyone after only pays to READ it, so launching ``concurrency`` units at once buys
# ``concurrency`` full-price writes where one would have done (measured at ~$8.92 of excess per
# extra cold unit on a real context arm). The pass therefore runs the first unit that will actually
# run ALONE, and only then goes wide.

_WIDE = [11, 22, 33, 44, 55]  # a 5-unit pass, so a resumed one still has several units left to run
_OVERLAP_WAIT = 0.25  # how long the warming unit stays in flight, waiting to catch an overlap


def _ordering_agent(warm: int) -> tuple[Callable[[Path, int, int], int], Dict[str, List[int]]]:
    """A fake agent that records ORDER, not elapsed time. Every unit except ``warm`` announces
    itself on an event as it starts, and ``warm`` waits on that event while it is in flight — so
    another unit running alongside it is recorded as an OBSERVED overlap instead of being inferred
    from a clock. ``log["order"]`` is the order the units finished in."""
    announced = threading.Event()
    log: Dict[str, List[int]] = {"order": [], "overlap": []}

    def agent(prov: Path, unit: int, task_idx: int) -> int:
        if unit == warm:
            if announced.wait(_OVERLAP_WAIT):
                log["overlap"].append(warm)
        else:
            announced.set()
        log["order"].append(unit)
        _seal(prov, task_idx, 1.0)
        return 0

    return agent, log


def test_pass_runs_the_first_unit_alone_to_warm_the_shared_prefix(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """With several units and room to run them in parallel, the first unit must FINISH before any
    other starts: it is the one that writes the shared prefix into the cache, and a unit racing it
    writes the whole prefix a second time for nothing."""
    agent, log = _ordering_agent(warm=0)
    ran = _stub_pass(monkeypatch, agent, indices=_WIDE)
    agg = _run(tmp_path / "end", indices=_WIDE, concurrency=3)

    assert log["overlap"] == [], "another unit ran while the warming unit was still in flight"
    assert log["order"][0] == 0, log["order"]
    assert sorted(ran) == [0, 1, 2, 3, 4]  # and the split costs nobody their turn
    assert agg == {"n": 5, "mean_reward": pytest.approx(1.0), "success_rate": pytest.approx(1.0)}
    # Observable: a reader of the log can see why the arm started serially.
    assert "unit 000 runs alone" in capsys.readouterr().out


def test_warm_start_picks_the_first_unit_that_actually_runs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """A re-run skips the units that already hold a sealed score, so warming on unit 000 would warm
    nothing at all: the warming unit is the first unit that will really launch an agent."""
    stream_dir = tmp_path / "end"
    for unit in (0, 1):
        _seal(stream_dir / f"prov-{unit:03d}", _WIDE[unit], 1.0)

    agent, log = _ordering_agent(warm=2)
    ran = _stub_pass(monkeypatch, agent, indices=_WIDE)
    agg = _run(stream_dir, indices=_WIDE, concurrency=3)

    assert sorted(ran) == [2, 3, 4], "already-scored units must be skipped, not replayed"
    assert log["overlap"] == [], "another unit ran while the warming unit was still in flight"
    assert log["order"][0] == 2, log["order"]
    assert agg["n"] == 5
    assert "unit 002 runs alone" in capsys.readouterr().out


def test_a_dead_warming_unit_does_not_abort_the_rest_of_the_arm(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One unit dying is routine — the harness records it and the arm reports INCOMPLETE. Going
    first must not promote that into an abort: the followers run anyway (they simply pay for a cold
    prefix), and the failure is reported exactly as it would be from any other position."""
    def agent(prov: Path, unit: int, idx: int) -> int:
        if unit == 0:
            return 1  # no sealed row: the warming unit died before it played
        _seal(prov, idx, 1.0)
        return 0

    ran = _stub_pass(monkeypatch, agent, indices=_WIDE)
    with pytest.raises(HeldoutIncomplete) as ei:
        _run(tmp_path / "end", indices=_WIDE, concurrency=3)

    assert sorted(ran) == [0, 1, 2, 3, 4], "the remaining units must still run"
    assert (ei.value.requested, ei.value.scored) == (5, 4)
    assert [f["unit"] for f in ei.value.failures] == [0]


@pytest.mark.parametrize("concurrency,scored", [(1, ()), (3, (0, 1, 2, 3))])
def test_warm_start_is_skipped_when_it_would_buy_nothing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        concurrency: int, scored: tuple) -> None:
    """A serial pass has no simultaneous miss to prevent, and a pass with ONE unit left to run would
    spend its only unit warming a prefix nobody else reads. Both must run in a single batch, exactly
    as they did before there was a warm start."""
    stream_dir = tmp_path / "end"
    for unit in scored:
        _seal(stream_dir / f"prov-{unit:03d}", _WIDE[unit], 1.0)

    ran = _stub_pass(monkeypatch, lambda prov, unit, idx: (_seal(prov, idx, 1.0), 0)[1],
                     indices=_WIDE)
    agg = _run(stream_dir, indices=_WIDE, concurrency=concurrency)

    assert sorted(ran) == [u for u in range(len(_WIDE)) if u not in scored]
    assert agg["n"] == 5
    assert "runs alone" not in capsys.readouterr().out


# --- one write per shared prefix ------------------------------------------------------------
#
# The warm start buys nothing unless every unit of an arm really does send the SAME prefix. A
# request is rendered ``tools`` -> ``system`` -> ``messages``, so the tool array is the FRONT of
# that prefix: a unit whose tool array differs by one entry shares no cached prefix at all with
# the rest, however byte-identical the ~900k tokens behind it are, and pays the full write again.
#
# So there are two ways to break "exactly one unit pays": launch units together before the prefix
# is warm, and let units disagree about their tool array. The tests above pin the ORDER units run
# in, which catches only the first. These price a pass against a fake prompt cache and assert the
# billing property itself, so both show up. They run the REAL ``run_agent`` with only ``docker
# run`` faked out, because the thing under test is the container the agent is launched in.

_PREFIX_TOKENS = 890_767  # the shared conversation prefix of a real context arm, in tokens

#: Units whose CLI wins the race between connecting to the task server and assembling its tool
#: array, and so loads that server's tools up front instead of deferring them behind ToolSearch.
#: Fixed here so the fake is deterministic; in the container it is a coin flip, which is the point.
_EAGER_UNITS = frozenset({2})


class _PromptCache:
    """The cache-economics half of the API: enough to price a pass, and nothing else.

    An entry is keyed by a request's prefix SIGNATURE and becomes readable only once the request
    that wrote it has come back — so units sending the same prefix at the same time all miss and
    all pay to write it, which is the burst the warm start exists to collapse."""

    def __init__(self) -> None:
        self._entries: set[str] = set()
        self._lock = threading.Lock()

    def send(self, signature: str) -> tuple[int, int]:
        """``(cache_creation, cache_read)`` for one request carrying ``signature``."""
        with self._lock:
            hit = signature in self._entries
        return (0, _PREFIX_TOKENS) if hit else (_PREFIX_TOKENS, 0)

    def store(self, signature: str) -> None:
        with self._lock:
            self._entries.add(signature)


def _prefix_signature(env: Dict[str, str], unit: int) -> str:
    """The prefix a unit's CLI would send, as the cache keys it.

    Everything after the tool array is identical across units of an arm by construction (same
    system prompt, same mounted transcript, same kickoff), so the tool array is the whole of the
    signature. Pinned, every unit sends the same one. Unpinned, the CLI decides per process."""
    pinned = env.get("ENABLE_TOOL_SEARCH")
    if pinned is not None:
        return f"tools:search={pinned}"
    return "tools:eager" if unit in _EAGER_UNITS else "tools:deferred"


def _stub_docker(monkeypatch: pytest.MonkeyPatch, cache: _PromptCache,
                 indices: Sequence[int] = _WIDE) -> Dict[int, str]:
    """Fake out ``docker run`` ONLY, so the real ``run_agent`` builds the real container.

    The fake reads the container environment back off the argv ``run_agent`` produced, works out
    which prefix that CLI would send, prices it against ``cache``, and tees a stream-json ``result``
    record carrying the usage — so a test reads per-unit ``cache_creation`` out of the persisted
    stream exactly the way the spend of a real arm is read. Returns the per-unit signature map."""
    import subprocess
    import time
    import types

    import experiments.selfopt.sandbox.study as study

    signatures: Dict[int, str] = {}

    def fake_docker_run(argv: List[str], **kw: object) -> subprocess.CompletedProcess:
        assert argv[:2] == ["docker", "run"], argv[:2]
        env = dict(pair.split("=", 1) for flag, pair in zip(argv, argv[1:]) if flag == "-e")
        out = kw["stdout"]
        stream_path = Path(out.name)                       # type: ignore[union-attr]
        unit = int(stream_path.stem.split("-")[-1])

        signature = _prefix_signature(env, unit)
        signatures[unit] = signature
        creation, read = cache.send(signature)
        time.sleep(0.05)        # stay in flight, so units launched together really do overlap
        cache.store(signature)  # readable only now: a cache entry lands when its request returns

        json.dump({"type": "result", "subtype": "success", "usage": {
            "cache_creation_input_tokens": creation, "cache_read_input_tokens": read}}, out)
        out.write("\n")
        _seal(stream_path.parent / f"prov-{unit:03d}", indices[unit], 1.0)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(study, "start_broker", lambda *a, **kw: None)
    monkeypatch.setattr(study, "_rm", lambda *names: None)
    monkeypatch.setattr(study, "build_filtered_home", lambda src, dst, **kw: dst)
    monkeypatch.setattr(study, "subprocess", types.SimpleNamespace(run=fake_docker_run))
    return signatures


def _prefix_writers(stream_dir: Path) -> List[int]:
    """The units that paid to WRITE the shared prefix, read out of their own persisted streams —
    the same ``usage.cache_creation_input_tokens`` a real arm's per-unit spend is read from."""
    writers: List[int] = []
    for path in sorted(stream_dir.glob("stream-*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("type") != "result":
                continue
            if record["usage"]["cache_creation_input_tokens"] > _PREFIX_TOKENS // 2:
                writers.append(int(path.stem.split("-")[-1]))
    return writers


def test_only_one_unit_pays_to_write_the_shared_prefix(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The property the warm start is FOR, asserted on the bill rather than on the running order:
    N units share one prefix, so exactly ONE may pay ``cache_creation`` for it and every other unit
    must read it.

    Running a unit alone first is necessary and not sufficient. It warms the prefix THAT unit sent,
    and a later unit whose tool array differs shares nothing with it, so it writes the whole ~900k
    tokens again at full price. Both failures land here as a second full-prefix writer."""
    stream_dir = tmp_path / "end"
    cache = _PromptCache()
    _stub_docker(monkeypatch, cache)
    agg = _run(stream_dir, indices=_WIDE, concurrency=3)

    writers = _prefix_writers(stream_dir)
    assert writers == [0], (
        f"units {writers} each paid to write the ~{_PREFIX_TOKENS // 1000}k-token shared prefix; "
        "only the warm-start unit may, the rest must read what it wrote")
    assert agg["n"] == len(_WIDE), "and pricing the pass must not cost anybody their turn"


def test_every_unit_of_an_arm_sends_the_same_prefix(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same property one level earlier, where it is fixable: the units must not merely be
    ordered so one prefix can be warmed, they must all send that one prefix. Left to the CLI, the
    tool array is decided per process and races the task server's connection, so an arm can send
    two prefixes and warm only one of them."""
    cache = _PromptCache()
    signatures = _stub_docker(monkeypatch, cache)
    _run(tmp_path / "end", indices=_WIDE, concurrency=3)

    assert len(signatures) == len(_WIDE)
    assert len(set(signatures.values())) == 1, (
        f"an arm sent {len(set(signatures.values()))} different prefixes ({signatures}); the tool "
        "array must be pinned, or the warm start warms a prefix some units never send")


# --- the two silent killers ---------------------------------------------------------------
#
# Both of these have mitigations in the tree and neither had a test. Both fail the same way: the
# agent boots, exits 0, and plays nothing — a healthy-looking run that measured nothing.

def test_detect_server_name_reads_the_name_the_context_was_trained_against() -> None:
    """A resumed probe MUST be handed the server name its recorded context used. An agent whose
    history is full of ``mcp__curriculum__*`` reads a differently-named server as its tools having
    disconnected and stops at turn 1 — with a clean exit and no score."""
    trace = Path(__file__).resolve().parents[1] / "runs" / "sandbox-1785221622" / \
        "treatment" / "stream.jsonl"
    if trace.exists():
        assert detect_server_name(trace) == "curriculum"


def test_detect_server_name_on_synthetic_and_missing_traces(tmp_path: Path) -> None:
    trace = tmp_path / "stream.jsonl"
    trace.write_text(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "mcp__curriculum__get_task"}]}}) + "\n")
    assert detect_server_name(trace) == "curriculum"

    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "mcp__tasks__get_task"}]}}) + "\n")
    assert detect_server_name(other) == "tasks"

    # A trace with no MCP tool call at all, and a trace that does not exist: fall back, never crash.
    (tmp_path / "empty.jsonl").write_text("")
    assert detect_server_name(tmp_path / "empty.jsonl") == "tasks"
    assert detect_server_name(tmp_path / "gone.jsonl", default="tasks") == "tasks"


# --- re-auditing an arm that already ran ---------------------------------------------------
#
# The pass stamps its own rows, which leaves the arms that ran BEFORE it did: their verdict exists
# only in ``context_audit.json`` and their rows read clean. Re-auditing must reach them without a
# credential, an image, or a docker daemon — and without replaying a single unit.

def _unit_stream(stream_dir: Path, unit: int, *, compacted: bool) -> None:
    """One finished unit's stream: fetch the task, optionally auto-compact, seal."""
    def turn(tokens: int, name: Optional[str] = None) -> dict:
        block = ({"type": "tool_use", "id": "t", "name": name} if name
                 else {"type": "text", "text": "working"})
        return {"type": "assistant", "message": {"content": [block],
                                                 "usage": {"input_tokens": tokens}}}

    events = [turn(958_000, "mcp__curriculum__get_task")]
    if compacted:
        events.append({"type": "system", "subtype": "compact_boundary"})
    events.append(turn(12_000 if compacted else 970_000, "mcp__curriculum__done"))
    stream_dir.mkdir(parents=True, exist_ok=True)
    (stream_dir / f"stream-{unit:03d}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events))


def _finished_arm(runs: Path, cp: str, compacted: set[int], units: int = 3) -> Path:
    """A run tree as a finished arm leaves it: per-unit streams and sealed rows, no flags."""
    stream_dir = runs / "sandbox-x" / "treatment" / "heldout_eval" / cp
    for unit in range(units):
        _unit_stream(stream_dir, unit, compacted=unit in compacted)
        _seal(stream_dir / f"prov-{unit:03d}", 100 + unit, 1.0)
    return stream_dir


def _row(prov: Path) -> dict:
    lines = [ln for ln in (prov / "results.jsonl").read_text().splitlines() if ln.strip()]
    assert len(lines) == 1, lines
    return json.loads(lines[0])


def test_audit_rows_stamps_an_already_finished_raw_arm(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw arm whose unit auto-compacted before the seal: the verdict lands in that unit's row,
    the clean units' rows are left exactly as the broker wrote them, and nothing is re-run."""
    from experiments.selfopt import config
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    stream_dir = _finished_arm(tmp_path / "runs", "pre-compact-2", compacted={1})

    verdict = audit_rows("sandbox-x", "treatment", ["pre-compact-2"])

    cp = verdict["checkpoints"]["pre-compact-2"]
    assert verdict["flagged"] == 1 and cp["units"] == 3 and cp["raw_segment"] is True
    assert [f["unit"] for f in cp["flagged"]] == [1]
    flagged = _row(stream_dir / "prov-001")
    assert flagged["context_ok"] is False
    assert "auto-compacted before the seal" in flagged["context_reason"]
    for unit in (0, 2):
        assert "context_ok" not in _row(stream_dir / f"prov-{unit:03d}")
    # The audit it re-derived is on disk beside the row, and the streams are untouched.
    audit = json.loads((stream_dir / "prov-001" / "context_audit.json").read_text())
    assert audit["stayed_raw"] is False and audit["arm"] == "pre-compact-2"


def test_audit_rows_leaves_a_summarised_or_fresh_arm_alone(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """A compaction is only a defect for an arm that was supposed to stay RAW. A post-compaction arm
    was already summarised, and a fresh-context arm carries no mounted prefix at all — neither may
    be stamped, and the fresh one is reported as unaudited rather than as clean."""
    from experiments.selfopt import config
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    summarised = _finished_arm(tmp_path / "runs", "post-compact-2", compacted={0, 1, 2})
    fresh = _finished_arm(tmp_path / "runs", "end", compacted={1})

    verdict = audit_rows("sandbox-x", "treatment", ["post-compact-2", "end"])

    assert verdict["flagged"] == 0
    assert verdict["checkpoints"]["post-compact-2"]["audited"] is True
    assert verdict["checkpoints"]["post-compact-2"]["raw_segment"] is False
    assert verdict["checkpoints"]["end"]["audited"] is False
    for unit in range(3):
        assert "context_ok" not in _row(summarised / f"prov-{unit:03d}")
        assert "context_ok" not in _row(fresh / f"prov-{unit:03d}")
    # A non-cut arm has nothing to audit, so it writes nothing at all.
    assert not (fresh / "prov-001" / "context_audit.json").exists()


def test_audit_rows_is_idempotent_and_survives_a_unit_with_no_row(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run it twice and the flagged row is byte-identical; a unit that died before sealing has no
    row to stamp and must not turn the audit into a crash."""
    from experiments.selfopt import config
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    stream_dir = _finished_arm(tmp_path / "runs", "pre-compact-1", compacted={1})
    _unit_stream(stream_dir, 3, compacted=True)          # a 4th unit that never sealed a row
    (stream_dir / "prov-003").mkdir(parents=True, exist_ok=True)

    first = audit_rows("sandbox-x", "treatment", ["pre-compact-1"])
    stamped = (stream_dir / "prov-001" / "results.jsonl").read_bytes()
    second = audit_rows("sandbox-x", "treatment", ["pre-compact-1"])

    assert [f["unit"] for f in first["checkpoints"]["pre-compact-1"]["flagged"]] == [1, 3]
    assert second == first
    assert (stream_dir / "prov-001" / "results.jsonl").read_bytes() == stamped
    assert not (stream_dir / "prov-003" / "results.jsonl").exists()


def test_dns_safe_keeps_every_container_name_resolvable() -> None:
    """Container names become DNS hostnames, and a label over 63 chars does not error — it fails to
    RESOLVE, so the agent's MCP connect fails, it is handed zero tools, and the unit scores nothing.
    Every arm's real container name is checked against the evaluator's ``-hoeval`` scope."""
    assert _dns_safe("short-name") == "short-name"
    exactly_63 = "a" * 63
    assert _dns_safe(exactly_63) == exactly_63, "63 is legal — do not fold it"
    assert len(_dns_safe("a" * 64)) == 63
    assert _dns_safe("a" * 200) == _dns_safe("a" * 200), "folding must be deterministic"
    assert _dns_safe("a" * 200) != _dns_safe("b" * 200), "and must not collide across arms"

    scope = "sandbox-1785221622-hoeval"
    for cp in CHECKPOINTS:
        name = _dns_safe(f"selfopt-broker-{scope}-treatment-ho-{cp}-000")
        assert len(name) <= 63, (cp, name)
