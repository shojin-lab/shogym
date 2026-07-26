"""Offline tests for the transcript-prefix ("context cut") arms.

Every failure mode here is one that would otherwise produce a healthy-looking run measuring the
wrong thing:

  (a) a prefix written with a broken parent chain or an unanswered ``tool_use`` — the agent boots a
      different conversation, or the API rejects the turn;
  (b) a prefix that no longer matches the facts its arm pins (the transcript moved under it);
  (c) a pre/post pair mounting two DIFFERENT memory homes — a compaction effect confounded with
      memory evolution;
  (d) a raw (pre-compaction) unit that auto-compacted before it sealed — a unit that quietly
      changed arms and would otherwise be averaged into the other one's mean.

The real transcript is checked too when the run artifacts are present (they are gitignored), so the
cut points are re-derived rather than trusted. No Docker / agent / OAuth anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from experiments.selfopt.sandbox.cuts import (
    CUT_PAIRS,
    CUTS,
    Cut,
    InvalidCut,
    PairMismatch,
    assert_pair_homes,
    acceptance_check,
    assert_future_terms,
    audit_prefix,
    cut_source,
    find_transcript,
    home_sealed_for,
    materialize_cut,
    train_rows,
    spoken_text,
    paired_home_hash,
    read_records,
    stream_audit,
    validate,
)
from experiments.selfopt.sandbox.study import (
    HeldoutIncomplete,
    _heldout_pass,
    resolve_cut_checkpoint,
    write_cut_manifest,
)

SESSION = "0680d5b2-c1b5-40b9-8769-9491295c3b7c"
SLUG = "projects/-work"


# --- a small synthetic transcript --------------------------------------------------------

def _episode(n: int, parent: Optional[str], *, server: str = "curriculum") -> List[dict]:
    """One task's records: fetch, answer, seal, answer — the shape a real episode has, with the
    uuid/parentUuid chain and matched tool_use/tool_result ids the CLI and the API require."""
    out: List[dict] = []
    for k, (role, block) in enumerate((
            ("assistant", {"type": "tool_use", "id": f"t{n}a", "name": f"mcp__{server}__get_task"}),
            ("user", {"type": "tool_result", "tool_use_id": f"t{n}a", "content": "task"}),
            ("assistant", {"type": "tool_use", "id": f"t{n}b", "name": f"mcp__{server}__done"}),
            ("user", {"type": "tool_result", "tool_use_id": f"t{n}b", "content": "scored"}))):
        uid = f"u{n}-{k}"
        rec = {"type": role, "uuid": uid, "parentUuid": parent,
               "message": {"role": role, "content": [block]}}
        if role == "assistant":
            rec["message"]["usage"] = {"input_tokens": 2, "cache_read_input_tokens": 1000 * n,
                                       "cache_creation_input_tokens": 7}
        out.append(rec)
        parent = uid
    return out


def _transcript(tasks: int = 6, boundary_after: Optional[int] = None,
                server: str = "curriculum") -> List[dict]:
    """``tasks`` episodes, optionally with a ``compact_boundary`` + summary after episode
    ``boundary_after`` (which is what makes a later prefix a *post*-compaction one)."""
    recs: List[dict] = []
    parent: Optional[str] = None
    for n in range(1, tasks + 1):
        ep = _episode(n, parent, server=server)
        recs.extend(ep)
        parent = ep[-1]["uuid"]
        if boundary_after == n:
            recs.append({"type": "system", "subtype": "compact_boundary", "uuid": f"cb{n}",
                         "parentUuid": parent, "compactMetadata": {"trigger": "auto"}})
            recs.append({"type": "user", "uuid": f"sum{n}", "parentUuid": f"cb{n}",
                         "message": {"role": "user",
                                     "content": [{"type": "text", "text": "Summary: " + "s" * 400}]}})
            parent = f"sum{n}"
    return recs


def _home(root: Path, records: List[dict], *, memory: str = "# index") -> Path:
    """A ``~/.claude`` holding one session transcript plus a memory knowledge base."""
    (root / SLUG).mkdir(parents=True, exist_ok=True)
    (root / SLUG / f"{SESSION}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    (root / SLUG / "memory").mkdir(parents=True, exist_ok=True)
    (root / SLUG / "memory" / "MEMORY.md").write_text(memory)
    return root


def _cut(name: str, records: List[dict], last: int, sealed: int, *, raw: bool = True,
         pair: str = "other") -> Cut:
    return Cut(name, last, str(records[last - 1]["uuid"]), sealed, raw=raw, pair=pair)


# --- materialization ---------------------------------------------------------------------

def test_materialize_writes_exactly_the_prefix(tmp_path: Path) -> None:
    recs = _transcript(tasks=4)
    src = _home(tmp_path / "train", recs)
    source = cut_source(src, _cut("pre", recs, 8, sealed=2))

    dst = tmp_path / "unit-home"
    manifest = materialize_cut(source, dst)

    written = read_records(dst / SLUG / f"{SESSION}.jsonl")
    assert len(written) == 8
    assert written == recs[:8], "the prefix must be the ORIGINAL records, byte-for-byte in order"
    assert manifest["sealed"] == 2 and manifest["fetched"] == 2
    assert manifest["last_uuid"] == recs[7]["uuid"]
    assert manifest["transcript_records"] == len(recs)
    assert len(manifest["transcript_sha256"]) == 64
    assert manifest["written_to"] == f"{SLUG}/{SESSION}.jsonl"
    # The truncated copy lands in the UNIT's home; the training transcript is untouched.
    assert len(read_records(src / SLUG / f"{SESSION}.jsonl")) == len(recs)


def test_materialize_leaves_the_contemporaneous_memory_alongside(tmp_path: Path) -> None:
    """A cut replaces only the conversation: whatever memory was copied into the home stays."""
    recs = _transcript(tasks=3)
    src = _home(tmp_path / "train", recs)
    dst = tmp_path / "unit-home"
    (dst / SLUG / "memory").mkdir(parents=True)
    (dst / SLUG / "memory" / "lesson.md").write_text("a lesson from task 2")

    materialize_cut(cut_source(src, _cut("pre", recs, 4, sealed=1)), dst)
    assert (dst / SLUG / "memory" / "lesson.md").read_text() == "a lesson from task 2"
    assert len(read_records(dst / SLUG / f"{SESSION}.jsonl")) == 4


def test_rejects_dangling_tool_use_and_writes_nothing(tmp_path: Path) -> None:
    """Cutting one record earlier leaves the ``done`` call unanswered — the API rejects a trailing
    ``tool_use``, so the arm must never be materialized."""
    recs = _transcript(tasks=3)
    src = _home(tmp_path / "train", recs)
    bad = _cut("pre", recs, 7, sealed=1)  # record 7 is the `done` tool_use; its result is record 8
    dst = tmp_path / "unit-home"

    with pytest.raises(InvalidCut) as ei:
        materialize_cut(cut_source(src, bad), dst)
    assert "unanswered tool_use" in str(ei.value)
    assert not (dst / SLUG / f"{SESSION}.jsonl").exists(), "an invalid cut must write NOTHING"


def test_rejects_orphaned_parent(tmp_path: Path) -> None:
    recs = _transcript(tasks=3)
    recs[5]["parentUuid"] = "a-uuid-that-is-not-in-the-prefix"
    src = _home(tmp_path / "train", recs)
    with pytest.raises(InvalidCut) as ei:
        materialize_cut(cut_source(src, _cut("pre", recs, 8, sealed=2)), tmp_path / "h")
    assert "parentUuid is outside the prefix" in str(ei.value)


def test_rejects_a_transcript_that_drifted_under_the_pin(tmp_path: Path) -> None:
    """The arm pins the last uuid and the sealed-task count. If the transcript changes, the same
    record number is a different conversation — fail rather than measure it."""
    recs = _transcript(tasks=4)
    src = _home(tmp_path / "train", recs)
    with pytest.raises(InvalidCut, match="the arm pins"):
        validate(Cut("pre", 8, "not-the-right-uuid", 2, raw=True, pair="p"), recs)
    with pytest.raises(InvalidCut, match="tasks sealed at the cut"):
        materialize_cut(cut_source(src, _cut("pre", recs, 8, sealed=99)), tmp_path / "h")


def test_audit_reports_context_from_recorded_usage_and_from_a_summary(tmp_path: Path) -> None:
    """A raw prefix reports the context its last turn actually held; a just-compacted prefix has no
    recorded usage, so it is estimated — and the manifest says which."""
    recs = _transcript(tasks=4, boundary_after=2)
    raw = audit_prefix(recs, 8)
    assert raw["context_tokens_method"] == "recorded-usage"
    assert raw["context_tokens"] == 2 + 2000 + 7  # episode 2's last assistant turn
    assert raw["segment_start_record"] == 1

    post = audit_prefix(recs, 10)  # boundary + summary
    assert post["segment_start_record"] == 10 and post["segment_records"] == 1
    assert post["context_tokens_method"] == "segment-chars/4"
    assert 50 < post["context_tokens"] < raw["context_tokens"]
    assert post["sealed"] == 2, "the summary segment still describes 2 sealed tasks"


def test_find_transcript_refuses_an_ambiguous_home(tmp_path: Path) -> None:
    home = _home(tmp_path / "h", _transcript(tasks=1))
    (home / SLUG / "another-session.jsonl").write_text("{}\n")
    with pytest.raises(InvalidCut, match="exactly one session transcript"):
        find_transcript(home)


# --- contemporaneous memory pairing -------------------------------------------------------

def _prov(root: Path, homes: List[str]) -> Path:
    """A provenance dir whose k-th train row is the k-th completed task, with the home hash the
    broker archived after it."""
    prov = root / "prov"
    (prov / "snapshots").mkdir(parents=True, exist_ok=True)
    rows = [{"seq": i + 1, "split": "train", "task_idx": 100 + i,
             "self_hash_before": "selfAAAA", "self_hash_after": "selfAAAA",
             "home_hash_before": h, "home_hash_after": h} for i, h in enumerate(homes)]
    (prov / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for h in set(homes):
        d = prov / "snapshots" / h / "projects" / "-work" / "memory"
        d.mkdir(parents=True, exist_ok=True)
        (d / "MEMORY.md").write_text(f"# {h}")
    return prov


def test_paired_home_is_the_last_completed_task_not_the_final_state(tmp_path: Path) -> None:
    prov = _prov(tmp_path, ["h1", "h1", "h2", "h2", "h3"])
    assert paired_home_hash(prov, 2) == "h1"
    assert paired_home_hash(prov, 4) == "h2"
    assert paired_home_hash(prov, 5) == "h3", "never the end-of-training home for an early cut"
    with pytest.raises(InvalidCut):
        paired_home_hash(prov, 9)


def test_pair_with_one_memory_state_is_accepted(tmp_path: Path) -> None:
    prov = _prov(tmp_path, ["h1"] * 6)
    a = Cut("pre", 8, "u2-3", 2, raw=True, pair="post")
    b = Cut("post", 12, "u3-3", 3, raw=False, pair="pre")
    assert assert_pair_homes(prov, a, b) == "h1"


def test_pair_across_two_memory_states_is_void(tmp_path: Path) -> None:
    """The binding constraint: a pre/post contrast whose members mount different memory measures
    memory evolution as well as compaction, so it must not run."""
    prov = _prov(tmp_path, ["h1", "h1", "h2", "h2"])
    a = Cut("pre", 8, "u2-3", 2, raw=True, pair="post")
    b = Cut("post", 16, "u4-3", 4, raw=False, pair="pre")
    with pytest.raises(PairMismatch) as ei:
        assert_pair_homes(prov, a, b)
    assert "'h1'" in str(ei.value) and "'h2'" in str(ei.value)


def test_resolve_cut_checkpoint_refuses_an_unarchived_home(tmp_path: Path) -> None:
    import shutil
    prov = _prov(tmp_path, ["h1", "gone"])
    shutil.rmtree(prov / "snapshots" / "gone")  # the paired home was never archived
    with pytest.raises(InvalidCut, match="no archived snapshot"):
        resolve_cut_checkpoint(prov, tmp_path / "self", Cut("pre", 8, "u2-3", 2, raw=True,
                                                            pair="post"))


def test_manifest_records_the_paired_home_and_the_folded_container(tmp_path: Path) -> None:
    recs = _transcript(tasks=4)
    src = _home(tmp_path / "train", recs)
    prov = _prov(tmp_path, ["h1"] * 4)
    home_src = prov / "snapshots" / "h1"
    manifest = write_cut_manifest(cut_source(src, _cut("pre", recs, 8, sealed=2)),
                                  tmp_path / "arm", home_src=home_src,
                                  container="selfopt-broker-x-000")

    on_disk = json.loads((tmp_path / "arm" / "cut_manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["paired_home_hash"] is not None
    assert manifest["broker_container"] == "selfopt-broker-x-000"
    assert manifest["last_record"] == 8 and manifest["sealed"] == 2


# --- raw stays raw --------------------------------------------------------------------------

def _stream(tmp_path: Path, events: List[dict]) -> Path:
    p = tmp_path / "stream-000.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in events))
    return p


def _turn(tokens: int, name: Optional[str] = None) -> dict:
    content = [{"type": "tool_use", "id": "x", "name": name}] if name else [
        {"type": "text", "text": "thinking"}]
    return {"type": "assistant",
            "message": {"content": content,
                        "usage": {"input_tokens": 1, "cache_read_input_tokens": tokens,
                                  "cache_creation_input_tokens": 0}}}


_COMPACTING = {"type": "system", "subtype": "status", "status": "compacting"}
_BOUNDARY = {"type": "system", "subtype": "compact_boundary"}


def test_stream_audit_passes_a_unit_that_stayed_raw(tmp_path: Path) -> None:
    audit = stream_audit(_stream(tmp_path, [
        {"type": "system", "subtype": "init"},
        _turn(900_000, "mcp__curriculum__get_task"),
        _turn(940_000),
        _turn(950_000, "mcp__curriculum__done"),
    ]))
    assert audit["stayed_raw"] is True
    assert audit["compactions_before_seal"] == 0
    assert audit["seal_event"] == 3
    assert audit["max_context_tokens"] == 950_001


def test_stream_audit_catches_a_unit_that_compacted_before_sealing(tmp_path: Path) -> None:
    audit = stream_audit(_stream(tmp_path, [
        _turn(900_000, "mcp__curriculum__get_task"),
        _COMPACTING,
        _BOUNDARY,
        _turn(9_000, "mcp__curriculum__done"),
    ]))
    assert audit["stayed_raw"] is False
    assert audit["compactions_before_seal"] == 2


def test_stream_audit_ignores_a_compaction_after_the_seal(tmp_path: Path) -> None:
    """Only the context the task was actually solved under matters."""
    audit = stream_audit(_stream(tmp_path, [
        _turn(900_000, "mcp__curriculum__done"), _COMPACTING, _BOUNDARY]))
    assert audit["stayed_raw"] is True and audit["compaction_events"] == [1, 2]


def test_stream_audit_treats_a_unit_that_never_sealed_as_unprotected(tmp_path: Path) -> None:
    audit = stream_audit(_stream(tmp_path, [_turn(900_000), _BOUNDARY]))
    assert audit["seal_event"] is None and audit["stayed_raw"] is False


# --- the pass refuses a mixed arm -------------------------------------------------------------

_INDICES = [11, 22, 33]


def _seal(prov: Path, idx: int, reward: float) -> None:
    prov.mkdir(parents=True, exist_ok=True)
    with (prov / "results.jsonl").open("a") as fh:
        fh.write(json.dumps({"seq": 1, "split": "heldout", "task_idx": idx,
                             "reward": reward, "success": True}) + "\n")


def _cut_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compact_units: set) -> tuple:
    """Run a 3-unit cut pass offline: the fake agent seals every unit, but the units in
    ``compact_units`` write a stream showing an auto-compaction before the seal."""
    import experiments.selfopt.sandbox.study as study

    recs = _transcript(tasks=4)
    train_home = _home(tmp_path / "train", recs)
    prov_dir = _prov(tmp_path, ["h1"] * 4)
    source = cut_source(train_home, _cut("pre-x", recs, 8, sealed=2))
    started: Dict[str, Path] = {}

    def fake_start_broker(name: str, *, prov: Path, indices: Optional[List[int]] = None,
                          **kw: object) -> None:
        prov.mkdir(parents=True, exist_ok=True)
        started[name] = prov

    def fake_run_agent(*, work: Path, stream_path: Path, broker_name: str, **kw: object) -> int:
        unit = int(stream_path.stem.split("-")[-1])
        events = [_turn(900_000, "mcp__curriculum__get_task")]
        if unit in compact_units:
            events += [_COMPACTING, _BOUNDARY]
        events.append(_turn(500_000 if unit in compact_units else 940_000,
                            "mcp__curriculum__done"))
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_path.write_text("".join(json.dumps(e) + "\n" for e in events))
        _seal(started[broker_name], _INDICES[unit], 1.0)
        return 0

    monkeypatch.setattr(study, "start_broker", fake_start_broker)
    monkeypatch.setattr(study, "run_agent", fake_run_agent)
    monkeypatch.setattr(study, "_rm", lambda *names: None)
    return source, prov_dir, tmp_path / "pre-x"


def test_pass_with_a_clean_cut_arm_aggregates_and_manifests(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    source, prov_dir, stream_dir = _cut_pass(tmp_path, monkeypatch, compact_units=set())
    agg = _heldout_pass(run_id="r", arm="treatment", cp="pre-x", indices=_INDICES, src=None,
                        home_src=prov_dir / "snapshots" / "h1", stream_dir=stream_dir,
                        disallowed=None, system="s", oauth="tok", wandb_key=None, project="p",
                        concurrency=2, resume=SESSION, cut=source)
    assert agg["n"] == 3
    # The manifest is written BEFORE any unit runs — the audit trail exists even for a failed arm.
    assert json.loads((stream_dir / "cut_manifest.json").read_text())["last_record"] == 8
    # Every unit mounted its own copy of the prefix, and its context audit is on disk.
    for i in range(3):
        assert len(read_records(stream_dir / f"home-{i:03d}" / SLUG / f"{SESSION}.jsonl")) == 8
        audit = json.loads((stream_dir / f"prov-{i:03d}" / "context_audit.json").read_text())
        assert audit["stayed_raw"] is True and audit["max_context_tokens"] == 940_001


def test_pass_fails_a_raw_unit_that_auto_compacted(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """The unit has a real sealed score — it just isn't a score for THIS arm any more. Averaging it
    in would mix treatments, so the arm fails instead."""
    source, prov_dir, stream_dir = _cut_pass(tmp_path, monkeypatch, compact_units={1})
    kw = dict(run_id="r", arm="treatment", cp="pre-x", indices=_INDICES, src=None,
              home_src=prov_dir / "snapshots" / "h1", stream_dir=stream_dir, disallowed=None,
              system="s", oauth="tok", wandb_key=None, project="p", concurrency=2,
              resume=SESSION, cut=source)
    with pytest.raises(HeldoutIncomplete) as ei:
        _heldout_pass(**kw)  # type: ignore[arg-type]
    assert [f["unit"] for f in ei.value.failures] == [1]
    assert "auto-compacted before the seal" in ei.value.failures[0]["reason"]

    # And it STAYS failed on a re-run: the resume-skip must not launder a mixed unit into the mean.
    with pytest.raises(HeldoutIncomplete) as again:
        _heldout_pass(**kw)  # type: ignore[arg-type]
    assert [f["unit"] for f in again.value.failures] == [1]


def _rows(prov: Path) -> List[dict]:
    return [json.loads(ln) for ln in (prov / "results.jsonl").read_text().splitlines() if ln.strip()]


def test_a_mixed_units_row_says_so_and_a_clean_row_is_untouched(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The arm refuses, but the ROW is what analysis reads. A unit that measured the other treatment
    must carry that verdict in its own sealed row — otherwise it is byte-indistinguishable from a
    clean one and joins a paired contrast unnoticed, which is exactly how 8 mixed units did.

    The clean rows must stay EXACTLY as the broker wrote them: readers key off the existing fields,
    and the flag is meaningful only because its absence is the normal case."""
    source, prov_dir, stream_dir = _cut_pass(tmp_path, monkeypatch, compact_units={1})
    with pytest.raises(HeldoutIncomplete):
        _heldout_pass(run_id="r", arm="treatment", cp="pre-x", indices=_INDICES, src=None,
                      home_src=prov_dir / "snapshots" / "h1", stream_dir=stream_dir,
                      disallowed=None, system="s", oauth="tok", wandb_key=None, project="p",
                      concurrency=2, resume=SESSION, cut=source)

    flagged = _rows(stream_dir / "prov-001")
    assert len(flagged) == 1, "the seal itself is untouched — one row, still scored"
    assert flagged[0]["context_ok"] is False
    assert "auto-compacted before the seal" in flagged[0]["context_reason"]
    assert flagged[0]["reward"] == 1.0 and flagged[0]["task_idx"] == _INDICES[1]

    for unit in (0, 2):
        row = _rows(stream_dir / f"prov-{unit:03d}")[0]
        assert "context_ok" not in row and "context_reason" not in row, row
        assert set(row) == {"seq", "split", "task_idx", "reward", "success"}


def test_the_row_flag_survives_a_re_run_and_the_unit_is_not_replayed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running the same command must not spend on the mixed unit (the prefix is the same size,
    so it would compact again) and must not stamp it twice: the flag is a fact about the row, not
    an append."""
    source, prov_dir, stream_dir = _cut_pass(tmp_path, monkeypatch, compact_units={1})
    kw = dict(run_id="r", arm="treatment", cp="pre-x", indices=_INDICES, src=None,
              home_src=prov_dir / "snapshots" / "h1", stream_dir=stream_dir, disallowed=None,
              system="s", oauth="tok", wandb_key=None, project="p", concurrency=2,
              resume=SESSION, cut=source)
    with pytest.raises(HeldoutIncomplete):
        _heldout_pass(**kw)  # type: ignore[arg-type]
    first = (stream_dir / "prov-001" / "results.jsonl").read_bytes()

    with pytest.raises(HeldoutIncomplete):
        _heldout_pass(**kw)  # type: ignore[arg-type]
    again = _rows(stream_dir / "prov-001")
    assert len(again) == 1, "the mixed unit was replayed — a second sealed row means real spend"
    assert again[0]["context_ok"] is False
    assert (stream_dir / "prov-001" / "results.jsonl").read_bytes() == first


# --- the real transcript ------------------------------------------------------------------

_RUN = Path(__file__).resolve().parents[1] / "runs" / "sandbox-1785221622" / "treatment"
_needs_run = pytest.mark.skipif(not _RUN.exists(), reason="run artifacts are not checked in")


@_needs_run
@pytest.mark.parametrize("name", list(CUTS))
def test_real_cut_points_are_clean_and_as_pinned(name: str) -> None:
    """Re-derive each arm's cut from the actual transcript: the pinned uuid and sealed-task count
    must hold, with no dangling tool call and no orphaned parent."""
    facts = validate(CUTS[name], read_records(find_transcript(_RUN / "self_home")))
    assert facts["dangling_tool_use"] == [] and facts["orphan_parents"] == []
    assert facts["context_tokens"] > 0


@_needs_run
@pytest.mark.parametrize("pre,post", CUT_PAIRS)
def test_real_pairs_report_their_memory_pairing_honestly(pre: str, post: str) -> None:
    """A pair either mounts one archived home (runnable) or is refused. Both outcomes are correct;
    silently mounting two different homes is not.

    A pair MAY deliberately pin both members to one memory state (``Cut.home_sealed``) when context
    headroom forces the arms onto different tasks and the agent wrote memory in between. That is
    still honest — the divergence is declared on the cut — so the mounted home is checked against
    what each cut DECLARES it mounts, not against contemporaneity. An UNdeclared difference stays
    forbidden."""
    prov = _RUN / "prov"
    a, b = CUTS[pre], CUTS[post]
    try:
        home = assert_pair_homes(prov, a, b)
    except PairMismatch:
        # Refused only when neither member pins — i.e. the mismatch was never declared.
        assert a.home_sealed is None and b.home_sealed is None
        assert paired_home_hash(prov, a.expect_sealed) != \
            paired_home_hash(prov, b.expect_sealed)
        return
    assert home == paired_home_hash(prov, home_sealed_for(a))
    assert home == paired_home_hash(prov, home_sealed_for(b))
    # A pin must be load-bearing and bounded: present only because the contemporaneous homes really
    # differ, and never reaching back for end-of-training memory.
    for cut in (a, b):
        if cut.home_sealed is not None:
            assert paired_home_hash(prov, cut.expect_sealed) != home, (
                f"{cut.name} pins a home it would have mounted anyway — drop the pin")
            assert cut.home_sealed < len(train_rows(prov)), (
                f"{cut.name} pins END-of-training memory into an earlier prefix")


# --- acceptance: does the model actually SEE the prefix? ------------------------------------

def test_spoken_text_collects_prose_wherever_the_answer_lands(tmp_path: Path) -> None:
    path = _stream(tmp_path, [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "I have done 46."}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Most recently ss_rca."}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "mcp__curriculum__get_task"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "later chatter"}]}},
    ])
    assert spoken_text(path, before_first_tool=True) == "I have done 46.\nMost recently ss_rca."
    # The answer usually lands in the CLOSING message instead, after the task is played — judging
    # only the opening turn would discard the evidence the probe exists to collect.
    assert "later chatter" in spoken_text(path)
    assert spoken_text(tmp_path / "missing.jsonl") == ""


_PROBE_CUT = Cut("pre", 8, "u2-3", 46, raw=True, pair="post",
                 recall=("ss_rca",), future=("exit interview",))


def test_acceptance_passes_only_on_all_three_signals() -> None:
    good = acceptance_check("I have completed 46 tasks; the last was the ss_rca sheet.", _PROBE_CUT)
    assert good["pass"] and good["counts_match"] and good["recalled"] == ["ss_rca"]

    # Right count, but it names a task it could not have seen: the wrong conversation was loaded.
    leaked = acceptance_check("46 tasks — the last scheduled an exit interview.", _PROBE_CUT)
    assert not leaked["pass"] and leaked["leaked_future"] == ["exit interview"]

    # Names something real, but counts a whole different trajectory.
    miscounted = acceptance_check("I have completed 223 tasks, latest ss_rca.", _PROBE_CUT)
    assert not miscounted["pass"] and not miscounted["counts_match"]

    # Remembers nothing at all — a fresh context that answered politely.
    blank = acceptance_check("I do not have any record of previous tasks.", _PROBE_CUT)
    assert not blank["pass"]


def test_future_terms_must_actually_discriminate(tmp_path: Path) -> None:
    """A term that is present in the prefix, or absent from the whole file, proves nothing — so the
    discriminator is itself validated before it is used to accept a run."""
    recs = _transcript(tasks=4)
    recs[2]["message"]["content"][0]["content"] = "an exit interview task"
    recs[13]["message"]["content"][0]["content"] = "another exit interview task"
    assert_future_terms(recs, Cut("c", 8, "u2-3", 2, raw=True, pair="p",
                                  future=()))
    with pytest.raises(InvalidCut, match="appears INSIDE the prefix"):
        assert_future_terms(recs, Cut("c", 8, "u2-3", 2, raw=True, pair="p",
                                      future=("exit interview",)))
    with pytest.raises(InvalidCut, match="discriminates nothing"):
        assert_future_terms(recs, Cut("c", 8, "u2-3", 2, raw=True, pair="p",
                                      future=("a term nobody wrote",)))


@_needs_run
@pytest.mark.parametrize("name", [n for n, c in CUTS.items() if c.future or c.recall])
def test_real_acceptance_discriminators_hold_against_the_transcript(name: str) -> None:
    recs = read_records(find_transcript(_RUN / "self_home"))
    cut = CUTS[name]
    assert_future_terms(recs, cut)
    prefix = "".join(json.dumps(r) for r in recs[:cut.last_record]).lower()
    for term in cut.recall:
        assert term.lower() in prefix, f"{term!r} is supposed to be recallable FROM the prefix"
