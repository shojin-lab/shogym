"""Transcript-prefix ("context cut") arms: resume an agent at an EARLIER point of a training run.

``--resume`` only ever resumes a session at its END, so probing what the agent's context held
part-way through a run requires **truncating the transcript**: keep records ``[0:N]`` of the
training session's JSONL, write that prefix into the probe's throwaway ``~/.claude``, and resume
the same session id. The CLI replays from the last ``compact_boundary`` in the file, so a prefix
that ends just BEFORE a compaction boots the whole raw segment, and one ending just AFTER it boots
only the model's own summary — the two representations of the same trajectory window.

Nothing here is safe by default, so nothing here is silent:

  - A prefix is **validated before it is written**. Every ``parentUuid`` must resolve inside the
    kept records, and every ``tool_use`` must have its ``tool_result`` — an unanswered ``tool_use``
    is rejected by the API, and a broken parent chain is rejected by the CLI. Either way the agent
    dies or boots a different conversation than the one being measured, so a bad cut raises
    :class:`InvalidCut` rather than being materialized.
  - Each arm pins the facts it asserts (last retained record, its uuid, the number of tasks sealed
    at that point). A transcript that no longer matches its pin fails loudly instead of quietly
    measuring a different cut.
  - Every materialization emits a **manifest** — the audit trail that ties a reported number back
    to an exact prefix of an exact file.
  - A raw (pre-compaction) prefix runs against a nearly-full context window. If a unit auto-compacts
    mid-episode it silently becomes a summary unit, so :func:`stream_audit` re-reads the unit's own
    stream and reports whether a compaction happened before the task was sealed.

The memory the probe mounts alongside the prefix is the CONTEMPORANEOUS one — the archived home
from the last task the agent had *completed* at that cut (:func:`paired_home_hash`). The
end-of-training memory would leak lessons the agent had not yet learned into an early context.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

# --- the arms -------------------------------------------------------------------------


@dataclass(frozen=True)
class Cut:
    """One transcript-prefix arm.

    ``last_record`` is 1-indexed and INCLUSIVE (record 1 is the first line of the JSONL), matching
    ``sed -n '<n>p'``. ``expect_uuid`` / ``expect_sealed`` are pinned so a drifted transcript is
    caught instead of silently measured. ``raw`` marks a pre-compaction prefix, whose whole segment
    replays uncompacted and which must therefore be verified to have STAYED raw. ``pair`` names the
    arm it is contrasted against; paired arms must mount a byte-identical memory home."""

    name: str
    last_record: int
    expect_uuid: str
    expect_sealed: int
    raw: bool
    #: The arm this one is contrasted against, or ``""`` for an arm that stands alone (a point on
    #: the learning curve rather than one side of a before/after pair). Paired arms must mount a
    #: byte-identical memory home; an unpaired arm has no such constraint to enforce.
    pair: str = ""
    #: Resolve the memory home as of THIS many sealed tasks instead of :attr:`expect_sealed`.
    #:
    #: Normally a cut mounts the home contemporaneous with its own last completed task. But a
    #: pre/post pair must mount byte-identical memory or the contrast carries memory evolution as
    #: well as compaction, and a ``pre`` arm is often forced EARLIER than its partner by context
    #: headroom — if the agent wrote memory in the gap, the pair is void. Setting this pins both
    #: members to one memory state, so the only thing that varies is the context.
    #:
    #: This is a deliberate, bounded anachronism: the ``pre`` arm is handed memory from a few tasks
    #: ahead of where it actually stood. It is justified only because the knowledge base was
    #: measured to contribute ~nothing once context is held fixed, and it must never be used to
    #: mount END-of-training memory into an early prefix. Always report it alongside the result.
    home_sealed: Optional[int] = None
    # Acceptance discriminators (see :func:`acceptance_check`): strings from the tasks INSIDE the
    # prefix that an agent holding it can recall, and strings from tasks AFTER it that it cannot
    # possibly know. ``future`` is verified to be future-only against the transcript itself, so the
    # discriminator is never taken on faith.
    recall: Tuple[str, ...] = ()
    future: Tuple[str, ...] = ()


#: Cut points for the ``sandbox-1785221622`` treatment transcript. Each ``pre-*`` sits at the last
#: task seal that still leaves a whole held-out episode of headroom under the context window; each
#: ``post-*`` sits just past the corresponding compaction, where the context is the model's own
#: summary of the same trajectory. Both members of a pair are clean cuts (no dangling ``tool_use``,
#: no orphaned ``parentUuid``) — asserted at materialization, never assumed.
CUTS: Dict[str, Cut] = {
    "pre-compact-1": Cut("pre-compact-1", 2310, "8b18d725-3aa1-405b-be35-3d016a173b58", 46,
                         raw=True, pair="post-compact-1",
                         # Recall: the last two tasks inside the prefix, as the agent is likely to
                         # name them (subject) or cite them (record id). Future: the very next task,
                         # fetched one record past the cut — knowable only from a longer context.
                         recall=("zoho", "event registration", "ss_rca", "ss_event_registration"),
                         future=("exit interview", "departing employee", "ss_exit")),
    # A post-compaction arm keeps the whole prefix on disk but the model only ever sees the segment
    # after the boundary — its own summary. Its discriminators are therefore properties of that
    # summary's text, not of the file prefix, so they are left empty until they are read off it.
    "post-compact-1": Cut("post-compact-1", 2528, "d93c6b24-85a1-460b-a7b1-e86592884e69", 51,
                          raw=False, pair="pre-compact-1"),
    # Context headroom forces this arm to 100 tasks while its partner sits at 104, and the agent
    # wrote memory in that gap — so the pair would otherwise measure memory evolution alongside
    # compaction. Both members are pinned to the task-104 home, holding memory fixed so the only
    # variable is raw context vs the agent's own summary of it.
    "pre-compact-2": Cut("pre-compact-2", 4773, "26c202a5-9939-4211-beca-6df20408d31b", 100,
                         raw=True, pair="post-compact-2", home_sealed=104),
    "post-compact-2": Cut("post-compact-2", 4943, "488191d7-a9c4-458b-9af2-1dfb94f9d847", 104,
                          raw=False, pair="pre-compact-2"),
    # Compactions 3 and 4, for the memory-vs-no-memory contrast at MAXIMUM context compression.
    # A just-compacted context is a few thousand tokens of the agent's own summary — the condition
    # under which a durable knowledge base has its best chance of mattering, since the detail the
    # summary dropped is exactly what the notes retain. Measuring that at one boundary is an
    # anecdote; measuring it at every boundary in the run is the claim. Unpaired: the contrast is
    # ``post-compact-N`` against ``post-compact-N-nomem``, which mount the same cut and the same
    # home and differ only in whether the memory files are withheld.
    # All four post-* cuts sit at boundary+2 (boundary, summary, one record) so the arms are the
    # same SHAPE and only the compression point differs.
    # The raw conversation just before compaction #3, for the summary-efficacy contrast at the
    # third boundary. Sits at 147 tasks rather than the last seal before the boundary: a ``pre-*``
    # arm must leave a whole held-out episode of headroom under the context window, or the probe
    # auto-compacts and stops being a pre-compaction measurement. 909,182 tokens leaves ~91K,
    # comparable to ``pre-compact-1``. Pairs with ``post-compact-3`` on the same archived home
    # (5cce8f37bb84) with no pin needed, unlike pair 2.
    "pre-compact-3": Cut("pre-compact-3", 7173, "32a4c1eb-1002-4a8e-bbb7-67998b93dea1", 147,
                         raw=True, pair="post-compact-3"),
    "post-compact-3": Cut("post-compact-3", 7416, "b04fcd0f-d8f5-4ecc-8459-6cd34c36af71", 153,
                          raw=False, pair="pre-compact-3"),
    # The raw conversation just before compaction #4, completing the summary-efficacy series. Sits
    # at 194 tasks rather than the last seal before the boundary: the segment reaches 999,704 tokens
    # by the boundary itself, so a later cut leaves no room for a held-out episode and the probe
    # auto-compacts, which stops it being a pre-compaction measurement at all. 909,174 leaves ~91K,
    # matching ``pre-compact-3``. Its partner is four tasks further on, so the home is pinned to the
    # partner's rather than taken contemporaneously — otherwise the contrast would carry four tasks
    # of memory growth as well as the compaction it is meant to isolate.
    "pre-compact-4": Cut("pre-compact-4", 9773, "abf37cd9-0982-480e-9ddd-6a7ba2f77dfa", 194,
                         raw=True, pair="post-compact-4", home_sealed=198),
    "post-compact-4": Cut("post-compact-4", 9984, "ba3c4ebc-465c-45c0-a86f-1d878e9da52f", 198,
                          raw=False, pair="pre-compact-4"),
    # Learning-curve arms: the context exactly as the agent held it after N tasks, unpaired. Each is
    # the FIRST clean record at which ``sealed == N`` — i.e. immediately after the Nth seal, so the
    # resumed agent's next move is a fetch rather than the middle of a task the probe's broker never
    # dispensed. ``raw`` is simply whether a ``compact_boundary`` falls inside the prefix: at 25 tasks
    # none has happened yet, while 75 and 125 replay from the summary the agent itself wrote.
    "ctx-25": Cut("ctx-25", 1413, "414394c1-b92c-43e6-b5e9-74a09d978f46", 25, raw=True),
    "ctx-75": Cut("ctx-75", 3718, "93352f4f-f3cd-4861-9789-4a35118af64c", 75, raw=False),
    "ctx-125": Cut("ctx-125", 6155, "324d11db-23d7-473a-8b8e-6c9cc1cf9895", 125, raw=False),
    # The last point on the curve, just past the final compaction: 200 tasks, two beyond the
    # boundary at 198. Placed here rather than at the end of the run because the curve is about
    # what the agent held while it was still working, and because the pair at that boundary
    # already measures the summary itself — this arm asks what the conversation was worth once
    # the agent had resumed from that summary and played on.
    "ctx-200": Cut("ctx-200", 10120, "e5303183-d8b4-475a-a547-5f26d84e871b", 200, raw=False),
}

#: The pre/post pairs, in the order they are reported.
CUT_PAIRS: Tuple[Tuple[str, str], ...] = (("pre-compact-1", "post-compact-1"),
                                          ("pre-compact-2", "post-compact-2"))


# Counts of tasks, not every number in the prose. A run report is full of scores, ids and dates, so
# an unqualified integer match would "confirm" the expected count out of noise; require the number
# to sit next to the word it is a count of.
_TASK_COUNT_RE = re.compile(r"(?:\b(\d{1,4})\b[^.\n]{0,30}?tasks?\b)|(?:\btasks?\b[^.\n]{0,30}?\b(\d{1,4})\b)",
                            re.IGNORECASE)


class InvalidCut(RuntimeError):
    """A prefix that must not be written: a broken chain, an unanswered tool call, or a transcript
    that no longer matches the facts the arm pins. Materializing it would boot the agent into a
    conversation other than the one being measured — or kill it outright — and either way the arm
    would report a number about nothing."""


# --- reading + auditing a prefix --------------------------------------------------------

def read_records(path: Path) -> List[dict]:
    """The transcript's records, in file order. Blank lines are skipped; a malformed line is fatal
    (a transcript we cannot fully parse is one we cannot honestly cut)."""
    out: List[dict] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise InvalidCut(f"{path}: record {lineno} is not valid JSON ({exc})") from exc
    return out


def _blocks(rec: dict) -> Sequence[dict]:
    content = (rec.get("message") or {}).get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _tool_calls(rec: dict, suffix: str) -> int:
    return sum(1 for b in _blocks(rec)
               if b.get("type") == "tool_use" and str(b.get("name", "")).endswith(suffix))


def audit_prefix(records: Sequence[dict], last_record: int) -> dict:
    """Everything knowable about ``records[:last_record]`` without running anything.

    ``orphans`` are records whose ``parentUuid`` points outside the prefix; ``dangling`` are
    ``tool_use`` blocks with no ``tool_result`` in it. ``sealed`` counts ``*__done`` calls (tasks
    submitted for scoring) and ``fetched`` counts ``*__get_task`` calls, both from parsed tool_use
    blocks — matching the tool name as a substring of the raw line also hits the names echoed back
    inside tool RESULTS and over-counts.

    ``context_tokens`` is the model-visible context at the cut: for a prefix whose replayed segment
    (everything after the last ``compact_boundary``) contains assistant turns, it is that last
    turn's own recorded usage — the tokens the model actually held. A segment with no assistant
    turn (a just-compacted prefix, which is a boundary plus a summary) has no recorded usage, so it
    falls back to a size estimate; ``context_tokens_method`` says which was used."""
    if last_record < 1 or last_record > len(records):
        raise InvalidCut(f"cut at record {last_record} is outside the transcript "
                         f"(1..{len(records)})")
    kept = list(records[:last_record])
    uuids = {r["uuid"] for r in kept if r.get("uuid")}
    orphans = [i for i, r in enumerate(kept, 1)
               if r.get("parentUuid") and r["parentUuid"] not in uuids]

    pending: Dict[str, Tuple[int, str]] = {}
    answered: set[str] = set()
    sealed = fetched = 0
    for i, rec in enumerate(kept, 1):
        for b in _blocks(rec):
            if b.get("type") == "tool_use":
                pending[str(b.get("id"))] = (i, str(b.get("name")))
            elif b.get("type") == "tool_result":
                answered.add(str(b.get("tool_use_id")))
        sealed += _tool_calls(rec, "__done")
        fetched += _tool_calls(rec, "__get_task")
    dangling = [{"record": rec, "name": name}
                for tid, (rec, name) in sorted(pending.items(), key=lambda kv: kv[1])
                if tid not in answered]

    boundary = 0
    for i, rec in enumerate(kept, 1):
        if rec.get("type") == "system" and rec.get("subtype") == "compact_boundary":
            boundary = i
    segment = kept[boundary:]
    recorded = _recorded_context(segment)
    tokens, method = ((recorded, "recorded-usage") if recorded is not None
                      else (_estimated_context(segment), "segment-chars/4"))

    last = kept[-1]
    return {
        "last_record": last_record,
        "last_uuid": last.get("uuid"),
        "last_type": last.get("type"),
        "records": last_record,
        "orphan_parents": orphans,
        "dangling_tool_use": dangling,
        "sealed": sealed,
        "fetched": fetched,
        "segment_start_record": boundary + 1,
        "segment_records": len(segment),
        "context_tokens": tokens,
        "context_tokens_method": method,
    }


def _recorded_context(segment: Sequence[dict]) -> Optional[int]:
    """The context the model held on the last assistant turn of ``segment``, off that turn's own
    usage accounting (fresh input + cached prefix read + cache written). ``None`` when the segment
    holds no assistant turn."""
    for rec in reversed(segment):
        usage = (rec.get("message") or {}).get("usage")
        if isinstance(usage, dict):
            return int(usage.get("input_tokens", 0)) + \
                int(usage.get("cache_read_input_tokens", 0)) + \
                int(usage.get("cache_creation_input_tokens", 0))
    return None


def _estimated_context(segment: Sequence[dict]) -> int:
    """A size estimate for a segment with no recorded usage: the message payloads at ~4 chars per
    token. Coarse by construction — it is only ever used where the transcript records nothing, and
    the manifest labels it as an estimate."""
    chars = sum(len(json.dumps(rec.get("message"), ensure_ascii=False))
                for rec in segment if rec.get("message") is not None)
    return chars // 4


def validate(cut: Cut, records: Sequence[dict]) -> dict:
    """Audit the arm's prefix and refuse it unless it is exactly what the arm claims: a clean chain,
    every tool call answered, and the pinned last-record uuid + sealed-task count."""
    facts = audit_prefix(records, cut.last_record)
    problems: List[str] = []
    if facts["dangling_tool_use"]:
        problems.append(
            f"{len(facts['dangling_tool_use'])} unanswered tool_use "
            f"(first at record {facts['dangling_tool_use'][0]['record']}: "
            f"{facts['dangling_tool_use'][0]['name']}) — the API rejects a trailing tool_use")
    if facts["orphan_parents"]:
        problems.append(f"{len(facts['orphan_parents'])} records whose parentUuid is outside the "
                        f"prefix (first at record {facts['orphan_parents'][0]})")
    if facts["last_uuid"] != cut.expect_uuid:
        problems.append(f"record {cut.last_record} is uuid {facts['last_uuid']!r}, "
                        f"the arm pins {cut.expect_uuid!r}")
    if facts["sealed"] != cut.expect_sealed:
        problems.append(f"{facts['sealed']} tasks sealed at the cut, the arm pins "
                        f"{cut.expect_sealed}")
    if problems:
        raise InvalidCut(f"cut {cut.name!r} at record {cut.last_record} is not usable:\n  - "
                         + "\n  - ".join(problems))
    return facts


# --- materializing a prefix into a probe's home -----------------------------------------

@dataclass(frozen=True)
class CutSource:
    """A cut bound to the transcript it truncates and the path that transcript occupies inside a
    ``~/.claude`` home (``projects/<cwd-slug>/<session>.jsonl``). The session id is the file stem,
    so the probe resumes the SAME session the training run used — just a shorter one."""

    cut: Cut
    transcript: Path
    rel: PurePosixPath

    @property
    def session_id(self) -> str:
        return self.transcript.stem


def find_transcript(home: Path) -> Path:
    """The one conversation transcript in a ``~/.claude`` home (``projects/<slug>/<session>.jsonl``).
    Ambiguity is an error: cutting the wrong session would boot the wrong conversation."""
    found = sorted(p for p in Path(home).glob("projects/*/*.jsonl") if p.is_file())
    if len(found) != 1:
        raise InvalidCut(f"expected exactly one session transcript under {home}, found "
                         f"{[str(p) for p in found]}")
    return found[0]


def cut_source(home: Path, cut: Cut) -> CutSource:
    """Bind ``cut`` to the transcript of a training home, keeping its in-home location."""
    transcript = find_transcript(home)
    return CutSource(cut=cut, transcript=transcript,
                     rel=PurePosixPath(transcript.relative_to(Path(home)).as_posix()))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize_cut(source: CutSource, home: Path) -> dict:
    """VALIDATE the prefix, then write it into ``home`` at the transcript's original in-home path.

    Returns the manifest facts for this cut. The source transcript is only read; the truncated copy
    is written into the probe's throwaway home (which the agent then appends to as it runs, which is
    why every unit gets its own copy)."""
    records = read_records(source.transcript)
    facts = validate(source.cut, records)
    dst = Path(home) / source.rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for rec in records[:source.cut.last_record]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {
        "arm": source.cut.name,
        "raw_segment": source.cut.raw,
        "paired_with": source.cut.pair,
        "session_id": source.session_id,
        "transcript": str(source.transcript),
        "transcript_sha256": sha256_file(source.transcript),
        "transcript_records": len(records),
        "written_to": source.rel.as_posix(),
        "written_bytes": dst.stat().st_size,
        **facts,
    }


# --- contemporaneous memory pairing ------------------------------------------------------

def train_rows(prov: Path) -> List[dict]:
    """The run's sealed TRAIN rows in play order — one per completed task, so row ``k`` is the k-th
    task the agent finished."""
    results = Path(prov) / "results.jsonl"
    if not results.exists():
        return []
    rows = [json.loads(x) for x in results.read_text().splitlines() if x.strip()]
    return sorted((r for r in rows if r.get("split") == "train"), key=lambda r: r["seq"])


def paired_home_hash(prov: Path, sealed: int) -> Optional[str]:
    """The memory home CONTEMPORANEOUS with a cut: ``home_hash_after`` of the last task the agent
    had completed there. Compactions fire mid-task, so no snapshot lands exactly on one — the
    nearest PRIOR task boundary is the closest archived state that the agent could actually have had
    at the cut. The end-of-training home is never the answer: mounting it into an early prefix would
    hand the agent lessons it had not yet learned."""
    rows = train_rows(prov)
    if sealed < 1 or sealed > len(rows):
        raise InvalidCut(f"cut claims {sealed} sealed tasks but the run has {len(rows)} train rows")
    return rows[sealed - 1].get("home_hash_after")


class PairMismatch(RuntimeError):
    """The two members of a pre/post pair would mount DIFFERENT memory homes.

    Then the contrast is not raw-vs-summarised context: it also carries whatever the agent wrote to
    its knowledge base in between, and any measured difference is confounded with memory evolution.
    The pair is void, so it does not run."""


def home_sealed_for(cut: Cut) -> int:
    """The task count whose archived memory home ``cut`` mounts — its own unless it deliberately
    pins a partner's (see :attr:`Cut.home_sealed`)."""
    return cut.home_sealed if cut.home_sealed is not None else cut.expect_sealed


def assert_pair_homes(prov: Path, a: Cut, b: Cut) -> str:
    """Both members of a pair must mount the byte-identical archived home. Returns that home hash."""
    ha, hb = paired_home_hash(prov, home_sealed_for(a)), paired_home_hash(prov, home_sealed_for(b))
    if ha != hb:
        raise PairMismatch(
            f"{a.name} (after {a.expect_sealed} tasks) pairs home {ha!r} but {b.name} (after "
            f"{b.expect_sealed} tasks) pairs home {hb!r} — a pre/post contrast across two different "
            f"memory states measures memory evolution as well as compaction. Pair void.")
    return str(ha)


# --- acceptance: does the model actually SEE the prefix? ----------------------------------

#: Asked (once, in the acceptance probe only — never in a scored arm) before the agent plays its
#: task. Exit code 0 says the process ran; this says the conversation the process resumed is the one
#: that was mounted. Deliberately open: it names no task and no count, so a correct answer cannot be
#: read off the question. It is delivered as the USER TURN, not appended to the system prompt — an
#: agent resumed mid-curriculum acts on the turn in front of it and reads a standing system prompt
#: as background, so a probe in the system prompt gets skipped straight past into the task loop.
CUT_PROBE = (
    "Before you do anything else, answer in plain prose, in your reply text: how many tasks have "
    "you already completed and submitted for scoring in this conversation, and what were the two "
    "most recent ones about — name the specific records, spreadsheets or systems you worked with. "
    "If you do not remember, say so. Then continue with your instructions."
)


def assert_future_terms(records: Sequence[dict], cut: Cut) -> None:
    """A "the model cannot know this" discriminator is only worth something if it is true. Each of
    ``cut.future`` must be absent from the prefix and present after it — otherwise the acceptance
    test would pass on a cut that loaded the wrong conversation."""
    before = "".join(json.dumps(r) for r in records[:cut.last_record]).lower()
    after = "".join(json.dumps(r) for r in records[cut.last_record:]).lower()
    for term in cut.future:
        t = term.lower()
        if t in before:
            raise InvalidCut(f"cut {cut.name!r}: {term!r} appears INSIDE the prefix, so its absence "
                             f"proves nothing")
        if t not in after:
            raise InvalidCut(f"cut {cut.name!r}: {term!r} never appears after the prefix either — "
                             f"it discriminates nothing")


def spoken_text(stream_path: Path, *, before_first_tool: bool = False) -> str:
    """Everything the agent said in prose over a unit — its answer to :data:`CUT_PROBE`.

    The answer does not reliably arrive first. An agent resumed deep in a task loop tends to play
    the task and then account for itself in its closing message, so judging only what it said before
    its first tool call throws away the evidence. ``before_first_tool=True`` isolates that opening
    turn for the record."""
    said: List[str] = []
    path = Path(stream_path)
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "assistant":
            continue
        for b in _blocks(ev):
            if b.get("type") == "tool_use" and before_first_tool:
                return "\n".join(said).strip()
            if b.get("type") == "text" and b.get("text"):
                said.append(str(b["text"]))
    return "\n".join(said).strip()


def acceptance_check(text: str, cut: Cut, *, tolerance: int = 3) -> dict:
    """Judge one acceptance answer against the cut it was supposed to boot from.

    Three independent things must hold, and each can fail on its own: the agent counts roughly the
    tasks the prefix sealed, it can name something from inside the prefix, and it names nothing from
    beyond it. A run that satisfies only the first is a run that guessed.

    The count is approximate on purpose — the agent legitimately counts the task it has just played
    and any task it had fetched but not sealed at the cut, so it lands a little above the sealed
    count. The tolerance is far tighter than the gap to any other point in this trajectory, so it
    still separates the prefix from a longer or shorter one; but it is the weakest of the three
    signals, and only the conjunction is evidence."""
    low = text.lower()
    numbers = [int(n) for m in _TASK_COUNT_RE.findall(text) for n in m if n]
    near = [n for n in numbers if abs(n - cut.expect_sealed) <= tolerance]
    recalled = [t for t in cut.recall if t.lower() in low]
    leaked = [t for t in cut.future if t.lower() in low]
    return {
        "text": text,
        "stated_counts": numbers,
        "expect_sealed": cut.expect_sealed,
        "counts_match": bool(near),
        "recalled": recalled,
        "recalls_the_prefix": bool(recalled) or not cut.recall,
        "leaked_future": leaked,
        "knows_nothing_later": not leaked,
        "pass": bool(near) and (bool(recalled) or not cut.recall) and not leaked,
    }


# --- raw must be verified to have STAYED raw ---------------------------------------------

def _usage_tokens(event: dict) -> Optional[int]:
    usage = (event.get("message") or {}).get("usage")
    if not isinstance(usage, dict):
        return None
    return int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0)) + \
        int(usage.get("cache_creation_input_tokens", 0))


def _is_compaction(event: dict) -> bool:
    if event.get("type") != "system":
        return False
    return (event.get("subtype") == "compact_boundary"
            or event.get("status") == "compacting"
            or "compact_result" in event)


def stream_audit(stream_path: Path) -> dict:
    """Re-read a finished unit's own stream and report whether its context stayed as it was mounted.

    A raw prefix boots close to the context limit. If the episode auto-compacts before the task is
    sealed, the model finishes the task off its own summary — the unit has quietly changed arms, and
    averaging it in mixes treatments. So: find the seal (the ``*__done`` call), and report every
    compaction event and whether any of them landed before it. Also reports the maximum
    model-visible context over the whole unit, which is what says how much headroom the cut had."""
    events: List[dict] = []
    path = Path(stream_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    seal_at: Optional[int] = None
    compactions: List[int] = []
    max_ctx = 0
    for i, ev in enumerate(events):
        if _is_compaction(ev):
            compactions.append(i)
        tokens = _usage_tokens(ev)
        if tokens is not None:
            max_ctx = max(max_ctx, tokens)
        if seal_at is None and _tool_calls(ev, "__done"):
            seal_at = i
    # No seal recorded: the unit did not submit anything, so ANY compaction happened "before" it.
    cutoff = seal_at if seal_at is not None else len(events)
    before = [i for i in compactions if i < cutoff]
    return {
        "events": len(events),
        "seal_event": seal_at,
        "compaction_events": compactions,
        "compactions_before_seal": len(before),
        "stayed_raw": not before,
        "max_context_tokens": max_ctx,
    }
