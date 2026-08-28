"""The seal-before-verdict lifecycle: state machine, protected terminal evidence, the
``finalize`` request/response types, and the **durable** finalization record.

An episode that scores a submission must seal *before* it evaluates, so the verdict is
computed from a frozen, core-owned record rather than from anything the agent can still
influence. This module owns three things:

- :class:`LifecycleState` — the per-episode state machine
  ``OPEN -> SEALED -> FINALIZING -> FINALIZED -> TEARING_DOWN -> CLOSED`` that every ingress
  path respects.
- :class:`FinalizeRequest` / :class:`TerminalEvidence` — the typed contract of the env's
  ``finalize`` hook. Evidence is **core-owned**: the env supplies a verdict/status/diagnostic,
  the serve layer stamps the non-forgeable ``provenance`` and the ``finalization_id`` a
  harness cannot supply.
- :class:`FinalizationStore` — a tiny, **zero-setup, local-file** durable record (one JSON
  file per ``(session_id, finalization_id)``, ``fsync``'d on every transition) holding
  ``SEALED | PENDING | FINALIZED | FAILED``, plus :func:`FinalizationStore.recover`, the
  fail-closed restart-recovery contract: a record found mid-finalize on restart resolves to a
  ``finalize_error`` verdict and the evaluator is **never** re-invoked.

No database, no credentials, no env-vars, no setup step — just a directory of small JSON
files next to the trace store.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import os
import tempfile
import threading
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

# The evidence-schema version. Stamped on every :class:`TerminalEvidence`, the durable
# record, and the trace ``terminal`` event so a reader can tell which envelope it is parsing.
EVIDENCE_SCHEMA_VERSION = 1

# The private durable-record filename schema version (independent of the evidence schema).
RECORD_SCHEMA_VERSION = 1

#: The largest value a pid can take on this platform, for the range check in the decoder. Read
#: from the kernel where it is published and otherwise a bound generous enough to admit every
#: real pid and reject the integers that are not pids at all.
try:  # pragma: no cover - one branch per platform
    _MAX_PID = int(Path("/proc/sys/kernel/pid_max").read_text(encoding="utf-8").strip())
except (OSError, ValueError):
    _MAX_PID = 2 ** 31 - 1

TerminalSource = Literal["explicit_tool", "horizon", "abort"]
TerminalStatus = Literal["ok", "finalize_error"]
FinalizationStatus = Literal["SEALED", "PENDING", "FINALIZED", "FAILED"]


class LifecycleState(enum.Enum):
    """The per-episode lifecycle. Only a ``score``-terminal env (one that declares a
    ``score`` tool **and** a ``finalize`` hook) drives it past ``OPEN``; every other env
    stays ``OPEN`` throughout and uses the legacy ``terminated`` flag, so its behaviour is
    unchanged by the seal machinery."""

    OPEN = "open"
    SEALED = "sealed"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    TEARING_DOWN = "tearing_down"
    CLOSED = "closed"


# ----- the finalize hook contract -----


@dataclass
class FinalizeRequest:
    """What the serve layer hands an env's ``finalize`` hook when it runs the evaluator on an
    already-sealed episode.

    - ``source`` — how the episode was terminated (``explicit_tool`` for a score-tool call,
      ``horizon`` for the step budget, ``abort`` for ``terminate``/close).
    - ``args`` — the validated, normalized score-tool arguments; **only** present for
      ``explicit_tool`` (``None`` for horizon/abort, which carry no submission).
    - ``deadline`` — the wall-clock seconds budget the evaluator has before the serve layer
      fails it closed; ``None`` disables the bound.
    - ``finalization_id`` / ``session_id`` — the durable-record key.
    - ``tool_name`` — the score tool that sealed (``None`` for horizon/abort).
    """

    source: TerminalSource
    finalization_id: str
    session_id: str
    args: Optional[Dict[str, Any]] = None
    deadline: Optional[float] = None
    tool_name: Optional[str] = None


@dataclass
class TerminalEvidence:
    """Core-owned, non-forgeable terminal evidence — the sole trusted input a verifier scores
    from (replacing marker-JSON scanning).

    An env's ``finalize`` returns this with ``verdict`` / ``status`` / ``args`` /
    ``diagnostic`` populated; the serve layer **stamps** ``source``, ``finalization_id``, and
    the ``provenance`` (a system field the harness cannot supply) before it is trusted, so a
    returned ``provenance`` is always overwritten. ``verdict`` is the public-safe authoritative
    score, public-safe because the RFC forbids returning judge reasoning or extracted answers —
    not because anything rewrote it, so it is the env's dict verbatim and a key in it that the
    core also owns is still only the env's word for that key. ``status`` is the env's *declared*
    outcome and the one channel that decides it; :attr:`finalize_error` reads that, never the
    verdict. ``diagnostic`` is private (server-side logs / the durable store only) and must
    never reach the agent.
    """

    source: TerminalSource
    status: TerminalStatus
    verdict: Dict[str, Any]
    # Validated, normalized args — ONLY for source=explicit_tool.
    args: Optional[Dict[str, Any]] = None
    # Stamped by the serve layer (non-forgeable). None until stamped.
    provenance: Optional[Dict[str, Any]] = None
    finalization_id: Optional[str] = None
    # Private diagnostic (never surfaced to the agent).
    diagnostic: Optional[str] = None
    # Structural description of the failure behind a `finalize_error`, for the harness-side
    # record only (see `failure_summary`). Like `diagnostic`, never surfaced to the agent.
    failure: Optional[Dict[str, Any]] = None
    schema_version: int = EVIDENCE_SCHEMA_VERSION

    @property
    def finalize_error(self) -> bool:
        return self.status == "finalize_error"


def fail_closed_verdict(confidence: Optional[Any] = None) -> Dict[str, Any]:
    """The canonical fail-closed verdict: an evaluator timeout/crash or a crashed-mid-finalize
    restart resolves here. Scores ``correct=False`` and flags ``finalize_error`` so a
    fail-closed zero is distinguishable in audit data from an honest wrong answer — without
    leaking any oracle. ``confidence`` echoes the caller's supplied value for calibration when
    known."""
    verdict: Dict[str, Any] = {"correct": False, "finalize_error": True}
    if confidence is not None:
        verdict["confidence"] = confidence
    return verdict


# How many distinct error kinds a failure summary keeps. One structured failure reports an entry
# per offending value, but the kinds behind them are few, so this bounds a pathological case
# rather than an ordinary one.
_MAX_FAILURE_ERROR_KINDS = 6


def _certified_error_kind(entry: Any) -> Optional[str]:
    """The error kind of one structured entry, but only when the validator itself vouches for it.

    A reported kind is a bare string, and a string is only safe to publish if something other than
    this function's optimism says where it came from. The validator's own documentation link is
    that witness: it is built from the fixed set of error kinds the library defines, so a link
    ending in the kind it reports proves the kind is one of them. An entry whose kind was supplied
    by whoever wrote the validator carries no link, and is refused.

    The witness holds against a genuine validation error, which is the case this exists for: the
    library builds the link and no caller can hand it one. It does not hold against an object that
    forges both halves, because a duck-typed ``errors()`` is the env's own code. That is the same
    trust boundary the verdict already sits on, and it is a different failure from the one guarded
    here: an env cannot leak its state through this by accident, only by writing something whose
    purpose is to.
    """
    kind = entry.get("type")
    url = entry.get("url")
    if not isinstance(kind, str) or not isinstance(url, str):
        return None
    return kind if url.endswith("/v/" + kind) else None


def failure_summary(exc: BaseException) -> Dict[str, Any]:
    """A structural description of a contained failure, for the harness-side record.

    Names the exception type that ended the terminal transaction and, when the exception reports
    structured validation errors, how many there were and which kinds of error they were. That is
    enough to act on: a type plus a kind says which layer failed and how, which is what a reader
    of an unscored row has to know before they can do anything about it.

    **Everything published here is drawn from a fixed vocabulary or is a count.** Nothing that
    could have originated in the data being validated is included, because for an env whose state
    is the thing being graded that data can be the answer, and this summary is written to a record
    that outlives the episode. Two exclusions carry that guarantee:

    - **Not the message.** It renders the offending values directly.
    - **Not the field locations.** They read like schema paths and are not: a location descends
      into the *input*, so a failure inside a mapping contributes that mapping's keys, and a
      rejected unknown key contributes the key itself. Either is a value wearing a field name's
      clothes, and neither can be told from a real field name without the model, which a caught
      exception does not carry.

    The kinds that do survive are filtered through :func:`_certified_error_kind`, so an error kind
    invented by whoever wrote the validator is dropped rather than trusted.

    What is left out of the row is not lost: the full diagnostic, locations and values included,
    is written to the private durable record, which is reachable by whoever runs the harness and
    by no one else.

    Every read of the exception is contained. Describing a caught failure runs code belonging to
    whoever raised it, a second time and outside the ``except`` that just caught it, so an
    accident in here would not stay caught. It would leave carrying the caller's job with it,
    and the caller's job at this point is committing a fail-closed verdict. ``SystemExit`` and
    ``KeyboardInterrupt`` still propagate, the same line the rest of this package holds: an
    interpreter-level signal costs the record loudly rather than being swallowed in a summary.
    """
    try:
        name = type(exc).__name__
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:  # noqa: BLE001 (a contained failure may not escape through its type)
        return {"error": "<unreadable>"}
    summary: Dict[str, Any] = {"error": name}
    try:
        # Structured errors are duck-typed rather than isinstance-checked against pydantic:
        # anything that reports its failures this way describes them the same useful way, and
        # anything that does not simply keeps the type-only summary.
        reported = list(exc.errors())  # type: ignore[attr-defined]
        kinds = sorted(
            {kind for kind in (_certified_error_kind(entry) for entry in reported) if kind}
        )
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:  # noqa: BLE001 (not a structured failure, or one that cannot say so)
        return summary
    if reported:
        summary["error_count"] = len(reported)
    if kinds:
        summary["error_kinds"] = kinds[:_MAX_FAILURE_ERROR_KINDS]
    return summary


def args_digest(args: Optional[Dict[str, Any]]) -> Optional[str]:
    """A stable digest of normalized args for the public trace event (never the raw args)."""
    if args is None:
        return None
    blob = json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# ----- the durable finalization record (local file, zero setup) -----


@dataclass
class FinalizationRecord:
    """One durable finalization record, keyed by ``(session_id, finalization_id)``.

    ``verdict`` is the public-safe score exactly as the env's ``finalize`` returned it — evidence
    of what the env said, not authority over what happened: :attr:`status` is where the core
    recorded the outcome, and :meth:`to_evidence` takes the outcome from that and only that,
    reconstructing the rest of the evidence around it. ``provenance`` and
    ``diagnostic`` are confidential (they live here, in the private store — never in the
    user-readable trace).
    """

    session_id: str
    finalization_id: str
    status: FinalizationStatus
    source: TerminalSource
    schema_version: int = RECORD_SCHEMA_VERSION
    args_digest: Optional[str] = None
    verdict: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None
    diagnostic: Optional[str] = None
    # The OS pid of the process that owns this in-flight finalization. Recovery uses it to tell
    # a **crashed** run (owner no longer alive) from a **live** concurrent worker sharing the
    # store (owner still alive), so it never resolves a record another episode is still writing.
    owner_pid: Optional[int] = None

    def to_evidence(self) -> TerminalEvidence:
        """Reconstruct the terminal evidence this record persisted (for a ``FINALIZED``
        replay or a recovered fail-closed resolution).

        **The outcome comes from** :attr:`status` **and from nothing else.** The core owns that
        field at every transition, and settles it at commit: ``FAILED`` exactly when the evidence
        it committed carried ``finalize_error``, ``FINALIZED`` otherwise. So it already *is* the
        decision the live path reached, not a restatement of one — an env
        reaches it only through ``TerminalEvidence.status``, the one channel that declares an
        outcome, and that is the same channel the live path answered from. Reading it back
        replays the live answer rather than deriving a second one. ``verdict`` is the opposite —
        persisted as the env's ``finalize`` returned it, verbatim — so consulting it as well
        would let an env overturn a decision the core had already made.
        Reading it by truthiness gets that wrong twice over, because an episode feedback value
        legally admits text and numbers and ``bool("false")`` is ``True``; but reading it
        *strictly* is wrong too, because even a literal ``True`` under a reserved name is still
        only the env's word — the live path overwrites that key from the core's own flag before
        the agent is ever shown it. Taken from ``status`` alone, a replay reproduces the outcome
        the live path published; taken from the verdict, a clean episode comes back as an
        infrastructure failure with its real result discarded.

        The verdict is still reconstructed verbatim: it is evidence of what the env returned,
        and only its authority over the outcome is withheld. Verbatim includes ``{}``, which is
        a verdict an env may legally return and is not the same thing as a record that never
        reached one — only ``None`` means no verdict was ever written, and only that gets the
        synthetic fail-closed stand-in. Testing the verdict for truthiness instead would answer
        a clean replay with an invented ``correct=False``."""
        status: TerminalStatus = "ok" if self.status == "FINALIZED" else "finalize_error"
        return TerminalEvidence(
            source=self.source,
            status=status,
            verdict=dict(self.verdict) if self.verdict is not None else fail_closed_verdict(),
            provenance=self.provenance,
            finalization_id=self.finalization_id,
            diagnostic=self.diagnostic,
        )


class FinalizationStore:
    """A directory of small JSON finalization records, one per ``(session_id,
    finalization_id)``, ``fsync``'d on every transition.

    Zero user setup: pass the trace directory (or accept the default under
    ``~/.cache/shogym/sessions``). No DB, no credentials, no env-vars.
    """

    def __init__(self, directory: Union[str, Path]) -> None:
        self._dir = Path(directory)

    @property
    def directory(self) -> Path:
        return self._dir

    @staticmethod
    def resolve_dir(
        session_id: str, trace_path: Optional[Union[str, Path]]
    ) -> Path:
        """Where an episode's records live. Both roots are **shared across sessions** so that
        startup recovery is *reachable*: a crashed prior session's record must sit in a
        directory the next process scans. Records are keyed by a globally-unique
        ``finalization_id`` (and each carries its own ``session_id``), so sharing a directory is
        safe — no per-session subdir is needed and, critically, a per-session subdir would hide
        a crashed run's record from recovery.

        - **With a trace path:** ``<trace_dir>/finalizations`` — next to the trace.
        - **Without one:** a stable, zero-config fallback root (``~/.cache/shogym/sessions``, or
          the system temp dir if that can't be created) — *not* keyed by session, so a later
          ``run_stdio`` startup recovers dangling records from a prior crashed run there.
        """
        if trace_path is not None:
            return Path(trace_path).parent / "finalizations"
        return _sessions_cache_root()

    def _path(self, finalization_id: str) -> Path:
        return self._dir / f"finalization-{finalization_id}.json"

    def write(self, record: FinalizationRecord) -> None:
        """Persist ``record`` durably: write a temp file, ``fsync`` it, atomically rename over
        the target, then ``fsync`` the directory. Every state transition calls this, so a crash
        can never leave a torn/partial record on disk.

        The directory holding it is made durable the same way, by :func:`_mkdir_durable`, and so
        is every level above it — a record fsync'd into a directory whose own entry is still
        only in the page cache is no more durable than that entry."""
        _mkdir_durable(self._dir)
        target = self._path(record.finalization_id)
        blob = json.dumps(asdict(record), sort_keys=True, allow_nan=False)
        fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        # fsync the directory so the rename itself is durable across a crash.
        _fsync_dir(self._dir)

    def read(self, finalization_id: str) -> Optional[FinalizationRecord]:
        path = self._path(finalization_id)
        if not path.exists():
            return None
        return _record_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_all(self) -> List[FinalizationRecord]:
        return self._load_all()[0]

    def _load_all(self) -> Tuple[List[FinalizationRecord], List[Path]]:
        """Every record this directory holds, and every entry in it that is not one.

        Skipping an unreadable entry is right for a caller that wants the records: one bad file
        is not a reason to refuse the rest. It is not enough for a caller that has to say the
        directory has been dealt with, because the entry it could not read may be exactly the
        record recovery exists for. So the entries are named rather than counted, and the caller
        can come back to those and only those."""
        if not self._dir.exists():
            return [], []
        out: List[FinalizationRecord] = []
        unreadable: List[Path] = []
        try:
            paths = sorted(self._dir.glob("finalization-*.json"))
        except OSError:
            # The directory itself could not be listed, so there is no entry to name and no
            # record to trust. Reported as an unreadable directory rather than an empty one.
            raise
        for path in paths:
            record = self._read_path(path)
            if record is None:
                unreadable.append(path)
            else:
                out.append(record)
        return out, unreadable

    def _read_path(self, path: Path) -> Optional[FinalizationRecord]:
        """One record from one entry, or ``None`` when that entry is not a record.

        **The name and the contents have to agree.** A record is written to the file its own
        ``finalization_id`` names, so an entry whose contents claim a different id is not a
        record this store wrote. Trusted, it is resolved and then *written back under the id it
        claims*: one malformed file overwrites a valid `FINALIZED` record belonging to another
        episode with its own fail-closed resolution, destroys that evidence, and stays
        unresolved itself because its own file is never touched."""
        record = None
        try:
            record = _record_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except _UNREADABLE:
            return None
        if self._path(record.finalization_id) != path:
            return None
        return record

    def recover(self) -> List[FinalizationRecord]:
        """Restart-recovery, fail-closed **and concurrency-safe**. For every record left
        ``SEALED`` or ``PENDING`` **by a process that is no longer alive** (a crash
        mid-finalize), rewrite it to ``FAILED`` with the fail-closed ``finalize_error`` verdict
        and return the resolved records. A record whose ``owner_pid`` is still alive belongs to a
        **live** concurrent episode sharing this (deliberately shared) store, so it is left
        untouched — recovery never clobbers an in-flight worker. The evaluator is **never**
        re-invoked — external judges/verifiers are not idempotent, so a mid-finalize crash is
        resolved to a safe zero rather than re-run. ``FINALIZED`` records are left as-is (they
        replay their stored evidence); ``FAILED`` is already terminal fail-closed.
        """
        return self._recover()[0]

    def _recover(self) -> Tuple[List[FinalizationRecord], List[Path]]:
        """:meth:`recover`, plus the entries the pass could not read. A write that fails is not
        caught here: a record this pass could not resolve is a record the next pass has to see
        again, so the failure belongs to the caller rather than to a flag."""
        records, unreadable = self._load_all()
        return self._resolve(records), unreadable

    def _resolve(self, records: List[FinalizationRecord]) -> List[FinalizationRecord]:
        resolved: List[FinalizationRecord] = []
        for record in records:
            if record.status in ("SEALED", "PENDING") and not _pid_alive(record.owner_pid):
                record.status = "FAILED"
                record.verdict = fail_closed_verdict(
                    (record.verdict or {}).get("confidence")
                )
                record.diagnostic = (
                    "recovered: crashed mid-finalize; resolved fail-closed "
                    "(evaluator not re-invoked)"
                )
                self.write(record)
                resolved.append(record)
        return resolved

    def _recover_paths(
        self, paths: Sequence[Path]
    ) -> Tuple[List[FinalizationRecord], List[Path]]:
        """Recovery over named entries only: the ones a previous pass could not read.

        A directory that has been scanned does not need scanning again for its own sake, and
        re-reading tens of thousands of known-good records because one file was corrupt is the
        cost this whole mechanism exists to remove. What can change is an entry that was not a
        record: a file half-written when its writer died, one a later process rewrote whole. Those
        come back here, and nothing else does."""
        records: List[FinalizationRecord] = []
        still: List[Path] = []
        for path in paths:
            if not path.exists():
                continue  # deleted since; there is nothing left to quarantine
            record = self._read_path(path)
            if record is None:
                still.append(path)
            else:
                records.append(record)
        return self._resolve(records), still

    def recover_once(self) -> List[FinalizationRecord]:
        """:meth:`recover` for the first caller in this process to ask about this directory, and
        nothing at all for every caller after it.

        Recovery is a **startup** question: which records did a process that is no longer here
        leave mid-finalize? The answer cannot change because this process opened another episode,
        so asking it again per episode buys nothing and costs a full read of the directory,
        which is shared across every session that runs without a trace path and therefore holds
        every record the machine has ever written. On a developer machine that is tens of
        thousands of files, so an episode that should open in milliseconds spends seconds reading
        JSON, and a suite that opens eighty episodes spends minutes of it. Asked once, the cost
        is paid once.

        What this gives up is small and named: a record another process abandons *while this one
        runs* is resolved by the next process to start rather than by this one's next episode.
        Recovery is for the run before this one, and a live owner's record is left alone.

        **A pass that finished counts, and one bad file does not undo it.** The directory is
        remembered after the scan, along with the entries the scan could not read. A later caller
        comes back to *those* and only those: an entry that was half-written when its writer died
        may be whole now, and one that is still not a record costs one open rather than a second
        read of every known-good record beside it. Remembering nothing because one file was
        corrupt is the O(machine history) behaviour this exists to remove, arrived at by a
        different route; and remembering the pass as complete would leave a record that may be
        exactly the dangling one unresolved for good.

        A write that fails is different and raises out of here having remembered nothing: the
        records it could not rewrite are known-good ones, and they have to be seen again.

        **Once per process means this process.** A forked child is another process: the record
        its parent left alone belongs to an owner that was alive when the parent looked and may
        not be now, so the child asks again. The cache is emptied in the child at fork, and keyed
        by pid as well, for a platform that cannot register a fork handler."""
        key = str(self._dir.resolve()) if self._dir.exists() else str(self._dir)
        with _RECOVERED_LOCK:
            quarantined = _recovered().get(key)
        if quarantined is not None:
            if not quarantined:
                return []
            resolved, still = self._recover_paths(quarantined)
            with _RECOVERED_LOCK:
                _recovered()[key] = tuple(still)
            return resolved
        # Outside the lock: this reads a directory that can hold every record the machine has
        # written, and holding a process-wide lock across it would stop episodes opening against
        # every other store. Two callers racing the same directory both scan, which costs a
        # second read and nothing else, because resolving a record is idempotent.
        resolved, unreadable = self._recover()
        with _RECOVERED_LOCK:
            _recovered()[key] = tuple(unreadable)
        if unreadable:
            warnings.warn(
                f"{len(unreadable)} entries in {self._dir} are not finalization records and were "
                f"skipped: {', '.join(sorted(path.name for path in unreadable)[:5])}"
                f"{' ...' if len(unreadable) > 5 else ''}. If one of them was a record left "
                "mid-finalize by a crashed run, its episode has not been resolved fail-closed. "
                "They are re-read on the next recovery in this process; the rest of the store is "
                "not.",
                RuntimeWarning,
                stacklevel=2,
            )
        return resolved


#: The store directories already recovered **by this process**, keyed by resolved path, each
#: mapped to the entries in it that were not records, alongside the pid that recovered them.
#: Recovery is a startup question asked once per directory (see
#: :meth:`FinalizationStore.recover_once`); the lock is here because episodes are opened
#: concurrently.
_RECOVERED: Dict[str, Tuple[Path, ...]] = {}
_RECOVERED_PID: Optional[int] = None
_RECOVERED_LOCK = threading.Lock()


def _recovered() -> Dict[str, Tuple[Path, ...]]:
    """The map for the process asking, from store directory to the entries in it that were not
    records. Call with :data:`_RECOVERED_LOCK` held.

    A fork handler empties this in the child, and the pid is checked here as well because a
    platform without ``register_at_fork`` would otherwise let a child answer out of its parent's
    memory of a directory the child has never looked at."""
    global _RECOVERED, _RECOVERED_PID
    pid = os.getpid()
    if _RECOVERED_PID != pid:
        _RECOVERED = {}
        _RECOVERED_PID = pid
    return _RECOVERED


def _forget_recovered() -> None:
    """A forked child inherits the parent's answers and, worse, whatever state the lock was in
    when the fork happened. Both are replaced."""
    global _RECOVERED, _RECOVERED_PID, _RECOVERED_LOCK
    _RECOVERED = {}
    _RECOVERED_PID = os.getpid()
    _RECOVERED_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_recovered)


def _has_unwritable_number(value: Any) -> bool:
    """Is there a ``NaN`` or an infinity anywhere in here?

    Recursive, because :meth:`FinalizationStore.write` is: it serialises with
    ``allow_nan=False``, which walks the whole structure, so a check that stops at the first level
    passes records the writer will never accept."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(
            _has_unwritable_number(key) or _has_unwritable_number(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_unwritable_number(item) for item in value)
    return False


def _pid_alive(pid: Optional[int]) -> bool:
    """Is process ``pid`` currently alive? ``None`` (a legacy/hand-written record with no owner)
    counts as **not alive** — it belongs to no tracked live episode, so recovery may resolve it.
    Uses the standard ``os.kill(pid, 0)`` liveness probe: success or ``EPERM`` (exists, other
    user) ⇒ alive; ``ESRCH`` ⇒ dead. On a platform without signal semantics, err on the side of
    *alive* so a live worker is never clobbered."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _sessions_cache_root() -> Path:
    """The shared, zero-config fallback root for durable records when no trace path is set.
    A module function so tests can redirect it without env-vars or config.

    Whoever creates a directory publishes it, so this creates the root through
    :func:`_mkdir_durable` rather than leaving it to whichever caller happens to write first.
    That keeps the contract local: the root exists and its entry is on disk when this returns,
    with no appeal to a later write that a recovery-only startup never performs."""
    try:
        base = Path(os.path.expanduser("~")) / ".cache" / "shogym" / "sessions"
        _mkdir_durable(base)
        return base
    except OSError:
        return Path(tempfile.gettempdir()) / "shogym-sessions"


def _mkdir_durable(directory: Path) -> None:
    """Create ``directory``, and make every entry on the path to it survive a host crash.

    Syncing a directory persists the entries *inside* it, never the entry that names it — that
    one lives in its parent. So each level has to be synced into the level above it, top down,
    and the walk runs the whole way up rather than stopping at the levels this call created.

    **An existing level is no evidence that anyone synced it.** ``mkdir`` makes a level visible
    immediately and durable never, and this store is deliberately shared between processes, so
    stopping at what this call created is a first-use race: one writer creates the chain and
    then stalls or dies before its own sync loop runs, and a writer starting a moment later
    finds every level present, concludes there is nothing left to publish, and returns success
    over a store whose directory entry is still only in the page cache. A crash then takes the
    whole store, with every write having reported success. The same hole opens without any
    concurrency at all whenever some other program created the path just before this one used
    it — the race is the sharpest instance, not the only one.

    Walking to the root costs one directory fsync per level. Not free, but a directory with
    nothing dirty behind it costs a syscall rather than a disk flush — measured on this path at
    roughly a twentieth of the record's own file sync, which the write pays anyway — and it is
    bounded by the depth of the path and charged only on the handful of writes a lifecycle
    makes. What it buys is that the publishing is unconditional: this call syncs every entry
    leading to ``directory``, regardless of who created it or how far they got, instead of
    inferring from a level's existence that someone else already did. The syncs themselves stay
    best-effort — see :func:`_fsync_dir`, where a filesystem that refuses one must not fail the
    write it was protecting — so what is guaranteed is that nothing on the path goes
    unattempted, not that a hostile filesystem was talked into it. Top down, so a crash
    part-way through leaves a durable prefix rather than a durable entry inside an ancestor
    that is still missing."""
    directory.mkdir(parents=True, exist_ok=True)
    holders: List[Path] = []
    level = directory
    while level.parent != level:  # the filesystem root names no entry above itself
        holders.append(level.parent)
        level = level.parent
    for holder in reversed(holders):
        _fsync_dir(holder)


def _fsync_dir(directory: Path) -> None:
    """Fsync a directory entry. Best-effort: not every platform or filesystem permits it, and a
    refusal must never fail the write it was protecting."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


#: What a decoded record may fail as. A store file is arbitrary bytes from the filesystem, not a
#: value this process produced: it can be valid JSON that is not an object (`AttributeError` on
#: the mapping), an object missing the fields a record is made of (`TypeError` from the
#: constructor), or an object whose fields are the wrong shape, which raises wherever they are
#: first used rather than where they were read. All of it means one thing to a reader, which is
#: that this file is not a record.
_UNREADABLE = (ValueError, TypeError, AttributeError, KeyError, IndexError, OSError)


def _record_from_dict(data: Dict[str, Any]) -> FinalizationRecord:
    """One record from one decoded file, refusing anything that is not the shape of a record.

    Validated here rather than caught later. A `verdict` that is a list decodes without complaint
    and raises on `.get` three frames away, in recovery, where the failure looks like a bug in
    recovery rather than a file that was never a record; and the caller that has to survive it is
    an episode opening against a directory shared with every session the machine has ever run."""
    if not isinstance(data, dict):
        raise TypeError(f"a finalization record is a JSON object, not {type(data).__name__}")
    for name in ("session_id", "finalization_id", "status", "source"):
        if not isinstance(data.get(name), str):
            raise TypeError(f"a record's {name} is a string, not {type(data.get(name)).__name__}")
    if data.get("status") not in ("SEALED", "PENDING", "FINALIZED", "FAILED"):
        raise ValueError(f"a record's status is not one this reader knows: {data.get('status')!r}")
    for name in ("verdict", "provenance"):
        value = data.get(name)
        if value is not None and not isinstance(value, dict):
            raise TypeError(
                f"a record's {name} is a JSON object or absent, not {type(value).__name__}"
            )
    for name in ("diagnostic",):
        value = data.get(name)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"a record's {name} is a string or absent, not {type(value).__name__}")
    # Every field recovery *operates* on, not only the ones it reads back. `owner_pid` decides
    # whether a record belongs to a live process, and it reaches `os.kill`: a string raises there,
    # three frames from here, in a caller that can only suppress it; and `True` is an `int` to
    # Python, so a boolean is pid 1, which is alive on every machine, so the record is left
    # untouched forever and the pass is recorded as having dealt with it.
    owner = data.get("owner_pid")
    if owner is not None:
        if isinstance(owner, bool) or not isinstance(owner, int):
            raise TypeError(
                f"a record's owner_pid is an int or absent, not {type(owner).__name__}"
            )
        # A *pid*, not any integer. `os.kill` reads 0 as "every process in my group" and a
        # negative number as "the group named by its absolute value", so those are not liveness
        # questions at all; and a number past the platform's pid range raises `OverflowError`
        # out of a probe whose only caller can suppress it, which leaves the directory unread
        # and every later episode reading the whole store again.
        if not 0 < owner <= _MAX_PID:
            raise ValueError(f"a record's owner_pid is not a pid a process could have: {owner}")
    version = data.get("schema_version", RECORD_SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError(f"a record's schema_version is an int, not {type(version).__name__}")
    if version != RECORD_SCHEMA_VERSION:
        # A record written by a schema this reader does not have is not a record this reader may
        # rewrite: the fields it does not know are dropped on decode, so resolving it would
        # publish a v-this record over v-that content while keeping the version that said so.
        # Quarantined instead, which leaves it for a reader that has the migration.
        raise ValueError(
            f"a record's schema_version is {version}, and this reader knows "
            f"{RECORD_SCHEMA_VERSION}"
        )
    # `json.loads` accepts `NaN` and `Infinity`, and the writer refuses them (`allow_nan=False`),
    # so a record carrying one can be read and never written back: recovery rewrites it, the
    # write raises, and the same failure repeats on every pass. Refused where it is read, and
    # refused the way the writer refuses it: all the way down.
    for name in ("verdict", "provenance"):
        if _has_unwritable_number(data.get(name)):
            raise ValueError(
                f"a record's {name} carries a number JSON can be read with and not written back"
            )
    digest = data.get("args_digest")
    if digest is not None and not isinstance(digest, str):
        raise TypeError(
            f"a record's args_digest is a string or absent, not {type(digest).__name__}"
        )
    fields = {
        "session_id",
        "finalization_id",
        "status",
        "source",
        "schema_version",
        "args_digest",
        "verdict",
        "provenance",
        "diagnostic",
        "owner_pid",
    }
    return FinalizationRecord(**{k: v for k, v in data.items() if k in fields})
