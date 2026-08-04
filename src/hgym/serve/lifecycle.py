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
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

# The evidence-schema version. Stamped on every :class:`TerminalEvidence`, the durable
# record, and the trace ``terminal`` event so a reader can tell which envelope it is parsing.
EVIDENCE_SCHEMA_VERSION = 1

# The private durable-record filename schema version (independent of the evidence schema).
RECORD_SCHEMA_VERSION = 1

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
    ``~/.cache/hgym/sessions``). No DB, no credentials, no env-vars.
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
        - **Without one:** a stable, zero-config fallback root (``~/.cache/hgym/sessions``, or
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
        if not self._dir.exists():
            return []
        out: List[FinalizationRecord] = []
        for path in sorted(self._dir.glob("finalization-*.json")):
            try:
                out.append(
                    _record_from_dict(json.loads(path.read_text(encoding="utf-8")))
                )
            except (ValueError, OSError):
                continue
        return out

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
        resolved: List[FinalizationRecord] = []
        for record in self.load_all():
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
        base = Path(os.path.expanduser("~")) / ".cache" / "hgym" / "sessions"
        _mkdir_durable(base)
        return base
    except OSError:
        return Path(tempfile.gettempdir()) / "hgym-sessions"


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


def _record_from_dict(data: Dict[str, Any]) -> FinalizationRecord:
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
