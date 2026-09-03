"""What a port has to build to end an attempt under the durable stream, built once.

The stream's stand-in grade computes from the shape of a filing: a terminal call with something in
it scores one and an empty one scores nothing. Almost every terminal in this package takes no
arguments at all, so under the stand-in almost every attempt scores nothing whatever the agent
did, and a number like that cannot be published to an agent as a grade. An environment replaces
its half of the terminal to fix that, and the same five pieces go around its own scoring: a
write-once record keyed by the seal id, a token that names what was captured, the four Activities
a Worker registers, a mapping from a verdict onto the numbers the environment declared it
publishes, and a digest of what the environment is configured as. Those five are here.

What is not here is anything that decides what a filing is worth. A seal reads a live world, and
only the environment knows how to read one; a submission is what the agent left behind, and only
the environment knows which part of that is the answer key; a verdict's names are what a body
prints, and only the environment can say which of its numbers an agent may be told. Those four
arrive as an environment's own answers in :class:`Grader`.

One of them is easy to get wrong and is worth naming. Under the stream a seal can arrive for an
attempt whose world this process never opened or has already let go, and a zero there is a
published grade for a world nobody read. :func:`routed` refuses instead, and it
refuses the two halves separately, because a seal that reached the wrong process and a seal that
reached the right one too late are different facts about a run.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import re
import secrets
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from temporalio import activity
from temporalio.exceptions import ApplicationError

from shogym.serve.protocol_v2 import (
    BlobRef,
    FilesystemBlobStore,
    blob_ref,
    length_prefixed,
)
from shogym.serve.protocol_v2.blobs import create_directory, flush_directory
from shogym.serve.protocol_v2.kernel.activities import (
    GRADE_ATTEMPT,
    SEAL_ATTEMPT,
    generate_payload_bundle_activity,
    verify_blobs_activity,
)
from shogym.serve.protocol_v2.kernel.messages import (
    GradeAttemptInput,
    GradeAttemptResult,
    SealAttemptInput,
    SealAttemptResult,
)
from shogym.serve.protocol_v2.policy import (
    GradeIdentity,
    PolicyViolation,
    check_grade,
    check_grade_result,
)

#: Which world each attempt was worked in, asked at the moment a seal has to read one. A generation
#: serves each task in a world of its own and these Activities are registered before the first of
#: those exists, so the pairing is resolved late rather than fixed at registration.
WorldRoute = Callable[[str], Optional[Tuple[Any, str]]]

#: The two records one seal id holds: what was captured off the world, and what grading it
#: produced. Each is written once, so a retry of either Activity returns the first call's bytes.
SEALED = "sealed"
GRADED = "graded"

#: The media type these records are installed under. The submission a digest is taken over and the
#: verdict a score is read out of are both JSON a port wrote.
_RECORD_MEDIA_TYPE = "application/json"

_SEAL_ID = re.compile(r"[0-9a-f]{64}")


def refusal(message: str, kind: str) -> ApplicationError:
    """A failure a port will not retry: nothing about it changes on a second attempt."""
    return ApplicationError(message, type=kind, non_retryable=True)


def encoded(value: Any) -> str:
    """One record as this module's own text: sorted keys, no spacing, and floats allowed.

    Not the protocol's canonical encoding, which holds whole numbers only because the wire has
    none, and a verdict has fractions in it. These bytes never travel as a wire record: they are
    what a capture is stored as, what a token is taken over, and what an evidence reference
    resolves to, so what they have to be is the same on every reading.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def private_root(name: str) -> Path:
    """Where this machine keeps one environment's sealed evidence.

    Beside the cache a served world is handed rather than inside it. What a record here holds is
    the state a score was taken from, and the directory an episode is given is one the code an
    agent runs can reach.
    """
    declared = os.environ.get("SHOGYM_CACHE")
    root = Path(declared).expanduser().resolve() if declared else Path.home() / ".cache" / "shogym"
    return root.parent / f"{root.name}-private" / name


def seal_id_path(seal_id: str) -> str:
    """Return one seal id, checked before anything makes a path or a key out of it."""
    if _SEAL_ID.fullmatch(seal_id) is None:
        raise ValueError("a seal id is 64 lower-case hexadecimal characters")
    return seal_id


class CaptureStore:
    """The captures one machine holds: one record per seal id per name, written once.

    Write once is the whole of it. An Activity retry finds the record the first call installed and
    returns its bytes, so the submission digest does not move and the acknowledgement a model may
    already have read stays true; and one filing is not graded twice, because the first verdict is
    what every later call reads.
    """

    def held(self, seal_id: str, name: str) -> Optional[Dict[str, Any]]:
        """The record under this key, or ``None`` if this machine holds none."""
        raise NotImplementedError

    def once(
        self, seal_id: str, name: str, build: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return the record under this key, building and installing one if there is none."""
        raise NotImplementedError


class MemoryCaptures(CaptureStore):
    """The captures of worlds that live in the serving process, held in that process.

    A world that is this process is gone when the process is, so writing its capture down buys
    nothing: a Worker that replaced the one which sealed has neither a world to read nor a record
    to grade, and it refuses either way. What the store is for is the retry that reaches the same
    process, which is every retry that could succeed.
    """

    def __init__(self) -> None:
        self._records: Dict[Tuple[str, str], str] = {}
        # Re-entrant, because a build runs under it: reading a live world is the work being
        # serialized, and two calls that both found a key empty must not both read one.
        self._lock = threading.RLock()

    def held(self, seal_id: str, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            held = self._records.get((seal_id_path(seal_id), name))
        return None if held is None else json.loads(held)

    def once(
        self, seal_id: str, name: str, build: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        key = (seal_id_path(seal_id), name)
        with self._lock:
            if key not in self._records:
                self._records[key] = encoded(build())
            return json.loads(self._records[key])


@dataclass(frozen=True)
class DirectoryCaptures(CaptureStore):
    """The captures of worlds that outlive this process, one directory per seal id.

    A container world is stopped by the host and what it left is on the disk, so a Worker that
    replaced the one which sealed can still grade what was sealed. That is worth a record, and the
    record is worth a claim: capturing such a world is a container being stopped and a tree being
    copied, and two of those over one directory would each undo the other. The claim is a file
    lock, so the machine takes it back from a process that died holding it.

    A record is installed by linking a staged file into place, so two writers that both found the
    key empty do not both replace it: the first name wins and the second is dropped.
    """

    root: Path

    def directory(self, seal_id: str) -> Path:
        """Where one seal's records live."""
        return self.root / seal_id_path(seal_id)

    def held(self, seal_id: str, name: str) -> Optional[Dict[str, Any]]:
        return self._read(self.directory(seal_id) / f"{name}.json")

    def once(
        self, seal_id: str, name: str, build: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        held = self.held(seal_id, name)
        if held is not None:
            return held
        with self.claim(seal_id):
            held = self.held(seal_id, name)
            if held is None:
                self._write(self.directory(seal_id) / f"{name}.json", build())
                held = self.held(seal_id, name)
        if held is None:  # pragma: no cover - the write above installed one or raised
            raise refusal(f"the {name} record for {seal_id} was not installed", "NotInstalled")
        return held

    @contextmanager
    def claim(self, seal_id: str) -> Iterator[None]:
        """Hold this seal id while its world is being captured or its capture graded."""
        directory = self.directory(seal_id)
        create_directory(directory)
        descriptor = os.open(str(directory / "claim.lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _read(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            held = json.loads(path.read_bytes().decode("utf-8"))
        except OSError:
            return None
        if not isinstance(held, dict):
            raise ValueError(f"{path} does not hold a record this store wrote")
        return held

    def _write(self, path: Path, value: Dict[str, Any]) -> None:
        if path.is_file():
            return
        create_directory(path.parent)
        descriptor, staged = tempfile.mkstemp(
            dir=str(path.parent), suffix=f".{secrets.token_hex(8)}.partial"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded(value).encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(staged, path)
            except FileExistsError:
                # Somebody installed this name while this call was writing its own copy. Theirs is
                # the record, and a second one under one key is what this store does not do.
                pass
        finally:
            Path(staged).unlink(missing_ok=True)
            # After both names, and never before the link: what has to survive a machine coming
            # back is the record under its own name and the staged name gone.
            flush_directory(path.parent)


def recovery_token(seal_id: str, capture: Dict[str, Any]) -> str:
    """Name one capture by its key and its own bytes, so a record that changed is caught."""
    return sha256(
        length_prefixed(b"shogym-seal-capture-1")
        + length_prefixed(seal_id.encode("utf-8"))
        + length_prefixed(encoded(capture).encode("utf-8"))
    ).hexdigest()


def check_recovery_token(seal_id: str, capture: Dict[str, Any], presented: str) -> None:
    """Refuse a capture that is not the one the seal this grade answers recovered."""
    if recovery_token(seal_id, capture) != presented:
        raise refusal(
            f"the capture under {seal_id} is not the one this seal recovered",
            "RecoveryTokenMismatch",
        )


def routed(
    route: WorldRoute, attempt_id: str, read: Callable[[str], Optional[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Read the world one attempt was worked in, or refuse rather than score an absent one.

    The two refusals are separate on purpose. A seal that resolves to no world at all reached a
    process that never served this attempt, which is what a resumed generation sees for every
    attempt; a seal that resolves to a world whose session is gone reached the right process after
    it let the world go. Both would otherwise be an honest zero, and a zero is a grade: it says
    the agent did nothing, and what happened is that nobody looked.
    """
    world = route(attempt_id)
    if world is None:
        raise refusal(
            f"attempt {attempt_id} was worked in no world this process opened, so there is "
            "nothing here to seal",
            "NoWorldHere",
        )
    _env, session_id = world
    state = read(session_id)
    if state is None:
        raise refusal(
            f"the world attempt {attempt_id} was worked in has been let go, so what it ended as "
            "cannot be read: a score for a world nobody read is not a score",
            "NoSessionHere",
        )
    return state


def installed(blob_root: Optional[str], text: str) -> BlobRef:
    """Put bytes a result names where an event may cite them, and return the name.

    A generation with no store gets the reference anyway: it is the hash of the same bytes, and
    what changes without a store is that nothing can be read back under it.
    """
    if blob_root is None:
        return blob_ref(text, _RECORD_MEDIA_TYPE)
    return FilesystemBlobStore(Path(blob_root)).put(
        text.encode("utf-8"), media_type=_RECORD_MEDIA_TYPE
    )


def graded_result(
    request: GradeAttemptInput, *, verdict: Dict[str, Any], grade: GradeIdentity
) -> GradeAttemptResult:
    """Project one verdict onto the numbers this grader declared, or refuse what it cannot.

    The projection is the roster and never the verdict. A grader returns whatever it computes and a
    body prints names, so what crosses is the headline the environment named as its score and the
    components it declared before the run; everything else stays in the evidence this installs,
    which a harness can resolve and a renderer cannot.

    The resolution is part of the projection. An environment declares how fine each of its numbers
    is, and a fraction it computed by division arrives with as many digits as a double holds: the
    ones under the resolution it declared say nothing about the measure and are a field in a body
    an agent reads. So what crosses is each number at the resolution its own declaration names, and
    the division it came out of stays in the verdict this installs as evidence.

    A field that is not a finite number is dropped where it is a component and refused where it is
    the score, and so is a score outside the unit interval. The kernel would refuse both a moment
    later, as an unusable batch that ends the attempt, and a grader that came back with a headline
    it cannot publish should say so here, where the message names the environment and the field.
    """
    try:
        check_grade(grade)
    except PolicyViolation as violation:
        raise refusal(str(violation), "GradeRosterRefused") from violation
    measured = _finite(verdict.get(grade.score_component))
    if measured is None:
        raise refusal(
            f"{grade.grader_id} scores by {grade.score_component} and this verdict carries "
            f"{verdict.get(grade.score_component)!r} under that name",
            "ScoreIsNotANumber",
        )
    score = round(measured, grade.score_places)
    components: Dict[str, float] = {}
    for number in grade.public_components:
        value = _finite(verdict.get(number.name))
        if value is not None:
            components[number.name] = round(value, number.places)
    try:
        check_grade_result(score=score, components=components, grade=grade)
    except PolicyViolation as violation:
        raise refusal(str(violation), "GradeOutsideItsDeclaration") from violation
    return GradeAttemptResult(
        attempt_id=request.attempt_id,
        seal_id=request.seal_id,
        score=score,
        decode_state=str(verdict["decode_state"]),
        evidence=installed(request.blob_root, encoded(verdict)),
        grade=grade,
        public_components=components,
    )


def _finite(value: Any) -> Optional[float]:
    """One verdict field as a number a body could print, or ``None`` if it is not one."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def configuration_digest(fields: Dict[str, Any]) -> str:
    """Return the digest of what an environment says it is configured as.

    The fields are the environment's own. What this adds is that they are hashed the same way
    everywhere, so a generation resumed against a differently configured world is refused rather
    than scored against a rule nobody committed to.
    """
    return sha256(encoded(fields).encode("utf-8")).hexdigest()


def terminal_activities(
    *,
    seal: Callable[[SealAttemptInput], SealAttemptResult],
    grade: Callable[[GradeAttemptInput], GradeAttemptResult],
) -> List[Any]:
    """Return the four Activities a Worker serving this environment registers.

    Two of them are the port's and two are the kernel's, unchanged. The list is the whole of the
    registration, so an environment that returned only its own two would leave a run unable to
    build a payload or verify a reference, which has nothing to do with grading.

    The port's two run off the event loop. The process that serves the model is the process that
    holds the world, so a scorer reading a database or walking a world state would otherwise stall
    the transport it is being asked through.
    """

    @activity.defn(name=SEAL_ATTEMPT)
    async def seal_attempt(request: SealAttemptInput) -> SealAttemptResult:
        """Capture what this attempt filed, or return what its key already holds."""
        return await asyncio.to_thread(seal, request)

    @activity.defn(name=GRADE_ATTEMPT)
    async def grade_attempt(request: GradeAttemptInput) -> GradeAttemptResult:
        """Score the capture, or return the score this key was already given."""
        return await asyncio.to_thread(grade, request)

    return [seal_attempt, grade_attempt, generate_payload_bundle_activity, verify_blobs_activity]


@dataclass(frozen=True)
class Grader:
    """What one environment answers when a generation asks how its attempts end.

    ``read`` turns a live world into the state a grade is taken from, and answers ``None`` for a
    session this process does not hold. ``submission`` is which part of that state is what the
    agent filed: it is the one field handed to the renderer that builds a payload, so an
    environment that put its answer key in it would be publishing the key through the
    acknowledgement's own evidence. ``score`` is what the environment makes of the state, under the
    environment's own names, and ``grade`` is which of those names reach an agent.
    """

    version: str
    grade: GradeIdentity
    read: Callable[[str], Optional[Dict[str, Any]]]
    submission: Callable[[Dict[str, Any]], Dict[str, Any]]
    score: Callable[[Dict[str, Any]], Dict[str, Any]]


def terminal_for(
    grader: Grader, route: WorldRoute, *, store: Optional[CaptureStore] = None
) -> Tuple[str, List[Any]]:
    """The version this environment declares, and the Activities that end an attempt in it.

    The seal reads the world once, under the seal id, and everything after that reads the record: a
    retry of the terminal finds the state the first call captured and returns the same submission
    text and the same token, so the digest does not move. The grade reads that state, checks it is
    the one this seal recovered, and scores it once under a record of its own.

    ``store`` is where those records go, and the default is this process. An environment whose
    world outlives the process that served it passes a :class:`DirectoryCaptures` rooted outside
    every mount a world is given.
    """
    captures = store if store is not None else MemoryCaptures()

    def seal(request: SealAttemptInput) -> SealAttemptResult:
        if request.canonicalization_version != grader.version:
            raise refusal(
                f"this world is captured as {grader.version!r} and the generation was started as "
                f"{request.canonicalization_version!r}",
                "CanonicalizationMismatch",
            )
        state = captures.once(
            request.seal_id, SEALED, lambda: routed(route, request.attempt_id, grader.read)
        )
        text = encoded(
            {
                "canonicalization_version": grader.version,
                "submission": grader.submission(state),
            }
        )
        return SealAttemptResult(
            attempt_id=request.attempt_id,
            seal_id=request.seal_id,
            canonicalization_version=grader.version,
            canonical_submission_text=text,
            canonical_submission=installed(request.blob_root, text),
            environment_recovery_token=recovery_token(request.seal_id, state),
        )

    def grade(request: GradeAttemptInput) -> GradeAttemptResult:
        state = captures.held(request.seal_id, SEALED)
        if state is None:
            raise refusal(
                f"this machine holds nothing sealed under {request.seal_id}, so there is nothing "
                "here to grade",
                "NoSealedCapture",
            )
        check_recovery_token(request.seal_id, state, request.environment_recovery_token)
        verdict = captures.once(request.seal_id, GRADED, lambda: grader.score(state))
        return graded_result(request, verdict=verdict, grade=grader.grade)

    return grader.version, terminal_activities(seal=seal, grade=grade)


__all__ = [
    "CaptureStore",
    "DirectoryCaptures",
    "GRADED",
    "Grader",
    "MemoryCaptures",
    "SEALED",
    "WorldRoute",
    "check_recovery_token",
    "configuration_digest",
    "encoded",
    "graded_result",
    "installed",
    "private_root",
    "recovery_token",
    "refusal",
    "routed",
    "seal_id_path",
    "terminal_activities",
    "terminal_for",
]
