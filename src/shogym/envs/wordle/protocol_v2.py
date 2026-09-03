"""How a wordle attempt ends under the durable stream, and what it is worth.

The stream's stand-in grade computes from the shape of a filing: a terminal call with something
in it scores one and an empty one scores nothing. Wordle's terminal is the empty abort, so every
attempt under the stand-in scores nothing whether the word was found on the first guess or never
guessed at all. That number cannot be published to an agent as a grade, because it is not one.

So this port replaces the environment's half of the terminal. The seal captures what was played
out of the server that ran the game, the grade scores those guesses against the target the task
loaded, and the score is the game's own result: the word was found within the allowed guesses or
it was not.

The evidence is written down, under the seal id, on the machine that took it. The seal and the
grade are two Activities and a Worker can be replaced between them, so a capture that lived only
in the process that made it would leave a sealed attempt with nothing left to score: the stream
would hold a successful seal, and the grade behind it would say there was no play. What is kept
is the target and the words played against it, beside the verdict taken from them, in a directory
of this port's own. An Activity retry finds the capture the first call made and returns its
bytes, so the digest does not move and the acknowledgement a model may already have read stays
true, and a Worker that replaced the one which sealed grades the play that was sealed.

The store is not the submission. What the agent filed is the guesses, and that is what the
canonical text and its digest cover; the target sits beside them, where the grade reaches it and
a payload cannot. It is machine-local for the same reason a stopped world is: a generation
resumed on another machine has neither, which is the answer any environment gives for evidence
that is not where it is being asked for.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from temporalio import activity
from temporalio.exceptions import ApplicationError

from shogym.envs.wordle import mcp_server
from shogym.envs.wordle.utils import score_guess
from shogym.serve.protocol_v2 import (
    BlobRef,
    FilesystemBlobStore,
    blob_ref,
    canonical_json,
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
from shogym.serve.protocol_v2.policy import GradeIdentity, PublishedNumber

#: Which world each attempt was played in, asked at the moment a seal has to capture one. A
#: generation serves each task in a world of its own and these Activities are registered before
#: the first of those exists, so the pairing is resolved late rather than fixed here.
WorldRoute = Callable[[str], Optional[Tuple[Any, str]]]

#: The version this environment declares for the submission its terminal captures. It names what
#: goes into the canonical text and what does not, so a run recorded under it can be compared
#: with another run recorded under it and with nothing else.
CANONICALIZATION_VERSION = "shogym.wordle.1"

MAX_GUESSES = mcp_server.MAX_GUESSES

#: What this environment's grader is. The score is the game's own result rather than a fact about
#: the shape of a filing, which is what lets a generation over this environment publish it, and
#: the roster is what a body may name beside that score: the guesses the game spent, and nothing
#: else this port computes. The roster carries the domain as well as the name, because a play
#: spends whole guesses and there are six of them: a number declared without its range and its
#: resolution is a field wide enough to write text in, whatever the name on it says. A guess is
#: whole, which is what a roster entry says by declaring no decimal places at all. The score is
#: whole by the same declaration and for the same reason: this game is won or it is not, so a
#: score arriving with digits after the point is not a finer answer to that question, it is
#: something else written in the field the answer goes in.
WORDLE_GRADE = GradeIdentity(
    grader_id="wordle-grade-v2",
    grader_version="1",
    stand_in=False,
    score_component="check_answer",
    score_places=0,
    public_components=(
        PublishedNumber(name="guesses_used", minimum=0, maximum=MAX_GUESSES),
    ),
)

_SEAL_ID = re.compile(r"[0-9a-f]{64}")

#: Where to keep sealed plays instead of the default. It is read from the environment rather than
#: passed, because the process that seals is not always the process that composed the run: a
#: served episode is spawned, and a caller that wants these records somewhere of its own has to be
#: able to say so across that boundary.
SEALS_ROOT_ENV_VAR = "SHOGYM_WORDLE_SEALS"


def seal_store_root() -> Path:
    """Where this machine keeps the plays it has sealed.

    Beside the port's cache rather than inside it, because what a record here holds is the target,
    and the directory a served world is given is not the directory an answer sits in.
    """
    declared = os.environ.get(SEALS_ROOT_ENV_VAR)
    if declared:
        return Path(declared).expanduser().resolve()
    base = os.environ.get("SHOGYM_CACHE")
    root = Path(base).expanduser().resolve() if base else Path.home() / ".cache" / "shogym"
    return root.parent / f"{root.name}-private" / "wordle"


@dataclass(frozen=True)
class SealStore:
    """The plays this machine has sealed, one directory per seal id, and their verdicts.

    Two records under one key and neither ever rewritten. The seal writes the play once and every
    later call under that key reads what the first one wrote, which is what makes an Activity
    retry return the same submission rather than capturing a second one, and what lets a Worker
    that replaced the one which sealed grade the play that was sealed rather than reporting that
    nothing was. The verdict is written the same way, so a retry of a grade returns the first
    verdict and never a second grading of the same evidence.

    A record is installed by linking a staged file into place, so two writers that both found the
    key empty do not both replace it: the first name wins and the second is dropped. A rename
    would let the second overwrite the first, and the whole value of the record is that it is the
    one the acknowledgement was built from.
    """

    root: Path

    def directory(self, seal_id: str) -> Path:
        """Where one seal's evidence lives. The key is checked before it becomes a path."""
        if _SEAL_ID.fullmatch(seal_id) is None:
            raise ValueError("a seal id is 64 lower-case hexadecimal characters")
        return self.root / seal_id

    def sealed(self, seal_id: str) -> Optional[Dict[str, Any]]:
        """The play captured under this key, or nothing if this machine captured none."""
        return self._read(self.directory(seal_id) / "played.json")

    def seal(self, seal_id: str, build: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Return the play under this key, capturing and installing one if there is none."""
        return self._once(self.directory(seal_id) / "played.json", build)

    def grade(self, seal_id: str, build: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Return the verdict under this key, taking and installing one if there is none."""
        return self._once(self.directory(seal_id) / "graded.json", build)

    def _once(self, path: Path, build: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        held = self._read(path)
        if held is not None:
            return held
        self._write(path, build())
        installed = self._read(path)
        if installed is None:  # pragma: no cover - the write above installed one or raised
            raise _refusal(f"the record under {path.name} was not installed", "NotInstalled")
        return installed

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
                handle.write(_encoded(value).encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(staged, path)
            except FileExistsError:
                # Somebody installed this name while this call was writing its own copy. Theirs
                # is the record, and a second one under one key is what this store does not do.
                pass
        finally:
            Path(staged).unlink(missing_ok=True)
            # After both names, and never before the link: what has to survive a machine coming
            # back is the record under its own name and the staged name gone.
            flush_directory(path.parent)


def wordle_terminal(
    route: WorldRoute, *, seals: Optional[Path] = None
) -> Tuple[str, List[Any]]:
    """The version this environment declares, and the Activities that end an attempt in it.

    The two the stream calls for a seal are this port's. The two behind them, which build the
    candidate payloads and verify blob references, are the kernel's and are unchanged: what is
    replaced here is how an attempt is captured and scored and nothing about how it is served.
    """
    store = SealStore(Path(seals) if seals is not None else seal_store_root())

    @activity.defn(name=SEAL_ATTEMPT)
    async def seal_wordle_attempt(request: SealAttemptInput) -> SealAttemptResult:
        """Capture what was played under ``seal_id``, or return what that key already holds."""
        return _seal(route, store, request)

    @activity.defn(name=GRADE_ATTEMPT)
    async def grade_wordle_attempt(request: GradeAttemptInput) -> GradeAttemptResult:
        """Score the captured guesses, or return the score this key was already given."""
        return _grade(store, request)

    return CANONICALIZATION_VERSION, [
        seal_wordle_attempt,
        grade_wordle_attempt,
        generate_payload_bundle_activity,
        verify_blobs_activity,
    ]


def configuration_digest(words: List[str], task_split: str) -> str:
    """Return the digest of what this environment is configured as.

    The word list decides what could be asked and the guess budget decides what could be done
    about it, so a generation resumed against a different one of those is a different
    measurement and is refused rather than scored against a key nobody drew for it.
    """
    return sha256(
        canonical_json(
            {
                "canonicalization_version": CANONICALIZATION_VERSION,
                "task_split": task_split,
                "max_guesses": MAX_GUESSES,
                "words_sha256": sha256("\n".join(words).encode("utf-8")).hexdigest(),
                "word_count": len(words),
            }
        )
    ).hexdigest()


def _refusal(message: str, kind: str) -> ApplicationError:
    """A failure this port will not retry: nothing about it changes on a second attempt."""
    return ApplicationError(message, type=kind, non_retryable=True)


def _seal(route: WorldRoute, store: SealStore, request: SealAttemptInput) -> SealAttemptResult:
    """Return the sealed play under this key, capturing it first if nothing has.

    The submission is the guesses and only the guesses. The target is captured beside them and
    never inside them: the canonical text is what the digest covers and what the payload
    renderer is handed, so the answer being in it would be an answer oracle reaching the agent
    through the acknowledgement's own evidence.
    """
    existing = store.sealed(request.seal_id)
    if existing is None:
        world = route(request.attempt_id)
        if world is None:
            raise _refusal(
                f"attempt {request.attempt_id} was played in no world this process opened, so "
                "there is nothing here to seal",
                "NoWorldHere",
            )
        _env, session_id = world
        played = mcp_server.played(session_id)
        if played is None:
            raise _refusal(
                f"the session attempt {request.attempt_id} was played in is not open, so what "
                "it played cannot be captured",
                "NoSessionHere",
            )
        existing = store.seal(request.seal_id, lambda: dict(played))
    submission = canonical_json(
        {
            "canonicalization_version": CANONICALIZATION_VERSION,
            "guesses": list(existing["entries"])[:MAX_GUESSES],
        }
    ).decode("utf-8")
    return SealAttemptResult(
        attempt_id=request.attempt_id,
        seal_id=request.seal_id,
        canonicalization_version=CANONICALIZATION_VERSION,
        canonical_submission_text=submission,
        canonical_submission=_installed(request.blob_root, submission),
        environment_recovery_token=_recovery_token(request.seal_id, existing),
    )


def _grade(store: SealStore, request: GradeAttemptInput) -> GradeAttemptResult:
    """Score the captured play against the target the task loaded.

    The guesses are read from the capture rather than from anything live, so the play that is
    scored and the play that was submitted are one play. What the server told the agent about
    each guess is not consulted: validity, the mask and the solved state are derived here from
    the recorded word and the target, which is what makes the grade the environment's rather
    than a summary of results the world reported.
    """
    record = store.sealed(request.seal_id)
    if record is None:
        raise _refusal(
            f"this machine holds no play sealed under {request.seal_id}, so there is nothing "
            "here to grade",
            "NoSealedPlay",
        )
    if _recovery_token(request.seal_id, record) != request.environment_recovery_token:
        raise _refusal(
            f"the capture under {request.seal_id} is not the one this seal recovered",
            "RecoveryTokenMismatch",
        )
    verdict = store.grade(request.seal_id, lambda: _score(record))
    return GradeAttemptResult(
        attempt_id=request.attempt_id,
        seal_id=request.seal_id,
        score=float(verdict["check_answer"]),
        decode_state=str(verdict["decode_state"]),
        evidence=_installed(request.blob_root, _encoded(verdict)),
        grade=WORDLE_GRADE,
        public_components={"guesses_used": float(verdict["guesses_used"])},
    )


def _score(record: Dict[str, Any]) -> Dict[str, Any]:
    """Score one captured play. The word was found within the allowed guesses, or it was not.

    The headline is the game's own result rather than a measure of how close the last guess came.
    Wordle either ends with the word or it does not, and a run of these is counted by how often
    it did, so the number a generation publishes is that one and the progress measures stay in
    the verdict beside it.
    """
    target = str(record["target"]).lower()
    entries = list(record["entries"])[:MAX_GUESSES]
    solved = False
    best_green = 0
    used = 0
    for word in entries:
        used += 1
        if not (isinstance(word, str) and len(word) == 5 and word.isalpha()):
            continue
        mask = score_guess(word.lower(), target)
        best_green = max(best_green, mask.count("G"))
        if mask == "GGGGG":
            solved = True
            break
    return {
        "check_answer": 1.0 if solved else 0.0,
        "partial_credit": 1.0 if solved else best_green / 5.0,
        "guesses_used": float(used),
        "decode_state": "decoded" if used else "ambiguous_zero",
    }


def _encoded(value: Dict[str, Any]) -> str:
    """One verdict as this port's own text: sorted keys, no spacing, and floats allowed.

    The canonical wire encoding takes integers only, and a verdict is fractions. This is not that
    encoding and is not used where that one is required: it names the bytes of a record this port
    wrote, which is what the evidence reference resolves to.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _recovery_token(seal_id: str, record: Dict[str, Any]) -> str:
    """Name one capture by its own bytes, so a record that changed under its key is caught."""
    return sha256(
        canonical_json({"seal_id": seal_id, "capture": record})
    ).hexdigest()


def _installed(blob_root: Optional[str], text: str) -> BlobRef:
    """Return the reference that names ``text``, with those bytes where the name resolves."""
    if blob_root is None:
        return blob_ref(text)
    return FilesystemBlobStore(Path(blob_root)).put(text.encode("utf-8"), media_type="text/plain")


__all__ = [
    "CANONICALIZATION_VERSION",
    "WORDLE_GRADE",
    "SealStore",
    "WorldRoute",
    "configuration_digest",
    "seal_store_root",
    "wordle_terminal",
]
