"""The scaffolding every ported terminal shares, tested where no environment is involved.

What a port authors is how to read its world and what its numbers mean. What it inherits is here:
a record written once under the seal id, a token that catches a record which changed under its
key, the four Activities a Worker registers, the projection from a verdict onto the numbers an
environment declared, and a refusal for a seal that arrived where its world is not. Each of those
is a rule the ports depend on and none of them is about any one environment, so they are checked
against fixtures rather than five times over against five upstreams.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.envs._grading import (  # noqa: E402
    SEALED,
    CaptureStore,
    DirectoryCaptures,
    MemoryCaptures,
    check_recovery_token,
    configuration_digest,
    graded_result,
    private_root,
    recovery_token,
    routed,
    seal_id_path,
    terminal_activities,
)
from shogym.serve.protocol_v2 import FilesystemBlobStore  # noqa: E402
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    GENERATE_PAYLOAD_BUNDLE,
    GRADE_ATTEMPT,
    SEAL_ATTEMPT,
    VERIFY_BLOBS,
)
from shogym.serve.protocol_v2.kernel.messages import GradeAttemptInput  # noqa: E402
from shogym.serve.protocol_v2.policy import GradeIdentity, PublishedNumber  # noqa: E402

SEAL_ID = "a" * 64
ATTEMPT = "b" * 32

# A grader that publishes one number beside its score, each over a range and at a resolution it
# declared: a reward measured to two decimals, and a whole count of steps.
FRACTION = GradeIdentity(
    grader_id="a-fixture-grade",
    grader_version="1",
    stand_in=False,
    score_component="reward",
    score_places=2,
    public_components=(PublishedNumber(name="steps_used", minimum=0, maximum=10),),
)


def grading(**over: Any) -> GradeAttemptInput:
    declared = dict(
        attempt_id=ATTEMPT,
        seal_id=SEAL_ID,
        submission_digest="c" * 64,
        canonical_submission_text="{}",
        environment_recovery_token="d" * 64,
    )
    declared.update(over)
    return GradeAttemptInput(**declared)  # type: ignore[arg-type]


@pytest.fixture(params=["memory", "directory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> CaptureStore:
    """Both implementations, held to one contract, because both ports depend on the contract."""
    if request.param == "memory":
        return MemoryCaptures()
    return DirectoryCaptures(tmp_path / "captures")


def test_a_capture_is_taken_once_and_every_later_call_reads_it(store: CaptureStore) -> None:
    """A retry finds the record the first call made rather than reading the world again.

    This is what makes the acknowledgement true. The digest a model may already have read is taken
    over the submission this record produces, so a second read of a world that has moved on would
    move the digest under a filing the stream has already committed to.
    """
    reads = []

    def build() -> Dict[str, Any]:
        reads.append(len(reads))
        return {"funds": len(reads)}

    assert store.held(SEAL_ID, SEALED) is None
    assert store.once(SEAL_ID, SEALED, build) == {"funds": 1}
    assert store.once(SEAL_ID, SEALED, build) == {"funds": 1}
    assert store.held(SEAL_ID, SEALED) == {"funds": 1}
    assert reads == [0]
    # And a caller that mutates what it read has not mutated the record.
    held = store.held(SEAL_ID, SEALED)
    assert held is not None
    held["funds"] = 99
    assert store.held(SEAL_ID, SEALED) == {"funds": 1}


def test_seals_that_arrive_together_still_read_the_world_once(store: CaptureStore) -> None:
    """The record is written once under contention and not only under one caller at a time.

    Retries of one Activity are what these stores exist for, and Temporal will start a retry while
    an earlier attempt is still running: a seal that timed out is retried against a Worker still
    holding the first call. Both calls find the key empty if the build is outside the lock, and both
    read a live world, which moves the digest under an acknowledgement the stream already committed
    to. The racers here all enter through one barrier so that they are inside the call together.
    """
    racers = 16
    entered = threading.Barrier(racers)
    builds: List[int] = []
    counting = threading.Lock()

    def build() -> Dict[str, Any]:
        with counting:
            builds.append(len(builds))
        # Wide enough that a second builder admitted alongside this one would overlap it.
        time.sleep(0.05)
        return {"funds": len(builds)}

    def race() -> Dict[str, Any]:
        entered.wait(timeout=20)
        return store.once(SEAL_ID, SEALED, build)

    with ThreadPoolExecutor(max_workers=racers) as pool:
        read = [future.result() for future in [pool.submit(race) for _ in range(racers)]]

    assert builds == [0]
    assert read == [{"funds": 1}] * racers
    assert store.held(SEAL_ID, SEALED) == {"funds": 1}


def test_a_seal_id_is_checked_before_anything_makes_a_key_out_of_it(
    store: CaptureStore,
) -> None:
    """The key reaches a filesystem in one of these stores, so it is a hash and not a path."""
    with pytest.raises(ValueError, match="64 lower-case hexadecimal"):
        store.held("../elsewhere", SEALED)
    with pytest.raises(ValueError, match="64 lower-case hexadecimal"):
        seal_id_path("A" * 64)


def test_a_record_that_changed_under_its_key_is_not_the_one_the_seal_recovered() -> None:
    """The token is over the capture's own bytes, so a substituted record is caught at the grade."""
    capture = {"funds": 1}
    check_recovery_token(SEAL_ID, capture, recovery_token(SEAL_ID, capture))
    with pytest.raises(ApplicationError, match="not the one this seal recovered"):
        check_recovery_token(SEAL_ID, {"funds": 2}, recovery_token(SEAL_ID, capture))
    # The key is in the token as well, so one capture cannot answer for another seal.
    with pytest.raises(ApplicationError, match="not the one this seal recovered"):
        check_recovery_token("e" * 64, capture, recovery_token(SEAL_ID, capture))


def test_a_seal_that_reached_no_world_refuses_rather_than_publishing_nothing() -> None:
    """The two ways a world is not here are two refusals, and neither of them is a zero.

    A zeroed verdict for a missing session is a grade published for a world nobody read, and it is
    indistinguishable from an agent that did nothing at all.
    """
    def read(session_id: str) -> Optional[Dict[str, Any]]:
        return {"funds": 1} if session_id == "here" else None

    assert routed(lambda _attempt: (None, "here"), ATTEMPT, read) == {"funds": 1}

    with pytest.raises(ApplicationError, match="no world this process opened"):
        routed(lambda _attempt: None, ATTEMPT, read)
    with pytest.raises(ApplicationError, match="has been let go"):
        routed(lambda _attempt: (None, "gone"), ATTEMPT, read)


def test_a_body_carries_the_numbers_the_environment_declared_and_no_others() -> None:
    """The projection is the roster, so a verdict's other fields stay in the evidence."""
    verdict = {
        "reward": 0.5,
        "steps_used": 3,
        "the_expected_answer": "crane",
        "decode_state": "decoded",
    }
    result = graded_result(grading(), verdict=verdict, grade=FRACTION)
    assert result.score == 0.5
    assert result.public_components == {"steps_used": 3.0}
    assert result.decode_state == "decoded"
    assert result.grade == FRACTION
    # The whole verdict is what the evidence reference names, which is where the rest of it lives.
    assert len(result.evidence.sha256) == 64


def test_a_number_crosses_at_the_resolution_its_grader_declared(tmp_path: Path) -> None:
    """Each number is published as fine as it was declared to be, and the arithmetic stays behind.

    Dividing two counts runs to as many digits as a double holds, and the digits under an
    environment's own resolution are not a finer measurement: they are room, in a field an agent
    reads. The projection is where that is settled, so a port hands over the verdict it computed
    and what crosses is each number at the resolution its declaration named.
    """
    verdict = {"reward": 2 / 3, "steps_used": 3.4, "decode_state": "decoded"}
    result = graded_result(grading(blob_root=str(tmp_path)), verdict=verdict, grade=FRACTION)
    assert result.score == 0.67
    assert result.public_components == {"steps_used": 3.0}
    # And the numbers the environment computed are in the evidence, where a harness reads them.
    held = json.loads(FilesystemBlobStore(tmp_path).read(result.evidence.sha256))
    assert held["reward"] == 2 / 3
    assert held["steps_used"] == 3.4


def test_a_verdict_the_grader_could_not_publish_is_refused_where_it_was_taken() -> None:
    """A headline outside the unit interval ends here rather than as an unusable batch.

    The kernel refuses it either way, and the attempt it refuses is ended as a seal the stream
    could not vouch for. A grader that came back with a number it cannot publish should say so
    where the message can name the environment and the field.
    """
    with pytest.raises(ApplicationError, match="unit interval"):
        graded_result(
            grading(),
            verdict={"reward": 20000.0, "decode_state": "decoded"},
            grade=FRACTION,
        )
    with pytest.raises(ApplicationError, match="scores by reward"):
        graded_result(grading(), verdict={"decode_state": "decoded"}, grade=FRACTION)
    with pytest.raises(ApplicationError, match="publishes steps_used between"):
        graded_result(
            grading(),
            verdict={"reward": 1.0, "steps_used": 40, "decode_state": "decoded"},
            grade=FRACTION,
        )
    # A component that is not a finite number is dropped; the score behind it still stands.
    dropped = graded_result(
        grading(),
        verdict={"reward": 1.0, "steps_used": float("nan"), "decode_state": "decoded"},
        grade=FRACTION,
    )
    assert dropped.public_components == {}


def test_a_worker_registers_the_kernels_two_activities_beside_the_ports() -> None:
    """The returned list is the whole registration, and forgetting the last two breaks a run.

    They have nothing to do with grading: one builds the candidate bodies and the other verifies
    the references a presentation carries, so an environment that returned only its own would fail
    a generation in a place that says nothing about what it was trying to replace.
    """
    activities = terminal_activities(
        seal=lambda request: None,  # type: ignore[arg-type,return-value]
        grade=lambda request: None,  # type: ignore[arg-type,return-value]
    )
    names = [getattr(fn, "__temporal_activity_definition").name for fn in activities]
    assert names == [SEAL_ATTEMPT, GRADE_ATTEMPT, GENERATE_PAYLOAD_BUNDLE, VERIFY_BLOBS]


def test_the_records_a_score_is_taken_from_are_not_under_a_directory_a_world_is_handed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Beside the cache and never inside it, because the cache is what a served world is given."""
    monkeypatch.setenv("SHOGYM_CACHE", "/tmp/a-cache-a-world-can-reach")
    root = private_root("an_environment")
    assert root.name == "an_environment"
    assert not root.is_relative_to(Path("/tmp/a-cache-a-world-can-reach"))


def test_two_configurations_that_differ_anywhere_are_two_measurements() -> None:
    """The digest is over the fields an environment says it is, hashed the same way everywhere."""
    fields: Dict[str, Any] = {"upstream_sha": "abc", "seeds": [1, 2]}
    assert configuration_digest(fields) == configuration_digest(dict(reversed(list(fields.items()))))
    assert configuration_digest(fields) != configuration_digest({**fields, "seeds": [1, 3]})


def test_a_directory_store_serializes_the_work_under_one_key_and_not_only_its_result(
    tmp_path: Path,
) -> None:
    """The claim is a file lock, so a machine takes it back from a process that died holding it."""
    store = DirectoryCaptures(tmp_path / "captures")
    with store.claim(SEAL_ID):
        assert (store.directory(SEAL_ID) / "claim.lock").is_file()
    held: Optional[Dict[str, Any]] = store.held(SEAL_ID, SEALED)
    assert held is None


#: One racer in the test below, as its own program. The claim this store takes is a file lock and
#: the record is installed by linking a name into place, and neither of those is about threads: what
#: they are for is the Worker that came back after another one died mid-capture, so the racers have
#: to be whole processes for the race to be the one the store is built against.
_RACER = """
import json
import sys
import time
from pathlib import Path

from shogym.envs._grading import SEALED, DirectoryCaptures

root, flags, builds, index, seal_id = sys.argv[1:6]


def build():
    (Path(builds) / index).write_text(index)
    time.sleep(0.2)
    return {"built_by": index}


(Path(flags) / index).write_text(index)
while not (Path(flags) / "go").exists():
    time.sleep(0.01)
sys.stdout.write(json.dumps(DirectoryCaptures(Path(root)).once(seal_id, SEALED, build)))
"""


def test_a_directory_store_installs_one_winner_when_whole_processes_race(tmp_path: Path) -> None:
    """Processes that all found the key empty leave one capture and read the same one back.

    A world this store is for outlives the process that served it, so the second racer is a Worker
    that replaced the first rather than a retry inside it. Two of them capturing at once would be
    one container stopped twice and one tree copied over itself, and two records under one key
    would mean the grade and the acknowledgement were taken over different bytes.
    """
    racers = 5
    flags, builds, root = tmp_path / "flags", tmp_path / "builds", tmp_path / "captures"
    flags.mkdir()
    builds.mkdir()
    running = [
        subprocess.Popen(
            [sys.executable, "-c", _RACER, str(root), str(flags), str(builds), str(index), SEAL_ID],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(racers)
    ]
    try:
        deadline = time.monotonic() + 20
        while len(list(flags.iterdir())) < racers and time.monotonic() < deadline:
            time.sleep(0.01)
        (flags / "go").write_text("go")
        finished = [process.communicate(timeout=20) for process in running]
    finally:
        for process in running:
            process.kill()

    for _out, err in finished:
        assert err == ""
    installed = [json.loads(out) for out, _err in finished]
    assert [path.name for path in builds.iterdir()] == [installed[0]["built_by"]]
    assert installed == [installed[0]] * racers
    assert DirectoryCaptures(root).held(SEAL_ID, SEALED) == installed[0]
