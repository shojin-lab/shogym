"""The ORCA-bench judge: its configuration, its preflight, and the parsing of its verdict.

Every task ships its own verifier (``tests/check_prediction.py``, byte-identical across all 755
tasks of the pinned revision), so the port fetches no code from the upstream repo. The verifier
is an LLM-as-a-judge: it scores the agent's ``/app/report.md`` against one rubric per plausible
root cause and writes two files under the task's verifier log directory:

  - ``reward.json``, a flat ``{name: number}`` payload: always ``reward`` (the rubric score
    mean, normalized to [0, 1]), plus ``rca_accuracy`` and ``hallucinate_any`` when they are
    defined.
  - ``reward-details.json``, the full judge transcript (prompt, raw response, per-rubric
    verdicts), **or**, when the judge itself failed, ``{reward, error, traceback}``.

**The ambiguity this module refuses to inherit.** The verifier's exception path writes
``{"reward": 0.0}``, byte-identical to a report the judge read and scored zero. Upstream that
failure is easy to hit: the shipped default judge model (:data:`UPSTREAM_DEFAULT_JUDGE_MODEL`)
no longer exists, so an unconfigured run silently records a whole benchmark of honest-looking
zeros. :func:`parse_verdict` therefore trusts a verdict only when ``reward.json`` carries
``rca_accuracy``, the key that exists exactly when at least one rubric was actually scored, and
grades everything else as an explicit judge error. :func:`preflight_judge` is the other half:
it refuses the dead default model and a missing key up front, hle-style, instead of discovering
both from a run of zeros.
"""

from __future__ import annotations

import hashlib
import json
import errno
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional

# Upstream's shipped default judge model. It was retired by the provider: a run that leaves it in
# place fails on every task with a 400 and records `{"reward": 0.0}`. Named here so the port can
# refuse it by name rather than by symptom.
UPSTREAM_DEFAULT_JUDGE_MODEL = "openai-gpt-5.4"

# The port's default. A current reasoning model, at the effort upstream's verifier defaults to
# (`--effort high`). The judge reads a long report against several rubrics, so a cheaper tier is
# a scoring change, not a cost tweak.
DEFAULT_JUDGE_MODEL = "gpt-5.6-terra"
DEFAULT_JUDGE_EFFORT = "high"
JUDGE_EFFORTS = ("low", "medium", "high")

# Where the task's verifier writes its two files (Harbor's convention, and the paths
# `tests/test.sh` bakes in).
VERIFIER_LOG_DIR = "/logs/verifier"
REWARD_FILENAME = "reward.json"
REWARD_DETAILS_FILENAME = "reward-details.json"

# A non-secret stand-in credential for an explicitly keyless endpoint. The verifier builds its
# client as `AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))`
# unconditionally, and the SDK refuses to construct on a missing or empty key ("Missing
# credentials") before it ever reaches the endpoint. So a local OpenAI-compatible server
# (Ollama/vLLM/LM Studio), which needs no real key, still needs *some* non-empty one, and this is
# what it gets. The same placeholder the hle env's judge uses, for the same reason.
PLACEHOLDER_API_KEY = "sk-no-key-required"


# The environment variables a judge run depends on. Read together, once, so that a process which
# changes one of them mid-episode cannot leave the run equipped with one endpoint and the score
# audited against another.
_JUDGE_ENVIRON_KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL")


def environment_snapshot(environ: Optional[Mapping[str, str]] = None) -> Mapping[str, str]:
    """One reading of the environment a judge run depends on, frozen.

    The preflight decision, the environment the verifier is given, and the endpoint recorded with
    the score all have to describe the same run. Derived from three separate reads of a mutable
    mapping they need not: the value can change in between, and the disagreement is invisible
    afterwards. So the reading happens once, here, and everything downstream takes this.

    Each variable is read exactly once, and an existing snapshot is passed through: a proxy over a
    dict that left with the frame that made it cannot change, so re-reading it would be a no-op,
    and passing it through keeps "one reading of the environment per episode" literally true
    rather than approximately."""
    if isinstance(environ, MappingProxyType):
        return environ
    source = os.environ if environ is None else environ
    values: Dict[str, str] = {}
    for key in _JUDGE_ENVIRON_KEYS:
        value = source.get(key)
        if value:
            values[key] = value
    return MappingProxyType(values)


class JudgePreflightError(RuntimeError):
    """The judge is not configured to be able to grade (dead model, or no key)."""


@dataclass(frozen=True)
class JudgeConfig:
    """How the task's own verifier should be invoked.

    ``base_url`` points the judge at an OpenAI-compatible endpoint (the verifier reads
    ``OPENAI_BASE_URL``); setting it opts out of the key requirement, since such an endpoint may
    be keyless. Mirrors how the hle env's ``judge_base_url`` opts out of its preflight, including
    the placeholder credential such an endpoint needs (see :data:`PLACEHOLDER_API_KEY`).

    An endpoint given through the process environment (``OPENAI_BASE_URL``) counts exactly the
    same as one passed here: :func:`preflight_judge` and :meth:`environment` read it through the
    one :meth:`endpoint` accessor, so they cannot disagree about whether a run is keyless.
    """

    model: str = DEFAULT_JUDGE_MODEL
    effort: str = DEFAULT_JUDGE_EFFORT
    base_url: Optional[str] = None

    def cli_args(self) -> list[str]:
        """The verifier arguments this config implies (``check_prediction.py --model … --effort …``)."""
        return ["--model", self.model, "--effort", self.effort]

    def endpoint(self, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
        """The OpenAI-compatible endpoint this run will use, if it is not the provider default."""
        source = os.environ if environ is None else environ
        return self.base_url or source.get("OPENAI_BASE_URL") or None

    def endpoint_id(self, environ: Optional[Mapping[str, str]] = None) -> str:
        """A publishable identity for the endpoint: ``default`` or ``custom:<digest>``.

        A custom endpoint's URL can name internal infrastructure, and it travels with every score
        into a trace someone else may read, so the URL itself is never published. The digest is
        stable, so two runs against the same endpoint match and two runs against different ones do
        not, which is all an audit needs from it."""
        endpoint = self.endpoint(environ)
        if not endpoint:
            return "default"
        return f"custom:{hashlib.sha256(endpoint.encode('utf-8')).hexdigest()[:12]}"

    def resolve(self, environ: Optional[Mapping[str, str]] = None) -> "ResolvedJudge":
        """Freeze this configuration against a concrete environment, once.

        Resolution is a moment, not a property, and it is **one** moment: ``OPENAI_BASE_URL`` is
        read from the process environment, and a process is free to change it while an episode
        runs. So the endpoint is
        resolved when the verifier is equipped with it, and the resulting record is what the score
        is audited against. Re-resolving at grading time would name whatever the environment said
        then, which need not be the endpoint that scored the report.
        """
        # One reading for both derivations: the endpoint recorded with the score and the endpoint
        # the verifier is handed are the same fact, and two reads of a live environment can
        # disagree about it.
        snapshot = environment_snapshot(environ)
        return ResolvedJudge(
            model=self.model,
            effort=self.effort,
            endpoint_id=self.endpoint_id(snapshot),
            # Read-only: this is what the verifier is equipped with *and* what the score is
            # audited against, so it is not a scratch dict for a later caller to edit.
            environment=MappingProxyType(self.environment(snapshot)),
        )

    def environment(self, environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        """The env vars the verifier needs, drawn from the caller's environment.

        A real ``OPENAI_API_KEY`` always wins. When there is none but an explicit endpoint was
        named, the verifier gets :data:`PLACEHOLDER_API_KEY` instead: that is what makes the
        keyless path actually reach the endpoint rather than dying in the SDK constructor. With
        no endpoint there is nowhere keyless to point at, so nothing is substituted and the
        preflight refuses the run.

        The **value** of a real key is never logged or echoed by this module. It is passed
        through to the verifier process and nowhere else."""
        source = os.environ if environ is None else environ
        out: Dict[str, str] = {}
        base_url = self.endpoint(source)
        api_key = source.get("OPENAI_API_KEY") or (PLACEHOLDER_API_KEY if base_url else "")
        if api_key:
            out["OPENAI_API_KEY"] = api_key
        if base_url:
            out["OPENAI_BASE_URL"] = base_url
        return out


@dataclass(frozen=True)
class ResolvedJudge:
    """A judge configuration resolved against one concrete environment, at one moment.

    Produced by :meth:`JudgeConfig.resolve` when an episode equips its verifier, and carried
    through grading. ``environment`` is what the verifier process is given and ``endpoint_id`` is
    what the score is audited against, so the two cannot drift apart: they were computed from the
    same reading.
    """

    model: str
    effort: str
    endpoint_id: str
    environment: Mapping[str, str]


def validate_judge_config(config: JudgeConfig) -> None:
    """Raise unless the model/effort are ones the verifier can be run with. Keyless and offline.

    Checked at env construction, so a configuration that could only ever produce zeros is
    rejected before a run starts. The retired upstream default is refused **by name**: every call
    to it 400s and the verifier records reward 0.0, which is byte-identical to a wrong answer.
    """
    if config.effort not in JUDGE_EFFORTS:
        raise JudgePreflightError(
            f"judge_effort {config.effort!r} is not one of {list(JUDGE_EFFORTS)}"
        )
    if not config.model.strip():
        raise JudgePreflightError("judge_model must be a non-empty model id")
    if config.model == UPSTREAM_DEFAULT_JUDGE_MODEL:
        raise JudgePreflightError(
            f"{UPSTREAM_DEFAULT_JUDGE_MODEL!r} is ORCA-bench's shipped default judge model and "
            "the provider retired it: every call 400s and the verifier records reward 0.0, which "
            f"is byte-identical to a wrong answer. Pass judge_model=… (default "
            f"{DEFAULT_JUDGE_MODEL!r})."
        )


def preflight_judge(
    config: JudgeConfig, *, environ: Optional[Mapping[str, str]] = None
) -> None:
    """Raise unless this judge configuration can actually grade. Offline, no network.

    :func:`validate_judge_config` plus the key requirement. Split in two because the halves are
    checked at different moments: the model is a configuration error and fails at construction,
    while the key is only needed when an episode is actually served, which is what keeps
    ``shogym.make("orca_bench")``, the tool manifest, and ``describe()`` keyless (the same split
    the hle env draws).

    An explicit OpenAI-compatible endpoint opts out, whether it came from ``judge_base_url`` or
    from ``OPENAI_BASE_URL`` in the environment: :meth:`JudgeConfig.endpoint` is the single
    reading of that question, so what this accepts is exactly what
    :meth:`JudgeConfig.environment` can equip the verifier for.

    Never reads or reports the key's value, only whether it is set.
    """
    validate_judge_config(config)
    source = os.environ if environ is None else environ
    if not config.endpoint(source) and not source.get("OPENAI_API_KEY"):
        raise JudgePreflightError(
            "ORCA-bench's verifier is an LLM judge and needs OPENAI_API_KEY to grade a report, "
            "but it is not set. Set OPENAI_API_KEY, or pass judge_base_url=… (or set "
            "OPENAI_BASE_URL) for a keyless OpenAI-compatible endpoint. Without it every task "
            "records reward 0.0 with no way to tell a failed judge from a wrong answer."
        )


# ----- the submission itself -----

# The largest report this port will hand to the judge. The verifier inlines the whole report into
# one request alongside the rubrics, and the recorded prompts for a real task are about 70 KB in
# total: ~65 KB of rubrics and a report of 2 to 5 KB. This bound is roughly fifty times the
# largest report seen in the port's own runs and still leaves the request comfortably inside a
# current model's context. Past it the request itself is what fails, and the cause of that failure
# is the agent's artifact, so it is a failed submission rather than a judge error.
MAX_REPORT_BYTES = 256 * 1024


@dataclass(frozen=True)
class CapturedReport:
    """The agent's submission as it stood at the seal, and the verdict on whether it can be graded.

    ``data`` is the bytes that were read, or ``None`` when there was nothing gradeable to read.
    ``problem`` is empty when those bytes can be graded, and otherwise says why they cannot, in
    terms of the agent's own artifact.
    """

    source: str
    data: Optional[bytes] = None
    problem: str = ""


def capture_report(report: Path) -> CapturedReport:
    """Read and validate the agent's report **once**, and hold the bytes that were read.

    **The classification principle, and the reason this function exists at all.** The upstream
    verifier wraps everything in one ``except``: reading the agent's ``/app/report.md`` and calling
    the judge model both land in the same ``{"reward": 0.0, "error", "traceback"}`` shape. This
    port grades a judge failure as an explicit ``judge_error`` and filters those out of results,
    which is right for grading infrastructure and catastrophic for anything the agent controls: an
    agent that never writes a report, or writes a directory, or writes 40 MB, would convert a
    failed attempt into no attempt at all, and would score better by doing so.

    So the class is decided by **cause**, not by the error string, and it is decided here, before
    the verifier runs:

    - the cause is the agent's own submission (absent, not a regular file, unreadable, not
      decodable, larger than the judge can take): an ordinary **failed attempt**, verified,
      reward 0, counted like any other zero;
    - the cause is grading infrastructure (the judge model, the endpoint, the key, a crash inside
      the judge, task data the agent cannot touch): a **judge error**, excluded.

    Everything the agent can reach inside that blanket handler is a report-read failure, which is
    what this checks. The other paths in it (the task's ``expected.json``, its rubrics, the judge
    call, parsing the judge's reply) read files the agent cannot write or talk to a service it
    cannot influence, so they stay judge errors.

    **Why capture rather than inspect.** Sealing an episode stops the agent's MCP calls; it does
    not stop processes the agent already started inside its container. Validating the live path
    and then letting the verifier reopen it grades whatever is there the second time, so a watcher
    started before the seal can delete the report in between and recreate the exclusion, or swap
    the bytes so the score belongs to something that did not exist when the episode was sealed. So
    this reads once, through a single descriptor: the checks are made against the bytes that were
    read, and those bytes are what the verifier is given. There is no second look at the path to
    disagree with the first.

    An **empty regular file is valid**: that is the required submission for a control task, and
    the emptiness check must not eat the one answer whose emptiness is the point.
    """
    source = str(report)
    try:
        # O_NOFOLLOW so a symlink is refused rather than resolved, and one descriptor for the
        # stat and the read so the path cannot be swapped between them.
        #
        # O_NONBLOCK because this path is agent-controlled and some file types make `open` itself
        # wait: a FIFO opened read-only blocks until a writer appears, so `mkfifo /app/report.md`
        # and then submitting would park the terminal call inside this syscall, with no verdict
        # and a teardown that drains the evaluator waiting on it too. With the flag the open
        # returns at once and the `S_ISREG` check below refuses the FIFO as an ordinary failed
        # submission. It is specified to have no effect on regular files, which is the case that
        # has to keep working; `test_a_regular_file_reads_whole_under_the_non_blocking_flag` pins
        # that a report at the size bound still reads whole and unaltered.
        descriptor = os.open(report, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return CapturedReport(source=source, problem=f"no report was written at {report}")
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            return CapturedReport(
                source=source,
                problem=f"the report at {report} is a symlink, not a regular file",
            )
        return CapturedReport(
            source=source,
            problem=f"the report at {report} could not be read: {type(exc).__name__}",
        )
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            return CapturedReport(
                source=source, problem=f"the report at {report} is not a regular file"
            )
        if status.st_size > MAX_REPORT_BYTES:
            return CapturedReport(
                source=source,
                problem=(
                    f"the report at {report} is {status.st_size} bytes, over the "
                    f"{MAX_REPORT_BYTES}-byte limit the judge can be given in one request"
                ),
            )
        # One byte past the bound, so a file that grew between the fstat and the read is caught
        # here rather than silently truncated into a grade.
        data = _read_exactly(descriptor, MAX_REPORT_BYTES + 1)
    except OSError as exc:
        return CapturedReport(
            source=source,
            problem=f"the report at {report} could not be read: {type(exc).__name__}",
        )
    finally:
        os.close(descriptor)
    if len(data) > MAX_REPORT_BYTES:
        return CapturedReport(
            source=source,
            problem=(
                f"the report at {report} is over the {MAX_REPORT_BYTES}-byte limit the judge can "
                "be given in one request"
            ),
        )
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return CapturedReport(
            source=source,
            problem=f"the report at {report} could not be read as UTF-8 text: UnicodeDecodeError",
        )
    return CapturedReport(source=source, data=data)


def _read_exactly(descriptor: int, limit: int) -> bytes:
    """Read up to ``limit`` bytes from an open descriptor."""
    chunks: List[bytes] = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def inspect_report(report: Path) -> Optional[str]:
    """Why the report at this path cannot be graded, or ``None`` when it can.

    A thin reading of :func:`capture_report`, kept for callers that only want the classification.
    Anything that then goes on to *grade* must use the capture itself: the bytes it validated are
    the bytes that have to be graded (see that function's note on why)."""
    return capture_report(report).problem or None


# ----- verdict parsing -----


# How much of a rejected payload the private diagnostic quotes. Enough to recognize it, bounded
# because a verifier that went wrong can write a very large file.
_PAYLOAD_SUMMARY_CHARS = 300


def _summarize(payload: Mapping[str, Any]) -> str:
    """A bounded rendering of a payload this module is refusing, for the private diagnostic."""
    text = repr(dict(payload))
    return text if len(text) <= _PAYLOAD_SUMMARY_CHARS else text[:_PAYLOAD_SUMMARY_CHARS] + "..."


def _valid_reward(value: Any) -> Optional[float]:
    """The reward if it is one: a finite number in [0, 1]. ``None`` means "not a grade".

    Upstream documents the reward as a normalized rubric score and derives it as ``depth / 3``, so
    anything outside that range is the verifier reporting something other than a grade for this
    report. Written as "finite, and within the range" rather than "not below and not above": every
    comparison with NaN is false, so the negative form waves NaN through into a published result.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _valid_flag(value: Any) -> Optional[bool]:
    """A 0/1 rollup as a bool. ``None`` means "not one".

    Harbor requires the rollups to be flat numbers, so the verifier writes ``int(bool)``. Exactly
    0 and 1 are accepted, and by value rather than by truthiness: ``rca_accuracy: 2`` is not a
    strict all-causes pass, it is a verifier that is not writing this contract."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


@dataclass(frozen=True)
class OrcaVerdict:
    """One task's graded outcome, as the port reads it back off the verifier's two files.

    ``rca_accuracy`` is the benchmark's headline per-task boolean: did the report name **every**
    listed plausible root cause (strict all-causes). ``hallucinate_any`` is ``None`` where
    upstream leaves it undefined: control tasks (nothing to hallucinate against) and empty
    reports. ``judge_error`` marks a verdict that is not a grade at all; its ``reward`` is 0.0 and
    must never be averaged in as an honest zero.

    ``judge_model`` / ``judge_effort`` are what the verifier **recorded** as having scored the
    report, which is not always what was configured, and are empty when the judge failed before
    writing them. :func:`judge_provenance` is what fills that gap from the configuration.
    """

    reward: float
    rca_accuracy: Optional[bool]
    hallucinate_any: Optional[bool]
    mode: str
    rca_depth: Optional[float]
    judge_error: bool
    judge_error_message: str = ""
    judge_model: str = ""
    judge_effort: str = ""
    # Why the agent's own submission could not be graded, when that is what happened. A grade,
    # not an exclusion: see :func:`inspect_report`.
    submission_error: str = ""

    def success(self, *, is_control: bool) -> bool:
        """The port's per-task binary outcome.

        For an incident task this is ``rca_accuracy``, the metric the leaderboard ranks. For a
        **control** task it is a full reward instead: the verifier synthesizes a single
        ``(no_incident)`` rubric whose ``feature_flag_match`` is hard-coded ``False``, so
        ``rca_accuracy`` is structurally 0 on every control task however well the agent did.
        The leaderboard's own subsets never mix the two, and neither does this.

        A judge error is never a success.
        """
        if self.judge_error:
            return False
        if is_control:
            return self.reward >= 1.0
        return bool(self.rca_accuracy)


def parse_verdict(
    reward: Optional[Mapping[str, Any]],
    details: Optional[Mapping[str, Any]] = None,
    *,
    submission_error: str = "",
) -> OrcaVerdict:
    """Parse the verifier's ``reward.json`` (+ optional ``reward-details.json``) into a verdict.

    ``submission_error`` is what :func:`inspect_report` found before the verifier ran, and it wins
    over anything in the payload: the verifier's blanket handler reports an unreadable report and
    a dead judge model identically, so the class has to be decided by cause, upstream of it.

    Pure. A verdict counts as a real grade only when ``reward.json`` carries ``rca_accuracy``:
    the verifier emits that key exactly when at least one rubric was scored, so its absence means
    the judge never produced a judgement, whether it raised (the details file then carries the
    error) or returned nothing scoreable. Everything else is a judge error with reward 0.0.
    """
    if submission_error:
        # A graded zero: the attempt happened, the artifact it produced cannot be scored, and that
        # is the agent's own doing. Never a judge error, or an agent could opt out of being scored
        # by not writing a report.
        return OrcaVerdict(
            reward=0.0,
            rca_accuracy=False,
            hallucinate_any=False,
            mode=str((details or {}).get("mode", "")),
            rca_depth=None,
            judge_error=False,
            submission_error=submission_error,
        )
    detail_error = _detail_error(details)
    if detail_error:
        return _judge_error(detail_error, details)
    if reward is None:
        return _judge_error("the verifier wrote no reward.json", details)
    graded = _valid_reward(reward.get("reward"))
    if graded is None:
        return _judge_error(
            "reward.json carries no finite reward in [0, 1], which is the only thing this "
            f"benchmark's verifier grades on: {_summarize(reward)}",
            details,
        )
    if "rca_accuracy" not in reward:
        return _judge_error(
            "reward.json carries no rca_accuracy, so no rubric was scored: the judge failed "
            f"without saying so ({_summarize(reward)})",
            details,
        )
    rca_accuracy = _valid_flag(reward.get("rca_accuracy"))
    if rca_accuracy is None:
        return _judge_error(
            "reward.json carries an rca_accuracy that is not the 0/1 rollup the verifier writes, "
            f"so nothing here says a rubric was scored: {_summarize(reward)}",
            details,
        )
    hallucinate_any: Optional[bool] = None
    if "hallucinate_any" in reward:
        hallucinate_any = _valid_flag(reward.get("hallucinate_any"))
        if hallucinate_any is None:
            return _judge_error(
                "reward.json carries a hallucinate_any that is not the 0/1 rollup the verifier "
                f"writes: {_summarize(reward)}",
                details,
            )
    return OrcaVerdict(
        reward=graded,
        rca_accuracy=rca_accuracy,
        hallucinate_any=hallucinate_any,
        mode=str((details or {}).get("mode", "")),
        rca_depth=_rca_depth(details),
        judge_error=False,
        judge_model=str((details or {}).get("model", "") or ""),
        judge_effort=str((details or {}).get("reasoning_effort", "") or ""),
    )


def judge_provenance(verdict: OrcaVerdict, judge: ResolvedJudge) -> Dict[str, str]:
    """Which scoring function produced this score, in publishable form.

    Changing the judge model or its effort changes the scoring function, so a score that does not
    say which one ran cannot be compared with another score, or re-checked later when a model id
    starts meaning something different. The verifier's own record wins where it exists (it is what
    actually scored the report); the resolved configuration fills in for a judge that failed
    before writing anything. The endpoint comes from that same resolution rather than from the
    environment as it stands now, and travels as an identity, never as a URL (see
    :meth:`JudgeConfig.endpoint_id`).
    """
    return {
        "judge_model": verdict.judge_model or judge.model,
        "judge_effort": verdict.judge_effort or judge.effort,
        "judge_endpoint": judge.endpoint_id,
    }


def read_verdict(verifier_dir: Path) -> OrcaVerdict:
    """Read a verifier log directory's two files and parse them. Missing files are judge errors."""
    return parse_verdict(
        _read_json(verifier_dir / REWARD_FILENAME),
        _read_json(verifier_dir / REWARD_DETAILS_FILENAME),
    )


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a JSON object, or ``None`` when it is absent, unreadable, or not an object."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _detail_error(details: Optional[Mapping[str, Any]]) -> str:
    """The verifier's own error string, when ``reward-details.json`` is its exception shape."""
    if not details:
        return ""
    error = details.get("error")
    return str(error) if error else ""


def _judge_error(message: str, details: Optional[Mapping[str, Any]]) -> OrcaVerdict:
    return OrcaVerdict(
        reward=0.0,
        rca_accuracy=None,
        hallucinate_any=None,
        mode=str((details or {}).get("mode", "")),
        rca_depth=None,
        judge_error=True,
        judge_error_message=message,
    )


def _rca_depth(details: Optional[Mapping[str, Any]]) -> Optional[float]:
    """The mean per-rubric score (0-3) the reward is derived from, when the details carry it."""
    if not details:
        return None
    nested = details.get("nested")
    if not isinstance(nested, Mapping):
        return None
    scores = [
        rubric.get("score")
        for rubric in (nested.get("rubrics") or [])
        if isinstance(rubric, Mapping) and isinstance(rubric.get("score"), (int, float))
    ]
    if not scores:
        return None
    return sum(float(s) for s in scores) / len(scores)  # type: ignore[arg-type]


__all__ = [
    "DEFAULT_JUDGE_EFFORT",
    "MAX_REPORT_BYTES",
    "CapturedReport",
    "capture_report",
    "environment_snapshot",
    "inspect_report",
    "DEFAULT_JUDGE_MODEL",
    "JUDGE_EFFORTS",
    "PLACEHOLDER_API_KEY",
    "JudgeConfig",
    "JudgePreflightError",
    "ResolvedJudge",
    "OrcaVerdict",
    "REWARD_DETAILS_FILENAME",
    "REWARD_FILENAME",
    "UPSTREAM_DEFAULT_JUDGE_MODEL",
    "VERIFIER_LOG_DIR",
    "judge_provenance",
    "parse_verdict",
    "preflight_judge",
    "read_verdict",
    "validate_judge_config",
]
