"""The bank: what exists before launch, and what is rendered after A seals.

Two things happen at two different times, and nothing here pretends otherwise.

BEFORE LAUNCH, materialized and hashed: every instance, meaning both surfaces and
both task texts, the convention commitments, the envelope template with its
registered slots, the committed filler stream, and the renderer configuration. A
bank names the generator, the master key and how many instances it holds, and
nothing else: WHICH instances is a deterministic consequence of those three and of
the admission rule, recomputed rather than listed. A bank that listed its passers
would be a bank whose population could be chosen, and choosing the population is the
whole of what admission is supposed to prevent.

AFTER A SEALS, in one act: the parser canonicalizes the filing, the three cells are
rendered, the envelope check runs, and the blobs are hashed. That is `render_fork`,
and it is the only place the three cells are made. What is committed once is the
record: a sequential retry or resume replays the published blobs and rerenders
nothing, because rerendering is how two branches of one fork come to differ, while
two callers arriving on one filing at the same instant can both render and one of
them discards its own bytes for the winner's.

THE BANK IS CONTROLLER-SIDE. It holds the master key, the drawn conventions and the
answer keys. It is never mounted in a lineage sandbox, and `Fork.agent_bytes` is the
only thing here shaped to cross the boundary.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from shogym.envs.receipts import streams
from shogym.envs.receipts.protocol import (
    Filing,
    Generator,
    Instance,
    NoFiling,
    RowOutcome,
    draw,
)
from shogym.envs.receipts.render import judge_cells
from shogym.envs.receipts.receipt_ast import (
    GRADED,
    ORACLE,
    PLACEBO,
    Envelope,
)

#: Bumped when anything about how a cell is built changes. It is recorded in every
#: bank, so a bank built by one renderer cannot be silently served by another.
RENDERER_CONFIGURATION = "receipts-render-v1"

#: The one label the settled gate set publishes. Kept here rather than imported so
#: that building a bank does not depend on the gate module at import time.
GATE_LABEL = "receipts-gates-v2"


@dataclass(frozen=True)
class Bank:
    """A generator, a key, and a count. Everything else about it is recomputed.

    There is no list of admitted ordinals here, and no summary of what they contain.
    The population is a function of these four fields, the admission rule and the
    current code, so it is derived wherever it is needed and compared against
    nothing. A field nobody writes is a field nobody can edit.
    """

    generator: str
    genre: str
    renderer: str
    master: bytes
    size: int

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("a bank of no instances is not a bank")
        if len(self.master) < 16:
            raise ValueError("a bank's master key is at least 16 bytes")


@dataclass(frozen=True)
class Population:
    """The instances a bank holds, recomputed by rerunning admission in order."""

    instances: tuple[Instance, ...]
    considered: int

    @property
    def ordinals(self) -> tuple[int, ...]:
        return tuple(i.ordinal for i in self.instances)

    @property
    def passing_fraction(self) -> float:
        return len(self.instances) / float(self.considered) if self.considered else 0.0

    def instance(self, ordinal: int) -> Instance:
        for held in self.instances:
            if held.ordinal == ordinal:
                return held
        raise KeyError(f"instance {ordinal} is not in this bank")


def is_hex(value: str) -> bool:
    """A digest is 64 hexadecimal characters. A 64-character string is not."""
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def commitment(master: bytes, ordinal: int, convention: Mapping[str, str]) -> str:
    """A binding, non-revealing commitment to one drawn convention."""
    ordered = [f"{k}={convention[k]}" for k in sorted(convention)]
    return streams.derive(master, streams.CONVENTION, "commit", ordinal, *ordered).hex()


def envelope_schema(envelope: Envelope) -> dict[str, Any]:
    """The registered shape, as a value that can be hashed and compared."""
    return {
        "size": envelope.size,
        "identifier_width": envelope.identifier_width,
        "observed_width": envelope.observed_width,
        "column_titles": list(envelope.column_titles),
        "slots": [
            {
                "name": spec.name,
                "width": spec.width,
                "vocabulary": list(spec.vocabulary),
                "allows_answers": spec.allows_answers,
                "allows_empty": spec.allows_empty,
            }
            for spec in envelope.slots
        ],
    }


def instance_record(instance: Instance, generator: Generator) -> dict[str, Any]:
    """Everything the bank commits to about one instance, as one canonical value.

    Task text and identifiers are not enough. What a link means is decided by the
    drawn convention, by both sides' answers under it, and by the shape the receipt
    will be rendered in. A commitment that omitted those would still match after code
    drift changed every answer, and the bank would keep asserting a rule nobody could
    now reproduce.

    AND BOTH TABLES. The table is opaque here, so the family says what it is through
    `table_record`, and this hashes what it said. Without them a generator whose rows
    carried a hidden serial rebuilt to different identifiers, different receipts and
    different bytes for the two branches of one fork, and the digest fixation compares
    stayed equal through all of it: text, identifier and key can all agree while the
    row an agent reads does not.
    """
    return {
        "generator": instance.generator,
        "genre": instance.genre,
        "ordinal": instance.ordinal,
        "convention": {k: instance.convention[k] for k in sorted(instance.convention)},
        "a": {
            "task_id": instance.a.task_id,
            "surface": instance.a.surface,
            "text": instance.a.text,
            "key": list(instance.a.key),
            "table": generator.table_record(instance.a.table),
        },
        "b": {
            "task_id": instance.b.task_id,
            "surface": instance.b.surface,
            "text": instance.b.text,
            "key": list(instance.b.key),
            "table": generator.table_record(instance.b.table),
        },
        "envelope": envelope_schema(instance.envelope),
        "filler": instance.envelope.filler,
        "neutral": {k: list(v) for k, v in sorted(instance.envelope.neutral.items())},
        "renderer": RENDERER_CONFIGURATION,
    }


def instance_digest(instance: Instance, generator: Generator) -> str:
    """The content hash of the canonical instance record."""
    return streams.digest(
        json.dumps(instance_record(instance, generator), sort_keys=True).encode()
    )


#: How many ordinals a build may consider before giving up on filling a bank. A
#: bounded loop rather than an open one, and the same bound at build and at verify,
#: so the population one computes is the population the other computes.
def _ceiling(size: int) -> int:
    return max(size * 8, size + 8)


def materialize(generator: Generator, master: bytes, size: int) -> Bank:
    """Freeze a bank of `size` gate passers under one key.

    The admission rule is built here, from the registered constants, and is not a
    caller's argument. A predicate that could be passed in is a predicate that could
    admit anything, and the bank's whole meaning is which rule filled it.
    """
    return materialized(generator, master, size)[0]


def materialized(
    generator: Generator, master: bytes, size: int
) -> tuple[Bank, Population]:
    """The same freeze, handing back the population it had to compute anyway.

    Filling a bank walks admission over every ordinal it considers, which is the
    expensive thing this package does. A caller that wants to print what was admitted
    reads it from here rather than running the walk a second time to learn what the
    first walk already knew.
    """
    bank = Bank(
        generator=generator.name,
        genre=generator.genre,
        renderer=RENDERER_CONFIGURATION,
        master=master,
        size=int(size),
    )
    # Proving the bank can be filled is part of freezing it: a bank whose population
    # cannot be recomputed is a bank nothing could ever serve.
    return bank, population(bank, generator)


def population(bank: Bank, generator: Generator, thresholds=None) -> Population:
    """The instances this bank holds, by rerunning admission over the ordinals in order.

    Ordinals are considered from zero and the failures are skipped, so the passers
    are whatever the rule admits and in what order, up to the size the bank was
    frozen at. Nothing is looked up and nothing is compared: there is no stored list
    to disagree with, no ordinal that can appear twice, and no prefix that can be
    shortened after the fact to change the fraction that passed.
    """
    from shogym.envs.receipts.admission import Thresholds
    from shogym.envs.receipts.admission import report as admission_report

    if bank.renderer != RENDERER_CONFIGURATION:
        raise ValueError(
            f"this bank was frozen under renderer {bank.renderer!r} and this build is "
            f"{RENDERER_CONFIGURATION!r}; the cells would not be the cells it gated"
        )
    if bank.generator != generator.name or bank.genre != generator.genre:
        raise ValueError(
            f"this bank names {bank.generator}/{bank.genre} and the generator offered is "
            f"{generator.name}/{generator.genre}"
        )
    bars = thresholds if thresholds is not None else Thresholds()
    if not bars.registered:
        # `settled` is R's arity and blocks alone, so a caller could move the copy,
        # flip, leverage and headroom bars and still fill a bank under them. Every
        # bar is what admitted the instances, so every bar is what a bank is filled
        # under: a predicate that could be loosened is a predicate that could admit
        # anything, and which rule filled it is the bank's whole meaning.
        registered = Thresholds().as_record()
        moved = ", ".join(
            f"{name} {value:g} against the registered {registered[name]:g}"
            for name, value in sorted(bars.as_record().items())
            if value != registered[name]
        )
        raise ValueError(f"only the registered bars may fill a bank; this moved {moved}")
    held: list[Instance] = []
    considered = 0
    for ordinal in range(_ceiling(bank.size)):
        if len(held) >= bank.size:
            break
        considered += 1
        instance = draw(generator, bank.master, ordinal)
        if admission_report(generator, instance, bank.master, bars).admitted:
            held.append(instance)
    if len(held) < bank.size:
        raise ValueError(
            f"only {len(held)} of {bank.size} instances passed admission in "
            f"{considered} draws"
        )
    return Population(instances=tuple(held), considered=considered)


#: Where the pin starts walking. Between them these three reach everything that
#: decides what a family means, what admitted it, and what a run seals: the generator
#: itself, the verifier that decides whether a bundle may be dealt, and the
#: environment that decides what score is sealed.
PIN_ROOTS = (
    "shogym.envs.receipts.bundle",
    "shogym.envs.receipts.env_v1",
)

#: The packages the pin covers, and the whole of what it covers. It is a closure over
#: these two only: first-party shogym code outside them is NOT pinned, including the
#: serve lifecycle and the core types that decide what a seal is, and neither are the
#: interpreter, the dependency lock or an immutable artifact store. Every one of those
#: is a stated boundary held by process rather than by the hash, and the module list a
#: bundle records is where a reader sees where the closure stopped.
PIN_PACKAGES = ("shogym.envs.receipts", "shogym.receipts")

#: Modules the walk reaches that decide none of it, each with the reason it is out.
#: Nothing else is excluded, and a module that appears in the closure without being
#: listed here is pinned, so a new decider joins the pin by existing rather than by
#: being remembered.
NOT_DECIDING = {
    # A name to a module and a name to a directory. Which generator is being pinned is
    # an argument here, so the lookup that found it decides nothing about the answer.
    "shogym.envs.receipts.registry": "it maps names to modules and directories",
    # Gate exhibits. They wear the protocol so the gates can be exercised through the
    # shipped path, they are never dealt, and they are reached only through the
    # registry's catalogue.
    "shogym.envs.receipts.generators.vectors": "they are exhibits nothing deals",
}


def _imported(module_name: str) -> set[str]:
    """The first-party modules one module imports, read out of its own source.

    Read from the syntax rather than from `sys.modules`, because half the imports here
    are inside functions to break cycles, and a walk over what happens to be loaded
    would depend on what a caller had touched first.
    """
    import ast
    import importlib.util

    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return set()
    # A package's own `__init__` resolves relative imports against the package, an
    # ordinary module against its parent. `resolve_name` knows which; counting dots
    # from the module name does not, and is one level too high inside every
    # `__init__.py`.
    anchor = module_name if _is_package(module_name) else module_name.rpartition(".")[0]
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    candidates: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            try:
                base = importlib.util.resolve_name(base, anchor)
            except (ImportError, ValueError):
                continue
            candidates.add(base)
            # `from shogym.envs.receipts import bank` names a module, not an object.
            candidates.update(f"{base}.{alias.name}" for alias in node.names)
    return {name for name in candidates if _is_module(name)}


def _is_package(name: str) -> bool:
    """Whether this dotted name is a package, whose `__init__` Python also executes."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    return bool(spec and spec.submodule_search_locations is not None)


def _ancestors(name: str) -> set[str]:
    """The package initializers Python runs on the way to importing this module.

    Importing `a.b.c` executes `a/__init__.py` and `a/b/__init__.py` first. Whatever
    they do is part of what the run does, so a pin that skipped them would hold still
    while deciding setup moved into one of them.
    """
    parts = name.split(".")
    return {
        ".".join(parts[:n]) for n in range(1, len(parts)) if _is_module(".".join(parts[:n]))
    }


def _is_module(name: str) -> bool:
    """Whether this dotted name is a first-party module the pin should consider."""
    import importlib.util

    if not any(
        name == package or name.startswith(package + ".") for package in PIN_PACKAGES
    ):
        return False
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    return bool(spec and spec.origin and spec.origin.endswith(".py"))


def pinned_modules(generator: Generator) -> list[str]:
    """Every first-party module that decides what this family means, in name order.

    Computed by walking imports from the roots rather than listed, so a module that
    starts deciding something is in the pin the moment something on the decision path
    imports it. The alternative is a list that has to be remembered, and the two
    modules that decided a bundle's eligibility were missing from the last one.
    """
    roots = [type(generator).__module__, *PIN_ROOTS]
    seen: set[str] = set()
    stack = [name for name in roots if _is_module(name)]
    # The generator's own module is never excluded. NOT_DECIDING names modules that
    # decide nothing, and a family implemented or wrapped inside one of them would
    # otherwise drop its own implementation out of the pin of what decides it.
    implementation = type(generator).__module__
    while stack:
        current = stack.pop()
        if current in seen or (current in NOT_DECIDING and current != implementation):
            continue
        seen.add(current)
        stack.extend((_imported(current) | _ancestors(current)) - seen)
    return sorted(seen)


def current_code_digest(generator: Generator) -> str:
    """The content hash of the code that decides what this family means.

    The generator, the protocol that decides what a draw is, the serializer and the
    shared grader, the gates and the checks, the screen, the verifier that decides
    whether a bundle may be dealt, and the environment that decides what score is
    sealed. A pin that covered only rendering would still match after any of those
    changed, and a bundle would keep naming an instrument that is no longer the
    instrument.
    """
    return streams.digest(*(blob for _, blob in _pinned_sources(generator)))


def _pinned_sources(generator: Generator) -> list[tuple[str, bytes]]:
    """Each pinned module's name and the bytes of its source, in name order."""
    import importlib.util

    out: list[tuple[str, bytes]] = []
    for name in pinned_modules(generator):
        spec = importlib.util.find_spec(name)
        if spec and spec.origin:
            out.append((name, Path(spec.origin).read_bytes()))
    return out


def code_pin(generator: Generator) -> dict[str, Any]:
    """What a bundle records about the code it was certified against.

    The digest AND the module list, each module with its own hash. One opaque digest
    says a bundle is stale and nothing about where: a reader is told that scoring,
    rendering or gating moved, with no way to find out which. The names are also what
    the pin does not cover, read directly: what is absent from the list is outside the
    closure, so the boundary is a thing to look at rather than a claim to believe.
    """
    sources = _pinned_sources(generator)
    return {
        "digest": streams.digest(*(blob for _, blob in sources)),
        "modules": {name: streams.digest(blob) for name, blob in sources},
    }


# --------------------------------------------------------------------------
# after A seals: the one atomic render
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fork:
    """The three cells of one fork, made in one act and hashed.

    `graded`, `placebo` and `oracle` are the serialized bytes, all of one envelope
    size. `component_score` is the sealed scalar. `canonical` is what the parser
    made of the filing, and it is the same object the scorer and both renderers
    saw, which is what stops a receipt grading something the score did not.

    `canonical` AND `outcomes` ARE None ON A REPLAY. What is persisted is the bytes,
    the digests and the scalar; the parsed filing and the row outcomes are not, and
    filling them with a stand-in would make a replayed substantive filing read as a
    genuine empty one to any caller that asked. A reader that needs them on a replay
    parses the raw filing itself, and one that reads them without checking fails
    where the absence is rather than three lines later on a fabricated answer.
    """

    task_id: str
    canonical: Filing | None
    component_score: float
    outcomes: tuple[RowOutcome, ...] | None
    graded: bytes
    placebo: bytes
    oracle: bytes
    digests: Mapping[str, str]
    #: What the filing hashed to, so a replay is keyed by the act that produced it.
    filing_digest: str = ""
    #: True when these bytes were read back rather than rendered.
    replayed: bool = False

    def agent_bytes(self, kind: str) -> bytes:
        """The one cell a branch is served. The only thing here that crosses outward."""
        if kind == GRADED:
            return self.graded
        if kind == PLACEBO:
            return self.placebo
        if kind == ORACLE:
            return self.oracle
        raise ValueError(f"a fork serves {GRADED}, {PLACEBO} or {ORACLE}, not {kind!r}")


def render_fork(
    generator: Generator,
    instance: Instance,
    side: str,
    raw: object,
) -> Fork:
    """Canonicalize the filing, render all three cells, judge them, hash.

    In one act, and only after the filing exists. Every branch of the fork is
    served out of what this returns, so the three cells of one fork are the same
    bytes for every branch by construction rather than by two renders agreeing.

    What "acceptable" means is not decided here. It is `judge_cells`, which admission
    also runs at every sample it takes, so a cell admission accepted is a cell this
    accepts. A fork that fails is never serialized or persisted: admission samples
    filings and the parser's legal space is open-ended, so the filing that actually
    seals is judged on its own terms rather than trusted because a sample passed.
    """
    task = instance.side(side)
    # Parsed ONCE. The scorer and both renderers have to have seen the same object,
    # which is what stops a receipt grading something the score did not, and a second
    # call to a stateful parser is a second object.
    canonical = generator.parse_and_canonicalize(task, raw)
    judged = judge_cells(
        generator, task, canonical, instance.convention, instance.envelope
    )
    if judged.problems:
        raise ValueError(judged.problems[0])
    return Fork(
        task_id=task.task_id,
        canonical=canonical,
        component_score=judged.score,
        outcomes=judged.outcomes,
        graded=judged.payloads[GRADED],
        placebo=judged.payloads[PLACEBO],
        oracle=judged.payloads[ORACLE],
        digests={
            kind: streams.digest(payload)
            for kind, payload in judged.payloads.items()
        },
        filing_digest=filing_digest(raw),
    )


def fork_path(
    directory: Path, task_id: str, digest: str, source_digest: str = ""
) -> Path:
    """Where one fork's committed bytes live: the source, the task, and the whole filing.

    The whole filing hash, not a prefix, because a truncated key is one two filings
    can share and sharing it replays one filing's feedback for another. The source
    digest goes in the directory for the same reason: two frozen bundles can hold the
    same task and the same filing, and a record that named neither would answer for
    both. In production that digest is the bundle's.
    """
    root = Path(directory) / (source_digest[:16] if source_digest else "unbound")
    return root / f"fork-{task_id}-{digest}.json"


def save_fork(fork: Fork, directory: Path, source_digest: str) -> Path:
    """Persist the three cells once. Written by exclusive creation, then only read.

    Exclusive creation rather than replace: two workers reaching the same fork must
    not race to overwrite each other, because the loser's bytes are the ones some
    branch already read.
    """
    if not is_hex(source_digest):
        raise ValueError(
            "a committed fork has to name the frozen source it belongs to; an unbound "
            "record in a shared directory answers for whichever source asks"
        )
    path = fork_path(directory, fork.task_id, fork.filing_digest, source_digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": fork.task_id,
        "filing_digest": fork.filing_digest,
        "source_digest": source_digest,
        "renderer": RENDERER_CONFIGURATION,
        "component_score": fork.component_score,
        "graded": fork.graded.decode("ascii"),
        "placebo": fork.placebo.decode("ascii"),
        "oracle": fork.oracle.decode("ascii"),
        "digests": dict(fork.digests),
    }
    if path.exists():
        return path
    text = json.dumps(payload, sort_keys=True)
    # A staging name no other writer can be using. Two writers sharing one staging name
    # share the file behind it, so a writer could rewrite the bytes another had already
    # linked into place, and a reader would be reading a file being written rather than a
    # record. The claim in `fork_for` keeps them apart in this package; the name keeps
    # them apart for any caller that reaches this door another way.
    scratch = path.with_name(path.name + f".{os.getpid()}.{secrets.token_hex(8)}.partial")
    scratch.write_text(text, encoding="utf-8")
    try:
        os.link(scratch, path)
    except FileExistsError:
        pass
    finally:
        scratch.unlink(missing_ok=True)
    return path


def load_fork(
    directory: Path, task_id: str, filing_digest: str, source_digest: str
) -> Fork | None:
    """The committed cells, or None when this fork has not been rendered yet.

    Every identity on the record is checked against what was asked for. A cell hash
    proves the bytes are the bytes that were written; it says nothing about whether
    they are the bytes for THIS task and THIS filing, and a file that arrived at the
    wrong name would otherwise replay another filing's feedback and score.
    """
    if not is_hex(source_digest):
        raise ValueError("reading a committed fork needs the source it belongs to")
    path = fork_path(directory, task_id, filing_digest, source_digest)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_id") != task_id:
        raise ValueError(
            f"the record at {path} is for task {payload.get('task_id')!r}, not {task_id!r}"
        )
    if payload.get("filing_digest") != filing_digest:
        raise ValueError(
            f"the record at {path} is for filing {str(payload.get('filing_digest'))[:12]}, "
            f"not {filing_digest[:12]}"
        )
    if payload.get("renderer") != RENDERER_CONFIGURATION:
        raise ValueError(f"the record at {path} was written by another renderer")
    if payload.get("source_digest") != source_digest:
        raise ValueError(f"the record at {path} belongs to another source")
    cells = {k: payload[k].encode("ascii") for k in (GRADED, PLACEBO, ORACLE)}
    for kind, blob in cells.items():
        if streams.digest(blob) != payload["digests"][kind]:
            raise ValueError(
                f"the committed {kind} cell at {path} does not match its recorded digest"
            )
    return Fork(
        task_id=payload["task_id"],
        canonical=None,
        component_score=float(payload["component_score"]),
        outcomes=None,
        graded=cells[GRADED],
        placebo=cells[PLACEBO],
        oracle=cells[ORACLE],
        digests=payload["digests"],
        filing_digest=payload["filing_digest"],
        replayed=True,
    )


def fork_for(
    generator: Generator,
    instance: Instance,
    side: str,
    raw: object,
    directory: Path | None = None,
    source_digest: str = "",
) -> Fork:
    """The fork for this filing: replayed if it is committed, rendered if it is not.

    This is the only entry a run should use. Rendering again on a retry is how two
    branches of one fork come to hold different bytes, so a fork that has been
    committed is read back rather than remade, and its digests are checked on the way
    in.

    Once is once under concurrency as well as over time, and that takes a claim. Two
    seals of one filing can arrive together, from two threads of one Worker or from two
    processes, and both would find the record absent and both would render it. What they
    would come to is the same bytes, because a render is a function of the filing and the
    instance, but the fork is where this environment says the cells are made once and a
    claim is what makes that true rather than likely. The claim is a lock on the fork's
    own name, so it is per fork rather than per store and two different filings never wait
    on each other, and the loser reads what the winner wrote.
    """
    if directory is None:
        return render_fork(generator, instance, side, raw)
    task = instance.side(side)
    digest = filing_digest(raw)
    existing = load_fork(directory, task.task_id, digest, source_digest)
    if existing is not None:
        return existing
    path = fork_path(directory, task.task_id, digest, source_digest)
    with one_writer(path):
        # Asked again under the claim, because the answer before it was taken is the
        # answer whoever held it has since changed.
        existing = load_fork(directory, task.task_id, digest, source_digest)
        if existing is not None:
            return existing
        fresh = render_fork(generator, instance, side, raw)
        save_fork(fresh, directory, source_digest)
        # Read back through the same door, so what a later branch will replay is what
        # this branch used, and a lost race resolves to the winner's bytes rather than
        # to whatever this worker happened to hold.
        committed = load_fork(directory, task.task_id, digest, source_digest)
    return committed if committed is not None else fresh


@contextmanager
def one_writer(path: Path) -> Iterator[None]:
    """Hold the right to render the fork this path names, for as long as rendering takes.

    A file lock rather than anything in this process, because the writers to keep apart
    are a Worker's threads and the Workers of two processes over one store, and the
    machine takes it back from a process that died holding it. The lock has a name of its
    own beside the record: a lock taken on the record would have to create the record to
    take it, and the record's absence is what is being decided.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path.with_name(path.name + ".claim")), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def filing_digest(raw: object) -> str:
    """A stable hash of the raw filing, so one fork is keyed by what produced it.

    TOTAL OVER WHAT THE TOOL ACCEPTS. The schema takes a Python string and a JSON
    string carries any code point, lone surrogates included, and strict UTF-8 refuses
    those. Refusing here is refusing before the parser has read anything: the value
    would have folded to a printable character and rendered three ordinary cells, and
    instead the seal raised, the episode failed closed, and the link kept no cell at
    all. `surrogatepass` gives every Python string one stable byte form, which is all
    a key has to be, and the folding stays where it belongs in the parser.
    """
    if raw is None:
        return streams.digest(b"")
    if isinstance(raw, str):
        return streams.digest(raw.encode("utf-8", "surrogatepass"))
    return streams.digest(
        json.dumps(raw, sort_keys=True, default=str).encode("utf-8", "surrogatepass")
    )


# There is no structures-only render path here. One existed, exported and called by
# nothing: it ran the three renderers with no acceptance predicate, no envelope check
# and no parity check, and handed the LIVE envelope to the placebo renderer, which is
# the exact hazard `frozen_envelope` was written for. A reader that wants the
# structures takes `judge_cells(...).asts`, so the one predicate still decides.


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def bank_record(bank: Bank) -> dict[str, Any]:
    """A bank as one canonical value. This is the whole of what a bank persists."""
    return {
        "generator": bank.generator,
        "genre": bank.genre,
        "renderer": bank.renderer,
        "master": bank.master.hex(),
        "size": bank.size,
    }


def bank_identity(bank: Bank) -> str:
    """What names THIS bank, independent of how any file happened to be written.

    The record, hashed with sorted names and no formatting to vary, so a bank has one
    identity whether it is sitting in a working file or canonically inside a bundle.
    Evidence taken on a bank names this, which is what stops a pilot or a human read
    of one bank being carried to another that drew a different convention.
    """
    return streams.digest(json.dumps(bank_record(bank), sort_keys=True).encode())


def bank_from_record(payload: object) -> Bank:
    """A bank out of its stored form, refusing anything that is not exactly one.

    The exact field set, so a file carrying a field nobody reads cannot travel inside
    a bundle asserting something, and a file missing one cannot have it defaulted.

    And the exact SPELLING of the key. A bundle is addressed by the hash of its own
    files, so one representation per value has to hold on arrival and not only after
    conversion: `bytes.fromhex` reads an uppercased or spaced master into the same
    bytes, which gave one bank two bundle addresses that both verified, and a genre
    with two bundles and no digest named is a genre production refuses to serve.
    """
    if not isinstance(payload, dict):
        raise ValueError("a bank record is a mapping")
    expected = {"generator", "genre", "renderer", "master", "size"}
    if set(payload) != expected:
        raise ValueError(
            "a bank record carries exactly %s, and this one carries %s"
            % (", ".join(sorted(expected)), ", ".join(sorted(payload)) or "nothing")
        )
    # Typed before conversion, never coerced. `int(1.9)` is 1 and `int(True)` is 1, so
    # a serializer that emitted a fractional or boolean count would be read as a
    # different bank than the one written, quietly and with everything else matching.
    for field in ("generator", "genre", "renderer", "master"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"a bank's {field} is a nonblank string")
    size = payload["size"]
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"a bank's size is a whole number, not {size!r}")
    try:
        master = bytes.fromhex(payload["master"])
    except ValueError as exc:
        raise ValueError(f"a bank's master key is hexadecimal: {exc}") from exc
    if payload["master"] != master.hex():
        raise ValueError(
            "a bank's master key is written as lowercase hexadecimal with no spacing, "
            "so one bank has one spelling and one address"
        )
    return Bank(
        generator=payload["generator"],
        genre=payload["genre"],
        renderer=payload["renderer"],
        master=master,
        size=size,
    )


def save_bank(bank: Bank, path: Path) -> str:
    """Write a bank, controller-side. Returns its digest."""
    text = json.dumps(bank_record(bank), indent=1, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return streams.digest(text.encode())


def load_bank(path: Path) -> Bank:
    """Read a bank back."""
    return bank_from_record(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# the filings a review pack renders against
# --------------------------------------------------------------------------

#: The registered filing shapes. A review pack needs a receipt with failures on it,
#: so a shape is named and drawn from its own stream rather than improvised.
FILING_SHAPES = ("canonical", "mixed", "empty", "malformed")


def review_filing(
    generator: Generator, instance: Instance, side: str, shape: str, master: bytes
) -> object:
    """A filing of one registered shape, for reading a rendered instance.

    Deterministic under the instance's own stream, so the same instance and shape
    always produce the same receipt and two readers are looking at one artifact.
    """
    task = instance.side(side)
    identifiers = generator.row_identifiers(task.table)
    if shape == "empty":
        return ""
    if shape == "malformed":
        return "no identifiers here\njust some prose about the schedule"
    if shape == "canonical":
        return "\n".join(f"{i},{v}" for i, v in zip(identifiers, task.key))
    if shape == "mixed":
        picker = streams.rng(master, streams.REVIEW_FILING, instance.ordinal, side)
        lines = []
        for identifier, correct in zip(identifiers, task.key):
            if picker.random() < 0.45:
                lines.append(f"{identifier},{_wrong(correct, picker)}")
            else:
                lines.append(f"{identifier},{correct}")
        return "\n".join(lines)
    raise ValueError(f"a filing shape is one of {FILING_SHAPES}, not {shape!r}")


def _wrong(correct: str, picker) -> str:
    """A value that is not the correct one, so the row reads FAIL."""
    alternatives = ["Provisional", "Deferred", "Held", "Cleared", ""]
    choices = [a for a in alternatives if a.strip().lower() != (correct or "").strip().lower()]
    return picker.choice(choices)


def outcome_summary(outcomes: Sequence[RowOutcome] | None) -> str:
    """A one-line controller-side summary. Never rendered into a cell.

    A replayed fork carries no outcomes and says so, rather than reporting that zero
    of zero rows matched on a filing that matched every one of them.
    """
    if outcomes is None:
        return "row outcomes were not persisted with these cells"
    passed = sum(1 for o in outcomes if o.matched)
    return f"{passed} of {len(outcomes)} rows matched"


def no_filing_reason(canonical: Filing | None) -> str:
    return canonical.reason if isinstance(canonical, NoFiling) else ""


__all__ = [
    "FILING_SHAPES",
    "GATE_LABEL",
    "RENDERER_CONFIGURATION",
    "Bank",
    "Fork",
    "Population",
    "bank_from_record",
    "bank_identity",
    "bank_record",
    "code_pin",
    "commitment",
    "instance_digest",
    "filing_digest",
    "fork_for",
    "fork_path",
    "instance_record",
    "is_hex",
    "load_bank",
    "load_fork",
    "materialize",
    "materialized",
    "no_filing_reason",
    "save_fork",
    "outcome_summary",
    "population",
    "render_fork",
    "review_filing",
    "save_bank",
]
