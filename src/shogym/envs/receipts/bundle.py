"""The admission bundle: one frozen directory, one digest, nothing read on trust.

Everything a family's admission rests on lives in one directory: the bank, the
instances it commits to, the thresholds it was filtered under, the code it was
certified against, the raw room screen, the human's review pack, and every artifact
that pack was read from. The directory is content-addressed once. A canonical
manifest lists every file with its size and its hash, and the bundle's digest is the
manifest's digest, so one name covers all of it.

WHY ONE DIGEST. Evidence that arrives as several separately named artifacts can be
recombined: a summary that says four instances where the file holds five, a threshold
set that says one rule while another one ran, a screen and a pack shuffled between
directories. Under a bundle there is nothing to shuffle: the files are in the same
hashed directory or they are not in it at all.

WHAT ONE DIGEST DOES NOT DO. It says these files were frozen together; it does not
say they are about the same family and the same draw. A screen taken on any family
and a pack read of any bank are files, and files can be put in a directory. So the
two pieces of evidence a person supplies each NAME what they are evidence of, and
verification refuses them when that is not this bundle: the screen names its family,
and the review pack names its family and the identity of the bank it was read from.

WHAT VERIFICATION DOES. It recomputes. The population is rebuilt by rerunning
admission from ordinal zero, so which instances a bank holds is a consequence rather
than a list. The screen is rerun on its own rows under its own recorded bars. The
review coverage is enumerated from the rebuilt instances and checked against files
the manifest hashed. The code digest is taken from the running code. Nothing here
reads a stored conclusion, and there is no field a writer can set to make this
return a shorter list of problems.

THE BUNDLE IS CONTROLLER-SIDE. It contains the master key, so it holds answers to
every task in it and is never mounted in a lineage sandbox.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import streams
from shogym.envs.receipts.protocol import Generator, Instance
from shogym.envs.receipts.review import identities, identity

#: Bumped when the bundle's own layout changes. A bundle naming another version is
#: refused rather than read leniently.
BUNDLE_VERSION = "receipts-bundle-v1"

MANIFEST = "manifest.json"
BANK = "bank.json"
INSTANCES = "instances.json"
THRESHOLDS = "thresholds.json"
CODE = "code.json"
SCREEN = "screen.json"
REVIEW = "review.json"
RENDERS = "renders"

#: The files every bundle has. Renders are additional and enumerated by the manifest.
CONTENTS = (BANK, INSTANCES, THRESHOLDS, CODE, SCREEN, REVIEW)

#: What one instance entry commits to. The digest fixes what the instance contains;
#: the commitment fixes which convention was drawn and can be published without the
#: master key, which is the only reason it is stored rather than recomputed silently.
INSTANCE_FIELDS = ("ordinal", "digest", "commitment")


@dataclass(frozen=True)
class Bundle:
    """A loaded bundle: its root, its digest, and the files the manifest hashed.

    Reaching this type means every file listed was found at the size and hash the
    manifest gave, and nothing else was in the directory. What it does not mean is
    that the evidence holds; that is `verify`.
    """

    root: Path
    digest: str
    files: Mapping[str, int]

    def path(self, name: str) -> Path:
        return self.root / name

    def payload(self, name: str) -> Any:
        """One of this bundle's JSON files, read strictly and required to be canonical."""
        return read_json(self.path(name))


def canonical_json(value: Any) -> str:
    """The one serialization of a value. Every file in a bundle is written this way.

    ONE REPRESENTATION PER VALUE, so a file's bytes and what it says are the same
    thing. Sorted names, no separators to vary, and the nonfinite extensions off. A
    bundle is content-addressed, and two byte strings that mean the same thing would
    be two addresses for one bundle; two values that share a byte string would be
    worse.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path: Path) -> Any:
    """One of a bundle's JSON files: no repeated names, no nonfinite, canonical bytes.

    The canonical check is what makes the parse and the file the same statement. A
    reordering, a space, or a `1.0` where the writer meant `1` would otherwise change
    the bytes the manifest hashed without changing what any reader sees, and the
    reverse is worse: a file whose bytes look settled while its meaning depends on
    which reader you ask.
    """
    from shogym.receipts import read_payload

    text = path.read_text(encoding="utf-8")
    parsed = read_payload(text)
    if canonical_json(parsed) != text:
        raise ValueError(
            f"{path.name} is not in canonical form, so its bytes and what it says are "
            "two different things"
        )
    return parsed


def canonical_manifest(entries: Sequence[Mapping[str, Any]]) -> str:
    """The manifest's one serialization. Its bytes are the bundle's name."""
    payload = {
        "bundle": BUNDLE_VERSION,
        "files": sorted(
            (
                {
                    "path": str(e["path"]),
                    "size": int(e["size"]),
                    "digest": str(e["digest"]),
                }
                for e in entries
            ),
            key=lambda e: e["path"],
        ),
    }
    return canonical_json(payload)


def _legal(path: str) -> bool:
    """Whether a manifest path names a file inside the bundle and nowhere else."""
    if not path or path == MANIFEST or path.startswith("/") or "\\" in path:
        return False
    parts = path.split("/")
    return all(part and part not in (".", "..") for part in parts)


def _walk(root: Path) -> set[str]:
    """Every file actually in the directory, manifest aside, as relative paths.

    A LINK IS NOT CONTENT, and the walk never follows one. A bundle is the bytes it is
    addressed by; a name for somebody else's bytes can be repointed, or written
    through, without any operation at the frozen path at all. So a symbolic link
    anywhere is refused, the root itself is checked before it is read, and a regular
    file with more than one link is refused too: a second hard link is a second name
    for the same inode, and the bytes under it can be replaced from outside.
    """
    if root.is_symlink():
        raise ValueError(
            f"{root} is a link, and a bundle is the directory it is addressed by rather "
            "than a name that can be repointed at another one"
        )
    found: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                item = Path(entry.path)
                shown = item.relative_to(root).as_posix()
                # follow_symlinks=False throughout: asking what a link points AT is
                # already following it.
                if entry.is_symlink():
                    raise ValueError(
                        f"{shown} is a link, and a bundle holds the bytes it is addressed "
                        "by rather than a name for somebody else's"
                    )
                if entry.is_dir(follow_symlinks=False):
                    stack.append(item)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"{shown} is not a file, and a bundle holds files")
                if entry.stat(follow_symlinks=False).st_nlink != 1:
                    raise ValueError(
                        f"{shown} has another name outside this bundle, so its bytes can "
                        "be replaced without an operation at the bundle's own path"
                    )
                found.add(shown)
    found.discard(MANIFEST)
    return found


def load(root: Path) -> Bundle:
    """Open a bundle by its directory, checking every byte against the manifest.

    The manifest is canonical, so it is reserialized and required to be the bytes on
    disk: a field nobody reads, a reordering or an extra space would otherwise be a
    way to change the file without changing what it says. The directory's name has to
    be the digest, which is what makes "load a bundle" and "load THIS bundle" the same
    operation. And the directory may hold nothing the manifest does not list, because
    a file that arrives unlisted is a file nothing hashed.
    """
    root = Path(root)
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"there is no bundle at {root}: it has no {MANIFEST}")
    try:
        parsed = read_json(manifest_path)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"the manifest at {manifest_path} is not readable: {exc}") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"bundle", "files"}:
        raise ValueError("a bundle manifest names exactly a bundle version and its files")
    if parsed.get("bundle") != BUNDLE_VERSION:
        raise ValueError(
            f"the bundle at {root} is version {parsed.get('bundle')!r} and this build "
            f"reads {BUNDLE_VERSION!r}"
        )
    entries = parsed.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"the bundle at {root} lists no files")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "digest"}:
            raise ValueError("a manifest entry is exactly a path, a size and a digest")
        if not _legal(str(entry["path"])):
            raise ValueError(f"the manifest names {entry['path']!r}, which is not a file in it")
    text = canonical_manifest(entries)
    if text != manifest_path.read_text(encoding="utf-8"):
        raise ValueError(
            f"the manifest at {manifest_path} does not list what it lists in canonical "
            "form, so its bytes and what it says are two different things"
        )
    digest = streams.digest(text.encode())
    if root.name != digest:
        raise ValueError(
            f"the bundle at {root} manifests to {digest[:16]} and sits under "
            f"{root.name[:16]}; a bundle is addressed by its own contents"
        )
    listed = {str(e["path"]): int(e["size"]) for e in entries}
    if len(listed) != len(entries):
        raise ValueError("the manifest lists one path twice")
    found = _walk(root)
    unlisted = sorted(found - set(listed))
    absent = sorted(set(listed) - found)
    if unlisted or absent:
        raise ValueError(
            "the bundle's directory is not its manifest: "
            + "; ".join(
                ([f"unlisted {', '.join(unlisted[:3])}"] if unlisted else [])
                + ([f"missing {', '.join(absent[:3])}"] if absent else [])
            )
        )
    for name, size in sorted(listed.items()):
        item = root / name
        if item.stat().st_size != size:
            raise ValueError(f"{name} is {item.stat().st_size} bytes and the manifest says {size}")
        if streams.file_digest(str(item)) != str(
            next(e["digest"] for e in entries if e["path"] == name)
        ):
            raise ValueError(f"{name} is not the file the manifest hashed")
    for name in CONTENTS:
        if name not in listed:
            raise ValueError(f"the bundle at {root} holds no {name}")
    return Bundle(root=root, digest=digest, files=listed)


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verification:
    """What recomputing a bundle established, and what it did not."""

    digest: str
    problems: tuple[str, ...]
    instances: tuple[Instance, ...] = ()
    considered: int = 0
    #: The screen this bundle records, once it has been read. Kept so a caller can
    #: print the bars a family was judged against without reading the file again.
    screen: Any = None

    @property
    def verified(self) -> bool:
        return not self.problems

    @property
    def ordinals(self) -> tuple[int, ...]:
        return tuple(i.ordinal for i in self.instances)

    @property
    def passing_fraction(self) -> float:
        return len(self.instances) / float(self.considered) if self.considered else 0.0


def verify(bundle: Bundle, generator: Generator) -> Verification:
    """Recompute everything this bundle asserts. Empty problems means it holds.

    This is the ONE eligibility operation. Production opens what it returns clean and
    refuses everything else, the roster prints it, and there is no second test
    composed beside it: two eligibility answers is one of them being wrong somewhere.
    """
    problems: list[str] = []

    got_pin = bank_mod.code_pin(generator)
    try:
        code = bundle.payload(CODE)
        if not isinstance(code, dict) or set(code) != set(got_pin):
            raise ValueError(
                "a code pin is a digest and the modules it covers, named %s"
                % ", ".join(sorted(got_pin))
            )
        if not isinstance(code["digest"], str):
            raise ValueError("a code pin's digest is a string")
        pinned = code["modules"]
        if not isinstance(pinned, dict) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in pinned.items()
        ):
            raise ValueError("a code pin's modules are names to digests")
        # Which modules, then which bytes. One digest over the whole closure says a
        # bundle is stale without saying where, and the closure itself can move: a
        # module that starts deciding something joins the pin, and a bundle frozen
        # before that was certified against a smaller instrument.
        gone = sorted(set(pinned) - set(got_pin["modules"]))
        added = sorted(set(got_pin["modules"]) - set(pinned))
        if gone or added:
            raise ValueError(
                "the code it pins is not the code that decides this family now: "
                + "; ".join(
                    part
                    for part in (
                        ("no longer pinned: " + ", ".join(gone[:3])) if gone else "",
                        ("newly pinned: " + ", ".join(added[:3])) if added else "",
                    )
                    if part
                )
            )
        moved = sorted(
            name for name, value in got_pin["modules"].items() if pinned[name] != value
        )
        if moved:
            raise ValueError(
                "%s changed under a frozen bundle, so scoring, rendering or gating is "
                "not what certified it" % ", ".join(moved[:3])
            )
        if str(code["digest"]) != got_pin["digest"]:
            raise ValueError(
                f"the code hashes to {got_pin['digest'][:12]} against the pinned "
                f"{str(code['digest'])[:12]}; scoring, rendering or gating has moved "
                "under a frozen bundle"
            )
    except (OSError, TypeError, ValueError) as exc:
        problems.append(f"its code pin does not hold: {exc}")

    thresholds = None
    try:
        thresholds = _thresholds(bundle)
    except (OSError, TypeError, ValueError) as exc:
        problems.append(f"its thresholds do not hold: {exc}")

    try:
        bank = bank_mod.bank_from_record(bundle.payload(BANK))
    except (OSError, TypeError, ValueError) as exc:
        # Nothing downstream means anything without the bank: the population, the
        # screen's family and the review's coverage are all functions of it.
        problems.append(f"its bank does not read: {exc}")
        return Verification(digest=bundle.digest, problems=tuple(problems))

    if thresholds is None:
        return Verification(digest=bundle.digest, problems=tuple(problems))
    try:
        held = bank_mod.population(bank, generator, thresholds)
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        problems.append(f"its population does not recompute: {exc}")
        return Verification(digest=bundle.digest, problems=tuple(problems))

    problems.extend(_verify_instances(bundle, bank, held, generator))
    trouble, record = _verify_screen(bundle, bank)
    problems.extend(trouble)
    problems.extend(_verify_review(bundle, generator, bank, held))
    return Verification(
        digest=bundle.digest,
        problems=tuple(problems),
        instances=held.instances,
        considered=held.considered,
        screen=record,
    )


def verify_at(root: Path, generator: Generator) -> Verification:
    """Load and verify in one step, turning any failure to read into a problem.

    A malformed file is a bundle that does not verify, never an exception out of a
    roster: `list` has to be able to say a bundle is not dealable, and a scalar of the
    wrong type is exactly the case where saying so matters.
    """
    try:
        opened = load(root)
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        return Verification(digest="", problems=(str(exc),))
    try:
        return verify(opened, generator)
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        return Verification(digest=opened.digest, problems=(str(exc),))


def _thresholds(bundle: Bundle):
    """The bars this bundle was filtered under, reconstructed field for field."""
    from shogym.envs.receipts.admission import Thresholds

    payload = bundle.payload(THRESHOLDS)
    registered = Thresholds().as_record()
    if not isinstance(payload, dict):
        raise ValueError("a threshold record is a mapping")
    if set(payload) != set(registered):
        raise ValueError(
            "a threshold record names exactly %s" % ", ".join(sorted(registered))
        )
    for field, value in sorted(payload.items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"the threshold {field} is a number, not {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"the threshold {field} is {value!r}, which is not a bar")
    if {k: float(v) for k, v in payload.items()} != registered:
        raise ValueError(
            "they are not the registered bars, and a bundle is dealt under the "
            "registered bars or not dealt"
        )
    return Thresholds(
        max_copy_score=float(payload["max_copy_score"]),
        max_flip_score=float(payload["max_flip_score"]),
        min_leverage=float(payload["min_leverage"]),
        min_arity=int(payload["min_arity"]),
        min_blocks=int(payload["min_blocks"]),
        min_headroom=float(payload["min_headroom"]),
        min_material_rows=int(payload["min_material_rows"]),
    )


def instance_entries(
    bank: bank_mod.Bank, held: bank_mod.Population, generator: Generator
) -> list[dict[str, Any]]:
    """What a bundle commits to about its instances, computed from the instances."""
    return [
        {
            "ordinal": instance.ordinal,
            "digest": bank_mod.instance_digest(instance, generator),
            "commitment": bank_mod.commitment(
                bank.master, instance.ordinal, instance.convention
            ),
        }
        for instance in held.instances
    ]


def _verify_instances(
    bundle: Bundle, bank: bank_mod.Bank, held: bank_mod.Population, generator: Generator
) -> list[str]:
    """The stored instance entries against the ones just recomputed, in order.

    A sequence, not a lookup. The recomputed population has each ordinal once and in
    admission order, so an entry that repeats an ordinal, omits one, or arrives in
    another order makes the two sequences differ and there is no per-entry search that
    could resolve to a different record than the one being checked.
    """
    try:
        stored = bundle.payload(INSTANCES)
    except (OSError, TypeError, ValueError) as exc:
        return [f"its instance record does not read: {exc}"]
    if not isinstance(stored, list):
        return ["its instance record is not a list of instances"]
    for entry in stored:
        if not isinstance(entry, dict) or set(entry) != set(INSTANCE_FIELDS):
            return [
                "an instance entry names exactly %s" % ", ".join(INSTANCE_FIELDS)
            ]
        ordinal = entry["ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            return [f"an instance ordinal is a whole number, not {ordinal!r}"]
        for field in ("digest", "commitment"):
            if not isinstance(entry[field], str) or not bank_mod.is_hex(entry[field]):
                return [f"an instance {field} is a 64-character hexadecimal digest"]
    computed = instance_entries(bank, held, generator)
    if stored != computed:
        want = [e["ordinal"] for e in computed]
        got = [e.get("ordinal") for e in stored]
        if want != got:
            return [
                f"it commits to instances {got} and its bank admits {want}"
            ]
        return [
            "its instance digests or commitments are not the ones its bank rebuilds to"
        ]
    return []


def _verify_screen(bundle: Bundle, bank: bank_mod.Bank) -> tuple[list[str], Any]:
    """Rerun the room screen on its own rows, against the REGISTERED bars.

    The bars a dealable bundle carries are the registered ones, exactly, the same way
    its gate thresholds are. A record is free to say what it was judged against and a
    diagnostic run is free to move a bar, but a family judged against an easier rule
    was not admitted under the rule the measurement is registered under, and printing
    that the bar was moved is not refusing to deal it.

    The recomputed statistics are then compared with those bars here, rather than
    inferred from a verdict another module composed: what production requires should
    be readable where production requires it.
    """
    from shogym.receipts import ScreenRecord
    from shogym.receipts.screen import (
        REGISTERED_MIN_PAIRS,
        REGISTERED_MIN_RATIO,
        REGISTERED_MIN_ROOM,
        at_least,
    )

    try:
        record = ScreenRecord.from_payload(bundle.payload(SCREEN))
    except (OSError, TypeError, ValueError) as exc:
        return [f"its screen artifact is not a readable record: {exc}"], None
    if record.run.family != bank.generator:
        # The rows carry three numbers each and the run says what they were taken on.
        # Without this the family label is supplied here at verification time, so a
        # pilot run on anything at all freezes into this bundle and reads as its room.
        return [
            f"its screen was taken on {record.run.family!r} and this bundle is "
            f"{bank.generator!r}, so its room was measured on another family"
        ], record
    if not record.dealable_selection:
        # The best of several clears a bar more easily than one does, and nothing
        # corrects for it: the interval, the bars and the verdict are identical for
        # one candidate and for a thousand. Until an adjustment is registered, a
        # selected winner has not met this stage. `screen` still scores and prints it.
        return [
            f"its screen was selected from {record.candidates_screened} candidates and "
            "no selection adjustment is registered, so the best of several would be "
            "dealt as though it were the only one"
        ], record
    if not record.registered:
        return [
            "its screen was judged against bars that are not the registered ones ("
            + "; ".join(record.overrides())
            + "), and a family admitted under an easier rule was not admitted under this one"
        ], record
    try:
        result = record.result(bank.generator)
    except (TypeError, ValueError) as exc:
        return [f"its screen does not rerun on its own rows: {exc}"], record
    problems: list[str] = []
    if record.run.distinct_instances < REGISTERED_MIN_PAIRS:
        problems.append(
            f"its screen was taken over {record.run.distinct_instances} distinct tasks "
            f"where {REGISTERED_MIN_PAIRS} is registered"
        )
    # `at_least` rather than `>=`, and the same helper the screen itself used, so a
    # family exactly at a registered bar is not admitted by one and refused by the
    # other on the last bit of a binary float.
    if not at_least(result.room, REGISTERED_MIN_ROOM):
        problems.append(
            f"its oracle beats its placebo by {result.room:.4f}, under the registered "
            f"{REGISTERED_MIN_ROOM:g}"
        )
    if not (math.isfinite(result.room_low) and result.room_low > 0.0):
        problems.append(
            f"its room interval reaches {result.room_low:.4f}, so its sample does not "
            "establish that there was any room at all"
        )
    if not at_least(result.ratio, REGISTERED_MIN_RATIO):
        problems.append(
            f"one graded receipt took {result.ratio:.4f} of the room its oracle had, "
            f"under the registered {REGISTERED_MIN_RATIO:g}"
        )
    if not result.verdict and not problems:
        problems.append(
            "its screen does not pass when rerun on its own rows: "
            + "; ".join(result.reasons[:2])
        )
    return problems, record


def _verify_review(
    bundle: Bundle,
    generator: Generator,
    bank: bank_mod.Bank,
    held: bank_mod.Population,
) -> list[str]:
    """Enumerate what a reader had to have seen, and find it in this bundle.

    Coverage comes from the rebuilt instances and the generator's own declarations,
    so a pack is checked against the family it is a read of rather than against a list
    written beside it. The pack also has to name the family and the bank it was read
    from, so a read of one draw cannot be offered as the read of another.
    """
    from shogym.envs.receipts.checks import FILING_CLASSES
    from shogym.envs.receipts.review import required_coverage
    from shogym.envs.receipts.review import verify as verify_pack

    try:
        pack = bundle.payload(REVIEW)
    except (OSError, TypeError, ValueError) as exc:
        return [f"its review pack does not read: {exc}"]
    if not isinstance(pack, dict):
        return ["its review pack is not a manifest"]
    counts = [i.a.n_rows for i in held.instances] + [i.b.n_rows for i in held.instances]
    envelope_size = min(i.envelope.size for i in held.instances)
    coverage = required_coverage(generator, FILING_CLASSES, counts)
    return verify_pack(
        pack,
        coverage,
        envelope_size,
        bundle.files,
        bank.generator,
        bank_mod.bank_identity(bank),
    )


# --------------------------------------------------------------------------
# building one
# --------------------------------------------------------------------------


def build(
    directory: Path,
    generator: Generator,
    bank: bank_mod.Bank,
    screen_artifact: Path,
    review_pack: Path,
) -> Bundle:
    """Freeze a bank, a screen run and a review pack into one addressed bundle.

    The bundle is assembled in a scratch directory, named by its own manifest, and
    only then moved into place, so a directory under a digest is always a complete
    bundle. It is verified before it is returned: a bundle that does not verify is
    removed rather than left somewhere to be found later.
    """
    from shogym.envs.receipts.admission import Thresholds
    from shogym.receipts import ScreenRecord, read_payload

    thresholds = Thresholds()
    held = bank_mod.population(bank, generator, thresholds)
    # Read both inputs before writing anything: a bundle half built from a screen
    # artifact that turns out to be unreadable is a directory someone has to clean up.
    screened = read_payload(Path(screen_artifact).read_text(encoding="utf-8"))
    ScreenRecord.from_payload(screened)
    pack = read_payload(Path(review_pack).read_text(encoding="utf-8"))
    if not isinstance(pack, dict):
        raise ValueError(f"the review pack at {review_pack} is not a manifest")
    if not isinstance(screened, dict):
        raise ValueError(f"the screen artifact at {screen_artifact} is not a record")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A staging name no other builder can be using. Naming it for the process meant
    # two threads of one process shared it, and one could remove the other's tree
    # mid-build.
    staging = directory / f".building-{os.getpid()}-{uuid.uuid4().hex}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        _write(staging / BANK, canonical_json(bank_mod.bank_record(bank)))
        _write(
            staging / INSTANCES,
            canonical_json(instance_entries(bank, held, generator)),
        )
        _write(staging / THRESHOLDS, canonical_json(thresholds.as_record()))
        _write(staging / CODE, canonical_json(bank_mod.code_pin(generator)))
        # The screen is written in canonical form rather than copied: a bundle holds
        # one representation of every value it is addressed by, and an operator's
        # export is under no obligation to be in it.
        _write(staging / SCREEN, canonical_json(screened))
        _write(staging / REVIEW, canonical_json(_gather(pack, Path(review_pack), staging)))
        entries = [
            {
                "path": name,
                "size": (staging / name).stat().st_size,
                "digest": streams.file_digest(str(staging / name)),
            }
            for name in sorted(_walk(staging))
        ]
        text = canonical_manifest(entries)
        _write(staging / MANIFEST, text)
        digest = streams.digest(text.encode())
        final = directory / digest
        if final.exists():
            # The same contents already sit under the same name. There is nothing to
            # write and nothing that could differ.
            shutil.rmtree(staging)
        else:
            staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    opened = load(final)
    checked = verify(opened, generator)
    if not checked.verified:
        shutil.rmtree(final, ignore_errors=True)
        raise ValueError(
            "this bundle does not verify, so it is not a bundle: "
            + "; ".join(checked.problems[:3])
        )
    return opened


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _gather(pack: Mapping[str, Any], source: Path, staging: Path) -> dict[str, Any]:
    """Copy every render the pack names into the bundle, under bundle paths.

    The artifacts come inside, because a pack that pointed outward would be a pack
    whose evidence could be replaced without changing the bundle. Paths are rewritten
    to where the bytes now live, and nothing else about the pack is touched.
    """
    renders = pack.get("renders")
    if not isinstance(renders, list) or not renders:
        raise ValueError(f"the review pack at {source} lists no renders")
    gathered: list[dict[str, Any]] = []
    for n, entry in enumerate(renders):
        if not isinstance(entry, dict):
            raise ValueError(f"render {n} of the pack at {source} is not a record")
        named = str(entry.get("path", ""))
        if not named:
            raise ValueError(f"render {n} of the pack at {source} names no file")
        artifact = Path(named)
        if not artifact.is_absolute():
            artifact = source.parent / artifact
        if not artifact.is_file():
            raise ValueError(f"the render for {entry.get('key')!r} is not at {artifact}")
        inside = f"{RENDERS}/{n:04d}-{artifact.name}"
        target = staging / inside
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifact, target)
        gathered.append(
            {
                "category": str(entry.get("category", "")),
                "key": str(entry.get("key", "")),
                "kind": str(entry.get("kind", "")),
                "path": inside,
            }
        )
    # Validated BEFORE it is copied, and never coerced. `str(None)` is the nonempty
    # string "None", so a pack export that lost the attesting person would otherwise be
    # written into the bundle as a reviewer named None, and the absence becomes
    # unrecoverable one line before anything would have noticed it.
    return {
        "reviewer": identity(pack.get("reviewer"), "reviewer"),
        "checklist": identities(pack.get("checklist"), "checklist item"),
        "seeds": identities(pack.get("seeds"), "seed"),
        # Carried through as the pack stated them, never filled in from the bank being
        # frozen. A binding the builder supplies is a binding that always holds.
        "family": identity(pack.get("family"), "family"),
        "bank": identity(pack.get("bank"), "bank"),
        "renders": gathered,
    }


__all__ = [
    "BANK",
    "BUNDLE_VERSION",
    "CODE",
    "CONTENTS",
    "INSTANCES",
    "MANIFEST",
    "RENDERS",
    "REVIEW",
    "SCREEN",
    "THRESHOLDS",
    "Bundle",
    "Verification",
    "build",
    "canonical_manifest",
    "instance_entries",
    "load",
    "verify",
    "verify_at",
]
