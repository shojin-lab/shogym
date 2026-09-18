"""Which genres exist, and how a bank for one is found.

The roster proper is a frozen manifest release, and it is not this. This is the
smaller thing the commands need before a release exists: a name to a generator
module, and a name to the bank file that generator's instances were materialized
into. Both are controller-side.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from shogym.envs.receipts.copy_profiles import require_profile
from shogym.envs.receipts.protocol import Generator

#: Genre name to the module that implements it. A module earns a line here by
#: implementing the protocol; it earns a place in a release by passing admission,
#: which is a different and later thing.
GENRES = {
    "components": "shogym.envs.receipts.generators.components",
    "ledger": "shogym.envs.receipts.generators.ledger",
    "soundchange": "shogym.envs.receipts.generators.soundchange",
}

#: The gate vectors. They wear the generator protocol so the gates can be exercised
#: through the shipped CLI path, and they are NOT families: they have no surface, no
#: room screen and no admission, and nothing may deal them. `list` says so and
#: `materialize` refuses them.
from shogym.envs.receipts.generators.vectors import VECTORS  # noqa: E402

FIXTURES = frozenset(VECTORS)

# There is no list of env names that may not be dealt. One was here, declared,
# exported and read by nothing, which reads as a control while enforcing nothing. What
# actually holds the line is that the development environment is a separate registered
# class whose `dealable` is False and which never opens a bundle.

#: Where materialized banks live. Controller-side, and overridable so a run can put
#: them somewhere a sandbox cannot reach, which is what an operator has to do: the
#: DEFAULT is under the home directory, a bank file holds the master key in the clear,
#: and the committed forks land beside it with the oracle cell stating the rule in
#: words. A run whose lineages mount any of the home tree sets this.
BANK_DIR_VAR = "SHOGYM_RECEIPTS_BANKS"
DEFAULT_BANK_DIR = Path.home() / ".cache" / "shogym" / "receipts" / "banks"


def load_generator(name: str) -> Generator:
    """The generator a genre name or a gate vector points at.

    EVERY GENERATOR THAT ARRIVES THROUGH THIS DOOR DECLARES ITS COPY PROFILE. The copy
    screen prices a registered family of maps and its bar is read against exactly that
    family, so a generator that declares none is refused here rather than priced under
    whichever family this package happened to try first. Registration is the right
    place for it because it is the one door every command, every check and every bundle
    reaches a generator through, and because the refusal a caller wants is "this module
    is not a generator yet", not a copy maximum taken over maps its answers cannot be.
    """
    if name in VECTORS and name in GENRES:
        # Vectors are looked up first, so a collision would silently serve a gate
        # exhibit wherever the family was meant. It is refused rather than resolved.
        raise KeyError(f"{name!r} names both a genre and a gate vector")
    if name in VECTORS:
        return require_profile(VECTORS[name])
    if name not in GENRES:
        raise KeyError(
            "no genre or vector named %r; this build carries %s"
            % (name, ", ".join(sorted(set(GENRES) | set(VECTORS))))
        )
    return require_profile(importlib.import_module(GENRES[name]).GENERATOR)


def is_fixture(name: str) -> bool:
    """Whether this name is a gate vector rather than a family."""
    return name in FIXTURES


def module_path(name: str) -> Path:
    """Where the generator's source lives, for the manifest's hash set."""
    target = "shogym.envs.receipts.generators.vectors" if name in VECTORS else GENRES[name]
    module = importlib.import_module(target)
    return Path(str(module.__file__))


def bank_dir() -> Path:
    return Path(os.environ.get(BANK_DIR_VAR) or DEFAULT_BANK_DIR)


def bank_path(name: str) -> Path:
    return bank_dir() / f"{name}.json"


#: Where the append-only key history lives, and why it is NOT under the bank directory.
#: The record beside the banks is a per-genre summary and it moves when the evidence
#: directory moves, so pointing `SHOGYM_RECEIPTS_BANKS` somewhere else produces an empty
#: record and a first attempt that nobody can see. The history is the other half: one
#: file, appended to and never rewritten, holding every attempt this machine has made
#: under any evidence directory. It is overridable for the same reason the bank directory
#: is, and a run that redirects the banks and leaves this alone is the case the separation
#: exists for.
HISTORY_VAR = "SHOGYM_RECEIPTS_HISTORY"
DEFAULT_HISTORY = Path.home() / ".cache" / "shogym" / "receipts" / "key-history.jsonl"


def history_path() -> Path:
    """The append-only key history for every genre on this machine."""
    return Path(os.environ.get(HISTORY_VAR) or DEFAULT_HISTORY)


def provenance_path(name: str) -> Path:
    """Where a genre's key provenance lives: beside the banks, one file per genre.

    Beside rather than inside, because it outlives any one bank: a materialization that
    could not fill writes no bank and still rolled a key, and that is the attempt the
    record exists for.
    """
    return bank_dir() / "provenance" / f"{name}.json"


def bundle_dir(name: str) -> Path:
    """Where a genre's admission bundles live, one directory per digest."""
    return bank_dir() / "bundles" / name


def bundles(name: str) -> list[Path]:
    """Every bundle directory this genre has, oldest name first.

    A directory whose name is not a digest is not a bundle and is not listed: a
    bundle is addressed by its contents, so anything else here is somebody's scratch
    space rather than something that could be served. Neither is a link: a bundle is
    the directory, and a name that can be repointed at another one is not it.
    """
    root = bundle_dir(name)
    if root.is_symlink() or not root.is_dir():
        return []
    return sorted(
        item for item in root.iterdir()
        if not item.is_symlink() and item.is_dir() and len(item.name) == 64
        and all(c in "0123456789abcdef" for c in item.name)
    )


__all__ = [
    "BANK_DIR_VAR",
    "DEFAULT_BANK_DIR",
    "DEFAULT_HISTORY",
    "HISTORY_VAR",
    "history_path",
    "FIXTURES",
    "GENRES",
    "is_fixture",
    "bank_dir",
    "bank_path",
    "bundle_dir",
    "bundles",
    "load_generator",
    "provenance_path",
    "module_path",
]
