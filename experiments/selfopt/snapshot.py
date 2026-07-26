"""Content-hashed, whole-directory snapshots — "the self" over time.

The treatment arm's persistent workdir *is* the self (issue #57: "the entire write-accessible
workdir persists and is snapshotted whole"). At each checkpoint we archive a full copy under
its content hash, so the self at any boundary is both **identifiable** (the hash) and
**recoverable** (the copy) — and the diffs between snapshots are the narrative.

The hash is taken over ``(relpath, bytes)`` of every file, sorted — order-stable and
tamper-evident. The snapshotter runs **outside** the agent's control (the harness, or in the
container topology a broker-only volume the agent can't reach), so the agent can neither forge
nor suppress the record of what its self looked like.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Optional

# A skip rule maps a path RELATIVE to the snapshot root to True (drop) / False (keep).
SkipFn = Callable[[Path], bool]

# Never snapshot these — noise, caches, or the agent's live scratch that would bloat the archive.
_SKIP_DIRS = {".git", ".jj", "node_modules", "__pycache__", ".venv", ".cache", "runs"}

# The agent's native Claude Code home (``~/.claude``) is the SECOND half of "the self": the
# memory/skills/settings the CLI writes there survive across the persistent session. Capture that
# durable self-surface, but skip transient session state, caches, and config backups so archives
# and W&B artifacts don't bloat. GOTCHA: Claude Code stores native memory at
# ``projects/<cwd-slug>/memory/`` — right beside the (large, transient) ``projects/<slug>/*.jsonl``
# conversation logs. So ``projects/`` is dropped WHOLESALE **except** any ``memory/`` subtree.
_HOME_SKIP_DIRS = {
    "shell-snapshots", "statsig", "sessions", "todos", "backups", "logs", "tmp",
    "ide", "file-history", ".cache", "cache", "downloads",
}
# Runtime/machine-state files at the home root — pure noise, no self-signal.
_HOME_SKIP_NAMES = {
    ".last-cleanup", "policy-limits.json", "remote-settings.json", "history.jsonl",
    "__store.db", ".flock",
}


def _default_skip(skip_dirs: Iterable[str]) -> SkipFn:
    skip = set(skip_dirs)
    return lambda rel: any(part in skip for part in rel.parts)


def home_skip(rel: Path) -> bool:
    """Skip rule for a snapshot of ``~/.claude`` (the native Claude Code home).

    Keeps the durable self-surface (``memory/``, ``skills/``, ``CLAUDE.md``, ``settings*.json``,
    ``MEMORY.md``); drops transient session logs/state, caches, and config backups. Defense in
    depth: any credential/token-shaped file is dropped no matter where it appears — a secret must
    never enter a snapshot or artifact (Claude Code with an ``-e`` OAuth token writes none, but we
    do not rely on that)."""
    parts = rel.parts
    name = rel.name.lower()
    if "credential" in name or "token" in name or name in (".netrc",):
        return True
    if any(part in _HOME_SKIP_DIRS for part in parts):
        return True
    if rel.name in _HOME_SKIP_NAMES:
        return True
    # Under projects/<slug>/: keep only the native-memory subtree, drop the conversation logs
    # (and any other per-project scratch) that live alongside it.
    if parts and parts[0] == "projects":
        return "memory" not in parts
    return False


def _iter_files(root: Path, skip: SkipFn) -> List[Path]:
    out: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not skip(p.relative_to(root)):
            out.append(p)
    return out


def _resolve_skip(skip: Optional[SkipFn], skip_dirs: Iterable[str]) -> SkipFn:
    return skip if skip is not None else _default_skip(skip_dirs)


def content_hash(
    root: Path,
    skip_dirs: Iterable[str] = _SKIP_DIRS,
    skip: Optional[SkipFn] = None,
) -> str:
    """A stable 12-char digest of the directory's file tree (relpath + bytes, sorted).

    Pass ``skip`` (a relpath→bool predicate, e.g. :func:`home_skip`) to override the default
    directory-name skip set — used to hash ``~/.claude`` by its durable self-surface only."""
    fn = _resolve_skip(skip, skip_dirs)
    h = hashlib.sha256()
    if not root.exists():
        return h.hexdigest()[:12]
    for p in _iter_files(root, fn):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


def snapshot(
    root: Path,
    dest_dir: Path,
    skip_dirs: Iterable[str] = _SKIP_DIRS,
    label: Optional[str] = None,
    skip: Optional[SkipFn] = None,
) -> str:
    """Archive ``root`` under ``dest_dir/<hash>/`` (copy-once by content) and return the hash.

    If a snapshot with this hash already exists, it is reused (identical self = identical
    archive). ``label`` (e.g. ``"start"``/``"mid"``/``"end"``) is recorded in a sidecar so the
    checkpoint sequence is readable without recomputing hashes. ``skip`` overrides the default
    skip (see :func:`content_hash`)."""
    fn = _resolve_skip(skip, skip_dirs)
    digest = content_hash(root, skip=fn)
    out = Path(dest_dir) / digest
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        if root.exists():
            for p in _iter_files(root, fn):
                rel = p.relative_to(root)
                target = out / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
    if label:
        (Path(dest_dir) / "checkpoints.log").open("a", encoding="utf-8").write(
            f"{label}\t{digest}\n"
        )
    return digest


def copy_tree(
    src: Path,
    dst: Path,
    skip_dirs: Iterable[str] = _SKIP_DIRS,
    skip: Optional[SkipFn] = None,
) -> None:
    """Copy the file tree under ``src`` into ``dst`` (same skip rules as snapshotting).

    Used to hand a held-out probe a **throwaway copy** of the self: the probe plays with the
    self's current skills/memory (in ``dst``), but every write it makes lands in ``dst`` and is
    discarded — it NEVER touches ``src`` (the training self is *measured*, never mutated by
    held-out). ``src`` is read-only here by construction; only ``dst`` is written."""
    src = Path(src)
    if not src.exists():
        return
    fn = _resolve_skip(skip, skip_dirs)
    for p in _iter_files(src, fn):
        target = Path(dst) / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
