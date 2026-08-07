"""Lazy loader for the ``cais/hle`` dataset (the ``hle`` extra).

Kept out of ``import shogym``: ``env_v1`` imports this only when the *registered* ``hle`` env
loads its real tasks (construction), never at module import — so registering the env stays
offline and the core install needs neither ``datasets`` nor a Hugging Face download.

The dataset is **gated** on the Hub (``cais/hle``) and must not be redistributed, so shogym
never ships or commits any of it — it is downloaded on demand (honoring the HF cache, under
``~/.cache/shogym/hle`` by default) once you have accepted the terms and authenticated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

HF_DATASET = "cais/hle"
HF_SPLIT = "test"  # cais/hle ships a single `test` split (2,500 questions)
# Fraction of the (text-only) set assigned to the train split; the rest is the test split.
TRAIN_FRACTION = 0.8


def cache_dir() -> Path:
    """Where to cache the downloaded dataset.

    Honors ``SHOGYM_HLE_DATA_DIR`` first, then the standard ``HF_HOME``, else
    ``~/.cache/shogym/hle`` — matching how the tau2 port caches its data under
    ``~/.cache/shogym``."""
    explicit = os.environ.get("SHOGYM_HLE_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    return Path(os.path.expanduser("~/.cache/shogym/hle"))


def load_hle_tasks(*, text_only: bool = True) -> List[Dict[str, Any]]:
    """Download (once) and return ``cais/hle`` tasks as plain dicts.

    ``text_only`` (default) drops questions that carry an image, since the env is text-only
    for now (multimodal is a follow-up). Raises an actionable error if the ``hle`` extra
    isn't installed or the gated dataset can't be fetched."""
    try:
        from datasets import Image, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the `hle` extra is required to load the cais/hle dataset — install it with "
            "`pip install 'shogym[hle]'` (or `uv sync`, which includes it in the dev group)."
        ) from exc

    try:
        # `load_dataset` is dynamically typed (its return union spans several dataset kinds);
        # with an explicit `split` it yields a row-iterable of Mappings, so treat it as `Any`
        # rather than fighting upstream types (mirrors the tau2 port's handling of its registry).
        dataset: Any = cast(Any, load_dataset(HF_DATASET, split=HF_SPLIT, cache_dir=str(cache_dir())))
    except Exception as exc:
        raise RuntimeError(_load_error_message(exc)) from exc

    # cais/hle carries Image-typed columns (e.g. `image_preview`, `rationale_image`); merely
    # *iterating* rows DECODES them, which pulls in Pillow (`ImportError: … install 'Pillow'`).
    # The env is text-only and never uses a decoded image, so disable decoding on every image
    # column — `row[col]` becomes the raw {bytes, path} dict — and we need no Pillow dependency.
    features = getattr(dataset, "features", {}) or {}
    for name, feature in features.items():
        if isinstance(feature, Image):
            dataset = dataset.cast_column(name, Image(decode=False))

    tasks: List[Dict[str, Any]] = []
    for row in dataset:
        if text_only and _row_has_image(row):
            continue
        tasks.append(
            {
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "answer_type": str(row.get("answer_type", "")),
                "category": str(row.get("category", "")),
            }
        )
    return tasks


def _row_has_image(row: Any) -> bool:
    """Whether a row carries an image (so the text-only filter drops it).

    Handles the field in whatever shape ``datasets`` yields it: a base64/URL **string**
    (cais/hle's `image` field), a non-decoded ``{bytes, path}`` **dict** (an Image column cast
    to ``decode=False``), or ``None``/absent (text-only)."""
    value = row.get("image")
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value.get("bytes") or value.get("path"))
    return bool(value)


def is_gating_error(exc: BaseException) -> bool:
    """True if ``exc`` (or its cause chain) looks like a Hugging Face gating/auth failure.

    Used to decide whether the "accept the terms + authenticate" hint is actually relevant —
    so a genuinely different failure (missing dep, network, corrupt cache) isn't mislabeled as
    a gating problem."""
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        name = type(cur).__name__.lower()
        msg = str(cur).lower()
        if any(k in name for k in ("gated", "unauthorized", "forbidden", "authentication")):
            return True
        if any(
            k in msg
            for k in ("gated", "401", "403", "not authorized", "must be authenticated",
                      "access to this dataset", "restricted", "log in", "huggingface-cli login")
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _load_error_message(exc: BaseException) -> str:
    """A load-failure message that surfaces the real error, adding the gating hint only when
    the failure is actually a gating/auth one."""
    base = f"could not load the dataset {HF_DATASET!r}: {type(exc).__name__}: {exc}"
    if is_gating_error(exc):
        return (
            f"{base}\ncais/hle is gated on the Hugging Face Hub. Accept the terms at "
            "https://huggingface.co/datasets/cais/hle and authenticate (`huggingface-cli "
            "login`, or set HF_TOKEN) before running the real env. Offline tests inject tasks "
            "instead and need neither the download nor a token."
        )
    return base


def split_tasks(tasks: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    """Positionally split the task list 80/20 into ``train`` / ``test``.

    cais/hle declares no train/test holdout, so — like wordle — we slice by index for a
    deterministic, non-overlapping split (indices are relative to the chosen split)."""
    if split not in ("train", "test"):
        raise ValueError(f"unknown task_split {split!r}; expected 'train' or 'test'")
    cut = int(len(tasks) * TRAIN_FRACTION)
    return tasks[:cut] if split == "train" else tasks[cut:]
