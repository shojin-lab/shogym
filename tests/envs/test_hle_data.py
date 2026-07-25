"""The real ``cais/hle`` load path — dataset parsing, the text-only filter, and (critically)
that loading needs **no Pillow**.

The offline env tests inject tasks and never exercise ``datasets.load_dataset``, which is how
a Pillow-decode bug slipped through: cais/hle carries Image-typed columns, and merely
*iterating* rows decodes them, pulling in Pillow (which the env doesn't need and doesn't
depend on). These tests build a synthetic ``datasets.Dataset`` shaped like cais/hle — with a
real Image column carrying bytes — and drive ``load_hle_tasks`` against it, so they reproduce
and guard that bug without any Hugging Face auth or download.

Gated on the ``hle`` extra (``importorskip("datasets")``); ``pyarrow`` ships with it.
"""

from __future__ import annotations

import sys

import pytest

datasets = pytest.importorskip("datasets", reason="hle extra not installed")

import pyarrow as pa  # noqa: E402  (comes with datasets)
from datasets import Dataset, DatasetInfo, Features, Image, Value  # noqa: E402

from hgym.envs.hle import data as hle_data  # noqa: E402
from hgym.envs.hle.data import is_gating_error  # noqa: E402

# Marker bytes standing in for an encoded image; never decoded (that would need Pillow).
_FAKE_IMAGE_BYTES = b"\x89PNG\r\n\x1a\n not-a-real-image"


def _synthetic_hle_dataset() -> Dataset:
    """A cais/hle-shaped dataset: one text-only row + one image row, with a real Image-typed
    ``image_preview`` column (bytes present). Built straight from an Arrow table so it needs no
    Pillow to *construct* — iterating it *with decoding* would, which is exactly the bug."""
    image_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    image_cell = {"bytes": _FAKE_IMAGE_BYTES, "path": None}
    table = pa.table(
        {
            "id": pa.array(["t_text", "t_image"]),
            "question": pa.array(["a text-only question", "a question with an image"]),
            "answer": pa.array(["42", "blue"]),
            "answer_type": pa.array(["exactMatch", "exactMatch"]),
            "category": pa.array(["Math", "Art"]),
            # cais/hle's `image` is a string field (base64/URL, empty for text-only).
            "image": pa.array(["", "data:image/png;base64,ZZZZ"]),
            # …plus an Image-typed preview column that would decode via Pillow on iteration.
            "image_preview": pa.array([image_cell, image_cell], type=image_struct),
        }
    )
    features = Features(
        {
            "id": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "answer_type": Value("string"),
            "category": Value("string"),
            "image": Value("string"),
            "image_preview": Image(),
        }
    )
    return Dataset(table, info=DatasetInfo(features=features))


def test_load_hle_tasks_is_text_only_and_needs_no_pillow(monkeypatch) -> None:
    ds = _synthetic_hle_dataset()

    # Sanity: this synthetic set genuinely reproduces the bug — iterating with decoding on
    # raises the Pillow ImportError (so the fix, not a toothless fixture, is what makes the
    # load below succeed).
    with pytest.raises(ImportError, match="Pillow"):
        for row in ds:
            _ = row["image_preview"]

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: ds)
    # Also block PIL outright, so any decode attempt fails loudly rather than silently working
    # in an environment that happens to have Pillow installed.
    monkeypatch.setitem(sys.modules, "PIL", None)

    tasks = hle_data.load_hle_tasks(text_only=True)

    assert [t["id"] for t in tasks] == ["t_text"]  # the image row is dropped
    task = tasks[0]
    assert task["question"] == "a text-only question"
    assert task["answer"] == "42"
    assert task["answer_type"] == "exactMatch"
    assert "image" not in task  # the image field isn't surfaced into the task dict


def test_load_hle_tasks_can_include_image_rows(monkeypatch) -> None:
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _synthetic_hle_dataset())
    monkeypatch.setitem(sys.modules, "PIL", None)

    tasks = hle_data.load_hle_tasks(text_only=False)
    assert {t["id"] for t in tasks} == {"t_text", "t_image"}  # both kept, still no decode


def test_split_tasks_is_positional_and_disjoint() -> None:
    tasks = [{"id": str(i)} for i in range(10)]
    train = hle_data.split_tasks(tasks, "train")
    test = hle_data.split_tasks(tasks, "test")
    assert [t["id"] for t in train] == [str(i) for i in range(8)]
    assert [t["id"] for t in test] == ["8", "9"]
    with pytest.raises(ValueError, match="unknown task_split"):
        hle_data.split_tasks(tasks, "holdout")


def test_is_gating_error_distinguishes_auth_from_other_failures() -> None:
    class GatedRepoError(Exception):
        pass

    assert is_gating_error(GatedRepoError("403 Client Error: you are not authorized"))
    assert is_gating_error(RuntimeError("Access to this dataset is restricted, please log in"))
    # The bug that started this: a Pillow ImportError must NOT be labeled a gating problem.
    assert not is_gating_error(ImportError("To support decoding images, please install 'Pillow'."))
    assert not is_gating_error(ConnectionError("temporary network failure"))
