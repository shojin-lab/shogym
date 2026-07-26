"""Integrity tests for the AutomationBench curriculum broker (issue #57).

These assert the properties that must hold *by construction*:
  - the train / held-out split is a disjoint, total partition (held-out can't leak into training);
  - the train stream is non-repeating and drawn from the train pool only;
  - ``get_task`` leaks no index / target / handle;
  - scoring is authoritative (read off the sealed episode's core-owned evidence), keyless.

Run:  uv run python -m experiments.selfopt.test_broker
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from .broker import Broker, public_split
from .policy import StubPolicy, _text
from .split import make_split, train_stream


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_split_is_disjoint_partition():
    # Pure — no dataset needed.
    s = make_split(600, seed=123, heldout_frac=0.2)
    assert len(s.heldout) == 120 and len(s.train) == 480
    assert set(s.train).isdisjoint(s.heldout)
    assert set(s.train) | set(s.heldout) == set(range(600))
    # Deterministic.
    assert make_split(600, seed=123, heldout_frac=0.2).heldout == s.heldout
    # Different seed → different split, still disjoint.
    s2 = make_split(600, seed=999, heldout_frac=0.2)
    assert s2.heldout != s.heldout and set(s2.train).isdisjoint(s2.heldout)
    print("PASS  split is a deterministic, disjoint, total partition")


def test_train_stream_never_holds_heldout():
    s = make_split(600, seed=123, heldout_frac=0.2)
    stream = train_stream(s, size=50)
    assert len(stream) == len(set(stream)), "train stream repeats"
    assert set(stream).issubset(set(s.train)), "train stream drew a held-out index"
    assert set(stream).isdisjoint(s.heldout)
    print("PASS  train stream is non-repeating and never holds a held-out index")


async def test_no_leakage():
    s = public_split()
    with tempfile.TemporaryDirectory() as tmp:
        br = Broker("train", queue_size=2, prov_dir=Path(tmp) / "prov")
        framing = await br.get_task()
        for forbidden in ("task_idx", "task_id", "idx", "handle", "target", "answer", "info"):
            assert forbidden not in framing, f"get_task leaked {forbidden!r}"
        # The dispensed index really is a train index (privileged peek — the agent never can).
        assert br.active["idx"] in set(s.train)
        assert br.active["idx"] not in set(s.heldout)
        await br._finalize(br.active)
    print("PASS  get_task returns no index/target/handle; dispensed index is train-only")


async def test_authoritative_keyless_scoring():
    with tempfile.TemporaryDirectory() as tmp:
        prov = Path(tmp) / "prov"
        br = Broker("train", queue_size=1, prov_dir=prov)
        framing = await br.get_task()
        # The stub plays without any privileged info; the broker scores via the seal.
        await StubPolicy().play(framing, br)
        assert (await br.get_task()).get("done")
        rows = [json.loads(line) for line in (prov / "results.jsonl").read_text().splitlines() if line.strip()]
        assert len(rows) == 1
        fb = rows[0]["feedback"]
        # AutomationBench's authoritative feedback: reward/partial_credit/success, all in [0,1].
        assert "success" in fb and ("reward" in fb or "partial_credit" in fb)
        assert 0.0 <= rows[0]["reward"] <= 1.0
    print("PASS  scoring is authoritative (seal evidence) and keyless")


async def test_forged_content_cannot_grant_credit():
    # A client could claim anything in its tool args; the score still comes from the sealed world.
    with tempfile.TemporaryDirectory() as tmp:
        br = Broker("train", queue_size=1, prov_dir=Path(tmp) / "prov")
        await br.get_task()
        # Feed a bogus api_fetch claiming success; then done. The recorded reward is the env's.
        out = json.loads(_text(await br.dispatch("done", {})))
        assert out["terminated"] and "result" in out
        assert 0.0 <= out["result"]["reward"] <= 1.0
    print("PASS  the agent cannot forge the score — it is read off the sealed evidence")


async def test_resume_skips_scored_and_continues_stream():
    # A resumed broker rebuilds the SAME seeded train stream but dispenses starting at the next
    # UNscored task: it drops exactly the already-scored tasks, re-runs none of them, loses none of
    # the remainder, and continues the seq numbering.
    with tempfile.TemporaryDirectory() as tmp:
        prov = Path(tmp) / "prov"
        # First pass: dispense + score one task, then interrupt (leave the rest undispensed).
        br = Broker("train", queue_size=3, prov_dir=prov)
        full_stream = list(br.queue)
        await br.get_task()
        await br._finalize(br.active)  # one task scored → one train row in results.jsonl
        rows = [json.loads(x) for x in (prov / "results.jsonl").read_text().splitlines() if x.strip()]
        assert len(rows) == 1
        scored_idx = rows[0]["task_idx"]

        # Resume against the SAME provenance: the scored task is gone from the queue, the rest stay
        # in order, and seq continues past the scored one.
        br2 = Broker("train", queue_size=3, prov_dir=prov, resume=True)
        assert scored_idx not in br2.queue, "resume re-queued an already-scored task"
        assert br2.queue == [i for i in full_stream if i != scored_idx], "resume lost/reordered tasks"
        assert br2.seq == 1 and br2.consumed == 1, "resume did not advance the seq past scored tasks"
        # A fresh (non-resume) broker over the same prov would wrongly re-dispense the scored task.
        assert Broker("train", queue_size=3, prov_dir=prov).queue == full_stream
    print("PASS  resume skips scored tasks, keeps the rest in order, continues the seq")


def main():
    test_split_is_disjoint_partition()
    test_train_stream_never_holds_heldout()
    _run(test_no_leakage())
    _run(test_authoritative_keyless_scoring())
    _run(test_forged_content_cannot_grant_credit())
    _run(test_resume_skips_scored_and_continues_stream())
    print("\nALL BROKER TESTS PASSED")


if __name__ == "__main__":
    main()
