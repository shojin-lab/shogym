"""Isolation test for the real held-out probe (issue #57): held-out must NEVER train the self.

The subtle correctness property of the measurement spine — a held-out pass is a *probe*, not a
training signal. The treatment probe plays with a **throwaway copy** of the current self, so even
if the (real) agent scribbles all over — or deliberately rewrites — its workdir during held-out,
the training self is byte-for-byte unchanged afterward. This proves that WITHOUT any model spend
by faking the Claude Code runner with an agent that mutates its workdir as hard as it can.

Run:  uv run python -m experiments.selfopt.test_heldout
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import List, Optional

from . import arms, heldout
from .sink import MetricSink
from .snapshot import content_hash


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _treatment_probe_never_mutates_self():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        self_dir = root / "self"
        (self_dir / ".claude" / "skills").mkdir(parents=True)
        (self_dir / "CLAUDE.md").write_text("# self\nlearned skill A\n")
        (self_dir / ".claude" / "skills" / "s.md").write_text("do X then Y\n")
        before = content_hash(self_dir)

        seen = {"copies": 0}

        def fake_run(cmd: List[str], stream_path: Path, cwd: Optional[Path] = None) -> int:
            assert cwd is not None
            work = Path(cwd)
            # The self's skills/memory rode along into the probe's workdir (a copy).
            assert (work / "CLAUDE.md").read_text() == "# self\nlearned skill A\n"
            assert (work / ".claude" / "skills" / "s.md").exists()
            seen["copies"] += 1
            # A (simulated) held-out agent mutating / scribbling on its workdir as hard as it can.
            (work / "CHEATED.txt").write_text("notes learned from the held-out task\n")
            (work / "CLAUDE.md").write_text("# self\nMUTATED DURING HELD-OUT\n")
            (work / ".claude" / "skills" / "s.md").write_text("REWRITTEN\n")
            Path(stream_path).parent.mkdir(parents=True, exist_ok=True)
            Path(stream_path).write_text('{"type":"fake_event"}\n')
            return 0

        arms.run_claude_stream = fake_run                 # no model spend
        heldout._heldout_indices = lambda size: [0, 1]    # no env load; 2 held-out tasks

        agg = await heldout.evaluate_heldout(
            run_dir=root, checkpoint="mid", sink=MetricSink(), size=2,
            real=True, self_src=self_dir, arm="treatment",
        )

        assert seen["copies"] == 2, "expected one fresh self-copy per held-out task"
        assert agg["measured_self"] == before, "checkpoint recorded the wrong self-version"
        # The training self is byte-for-byte unchanged — held-out leaked NOTHING back.
        assert content_hash(self_dir) == before, "held-out mutated the training self"
        assert (self_dir / "CLAUDE.md").read_text() == "# self\nlearned skill A\n"
        assert not (self_dir / "CHEATED.txt").exists()
        assert (self_dir / ".claude" / "skills" / "s.md").read_text() == "do X then Y\n"
    print("PASS  treatment held-out probes a throwaway copy; training self is never mutated")


async def _control_probe_has_no_self():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        seen = {"empty": 0}

        def fake_run(cmd: List[str], stream_path: Path, cwd: Optional[Path] = None) -> int:
            assert cwd is not None
            work = Path(cwd)
            # Control = the default harness: a fresh, empty workdir. No persistent self at all.
            assert not (work / "CLAUDE.md").exists()
            assert list(p for p in work.iterdir() if p.name != ".mcp.json") == []
            seen["empty"] += 1
            Path(stream_path).parent.mkdir(parents=True, exist_ok=True)
            Path(stream_path).write_text('{"type":"fake_event"}\n')
            return 0

        arms.run_claude_stream = fake_run
        heldout._heldout_indices = lambda size: [0, 1]

        await heldout.evaluate_heldout(
            run_dir=root, checkpoint="end", sink=MetricSink(), size=2,
            real=True, self_src=None, arm="control",
        )
        assert seen["empty"] == 2
    print("PASS  control held-out runs the default harness (fresh, no persistent self)")


def main():
    _run(_treatment_probe_never_mutates_self())
    _run(_control_probe_has_no_self())
    print("\nALL HELD-OUT ISOLATION TESTS PASSED")


if __name__ == "__main__":
    main()
