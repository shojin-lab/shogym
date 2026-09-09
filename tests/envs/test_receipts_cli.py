"""``shogym receipts``: materialize, draw, gate, check, screen, bundle, verify, list.

There is deliberately no way to draw an ordinal the bank does not hold, and no
`--seed`. A free seed makes the gate universe and the review cherry-pickable, and a
live run must never serve a draw nobody gated.

A bank is not dealable. What is dealable is a bundle, and `verify` recomputes one
rather than reading anything off it.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from shogym.cli import main
from shogym.envs.receipts import admission as admission_mod
from shogym.envs.receipts.registry import BANK_DIR_VAR
from tests._fixtures.receipts_bundle import private_bundle, screen_artifact

#: The registered parameters, supplied explicitly because there is no default to fall
#: back on. These are test values and nothing here claims they are the registration.
#: The registered bars are the defaults, so nothing here has to pass them. The list
#: is kept for the cases that deliberately override one.
BARS: list[str] = []


def _run(argv: list[str]) -> int:
    try:
        main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


@pytest.fixture(autouse=True)
def _banks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own bank directory, controller-side and disposable."""
    monkeypatch.setenv(BANK_DIR_VAR, str(tmp_path / "banks"))
    return tmp_path


#: The instances each bank this module has walked holds. A population is a function of the bank
#: and the running code and is recomputed rather than stored, so one bank walked twice in one
#: process is one answer twice. Every command below still recomputes its own; this is only for
#: the helper that builds a command's input.
_POPULATIONS: dict = {}


def _held(bank):
    """The instances a bank holds, walked once per bank."""
    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts.registry import load_generator

    known = _POPULATIONS.get(bank)
    if known is None:
        known = bank_mod.population(bank, load_generator("ledger"))
        _POPULATIONS[bank] = known
    return known


@functools.lru_cache(maxsize=4)
def _filled(size: int):
    """One real bank of this size, filled once for the whole module, with its population.

    Filling a bank walks admission over every ordinal it considers, which is what every command
    below is measured in. A test whose subject is `gate`, `check` or `draw` needs a frozen bank
    to run against and does not need a freshly filled one, so the fill happens once here and
    :func:`_frozen_bank` writes the same frozen input into each test's own directory.

    The real materialize command is what five of the tests below are about, and they still run
    it: its report, its refusal to overwrite, its forced replacement, its refusal of a gate
    vector, and the whole materialize/bundle/verify/list chain.
    """
    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts import streams
    from shogym.envs.receipts.registry import load_generator

    bank, held = bank_mod.materialized(
        load_generator("ledger"), streams.new_master_key(), size
    )
    _POPULATIONS[bank] = held
    return bank, held


def _frozen_bank(size: int) -> None:
    """Freeze that bank where this test's commands will look for it.

    The same file the materialize command writes, written the same way: a bank is five fields
    and every command below recomputes everything else from them.
    """
    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts.registry import bank_path

    path = bank_path("ledger")
    path.parent.mkdir(parents=True, exist_ok=True)
    bank_mod.save_bank(_filled(size)[0], path)


def test_list_says_when_nothing_is_materialized(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["receipts", "list"]) == 0
    out = capsys.readouterr().out
    assert "ledger" in out
    assert "no bank; nothing to bundle yet" in out
    assert "NOT DEALABLE: no admission bundle" in out


def test_materialize_then_list_reports_the_frozen_bank(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(["receipts", "materialize", "ledger", "--size", "2", *BARS]) == 0
    made = capsys.readouterr().out
    assert "materialized 2 instances" in made
    assert "bank digest" in made
    assert admission_mod.GATE_VERSION in made
    assert "passed" in made

    assert _run(["receipts", "list"]) == 0
    listed = capsys.readouterr().out
    # The roster describes nothing it did not recompute. A development bank is
    # mentioned, and its stored size and renderer are not printed beside a verified
    # bundle as though something had checked them.
    assert "a development bank is present, unverified and not dealable" in listed
    assert "bank of 2" not in listed
    assert "receipts-render-v1" not in listed
    assert "NOT DEALABLE: no admission bundle" in listed
    assert "gate vectors (never dealt)" in listed


def test_a_bank_alone_is_not_dealable(capsys: pytest.CaptureFixture[str]) -> None:
    """Materializing says so, and the roster says so: only a bundle can be dealt."""
    assert _run(["receipts", "materialize", "ledger", "--size", "1", *BARS]) == 0
    assert "a bank is not dealable" in capsys.readouterr().out
    assert _run(["receipts", "list"]) == 0
    listed = capsys.readouterr().out
    assert "NOT DEALABLE: no admission bundle" in listed
    assert "unverified and not dealable" in listed


def test_a_gated_bank_holds_only_passers(capsys: pytest.CaptureFixture[str]) -> None:
    _frozen_bank(3)
    assert _run(["receipts", "gate", "ledger", "--instances", "3"]) == 0
    assert "0 of 3 instances rejected" in capsys.readouterr().out


def test_materialize_refuses_a_gate_vector(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["receipts", "materialize", "slots-c4", "--size", "1", *BARS]) == 1
    assert "never dealt" in capsys.readouterr().out


def test_gate_reports_a_vector_through_the_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["receipts", "gate", "slots-c4", "--instances", "1"]) == 1
    out = capsys.readouterr().out
    assert "GATE R  resolution     FAIL" in out
    assert "HEADROOM                 0.0000" in out
    assert f"1 of 1 instances rejected by {admission_mod.GATE_VERSION}" in out


def test_gate_admits_the_merging_vector(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["receipts", "gate", "merge", "--instances", "1"]) == 0
    assert "VERDICT                USABLE" in capsys.readouterr().out


def test_check_reports_every_named_check(capsys: pytest.CaptureFixture[str]) -> None:
    _frozen_bank(1)
    assert _run(["receipts", "check", "ledger", "--instances", "1", *BARS]) == 0
    out = capsys.readouterr().out
    for name in ("exercise", "materiality", "copy", "fixation", "envelope", "graded",
                 "placebo", "neutral", "oracle", "lint", "invariance"):
        assert name in out
    assert "0 of 1 instances failed a named check" in out


def test_gate_and_check_refuse_to_invent_out_of_bank_instances(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """These commands report on what would be dealt, so with no bank there is nothing
    to report on. Generating fresh ones would reopen a cherry-pickable universe.

    Refused in a line and a nonzero exit, the way `draw` refuses. A traceback is what
    an operator got before, and which command printed a message depended on which one
    they ran.
    """
    assert _run(["receipts", "gate", "ledger", "--instances", "1"]) == 1
    assert "Materialize a bank" in capsys.readouterr().out
    assert _run(["receipts", "check", "ledger", "--instances", "1", *BARS]) == 1
    assert "Materialize a bank" in capsys.readouterr().out


def test_an_unknown_genre_is_refused_in_a_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(["receipts", "materialize", "nosuchgenre"]) == 1
    assert "no genre or vector named" in capsys.readouterr().out


def test_a_malformed_bank_is_refused_in_a_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every shape a bad input takes ends in a line and a nonzero exit.

    A file that is not there, a name nothing maps to, a file that will not read, and a
    value that is not what its record says it is. Only `FileNotFoundError` and
    `KeyError` were caught, so a malformed bank left a traceback on three commands.
    """
    from shogym.envs.receipts.registry import bank_path

    bank_path("ledger").parent.mkdir(parents=True, exist_ok=True)
    bank_path("ledger").write_text('{"generator": "ledger"}', encoding="utf-8")
    for command in (
        ["receipts", "draw", "ledger"],
        ["receipts", "gate", "ledger", "--instances", "1"],
        ["receipts", "check", "ledger", "--instances", "1", *BARS],
    ):
        assert _run(command) == 1
        assert "a bank record carries exactly" in capsys.readouterr().out


def test_a_screen_taken_on_another_family_is_refused_in_a_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json as _json

    artifact = tmp_path / "screen.json"
    artifact.write_text(
        _json.dumps({
            "family": "ledger", "model": "a scripted policy",
            "task_seeds": [str(i) for i in range(40)],
            "pairs": [
                {"instance": f"t{i:02d}", "filing": f"f{i:02d}",
                 "placebo": 0.4, "graded": 0.6, "oracle": 0.9}
                for i in range(40)
            ],
            "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36, "floor": 0.0,
            "floor_rule": "drop", "candidates_screened": 1, "selection_note": "",
        }),
        encoding="utf-8",
    )
    assert _run(["receipts", "screen", "binary", "--outcomes", str(artifact)]) == 1
    assert "is being read as evidence for" in capsys.readouterr().out


def test_a_screen_carrying_an_out_of_range_number_is_refused_in_a_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """JSON integers are unbounded and Python floats are not.

    A record carrying 10**400 raised `OverflowError` out of the float conversion, past
    every range check and out of the command that promises to refuse rather than
    crash. An out-of-range number is a malformed record like any other.
    """
    import json as _json

    payload = {
        "family": "ledger", "model": "a scripted policy",
        "task_seeds": [str(i) for i in range(40)],
        "pairs": [
            {"instance": f"t{i:02d}", "filing": f"f{i:02d}",
             "placebo": 0.4, "graded": 0.6, "oracle": 0.9}
            for i in range(40)
        ],
        "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36, "floor": 0.0,
        "floor_rule": "drop", "candidates_screened": 1, "selection_note": "",
    }
    payload["pairs"][0]["graded"] = 10**400  # type: ignore[index]
    artifact = tmp_path / "screen.json"
    artifact.write_text(_json.dumps(payload), encoding="utf-8")
    assert _run(["receipts", "screen", "ledger", "--outcomes", str(artifact)]) == 1
    assert "out of range" in capsys.readouterr().out


def test_a_gate_vector_needs_no_bank() -> None:
    assert _run(["receipts", "gate", "merge", "--instances", "1"]) == 0


def test_check_exits_nonzero_when_a_threshold_bites(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _frozen_bank(1)
    assert _run(
        ["receipts", "check", "ledger", "--instances", "1",
         "--max-copy-score", "0.0", "--max-flip-score", "0.95", "--min-leverage", "0.05"]
    ) == 1
    assert "1 of 1 instances failed a named check" in capsys.readouterr().out


def test_materialize_refuses_to_overwrite_without_force(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(["receipts", "materialize", "ledger", "--size", "1", *BARS]) == 0
    capsys.readouterr()
    assert _run(["receipts", "materialize", "ledger", "--size", "1", *BARS]) == 1
    assert "pass --force" in capsys.readouterr().out
    assert _run(["receipts", "materialize", "ledger", "--size", "1", "--force", *BARS]) == 0


def test_draw_needs_a_bank_before_it_will_render(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(["receipts", "draw", "ledger"]) == 1
    assert "materialize one first" in capsys.readouterr().out


def test_draw_prints_both_siblings_and_all_three_cells(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _frozen_bank(2)
    assert _run(["receipts", "draw", "ledger"]) == 0
    out = capsys.readouterr().out
    assert "TASK A" in out and "TASK B" in out
    assert "GRADED cell" in out and "PLACEBO cell" in out and "ORACLE cell" in out
    assert "HOUSE CONVENTIONS" in out
    assert "all three match the envelope: True" in out
    # the drawn convention is printed for the reader, never inside a task text
    assert "drawn convention:" in out
    assert "commitment" in out


def test_draw_refuses_an_ordinal_the_bank_does_not_hold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _frozen_bank(2)
    assert _run(["receipts", "draw", "ledger", "--instance", "99"]) == 1
    assert "is not in this bank" in capsys.readouterr().out


def test_draw_takes_no_seed() -> None:
    _frozen_bank(1)
    with pytest.raises(SystemExit):
        main(["receipts", "draw", "ledger", "--seed", "4"])


def test_draw_renders_every_registered_filing_shape(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _frozen_bank(1)
    for shape, expected in (
        ("canonical", "component score 1.000000"),
        ("empty", "no filing (empty)"),
        ("malformed", "no filing (no_known_identifier)"),
    ):
        assert _run(["receipts", "draw", "ledger", "--filing", shape]) == 0
        assert expected in capsys.readouterr().out


def test_tasks_only_stops_before_the_cells(capsys: pytest.CaptureFixture[str]) -> None:
    _frozen_bank(1)
    assert _run(["receipts", "draw", "ledger", "--tasks-only"]) == 0
    out = capsys.readouterr().out
    assert "TASK B" in out
    assert "GRADED cell" not in out


# ----- the bundle is what can be dealt -----


def _artifacts(room: Path) -> tuple[Path, Path]:
    """A screen artifact and a review pack for the materialized ledger bank."""
    import json

    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts.checks import FILING_CLASSES
    from shogym.envs.receipts.registry import bank_path, load_generator
    from shogym.envs.receipts.review import required_coverage

    generator = load_generator("ledger")
    bank = bank_mod.load_bank(bank_path("ledger"))
    held = _held(bank)
    screen = room / "screen.json"
    screen.write_text(json.dumps({
        "family": generator.name,
        "model": "a scripted policy",
        "task_seeds": [str(i) for i in range(40)],
        "pairs": [
            {"instance": f"t{i:02d}", "filing": f"f{i:02d}",
             "placebo": 0.4, "graded": 0.6, "oracle": 0.9}
            for i in range(40)
        ],
        "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36, "floor": 0.0,
        "floor_rule": "drop", "candidates_screened": 1, "selection_note": "",
    }), encoding="utf-8")
    coverage = required_coverage(
        generator, FILING_CLASSES,
        [i.a.n_rows for i in held.instances] + [i.b.n_rows for i in held.instances],
    )
    envelope = min(i.envelope.size for i in held.instances)
    folder = room / "renders"
    folder.mkdir(exist_ok=True)
    renders = []
    for index, (category, key) in enumerate(coverage.required):
        kind = "task" if category == "surface" else "cell"
        artifact = folder / f"{index:03d}.txt"
        artifact.write_text("R" * ((400 if kind == "task" else envelope) + 8))
        renders.append({"category": category, "key": key, "kind": kind,
                        "path": f"renders/{artifact.name}"})
    pack = room / "pack.json"
    pack.write_text(json.dumps({
        "reviewer": "andrew", "checklist": ["read one"], "seeds": [0],
        "family": generator.name, "bank": bank_mod.bank_identity(bank),
        "renders": renders,
    }), encoding="utf-8")
    return screen, pack


def test_screen_scores_the_artifact_it_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command reads the artifact it is given, and the artifact is the whole input.

    `screen` opens no bank: the rows, the model, the seeds and the bars are all in the file, and
    that is the design, because a screen scored under bars supplied elsewhere is a screen whose
    verdict depends on who reran it. So the rows here are the suite's own forty-pair artifact
    rather than a bank materialized and a review pack built to obtain one file.
    """
    import json

    artifact = tmp_path / "screen.json"
    artifact.write_text(json.dumps(screen_artifact()), encoding="utf-8")
    assert _run(["receipts", "screen", "ledger", "--outcomes", str(artifact)]) == 0
    out = capsys.readouterr().out
    assert "VERDICT                ADMITTED" in out
    assert "model a scripted policy, 40 task seeds" in out
    assert "screen bars: room 0.05, ratio 0.25, pairs 36 (registered)" in out


def test_bundle_then_verify_then_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["receipts", "materialize", "ledger", "--size", "1"]) == 0
    capsys.readouterr()
    screen, pack = _artifacts(tmp_path)
    assert _run([
        "receipts", "bundle", "ledger", "--screen", str(screen), "--review", str(pack)
    ]) == 0
    made = capsys.readouterr().out
    assert made.startswith("bundle ")
    digest = made.split()[1]
    assert len(digest) == 64

    assert _run(["receipts", "verify", "ledger"]) == 0
    verified = capsys.readouterr().out
    assert "VERIFIED" in verified
    assert digest[:16] in verified
    assert (
        "gate bars:   max_copy_score=%g" % admission_mod.REGISTERED_MAX_COPY_SCORE
    ) in verified
    assert "screen bars: room 0.05, ratio 0.25, pairs 36 (registered)" in verified

    assert _run(["receipts", "list"]) == 0
    listed = capsys.readouterr().out
    assert "DEALABLE" in listed and "NOT DEALABLE" not in listed
    assert "screen bars: room 0.05, ratio 0.25, pairs 36 (registered)" in listed


def test_verify_refuses_a_bundle_whose_file_was_edited(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An honest bundle, cloned into this genre's directory and then edited.

    What is under test is the recomputation `verify` makes, so the bundle it is given is frozen
    input: a private copy of one built once for this process. Building it again here would pay
    for a bank, a review pack and a build to reach the one line that matters, and the copy is
    the same bytes under the same address.
    """
    import json

    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.envs.receipts.registry import bundle_dir, bundles

    private_bundle(bundle_dir("ledger"))
    root = bundles("ledger")[0]
    stored = json.loads((root / bundle_mod.BANK).read_text(encoding="utf-8"))
    (root / bundle_mod.BANK).write_text(
        json.dumps({**stored, "size": 9}), encoding="utf-8"
    )
    assert _run(["receipts", "verify", "ledger"]) == 1
    assert "NOT VERIFIED" in capsys.readouterr().out


def test_verify_says_so_when_there_is_nothing_to_verify(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(["receipts", "verify", "ledger"]) == 1
    assert "no bundles for" in capsys.readouterr().out


def test_a_bundle_that_does_not_verify_is_not_left_behind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from shogym.envs.receipts.registry import bundles

    _frozen_bank(1)
    screen, pack = _artifacts(tmp_path)
    stored = json.loads(pack.read_text(encoding="utf-8"))
    kept = [e for e in stored["renders"] if e["category"] != "option"]
    pack.write_text(json.dumps({**stored, "renders": kept}), encoding="utf-8")
    assert _run([
        "receipts", "bundle", "ledger", "--screen", str(screen), "--review", str(pack)
    ]) == 1
    assert "does not make a bundle" in capsys.readouterr().out
    assert bundles("ledger") == []


def test_the_roster_prints_no_claim_from_an_unverified_bank(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stale development bank beside a verified bundle would give an operator two
    descriptions of what looks like one family, with nothing saying which was checked.

    The verified bundle is a private copy of the one built once for this process, because what
    the roster is being held to is what it prints over a bundle it verified and a bank it did
    not, and building the bundle here again would say nothing more about that.
    """
    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts.registry import bank_path, bundle_dir

    private_bundle(bundle_dir("ledger"))
    bank_mod.save_bank(
        bank_mod.Bank(generator="ledger", genre="invented-genre",
                      renderer="invented-renderer", master=bytes(range(32)), size=999),
        bank_path("ledger"),
    )
    assert _run(["receipts", "list"]) == 0
    listed = capsys.readouterr().out
    assert "999" not in listed
    assert "invented-renderer" not in listed
    assert "invented-genre" not in listed
    assert "DEALABLE" in listed
