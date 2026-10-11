"""The second versions of the row-banded families: the first versions' rules, truer texts.

`scorecard_v2`, `standing_v2` and `judging_v2` are the first versions with their task
text corrected, registered beside them under names of their own. Each test states the
failure it is here to catch. Each correction is held against the rule code that grades
it, everything else is held against the first version, and a bank of twenty four of each
is filled through the registered admission rule.

THE KEYS ARE FIXED. A bank's population is a function of its key, so a test that reads
an admitted bank reads the same one on every run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import itertools
import json
import re
from collections import Counter
from types import MappingProxyType
from typing import Any

import pytest

from shogym.envs.receipts import admission, checks, streams
from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts.admission import Thresholds, report
from shogym.envs.receipts.generators import banded_rows, judging, scorecard
from shogym.envs.receipts.protocol import (
    FULL_RECEIPT,
    SAMPLED_TWO_OF_TWENTY_FOUR,
    Instance,
    NoFiling,
    Task,
    conventions,
    draw,
    receipt_mask,
    support_of,
)
from shogym.envs.receipts.receipt_ast import frozen_envelope
from shogym.envs.receipts.receipt_law import law_for
from shogym.envs.receipts.registry import GENRES, load_generator
from shogym.envs.receipts.render import Feedback, feedback_for

#: Each second version, and the first version it corrects.
FIRST = {"scorecard_v2": "scorecard", "standing_v2": "standing", "judging_v2": "judging"}
FAMILIES = tuple(FIRST)

#: How many admitted families each bank holds, as for the first versions.
BANK_SIZE = 24

#: Where the second versions send a reader for an id the format prints as a block header.
IN_BRACKETS = "in the brackets that open its block"


def _master(name: str) -> bytes:
    return hashlib.sha256(f"banded-v2:{name}:1".encode()).digest()


def _module(name: str) -> Any:
    return importlib.import_module(GENRES[name])


def _group(name: str) -> pytest.MarkDecorator:
    """One family's tests as one group. Under ``--dist loadgroup`` a group goes to one
    worker, so each family's bank is filled once, on the worker that runs every test that
    reads it, and the three banks fill in parallel on different workers."""
    return pytest.mark.xdist_group(f"receipts-banded-v2-{name}")


PARAMS = [pytest.param(name, marks=_group(name)) for name in FAMILIES]


@pytest.fixture(scope="module", params=PARAMS)
def family(request: pytest.FixtureRequest) -> str:
    return str(request.param)


_BANKS: dict[str, tuple[bank_mod.Bank, bank_mod.Population]] = {}


def _bank(name: str) -> tuple[bank_mod.Bank, bank_mod.Population]:
    """The admitted bank of this family under its fixed key, filled once per process."""
    if name not in _BANKS:
        _BANKS[name] = bank_mod.materialized(load_generator(name), _master(name), BANK_SIZE)
    return _BANKS[name]


def _held(name: str, count: int = BANK_SIZE) -> list[Instance]:
    return list(_bank(name)[1].instances[:count])


def _tasks(name: str, count: int = BANK_SIZE) -> list[Task]:
    return [task for instance in _held(name, count) for task in (instance.a, instance.b)]


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# reading a printed body the way the task text says to
# --------------------------------------------------------------------------


def _column_values(body: str, fmt: str, column: str) -> list[str]:
    """The values printed under `column`, read off the body in its own format.

    Empty when the body never prints a column of that name, which is what the first
    versions' filing lines send a reader to on the bracketed-block surface.
    """
    lines = body.splitlines()
    if fmt in ("csv", "tsv", "semicsv"):
        separator = {"csv": ",", "tsv": "\t", "semicsv": ";"}[fmt]
        rows = [[cell.strip('"') for cell in line.split(separator)] for line in lines]
        if column not in rows[0]:
            return []
        return [row[rows[0].index(column)] for row in rows[1:]]
    if fmt == "pipe":
        rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
        if column not in rows[0]:
            return []
        return [row[rows[0].index(column)] for row in rows[2:]]
    if fmt == "fixed":
        starts = [found.start() for found in re.finditer(r"\S+", lines[0])]
        names = lines[0].split()
        if column not in names:
            return []
        n = names.index(column)
        end = starts[n + 1] if n + 1 < len(starts) else None
        return [line[starts[n]:end].strip() for line in lines[1:]]
    if fmt == "jsonl":
        return [str(record[column]) for record in map(json.loads, lines) if column in record]
    separator = ": " if fmt == "yaml" else " = "
    out = []
    for line in lines:
        key, found, value = line.lstrip("- ").partition(separator)
        if found and key == column:
            out.append(value)
    return out


def _ids_where(text: str, lead: str, body: str, fmt: str) -> list[str]:
    """The ids a reader finds by following the text's sentence that begins with `lead`."""
    found = re.search(
        re.escape(lead) + r" (?:in column (?P<column>[^\s,]+)|(?P<brackets>"
        + re.escape(IN_BRACKETS) + "))",
        _flat(text),
    )
    assert found, lead
    if found.group("brackets"):
        return re.findall(r"^\[([^\]\n]+)\]$", body, re.M)
    return _column_values(body, fmt, found.group("column"))


def _filing_lead(name: str, task: Task) -> str:
    """The words the filing line opens with, up to where it says the id is printed."""
    dom = task.table.dom
    if name.startswith("judging"):
        return "File one line per %s: the %s number" % (dom["entry"], dom["entry"])
    if name.startswith("scorecard"):
        return "File one line per %s: the id" % dom["entity"]
    return "File one line per entry: the id"


# --------------------------------------------------------------------------
# registration, the pin and the streams
# --------------------------------------------------------------------------


def test_each_version_is_registered_beside_the_first_and_keeps_its_rule(family: str) -> None:
    """FAILS IF the second version does not reach its module through the registry under a
    name of its own, if the first version stopped reaching its own, or if the second
    overrides anything but its name and its task text: its axes, oracle, surfaces,
    envelope, copy profile and receipt policy are the first version's objects."""
    generator = load_generator(family)
    first = load_generator(FIRST[family])
    assert generator is _module(family).GENERATOR
    assert first is _module(FIRST[family]).GENERATOR
    assert generator.name == family and first.name == FIRST[family]
    assert type(generator).__mro__[1] is type(first)
    own = {key for key in vars(type(generator)) if not key.startswith("__")}
    assert own == {"name", "describe"}
    assert generator.AXES is first.AXES
    assert generator.ORACLE is first.ORACLE
    assert generator.SHAPE is first.SHAPE
    assert generator.ENVELOPE_SIZE == first.ENVELOPE_SIZE
    assert generator.COPY_PROFILE == first.COPY_PROFILE
    assert generator.RECEIPT_POLICY == FULL_RECEIPT == first.RECEIPT_POLICY
    assert generator.surface_templates() == first.surface_templates()
    assert support_of(generator) == support_of(first)


def test_each_version_is_in_its_own_pin_and_in_no_one_else_s(family: str) -> None:
    """FAILS IF the second version's module or the first version's module is outside the
    second version's code pin, if the gate label or the renderer configuration moved to
    register it, or if the second version's module is inside the pin of any other
    registered genre, its own first version above all: a change to it would then refuse
    their bundles as stale for a change that decides nothing about them."""
    assert admission.GATE_VERSION == "receipts-gates-v4"
    assert bank_mod.RENDERER_CONFIGURATION == "receipts-render-v3"
    generator = load_generator(family)
    pinned = bank_mod.pinned_modules(generator)
    for name in (GENRES[family], GENRES[FIRST[family]], banded_rows.__name__):
        assert name in pinned
    for other in sorted(GENRES):
        if other != family:
            assert GENRES[family] not in bank_mod.pinned_modules(load_generator(other)), other


def test_each_version_draws_under_its_own_name(family: str) -> None:
    """FAILS IF the second version's tables, task identifiers or filler are drawn from the
    first version's coordinates, or if the two stop sharing the convention stream every
    genre shares, which is what makes one key draw one rule for both."""
    generator = load_generator(family)
    first = load_generator(FIRST[family])
    master = _master(family)
    mine, theirs = draw(generator, master, 0), draw(first, master, 0)
    assert dict(mine.convention) == dict(theirs.convention)
    for side in ("a", "b"):
        assert mine.side(side).task_id != theirs.side(side).task_id
        assert generator.table_record(mine.side(side).table) != first.table_record(
            theirs.side(side).table
        )
    for instance, name in ((mine, family), (theirs, FIRST[family])):
        assert instance.envelope.filler == streams.filler_stream(
            master, banded_rows.FILLER_ALPHABET, generator.ENVELOPE_SIZE, name, 0, "pad"
        )


# --------------------------------------------------------------------------
# the text: the first version's, but for the stated sentences
# --------------------------------------------------------------------------


def _id_place(fmt: str, column: str) -> str:
    return IN_BRACKETS if fmt == "ini" else "in column %s" % column


def _stated_changes(name: str, task: Task) -> list[tuple[str, str]]:
    """Each sentence the second version changes, as the first prints it and as it is now."""
    dom = task.table.dom
    cols = dom["cols"]
    place = _id_place(dom["fmt"], cols[0])
    if name == "scorecard_v2":
        return [
            (
                "which is the sum of the points it earns on the house scorecard's four "
                "factors: its hazard grade (column %s), the criticality of the asset it "
                "concerns (column %s), its exposure, and whether it repeats an earlier fault "
                "(column %s)." % (cols[1], cols[2], cols[5]),
                "which is the total of the points it earns. It earns points on the house "
                "scorecard's three factors: its hazard grade (column %s), the criticality of "
                "the asset it concerns (column %s) and its exposure." % (cols[1], cols[2]),
            ),
            (
                "The band is read off the score with this table:",
                "The band is read off the score with this table, and a repeat of an "
                "earlier fault (column %s) can move the %s to a higher band:"
                % (cols[5], dom["entity"]),
            ),
            ("the id in column %s," % cols[0], "the id %s," % place),
        ]
    if name == "judging_v2":
        entry = dom["entry"]
        return [
            ("Some %ss are disqualified," % entry, "Some %ss may be disqualified," % entry),
            ("(column %s) is penalised." % cols[6], "(column %s) may be penalised." % cols[6]),
            (
                "running order means the %s number in column %s," % (entry, cols[0]),
                "running order means the %s number %s," % (entry, place),
            ),
            (
                "File one line per %s: the %s number in column %s,"
                % (entry, entry, cols[0]),
                "File one line per %s: the %s number %s," % (entry, entry, place),
            ),
        ]
    return [("the id in column %s," % cols[0], "the id %s," % place)]


def test_the_text_is_the_first_version_s_but_for_the_stated_sentences(family: str) -> None:
    """FAILS IF, on any surface, the second version's task text differs from the first
    version's text for the same table by anything but the sentences listed here, if a
    first version sentence listed here is not one the first version prints, or if the
    text names an option or moves with the drawn rule.

    Compared word for word, because the second versions fill the filing line to width
    where the first wrapped it by hand. Every admitted instance is read, so every surface
    of both pools is.
    """
    generator = load_generator(family)
    first = load_generator(FIRST[family])
    surfaces = set()
    for task in _tasks(family):
        surfaces.add(task.surface)
        assert task.text == generator.describe(task.public())
        before = _flat(first.describe(task.public()))
        for old, new in _stated_changes(family, task):
            assert old in before, (task.surface, old)
            before = before.replace(old, new)
        assert before == _flat(task.text), task.surface
    assert surfaces == set(generator.surface_templates())
    instance = _held(family, 1)[0]
    assert checks.check_lint(generator, instance).passed
    assert checks.check_invariance(generator, instance).passed


def test_the_id_is_where_the_text_says_it_is_on_every_surface(family: str) -> None:
    """FAILS IF following the filing line's directions for the id on any surface does not
    read off the identifiers the grader reads, in printed order, or if a filing built from
    them with the key does not score 1.0 under the registered reading and scorer. It also
    fails if the first version's directions find the ids on the bracketed-block surface,
    which would mean this test no longer sees the defect the second version corrects."""
    generator = load_generator(family)
    first = load_generator(FIRST[family])
    for task in _tasks(family):
        dom = task.table.dom
        lead = _filing_lead(family, task)
        ids = _ids_where(task.text, lead, task.table.body, dom["fmt"])
        assert ids == list(generator.row_identifiers(task.table)), task.surface
        filed = "\n".join("%s,%s" % pair for pair in zip(ids, task.key))
        score, _ = generator.score(task, generator.parse_and_canonicalize(task, filed))
        assert score == 1.0, task.surface
        old = _ids_where(first.describe(task.public()), lead, task.table.body, dom["fmt"])
        assert (old == []) == (dom["fmt"] == "ini"), task.surface


# --------------------------------------------------------------------------
# each correction, against the rule that grades it
# --------------------------------------------------------------------------


@_group("scorecard_v2")
def test_scorecard_v2_states_the_band_every_repeat_rule_gives() -> None:
    """FAILS IF, under any of the 96 rules on an admitted list, a first fault's band is not
    the printed table's band for its three-factor score, if a repeat's band is below it,
    or if either repeat rule never lifts a repeat above it. Those are the second version's
    sentences: the score totals the points earned on three factors, the band is read off
    the table, and a repeat can move an order to a higher band. Under `oneband_up` that move adds no
    points, which is why the first version's sum over four factors misstated it."""
    generator = load_generator("scorecard_v2")
    lifted: Counter[str] = Counter()
    for task in _tasks("scorecard_v2", 3):
        dom = task.table.dom
        for rule in conventions(generator.AXES):
            key = generator.key_for(task.table, rule)
            for row, band in zip(task.table.rows, key):
                score = (
                    scorecard.HAZARD[rule["hazard_points"]][row.hazard]
                    + scorecard.CRITICALITY[rule["criticality_points"]][row.criticality]
                    + scorecard.exposure_points(row, rule["exposure_basis"])
                )
                read = scorecard.band_index(score, dom["cuts"])
                got = dom["bands"].index(band)
                if not row.repeat:
                    assert got == read, (task.surface, rule, row.row_id)
                    continue
                assert got >= read, (task.surface, rule, row.row_id)
                if got > read:
                    lifted[rule["repeat_rule"]] += 1
    assert set(lifted) == set(generator.AXES[3].options), lifted
    text = _flat(_tasks("scorecard_v2", 1)[0].text)
    assert "three factors" in text and "four factors" not in text
    assert "can move the" in text and "to a higher band:" in text


def _placed(rows: list[judging.EntryRow], rule: dict[str, str], position: int) -> int:
    return judging.heat_places(rows, rule)[position]


@_group("judging_v2")
def test_judging_v2_says_what_the_rules_may_do() -> None:
    """FAILS IF the rule code disqualifies anybody under `bar_none` or penalises every late
    entry under `late_last`, which is what makes "may" the true word in both sentences,
    or if the second version's extract still says entries are disqualified and late
    entries penalised.

    The late case is asked of the heat itself: a late entry is placed where it would be
    placed on time, under a rule that disqualifies it and places late entries last. Every
    other bar disqualifies somebody on an admitted table, so "may" is not "never".
    """
    generator = load_generator("judging_v2")
    unpenalised = 0
    for task in _tasks("judging_v2", 3):
        scores = [row.scores for row in task.table.rows]
        assert not any(judging.disqualified(s, "bar_none") for s in scores)
        for bar in generator.AXES[1].options[1:]:
            assert any(judging.disqualified(s, bar) for s in scores), bar
        for heat in range(judging.HEATS):
            rows = [row for row in task.table.rows if row.heat == heat]
            for rule in conventions(generator.AXES):
                if rule["late"] != "late_last":
                    continue
                for position, row in enumerate(rows):
                    if not row.late:
                        continue
                    on_time = list(rows)
                    on_time[position] = dataclasses.replace(row, late=False)
                    late, prompt = _placed(rows, rule, position), _placed(on_time, rule, position)
                    if judging.disqualified(row.scores, rule["bar"]):
                        assert late == prompt, (rule, row.row_id)
                        unpenalised += 1
                    else:
                        assert late >= prompt, (rule, row.row_id)
    assert unpenalised > 0
    task = _tasks("judging_v2", 1)[0]
    entry, late_column = task.table.dom["entry"], task.table.dom["cols"][6]
    text = _flat(task.text)
    assert "Some %ss may be disqualified," % entry in text
    assert "(column %s) may be penalised." % late_column in text
    assert "Some %ss are disqualified" % entry not in text
    assert "(column %s) is penalised" % late_column not in text


@_group("judging_v2")
def test_judging_v2_running_order_is_the_number_the_text_points_at() -> None:
    """FAILS IF, on any surface, the entry numbers read where the extract says running
    order is printed, lower first, do not give each heat's entries in the running order
    the tie rules fall back on. Both tie rules end there, so a reader who cannot find the
    number cannot break a tie the way the key does."""
    for task in _tasks("judging_v2"):
        dom = task.table.dom
        lead = "running order means the %s number" % dom["entry"]
        ids = _ids_where(task.text, lead, task.table.body, dom["fmt"])
        by_id = {row.row_id: row for row in task.table.rows}
        assert sorted(ids) == sorted(by_id), task.surface
        for heat in range(judging.HEATS):
            mine = [i for i in ids if by_id[i].heat == heat]
            numbered = sorted(mine, key=lambda i: int(re.search(r"\d+$", i).group()))
            assert [by_id[i].order for i in numbered] == sorted(by_id[i].order for i in mine)


# --------------------------------------------------------------------------
# the perfect reader
# --------------------------------------------------------------------------


def _graded(generator, instance: Instance, raw: str, feedback: Feedback):
    task = instance.a
    canonical = generator.parse_and_canonicalize(task, raw)
    return canonical, generator.render_receipt(task, canonical, task.key, feedback)


def _raw(generator, task: Task, filed: tuple[str, ...]) -> str:
    return "\n".join(f"{i},{v}" for i, v in zip(generator.row_identifiers(task.table), filed))


def test_the_perfect_reader_of_a_full_receipt_recovers_the_rule(family: str) -> None:
    """FAILS IF the ideal reader of a full receipt leaves more than the drawn rule standing
    or files B imperfectly, whatever A was filed with.

    The receipt is rendered from four filings: the right answers, the answers of a rule
    one decision away, an explicit empty value on every row, and nothing at all, which
    the reading takes as no filing. Every one reports a verdict and the correct answer on
    every row, so the reader holds exactly one rule and scores 1.0 on B.
    """
    generator = load_generator(family)
    for instance in _held(family, 2):
        task = instance.a
        feedback = feedback_for(generator, task, frozen_envelope(instance.envelope))
        near = dict(instance.convention)
        axis = generator.AXES[0]
        near[axis.name] = next(o for o in axis.options if o != near[axis.name])
        raws = [
            _raw(generator, task, tuple(task.key)),
            _raw(generator, task, tuple(generator.key_for(task.table, near))),
            _raw(generator, task, ("",) * task.n_rows),
            "",
        ]
        for raw in raws:
            canonical, receipt = _graded(generator, instance, raw, feedback)
            if raw == "":
                assert canonical == NoFiling("empty")
            evidence = banded_rows.receipt_evidence(receipt)
            assert [evidence[row] for row in range(task.n_rows)] == list(task.key)
            reading = generator.perfect_reader(instance, receipt)
            assert reading == banded_rows.perfect_reader(generator, instance, receipt)
            assert reading.consistent == 1
            assert reading.score == 1.0 and reading.expected == 1.0
            assert reading.filing == tuple(instance.b.key)


def test_the_perfect_reader_agrees_with_the_receipt_law(family: str) -> None:
    """FAILS IF the library's receipt law and an independent law calculation disagree
    about the ideal level of a two-of-twenty-four receipt, or if the domain's perfect
    reader of a rendered receipt files anything but the posterior's modes under the
    published tie rule, or claims an expected share other than theirs.

    The first part is a law calculation, not a reading: over every one of the 276 masks
    it groups the rules by what the receipt would show and takes the top share of B's
    answers in each group, which is the ideal level the law averages. The second part
    goes through real rendered receipts, on eight masks and three truths each, and counts
    the posterior's modes itself: the rules that agree with the truth on the masked rows,
    and per row of B the answer most of them give, ties going to the answer the task's
    published order lists first. Ties have to arise, so that the tie rule is exercised.
    """
    generator = load_generator(family)
    instance = _held(family, 1)[0]
    rules = conventions(generator.AXES)
    fold = generator.normalize_answer
    a_keys = [generator.key_for(instance.a.table, rule) for rule in rules]
    b_keys = [generator.key_for(instance.b.table, rule) for rule in rules]
    a_codes = banded_rows.coded(a_keys, fold)
    b_codes = banded_rows.coded(b_keys, fold)

    total = 0.0
    count = 0
    for mask in itertools.combinations(range(instance.a.n_rows), 2):
        classes: dict[bytes, list[int]] = {}
        for index in range(len(rules)):
            classes.setdefault(a_codes[index, list(mask)].tobytes(), []).append(index)
        for members in classes.values():
            top = [
                Counter(int(b_codes[m, row]) for m in members).most_common(1)[0][1]
                for row in range(b_codes.shape[1])
            ]
            total += sum(top) / float(b_codes.shape[1])
            count += len(members)
    law = law_for(generator, instance, "a", policy=SAMPLED_TWO_OF_TWENTY_FOUR)
    assert abs(law.ideal - total / count) < 1e-9

    envelope = frozen_envelope(instance.envelope)
    published = [fold(token) for token in generator.answer_ranks(instance.b.table)]
    tied_rows = 0
    for n in range(8):
        mask = receipt_mask(SAMPLED_TWO_OF_TWENTY_FOUR, _master(family), family, n, "A", 24)
        feedback = Feedback(
            policy=SAMPLED_TWO_OF_TWENTY_FOUR, mask=mask,
            neutral=MappingProxyType(dict(envelope.neutral)),
        )
        for index in (0, len(rules) // 2, len(rules) - 1):
            task = instance.a
            truthful = Instance(
                generator=instance.generator, genre=instance.genre,
                ordinal=instance.ordinal, convention=MappingProxyType(rules[index]),
                a=Task(
                    label=task.label, task_id=task.task_id, surface=task.surface,
                    table=task.table, text=task.text, key=a_keys[index], mask=mask,
                ),
                b=Task(
                    label="B", task_id=instance.b.task_id, surface=instance.b.surface,
                    table=instance.b.table, text=instance.b.text, key=b_keys[index],
                    mask=instance.b.mask,
                ),
                envelope=instance.envelope,
            )
            raw = _raw(generator, task, ("",) * 24)
            _, receipt = _graded(generator, truthful, raw, feedback)
            assert set(banded_rows.receipt_evidence(receipt)) == set(mask)
            reading = banded_rows.perfect_reader(generator, truthful, receipt)

            members = [
                m for m in range(len(rules))
                if all(fold(a_keys[m][r]) == fold(a_keys[index][r]) for r in mask)
            ]
            assert reading.consistent == len(members)
            share = 0.0
            for row in range(24):
                tally = Counter(fold(b_keys[m][row]) for m in members)
                most = max(tally.values())
                modes = {answer for answer, times in tally.items() if times == most}
                tied_rows += len(modes) > 1
                mode = next((answer for answer in published if answer in modes), min(modes))
                assert fold(reading.filing[row]) == mode, (n, index, row)
                share += most / float(len(members))
            assert reading.expected == pytest.approx(share / 24.0, abs=1e-12)
            hits = sum(1 for got, want in zip(reading.filing, b_keys[index]) if fold(got) == fold(want))
            assert reading.score == round(hits / 24.0, 6)
    assert tied_rows > 0


# --------------------------------------------------------------------------
# admission and the bank
# --------------------------------------------------------------------------


def test_an_admitted_bank_of_twenty_four_builds(family: str) -> None:
    """FAILS IF the second version cannot fill a bank of twenty four instances under the
    registered bars and the library's bank law, or if an instance it holds is not one
    admission admits when asked again. Admission runs the gates and every named check,
    copy, materiality, the placebo, the oracle, the lint and invariance among them, on
    each instance it holds."""
    generator = load_generator(family)
    bank, held = _bank(family)
    assert bank.generator == family
    assert len(held.instances) == BANK_SIZE
    assert held.considered <= 2 * BANK_SIZE
    made = report(generator, held.instances[0], bank.master, Thresholds())
    assert made.admitted, made.lines()
    assert made.gates.verdict and made.gates.ceiling == 1.0
