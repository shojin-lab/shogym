"""The instrument's hand-checkable gate vectors, run through the shipped CLI path.

Every expected value here is arithmetic and every one is the instrument's. They are
not asserted against hand-built matrices: each vector is a real generator, so the
assertion runs the whole path a family runs, renderer to serializer to observer to
gate. A gate validated only against matrices would be a gate nobody had run.

The payload every section turns on is the KNOWN-DEAD one: a receipt with one verdict
per named convention axis, whose readout is the option itself. Its ceiling is real
and every point of it is reachable by a rule that costs no induction, so a change
that let the gates approve it breaks these tests and nothing else.
"""

from __future__ import annotations

import pytest

from shogym.envs.receipts.observe import observe
from shogym.envs.receipts.protocol import conventions, draw, support_of
from shogym.envs.receipts.registry import load_generator
from shogym.receipts import bits, gate

MASTER = bytes(range(32))


def _gate(name: str, **kwargs):
    generator = load_generator(name)
    return gate(observe(generator, draw(generator, MASTER, 0), "a"), **kwargs)


# ----- 1. an injective readout pins every axis at two blocks -----


@pytest.mark.parametrize("c", [3, 4, 6])
def test_a_verbatim_slot_receipt_is_pinned_at_two_blocks(c: int) -> None:
    result = _gate(f"slots-c{c}")
    assert set(result.blocks.values()) == {2}
    assert not result.r_pass
    assert result.r_axes == []


@pytest.mark.parametrize("c", [3, 4, 6])
def test_a_verbatim_slot_receipt_scores_two_over_c_and_lifts_by_one_over_c(c: int) -> None:
    """The closed form, and it survives the rowwise action rule exactly.

    Each row is one axis and the rows are independent, so answering row by row and
    committing to a whole convention are the same act here. That is why the
    instrument's number is still the number.
    """
    result = _gate(f"slots-c{c}")
    assert result.ceiling == pytest.approx(2.0 / c, abs=1e-12)
    assert result.placebo == pytest.approx(1.0 / c, abs=1e-12)
    assert result.ceiling - result.placebo == pytest.approx(1.0 / c, abs=1e-12)


@pytest.mark.parametrize("c", [3, 4, 6])
def test_a_verbatim_slot_receipt_has_exactly_zero_headroom(c: int) -> None:
    """Floor and ceiling are the same partition, so no action rule can separate them."""
    result = _gate(f"slots-c{c}")
    assert result.floor == pytest.approx(result.ceiling, abs=1e-12)
    assert result.headroom == pytest.approx(0.0, abs=1e-12)
    assert not result.h_pass


@pytest.mark.parametrize("c", [3, 4, 6])
def test_the_placebo_would_have_approved_what_the_floor_rejects(c: int) -> None:
    result = _gate(f"slots-c{c}")
    assert result.ceiling - result.placebo > 0.0
    assert result.ceiling - result.floor <= 0.0
    assert not result.verdict


# ----- 2. item count with an unchanged readout is worth exactly zero -----


def test_twenty_copies_of_a_row_carry_what_one_copy_carries() -> None:
    many, one = _gate("copies-20"), _gate("copies-1")
    assert many.blocks == one.blocks
    assert many.ceiling == pytest.approx(one.ceiling, abs=1e-12)
    assert many.floor == pytest.approx(one.floor, abs=1e-12)
    assert many.placebo == pytest.approx(one.placebo, abs=1e-12)
    assert not many.verdict and not one.verdict


# ----- 3. a merging readout is what lifts the partition past two blocks -----


def test_two_crossed_merges_resolve_all_three_options() -> None:
    result = _gate("merge")
    assert result.blocks["k"] == 3
    assert result.r_pass and result.r_axes == ["k"]
    assert result.s_pass
    assert result.h_pass
    assert result.headroom > 0.0
    assert result.verdict


def test_one_row_can_produce_at_most_two_signatures() -> None:
    """However rich the readout, a single verdict-only row prints two things."""
    result = _gate("one-row")
    assert result.blocks["k"] == 2
    assert not result.r_pass
    assert not result.verdict


# ----- 4. gate R is not asked of a binary axis -----


def test_a_binary_axis_is_outside_gate_r_and_cannot_carry_an_instance() -> None:
    result = _gate("binary")
    assert result.arity == {"bin": 2}
    assert result.blocks["bin"] == 2
    assert not result.r_pass
    assert any("no axis has 3 or more options" in reason for reason in result.reasons)
    assert not result.verdict


# ----- 5. gate S -----


def test_an_axis_labelled_receipt_prints_its_own_interpretation() -> None:
    result = _gate("slots-c4")
    assert not result.s_pass
    assert "labelled by axis" in result.s_structural
    assert any("names the axis" in leak for leak in result.s_leaks)


def test_a_row_labelled_receipt_that_merges_passes_s() -> None:
    result = _gate("merge")
    assert result.s_pass
    assert result.s_leaks == []
    assert not result.s_order_moves
    assert "labelled by scored row" in result.s_structural


# ----- 6. the correlated sampler is rejected -----


def test_the_correlated_affine_sampler_is_rejected() -> None:
    """Four declared slot axes, and a sampler that reaches nine of their 81 assignments.

    This is the source failure shape, not an affine readout over two independent
    axes: the axis catalogue advertises 81 rules and the latent reaches 9, and the
    gate reads the support rather than the advertisement. Holding three slots at the
    drawn convention and varying the fourth leaves the support entirely, so no axis
    resolves anything and R kills it.
    """
    result = _gate("affine")
    assert set(result.blocks.values()) == {1}
    assert not result.r_pass
    assert not result.verdict


def test_two_resolved_rows_pin_every_other_row_under_the_latent_sampler() -> None:
    """The failure the affine exhibit exists to show, stated as it actually is.

    The claim is about what the rows READ, not about what the receipt prints of
    them. Under a two-parameter latent, any two conventions that agree on the first
    two rows agree on all of them, so the rule space is the size of the latent and
    not of the assignment. Under a product sampler the same two rows leave the rest
    open, which is what makes the latent case a finding rather than arithmetic about
    row counts.
    """
    generator = load_generator("affine")
    instance = draw(generator, MASTER, 0)
    latent = [
        generator.key_for(instance.a.table, convention)
        for convention in support_of(generator)
    ]
    # four axes of three options each, read verbatim: the product sampler's shape
    product = [
        (a, b, c, d)
        for a in "012"
        for b in "012"
        for c in "012"
        for d in "012"
    ]

    def pinned(rows) -> bool:
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i][:2] == rows[j][:2] and rows[i] != rows[j]:
                    return False
        return True

    assert pinned(latent)
    assert not pinned(product)


def test_the_correlated_sampler_s_support_is_pinned_directly() -> None:
    """The support, counted, rather than inferred by comparing two constructions.

    Four rows of three options each would be 81 assignments if the rows were drawn
    independently. This sampler realizes 9, one per latent pair, and that number is
    the whole finding: anyone counting assignments reports 81 and is wrong by an
    order of magnitude about how much there is to learn.
    """
    generator = load_generator("affine")
    instance = draw(generator, MASTER, 0)
    realized = {
        tuple(generator.key_for(instance.a.table, convention))
        for convention in support_of(generator)
    }
    assert len(realized) == 9
    # the catalogue advertises four slot axes of three options: 81 assignments
    assert len(generator.AXES) == 4
    assert all(len(axis.options) == 3 for axis in generator.AXES)
    assert len(conventions(generator.AXES)) == 81
    # and the sampler reaches nine of them, which is the whole finding
    assert len(support_of(generator)) == 9


def test_bits_alone_would_not_have_caught_the_correlated_sampler() -> None:
    """The entropy reading approves it. That is why R, S and H exist."""
    assert bits(_observation("affine")) > 0.0
    assert not _gate("affine").verdict


def _observation(name: str):
    generator = load_generator(name)
    return observe(generator, draw(generator, MASTER, 0), "a")


# ----- 7. the thresholds are arguments -----


def test_raising_the_headroom_bar_rejects_what_zero_admitted() -> None:
    assert _gate("merge").verdict
    assert not _gate("merge", min_headroom=0.99).verdict


def test_lowering_the_block_bar_admits_what_two_rejected() -> None:
    """The pin is where it is by arithmetic, and the gate still takes it as a number."""
    assert not _gate("slots-c4").r_pass
    assert _gate("slots-c4", min_blocks=1).r_pass
