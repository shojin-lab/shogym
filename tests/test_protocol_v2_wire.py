"""The v2 wire layer: what a record must be, what it serializes to, and what it hashes to.

Three properties are under test. A record is exactly what the protocol declares or it is
refused, so every missing, extra, mistyped, out-of-range, and wrong-version field is rejected
and the two result unions do not accept each other's members. A record has one serialization,
pinned here by byte fixtures rather than by round-tripping through the same code that produced
them. And every identity is a function of its declared inputs alone, so the submission digest
cannot move when a verdict does, and the message ID stream says nothing about its ordinal.

The byte and hash fixtures were computed by hand from the protocol's own formulas, not read
back out of this implementation.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Dict, Tuple, Type

import pytest

from shogym.serve.protocol_v2 import (
    PROTOCOL_ERROR_CODES,
    Done,
    Payload,
    PresentationAck,
    PresentationCommit,
    ProtocolError,
    PullRequest,
    SealAck,
    SealReject,
    Task,
    TerminalMetadata,
    Wait,
    WireFormatError,
    canonical_bytes,
    canonical_json,
    decode_pull_result,
    decode_terminal_result,
    mcp_text_content,
    presentation_request_identity,
    pull_request_identity,
    require_opaque_id,
    stream_message_id,
    submission_digest,
    terminal_request_identity,
    visible_bytes,
)
from shogym.serve.protocol_v2 import records as records_module

ID_A = "0" * 32
ID_B = "1" * 32
ID_C = "2" * 32
ID_D = "3" * 32
ID_E = "4" * 32
H_A = "a" * 64
H_B = "b" * 64
H_C = "c" * 64
H_D = "d" * 64

TERMINAL_TOOL = "apis.supervisor.complete_task"
REJECT_BODY = 'arg "answer" is \\ required\n\tstop\x07'
UNICODE_BODY = "π ≥ 3, 日本語, \U0001f600"

VALID: Dict[Type[Any], Dict[str, Any]] = {
    PullRequest: {"request_id": ID_E, "last_presented_cursor": ID_D, "protocol_version": 2},
    Task: {
        "message_id": ID_A,
        "attempt_id": ID_B,
        "body": "Reconcile the ledger.",
        "protocol_version": 2,
        "kind": "task",
    },
    Payload: {
        "message_id": ID_A,
        "attempt_id": ID_B,
        "body": "The filing scored 1.",
        "protocol_version": 2,
        "kind": "payload",
    },
    Wait: {
        "message_id": ID_A,
        "retry_after_ms": 4294967295,
        "protocol_version": 2,
        "kind": "wait",
    },
    Done: {"message_id": ID_A, "protocol_version": 2, "kind": "done"},
    SealAck: {
        "message_id": ID_A,
        "attempt_id": ID_B,
        "submission_digest": H_A,
        "canonicalization_version": "appworld.1",
        "protocol_version": 2,
        "kind": "seal_ack",
    },
    SealReject: {
        "message_id": ID_A,
        "attempt_id": ID_B,
        "body": REJECT_BODY,
        "protocol_version": 2,
        "kind": "seal_reject",
        "code": "invalid_arguments",
    },
    ProtocolError: {"code": "invalid_cursor", "protocol_version": 2, "kind": "protocol_error"},
    TerminalMetadata: {
        "request_id": ID_E,
        "last_presented_cursor": ID_D,
        "attempt_id": ID_B,
        "protocol_version": 2,
    },
    PresentationCommit: {
        "attestation_id": ID_C,
        "cursor_before": ID_D,
        "message_id": ID_A,
        "visible_bytes_sha256": H_A,
        "transcript_blob": H_B,
        "provider_turn_blob": H_C,
        "task_start_checkpoint_blob": None,
        "completed_turn": True,
        "stream_state_before_sha256": H_D,
        "protocol_version": 2,
    },
    PresentationAck: {
        "attestation_id": ID_C,
        "cursor": ID_A,
        "stream_state_sha256": H_B,
        "protocol_version": 2,
        "kind": "presentation_ack",
    },
}
CASES = tuple(VALID.items())
FIELD_CASES = tuple((cls, name) for cls, wire in CASES for name in wire)


def wire(cls: Type[Any], **overrides: Any) -> Dict[str, Any]:
    payload = dict(VALID[cls])
    payload.update(overrides)
    return payload


# ----- canonical JSON -----


def test_canonical_json_sorts_keys_by_utf16_code_unit() -> None:
    """The scheme's own sorting example, whose point is that a character outside the basic
    plane sorts by its leading surrogate and so lands before U+FB33, not after it."""
    value = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        "\u0080": "Control",
        "ö": "Latin Small Letter O With Diaeresis",
    }
    expected = (
        '{"\\r":"Carriage Return","1":"One","\u0080":"Control",'
        '"ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign",'
        '"\U0001f600":"Emoji: Grinning Face","דּ":"Hebrew Letter Dalet With Dagesh"}'
    )
    assert canonical_json(value) == expected.encode("utf-8")


def test_canonical_json_escapes_strings() -> None:
    """The scheme's own string example: two-character escapes where they exist, \\u00xx for
    the rest of the control range, and nothing else touched."""
    value = '\u20ac$\u000f\nA\'B"\\\\"/'
    expected = '"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"'
    assert canonical_json(value) == expected.encode("utf-8")


def test_canonical_json_writes_literals_and_integers() -> None:
    value = {"b": [None, True, False], "a": 0, "c": -1, "d": 9007199254740991}
    assert canonical_json(value) == b'{"a":0,"b":[null,true,false],"c":-1,"d":9007199254740991}'


@pytest.mark.parametrize(
    "value",
    [
        4.5,
        float("nan"),
        float("inf"),
        {"a": 1.0},
        {1: "a"},
        {"a": {"b"}},
        "\ud800",
        {"\ud800": 1},
        9007199254740992,
        -9007199254740992,
    ],
)
def test_canonical_json_refuses_what_has_no_canonical_form(value: Any) -> None:
    with pytest.raises(WireFormatError):
        canonical_json(value)


# ----- strict schema -----


@pytest.mark.parametrize("cls,payload", CASES, ids=lambda case: getattr(case, "__name__", ""))
def test_a_valid_payload_decodes_and_re_encodes(cls: Type[Any], payload: Dict[str, Any]) -> None:
    record = cls.from_wire(payload)
    assert record.to_wire() == payload
    assert cls.from_wire(record.to_wire()) == record


@pytest.mark.parametrize("cls,name", FIELD_CASES)
def test_a_missing_field_is_rejected(cls: Type[Any], name: str) -> None:
    payload = wire(cls)
    del payload[name]
    with pytest.raises(WireFormatError):
        cls.from_wire(payload)


@pytest.mark.parametrize("cls,payload", CASES)
def test_an_unknown_field_is_rejected(cls: Type[Any], payload: Dict[str, Any]) -> None:
    with pytest.raises(WireFormatError):
        cls.from_wire({**payload, "hint": "hurry"})
    with pytest.raises(WireFormatError):
        cls.from_wire({**payload, 7: "hurry"})


def _wrong_type(value: Any) -> Any:
    if isinstance(value, bool):
        return "true"
    if isinstance(value, int):
        return "2"
    return 7


@pytest.mark.parametrize("cls,name", FIELD_CASES)
def test_a_mistyped_field_is_rejected(cls: Type[Any], name: str) -> None:
    with pytest.raises(WireFormatError):
        cls.from_wire(wire(cls, **{name: _wrong_type(VALID[cls][name])}))


@pytest.mark.parametrize("cls,payload", CASES)
@pytest.mark.parametrize("version", [1, 3, "2", 2.0, None])
def test_a_wrong_version_is_rejected(cls: Type[Any], payload: Dict[str, Any], version: Any) -> None:
    with pytest.raises(WireFormatError):
        cls.from_wire({**payload, "protocol_version": version})


@pytest.mark.parametrize("cls", list(VALID))
def test_a_payload_that_is_not_an_object_is_rejected(cls: Type[Any]) -> None:
    for payload in ([], "task", 2, None):
        with pytest.raises(WireFormatError):
            cls.from_wire(payload)


@pytest.mark.parametrize(
    "cls,name,value",
    [
        (PullRequest, "request_id", "0" * 31),
        (PullRequest, "last_presented_cursor", "F" * 32),
        (Wait, "retry_after_ms", -1),
        (Wait, "retry_after_ms", 4294967296),
        (Wait, "retry_after_ms", True),
        (Task, "message_id", "0" * 31),
        (Task, "message_id", "0" * 33),
        (Task, "attempt_id", "A" * 32),
        (Task, "attempt_id", " " + "0" * 31),
        (Task, "body", "\ud800"),
        (Task, "kind", "payload"),
        (Payload, "kind", "task"),
        (SealAck, "submission_digest", "a" * 63),
        (SealAck, "submission_digest", "A" * 64),
        (SealAck, "canonicalization_version", ""),
        (SealAck, "canonicalization_version", "Appworld.1"),
        (SealAck, "canonicalization_version", "-appworld"),
        (SealAck, "canonicalization_version", "a" * 65),
        (SealAck, "canonicalization_version", "appworld 1"),
        (SealReject, "code", "invalid_cursor"),
        (ProtocolError, "code", "unknown_code"),
        (ProtocolError, "code", ""),
        (PresentationCommit, "attestation_id", "2" * 33),
        (PresentationCommit, "cursor_before", "3" * 31),
        (PresentationCommit, "visible_bytes_sha256", "a" * 63),
        (PresentationCommit, "transcript_blob", "B" * 64),
        (PresentationCommit, "stream_state_before_sha256", "d" * 65),
        (PresentationCommit, "provider_turn_blob", "c" * 63),
        (PresentationCommit, "task_start_checkpoint_blob", ""),
        (PresentationCommit, "completed_turn", "true"),
        (PresentationCommit, "completed_turn", 1),
        (PresentationAck, "cursor", "0" * 31),
        (PresentationAck, "stream_state_sha256", "z" * 64),
    ],
)
def test_an_out_of_range_value_is_rejected(cls: Type[Any], name: str, value: Any) -> None:
    with pytest.raises(WireFormatError):
        cls.from_wire(wire(cls, **{name: value}))


def test_a_boundary_value_is_accepted() -> None:
    assert Wait.from_wire(wire(Wait, retry_after_ms=0)).retry_after_ms == 0
    assert SealAck.from_wire(wire(SealAck, canonicalization_version="a")) is not None
    assert SealAck.from_wire(wire(SealAck, canonicalization_version="a" + "z" * 63)) is not None
    assert PresentationCommit.from_wire(wire(PresentationCommit, provider_turn_blob=None))


@pytest.mark.parametrize("code", PROTOCOL_ERROR_CODES)
def test_every_declared_error_code_is_accepted(code: str) -> None:
    assert ProtocolError.from_wire(wire(ProtocolError, code=code)).code == code


def test_the_error_code_set_is_closed() -> None:
    """The codes are named here, not derived from the production tuple, so swapping a
    declared code for an invented one fails instead of redefining the refusal surface."""
    assert set(PROTOCOL_ERROR_CODES) == {
        "unsupported_version",
        "invalid_message",
        "consumer_conflict",
        "overlapping_call",
        "invalid_cursor",
        "request_conflict",
        "outstanding_response",
        "already_presented",
        "invalid_attempt",
        "conflicting_seal",
        "frozen_stream",
        "failed_stream",
        "closed_stream",
        "fenced_writer",
        "configuration_mismatch",
    }
    assert len(PROTOCOL_ERROR_CODES) == 15


@pytest.mark.parametrize("cls,name", FIELD_CASES)
def test_every_field_carries_a_rule(cls: Type[Any], name: str) -> None:
    """A field is checked by name or fixed by a string default. A new field with neither would
    otherwise be accepted unvalidated, which is the one way this layer could go quiet."""
    default = cls.__dataclass_fields__[name].default
    assert isinstance(default, str) or name in records_module._CHECKS


@pytest.mark.parametrize("cls,payload", CASES)
def test_a_constructor_validates_what_the_decoder_validates(
    cls: Type[Any], payload: Dict[str, Any]
) -> None:
    with pytest.raises(WireFormatError):
        cls(**{**payload, "protocol_version": 1})


def test_a_record_is_frozen() -> None:
    """A validated record stays validated, so the bytes a later layer hashes are the bytes
    that were checked."""
    task = Task.from_wire(VALID[Task])
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.body = "something else"  # type: ignore[misc]


# ----- the unions -----


@pytest.mark.parametrize("cls", [Task, Payload, Wait, Done])
def test_a_pull_result_decodes_to_its_own_class(cls: Type[Any]) -> None:
    assert type(decode_pull_result(VALID[cls])) is cls


@pytest.mark.parametrize("cls", [SealAck, SealReject])
def test_a_terminal_result_decodes_to_its_own_class(cls: Type[Any]) -> None:
    assert type(decode_terminal_result(VALID[cls])) is cls


@pytest.mark.parametrize("cls", [SealAck, SealReject, ProtocolError, PresentationAck])
def test_a_pull_result_refuses_the_other_unions(cls: Type[Any]) -> None:
    with pytest.raises(WireFormatError):
        decode_pull_result(VALID[cls])


@pytest.mark.parametrize("cls", [Task, Payload, Wait, Done, ProtocolError, PresentationAck])
def test_a_terminal_result_refuses_the_other_unions(cls: Type[Any]) -> None:
    with pytest.raises(WireFormatError):
        decode_terminal_result(VALID[cls])


@pytest.mark.parametrize("payload", [{"kind": "ack"}, {"kind": 7}, {}, [], "task"])
def test_an_unrecognized_kind_is_rejected(payload: Any) -> None:
    with pytest.raises(WireFormatError):
        decode_pull_result(payload)


def test_a_record_refuses_a_sibling_payload() -> None:
    with pytest.raises(WireFormatError):
        Task.from_wire(VALID[Payload])


# ----- serialization -----


@pytest.mark.parametrize("cls", [Task, Payload, Wait, Done, SealAck, SealReject])
def test_a_result_is_exactly_one_text_item(cls: Type[Any]) -> None:
    result = cls.from_wire(VALID[cls])
    items = mcp_text_content(result)
    assert len(items) == 1
    assert set(items[0]) == {"type", "text"}
    assert items[0]["type"] == "text"
    assert items[0]["text"].encode("utf-8") == visible_bytes(result)


@pytest.mark.parametrize(
    "cls", [PullRequest, TerminalMetadata, ProtocolError, PresentationCommit, PresentationAck]
)
def test_what_is_never_presented_has_no_content_item(cls: Type[Any]) -> None:
    record = cls.from_wire(VALID[cls])
    with pytest.raises(WireFormatError):
        visible_bytes(record)
    with pytest.raises(WireFormatError):
        mcp_text_content(record)
    assert canonical_bytes(record)


def test_canonical_bytes_refuses_a_foreign_object() -> None:
    with pytest.raises(WireFormatError):
        canonical_bytes({"kind": "task"})


GOLDEN: Tuple[Tuple[Type[Any], Dict[str, Any], bytes], ...] = (
    (
        Task,
        {},
        b'{"attempt_id":"11111111111111111111111111111111","body":"Reconcile the ledger.",'
        b'"kind":"task","message_id":"00000000000000000000000000000000","protocol_version":2}',
    ),
    (
        Task,
        {"body": UNICODE_BODY},
        '{"attempt_id":"11111111111111111111111111111111","body":"π ≥ 3, '
        '日本語, \U0001f600","kind":"task",'
        '"message_id":"00000000000000000000000000000000","protocol_version":2}'.encode("utf-8"),
    ),
    (
        SealReject,
        {},
        b'{"attempt_id":"11111111111111111111111111111111",'
        b'"body":"arg \\"answer\\" is \\\\ required\\n\\tstop\\u0007",'
        b'"code":"invalid_arguments","kind":"seal_reject",'
        b'"message_id":"00000000000000000000000000000000","protocol_version":2}',
    ),
    (
        SealAck,
        {},
        b'{"attempt_id":"11111111111111111111111111111111",'
        b'"canonicalization_version":"appworld.1","kind":"seal_ack",'
        b'"message_id":"00000000000000000000000000000000","protocol_version":2,'
        b'"submission_digest":"' + H_A.encode() + b'"}',
    ),
    (
        Wait,
        {},
        b'{"kind":"wait","message_id":"00000000000000000000000000000000","protocol_version":2,'
        b'"retry_after_ms":4294967295}',
    ),
    (
        Done,
        {},
        b'{"kind":"done","message_id":"00000000000000000000000000000000","protocol_version":2}',
    ),
    (
        ProtocolError,
        {},
        b'{"code":"invalid_cursor","kind":"protocol_error","protocol_version":2}',
    ),
    (
        PresentationAck,
        {},
        b'{"attestation_id":"22222222222222222222222222222222",'
        b'"cursor":"00000000000000000000000000000000","kind":"presentation_ack",'
        b'"protocol_version":2,"stream_state_sha256":"' + H_B.encode() + b'"}',
    ),
)


@pytest.mark.parametrize("cls,overrides,expected", GOLDEN)
def test_a_record_serializes_to_its_pinned_bytes(
    cls: Type[Any], overrides: Dict[str, Any], expected: bytes
) -> None:
    assert canonical_bytes(cls.from_wire(wire(cls, **overrides))) == expected


def test_the_field_order_a_payload_arrives_in_does_not_reach_the_bytes() -> None:
    forward = Task.from_wire(VALID[Task])
    reversed_payload = dict(reversed(list(VALID[Task].items())))
    assert canonical_bytes(Task.from_wire(reversed_payload)) == canonical_bytes(forward)


# ----- identities -----

PULL_REQUEST = PullRequest.from_wire(VALID[PullRequest])
TERMINAL_METADATA = TerminalMetadata.from_wire(VALID[TerminalMetadata])
PRESENTATION_COMMIT = PresentationCommit.from_wire(VALID[PresentationCommit])
NATIVE_ARGUMENTS = {"answer": "42", "confidence": 3, "notes": None}
ID_KEY = bytes(range(32))


def test_the_pull_request_identity_is_pinned() -> None:
    assert pull_request_identity(PULL_REQUEST) == (
        "38ebf67248e9ada37ac043a26445e7012251f505a419c25988f468145b6e99c1"
    )


def test_the_terminal_request_identity_is_pinned() -> None:
    assert (
        terminal_request_identity(TERMINAL_METADATA, "submit", TERMINAL_TOOL, NATIVE_ARGUMENTS)
        == "4d956d1eb6f9fe3a27a37e9c25b7e0e2e89f806c3be8308e524ba66cd8a1c689"
    )


def test_the_presentation_identity_is_pinned() -> None:
    assert presentation_request_identity(PRESENTATION_COMMIT) == (
        "b1052b2d780daf52f80568ab107cd82ccfae2b03fe4d052b68cd1c9333c46ca3"
    )


def test_an_identity_moves_when_any_of_its_inputs_moves() -> None:
    base = terminal_request_identity(TERMINAL_METADATA, "submit", TERMINAL_TOOL, NATIVE_ARGUMENTS)
    other_metadata = TerminalMetadata.from_wire(wire(TerminalMetadata, request_id=ID_C))
    moved = (
        terminal_request_identity(other_metadata, "submit", TERMINAL_TOOL, NATIVE_ARGUMENTS),
        terminal_request_identity(TERMINAL_METADATA, "file", TERMINAL_TOOL, NATIVE_ARGUMENTS),
        terminal_request_identity(TERMINAL_METADATA, "submit", "apis.other", NATIVE_ARGUMENTS),
        terminal_request_identity(
            TERMINAL_METADATA, "submit", TERMINAL_TOOL, {**NATIVE_ARGUMENTS, "answer": "43"}
        ),
    )
    assert base not in moved
    assert len(set(moved)) == 4
    assert pull_request_identity(PULL_REQUEST) != presentation_request_identity(PRESENTATION_COMMIT)


def test_the_length_prefix_keeps_a_field_from_borrowing_its_neighbours_bytes() -> None:
    """Concatenating without lengths would make these two the same preimage."""
    assert submission_digest(ID_B, "ab", b"c") != submission_digest(ID_B, "a", b"bc")


def test_the_submission_digest_is_pinned() -> None:
    assert submission_digest(ID_B, TERMINAL_TOOL, b"answer=42\n") == (
        "76b70a1929654ae1910658739d762a7353ae549be8c7469e3ca1070f1cb56d3b"
    )


EXCLUDED_FROM_THE_DIGEST = (
    ("score", 1.0),
    ("verdict", "correct"),
    ("hidden_rule", "prefix"),
    ("cell", "placebo"),
    ("branch", "yoked"),
    ("schedule", "delayed"),
    ("failure_class", "quota"),
    ("sealed_at_ns", 1724900000000000000),
    ("payload_bytes", b"the filing scored 1"),
    ("delivery_state", "presented"),
)


def _seal_digest(context: Dict[str, Any]) -> str:
    """What a seal does: build the digest from the three permitted facts in its context."""
    return submission_digest(
        context["attempt_id"], context["native_terminal_name"], context["canonical_submission"]
    )


@pytest.mark.parametrize("name,value", EXCLUDED_FROM_THE_DIGEST)
def test_the_digest_does_not_move_with_what_it_excludes(name: str, value: Any) -> None:
    context = {
        "attempt_id": ID_B,
        "native_terminal_name": TERMINAL_TOOL,
        "canonical_submission": b"answer=42\n",
    }
    assert _seal_digest({**context, name: value}) == _seal_digest(context)


def test_the_digest_takes_only_the_three_facts_it_is_allowed() -> None:
    """The invariance above holds because nothing else can reach the function at all."""
    assert tuple(inspect.signature(submission_digest).parameters) == (
        "attempt_id",
        "native_terminal_name",
        "canonical_submission",
    )


def test_the_digest_moves_with_the_submission_or_the_tool() -> None:
    base = submission_digest(ID_B, TERMINAL_TOOL, b"answer=42\n")
    assert submission_digest(ID_B, TERMINAL_TOOL, b"answer=43\n") != base
    assert submission_digest(ID_B, TERMINAL_TOOL, b"answer=42") != base
    assert submission_digest(ID_B, "apis.supervisor.give_up", b"answer=42\n") != base
    assert submission_digest(ID_C, TERMINAL_TOOL, b"answer=42\n") != base


@pytest.mark.parametrize(
    "attempt_id,terminal,submission",
    [
        (ID_B[:31], TERMINAL_TOOL, b"x"),
        ("lease-7", TERMINAL_TOOL, b"x"),
        (ID_B, TERMINAL_TOOL, "x"),
    ],
)
def test_the_digest_refuses_an_input_that_is_not_what_it_says(
    attempt_id: Any, terminal: Any, submission: Any
) -> None:
    with pytest.raises(WireFormatError):
        submission_digest(attempt_id, terminal, submission)


def test_the_message_id_stream_is_pinned() -> None:
    assert [stream_message_id(ID_KEY, n) for n in (0, 1, 2, 4294967296)] == [
        "a774ae2a2e5adbc2aa445dc36a49d664",
        "65e69e74b7b1ea0cb1a48e793b1251e2",
        "2b1bca9bb3a22851ce78c211e86f8645",
        "cdeee618914196ed63e23898f17f936e",
    ]


def _shared_prefix(left: str, right: str) -> int:
    length = 0
    while length < len(left) and left[length] == right[length]:
        length += 1
    return length


def test_a_message_id_is_an_opaque_id_and_says_nothing_about_its_ordinal() -> None:
    ids = [stream_message_id(ID_KEY, n) for n in range(1000)]
    for value in ids:
        require_opaque_id("message_id", value)
    assert len(set(ids)) == 1000
    # A counter written into the ID would sort back into ordinal order and would leave each ID
    # sharing a long prefix with its neighbour. Neither holds here.
    assert ids != sorted(ids)
    assert max(_shared_prefix(a, b) for a, b in zip(ids, ids[1:])) < 8
    # The same ordinal under another key is another ID, so the ordinal is not what is being
    # read back out. The same key and ordinal always give the same ID.
    assert all(stream_message_id(bytes(32), n) != ids[n] for n in range(100))
    assert stream_message_id(ID_KEY, 7) == stream_message_id(bytes(range(32)), 7)


@pytest.mark.parametrize(
    "id_key,ordinal",
    [
        (bytes(31), 0),
        (bytes(33), 0),
        ("0" * 32, 0),
        (ID_KEY, -1),
        (ID_KEY, 2**64),
        (ID_KEY, True),
        (ID_KEY, "3"),
    ],
)
def test_the_message_id_stream_refuses_a_bad_key_or_ordinal(id_key: Any, ordinal: Any) -> None:
    with pytest.raises(WireFormatError):
        stream_message_id(id_key, ordinal)
