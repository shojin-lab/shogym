"""The receipts environment: sibling tasks under one drawn hidden convention.

`protocol` is what a genre implements, `generators/` holds the genres,
`receipt_ast` is the receipt structure and the one serializer that turns it into
bytes, `streams` is the keyed randomness and the opaque identifiers, `bank` is what
is materialized before launch and what is rendered after a filing seals, `registry`
maps a genre name to its module and its bank, and `env_v1` serves one side of an
admitted instance as `receipts_v1`.

Everything except the rendered bytes is controller-side.
"""
