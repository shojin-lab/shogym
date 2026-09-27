"""One walk of admission over the ledger's first twelve ordinals, made once for two tests.

Two tests make exactly this walk. tests/envs/test_receipts_checks.py holds every instance it admits
to every registered bar, and tests/envs/test_receipts_sampled.py holds the set it admits to the set
the law admits. Both draw ordinals 0 to 11 of the ledger under the key ``bytes(range(32))`` and
report each under the registered bars, and a report is a whole admission, so the second test paid
for the same twelve reports again to read a different fact off them.

So both ask :func:`registered_reports` for them, and both carry :data:`SHARES_REGISTERED_REPORTS`,
which sends them to one worker under ``--dist loadgroup`` so that the walk is made there once.

WHAT THIS DOES NOT DO. It replaces nothing: ``admission.report`` and ``bank.population`` are called
the ordinary way everywhere, and this is twelve of those calls, remembered for the two tests that
ask. A test that changed a check and then asked would get reports made before its change, which is
the stale answer src/shogym/envs/receipts/bank.py records a population cache giving, so only a test
that changes nothing may call it.
"""

from __future__ import annotations

import functools

import pytest

from shogym.envs.receipts import admission
from shogym.envs.receipts.protocol import Generator, Instance, draw

#: How many ordinals the walk covers, from zero.
ORDINALS = 12

#: The group the two tests carry, so that one worker runs both and walks once.
SHARES_REGISTERED_REPORTS = pytest.mark.xdist_group("receipts-ledger-registered-reports")


@functools.cache
def registered_reports(
    generator: Generator, master: bytes
) -> tuple[tuple[Instance, admission.Report], ...]:
    """Each of the first twelve instances with its admission report under the registered bars.

    Remembered by the generator object and the key, so a caller that passes another generator or
    another key gets its own walk rather than this one.
    """
    registered = admission.Thresholds()
    made = []
    for ordinal in range(ORDINALS):
        instance = draw(generator, master, ordinal)
        made.append((instance, admission.report(generator, instance, master, registered)))
    return tuple(made)


__all__ = ["ORDINALS", "SHARES_REGISTERED_REPORTS", "registered_reports"]
