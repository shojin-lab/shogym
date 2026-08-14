"""shogym port of ORCA-bench: root-cause analysis over a recorded telemetry stack.

ORCA-bench (CC BY 4.0) gives an SRE agent a live observability stack replayed from a frozen
snapshot and asks it to write an incident RCA report; the report is graded by an LLM judge that
ships inside every task. The dataset is 755 tasks over 77 incidents, published on the Harbor hub.

This package is *registration-safe to import*: ``env_v1`` imports nothing heavy at module load,
so ``import shogym`` (which registers the ``orca_bench`` env) stays offline and needs no dataset,
no key, and no Docker. The dataset downloads on demand into ``~/.cache/shogym/orca_bench`` when
the env is first *constructed*; nothing from it is ever committed to this repo.

The port landed in two phases (issue #77). Phase 1 is the dataset loader, the task index, the
redacted ``describe()``, the judge's verdict parsing, and the judge preflight, i.e. everything
that works and is tested offline; phase 2 is the compose backend that runs a task's 28-service
stack for real (privileged docker-out-of-docker, named-volume staging of the snapshot cache,
sibling teardown). Constructing and describing tasks needs neither Docker nor a key; *serving*
an episode needs both, and raises
:class:`~shogym.envs.orca_bench.backend.BackendUnavailableError` when the daemon is missing.

The benchmark's recorded telemetry had aged out of Jaeger's configured lookback, which made a
live stack answer with no services at all. Two runtime knobs restore the pre-expiry window without
editing the pinned image; the mechanism the design originally chose turned out to be inert against
this Jaeger build. The env README explains both under "The clock", and every run records what the
stack could actually see.
"""
