"""shogym port of ORCA-bench: root-cause analysis over a recorded telemetry stack.

ORCA-bench (CC BY 4.0) gives an SRE agent a live observability stack replayed from a frozen
snapshot and asks it to write an incident RCA report; the report is graded by an LLM judge that
ships inside every task. The dataset is 755 tasks over 77 incidents, published on the Harbor hub.

This package is *registration-safe to import*: ``env_v1`` imports nothing heavy at module load,
so ``import shogym`` (which registers the ``orca_bench`` env) stays offline and needs no dataset,
no key, and no Docker. The dataset downloads on demand into ``~/.cache/shogym/orca_bench`` when
the env is first *constructed*; nothing from it is ever committed to this repo.

The port lands in two phases (issue #77). **This is phase 1**: the dataset loader, the task index,
the redacted ``describe()``, the judge's verdict parsing, and the judge preflight, i.e. everything
that works and is tested offline. **Phase 2** adds the compose backend that actually runs a task's
28-service stack (privileged docker-out-of-docker, named-volume staging of the snapshot cache,
sibling teardown). The calls it needed are settled and recorded in the env README, so what
remains there is implementation. Until it lands, constructing and describing tasks works;
*serving* an episode raises
:class:`~shogym.envs.orca_bench.backend.BackendUnavailableError`.
"""
