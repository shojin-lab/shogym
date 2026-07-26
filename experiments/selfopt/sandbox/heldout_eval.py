"""Standalone, parallel held-out evaluator for an ALREADY-EXISTING treatment run.

The training study (``study.py``) grows a persistent self over a stream of tasks; the self's real
self-improvement surface is the memory/knowledge-base the CLI writes to ``~/.claude``
(``projects/<slug>/memory/*.md``), NOT the inert workdir ``self/``. This tool measures how much
that accumulated self generalizes, by replaying a checkpoint of it over the held-out split — WITHOUT
re-training. It reads an existing run's provenance, resolves each checkpoint's PAIR of (workdir
self, filtered memory home) at the same task boundary, mounts them into a fresh throwaway agent per
held-out task, and seal-scores the pass authoritatively through the broker.

It reuses the training machinery verbatim (``run_agent`` / ``start_broker`` / ``resolve_checkpoints``
/ ``build_filtered_home`` / ``_heldout_pass`` / ``aggregate``) — so the eval and the training arms
can never drift. It NEVER mutates the training self: the workdir self and the home are only copied
from (throwaway per-unit copies), and both the workdir-self hash and the memory-home hash are
asserted unchanged afterwards.

Beyond the fresh-context checkpoints it also probes the agent's CONTEXT: ``end-context`` resumes the
whole training conversation, and the ``pre-``/``post-compact-N`` arms resume a validated PREFIX of it
(see :mod:`cuts`) — the same trajectory window held raw versus held as the model's own summary.

Held-out runs WEB+BASH-off (the held-out split is public — deny answer-lookup via web AND
curl-via-Bash; tasks are solved through the ``tasks`` MCP ``api_*`` tools, so denying Bash costs
nothing). The (checkpoint × task) units are fully independent and run through a bounded worker pool.

    # plan only (no spend):
    uv run python experiments/selfopt/sandbox/heldout_eval.py --run sandbox-1785221622
    # the real eval (spend) for begin+end over 40 held-out tasks, 4 in parallel:
    uv run python experiments/selfopt/sandbox/heldout_eval.py --run sandbox-1785221622 --go \
        --checkpoints start,end --n 40 --effort low --concurrency 4
    # re-audit an arm that already ran and stamp the verdict onto its rows (no spend, no containers):
    uv run python experiments/selfopt/sandbox/heldout_eval.py --run sandbox-1785221622 \
        --checkpoints pre-compact-2 --audit-rows
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

# Run as a script (sys.path[0] is this dir); put the repo root on the path so the shared experiment
# code — the SAME prompts / split / snapshot / docker plumbing the training arms use — is importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.selfopt import config  # noqa: E402
from experiments.selfopt.arms import HELDOUT_PROMPT, KICKOFF, _HELDOUT_OFF  # noqa: E402
from experiments.selfopt.broker import public_split  # noqa: E402
from experiments.selfopt.sandbox.cuts import (  # noqa: E402
    CUT_PROBE,
    CUTS,
    CutSource,
    InvalidCut,
    PairMismatch,
    acceptance_check,
    assert_future_terms,
    assert_pair_homes,
    cut_source,
    spoken_text,
    read_records,
)
from experiments.selfopt.sandbox.study import (  # noqa: E402
    HeldoutIncomplete,
    _dns_safe,
    _heldout_pass,
    audit_unit_context,
    build_images,
    ensure_network,
    read_session_id,
    resolve_checkpoints,
    resolve_cut_checkpoint,
    write_cut_manifest,
)
from experiments.selfopt.snapshot import content_hash, home_skip  # noqa: E402

#: Suffix marking a MEMORY-WITHHELD variant of any context-carrying arm: same conversation, same
#: workdir self, memory files removed. Isolating the knowledge base has to be available at EVERY
#: context arm, not just the final one — a null measured where the context is still rich says little
#: about the case where a knowledge base should matter most (a just-compacted context, which is a
#: few thousand tokens of summary). So the variant is a name suffix rather than one special-cased
#: checkpoint: ``<arm>-nomem`` runs ``<arm>`` with the memory withheld.
_NOMEM_SUFFIX = "-nomem"

# ``end-context`` is the CONTEXT-LOADED probe: same final self as ``end``, but the agent boots by
# ``--resume``-ing the training session, so it carries the original IN-CONTEXT HISTORY as well as
# the memory. ``end`` (fresh context, memory only) measures the DURABLE artifact; the gap between
# them is how much of the learning lived in the conversation and would be lost on a restart.
_CONTEXT_CP = "end-context"
# Same resumed conversation, but the memory FILES withheld — completes the 2x2 (context x memory)
# and isolates what the on-disk knowledge base adds once the conversation is already loaded.
_CONTEXT_NOMEM_CP = _CONTEXT_CP + _NOMEM_SUFFIX
_CONTEXT_CPS = (_CONTEXT_CP, _CONTEXT_NOMEM_CP)

# Same arm again with the model's own reasoning removed from the replayed turns. The recorded
# thinking TEXT is already empty (the API returns none), so what a resumed agent actually carries
# is the signature; stripping it asks whether the conversation's value is the reasoning or only
# the graded worked examples and the summary the agent wrote of them.
_NOTHINK_SUFFIX = "-nothink"

CHECKPOINTS = ("start", "mid", "end", _CONTEXT_CP, _CONTEXT_NOMEM_CP,
               *CUTS, *(c + _NOMEM_SUFFIX for c in CUTS),
               *(c + _NOTHINK_SUFFIX for c in (_CONTEXT_CP, _CONTEXT_NOMEM_CP, *CUTS)),
               *(c + _NOMEM_SUFFIX + _NOTHINK_SUFFIX for c in CUTS))


def base_checkpoint(cp: str) -> str:
    """The arm ``cp`` measures, with any withheld-surface markers stripped.

    Order matters and mirrors how the names are built: ``-nothink`` is the outermost suffix, so
    ``end-context-nomem-nothink`` resolves through ``end-context-nomem`` to ``end-context``."""
    if cp.endswith(_NOTHINK_SUFFIX):
        cp = cp[: -len(_NOTHINK_SUFFIX)]
    return cp[: -len(_NOMEM_SUFFIX)] if cp.endswith(_NOMEM_SUFFIX) else cp


def withholds_memory(cp: str) -> bool:
    """Whether this arm mounts the conversation but NOT the memory files."""
    return cp[: -len(_NOTHINK_SUFFIX)].endswith(_NOMEM_SUFFIX) \
        if cp.endswith(_NOTHINK_SUFFIX) else cp.endswith(_NOMEM_SUFFIX)


def strips_thinking(cp: str) -> bool:
    """Whether this arm replays the conversation with its ``thinking`` blocks removed."""
    return cp.endswith(_NOTHINK_SUFFIX)


def detect_server_name(stream_path: Path, default: str = "tasks") -> str:
    """The MCP server name THIS run was trained against, read off its trace (tools appear as
    ``mcp__<name>__get_task``). A resumed probe must be given the same name: an agent whose recorded
    context is full of ``mcp__curriculum__*`` reads a differently-named server as its tools having
    disconnected, and stops without playing the task."""
    if stream_path.exists():
        m = re.search(r"mcp__([a-z0-9_]+)__get_task",
                      stream_path.read_text(encoding="utf-8", errors="ignore")[:2_000_000])
        if m:
            return m.group(1)
    return default


def resolve_all(prov: Path, self_dir: Path, self_home: Path, checkpoints: List[str]) -> tuple:
    """The (checkpoint → self/home pair) map plus, for the transcript-cut arms, the bound cut each
    one materializes. Reuses :func:`resolve_checkpoints` for the fresh-context arms and
    :func:`resolve_cut_checkpoint` for the cuts, so every arm resolves through one code path.

    A cut arm's pair is CHECKED here, before any spend: both members of a pre/post pair must resolve
    to the byte-identical archived memory home, or the contrast is confounded with memory evolution
    and the arm does not run."""
    cps = resolve_checkpoints(prov, self_dir, self_home)
    cps[_CONTEXT_CP] = dict(cps["end"])
    cps[_CONTEXT_NOMEM_CP] = dict(cps["end"])
    sources: dict = {}
    for name in checkpoints:
        # A ``-nomem`` / ``-nothink`` arm resolves to exactly the same self/home/cut as the arm it
        # mirrors; the ONLY difference is what is withheld when the home is materialized, so the
        # pair differs in one variable and nothing else.
        cut = CUTS.get(base_checkpoint(name))
        if cut is None:
            # A withheld-surface variant of a non-cut arm (the context probes) mirrors its base.
            if name not in cps and base_checkpoint(name) in cps:
                cps[name] = dict(cps[base_checkpoint(name)])
            continue
        if cut.pair:
            assert_pair_homes(prov, cut, CUTS[cut.pair])
        cps[name] = resolve_cut_checkpoint(prov, self_dir, cut)
        sources[name] = cut_source(self_home, cut)
    return cps, sources


def plan(run: str, arm: str, checkpoints: List[str], n: int, cps: dict) -> dict:
    return {
        "run": run, "arm": arm, "checkpoints": checkpoints, "n_heldout": n,
        "model": config.MODEL, "effort": config.EFFORT,
        "held_out_tools": "web OFF + Bash OFF (public split — deny lookup and curl-the-answer)",
        "measures": {cp: {"self": cps[cp]["self_hash"], "home": cps[cp]["home_hash"]}
                     for cp in checkpoints},
        "isolation": "throwaway copies only; training self/home asserted unchanged after",
    }


def acceptance_verdict(out_root: Path, checkpoints: List[str], indices: List[int]) -> dict:
    """Judge an acceptance run: for every unit, what the agent SAID about the conversation it woke
    up in, and what its own stream says about the context it held.

    A unit passes only if all of it holds — the count it states matches the prefix, it names
    something from inside the prefix, it names nothing from beyond it, the context it actually
    carried is the size the manifest says the prefix is, and (for a raw arm) it stayed raw. Any one
    of these failing alone is the signature of a probe that ran beautifully and measured nothing."""
    out: dict = {"pass": True, "checkpoints": {}}
    for cp in checkpoints:
        cut = CUTS[cp]
        stream_dir = out_root / cp
        manifest_path = stream_dir / "cut_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        # The prefix is only "loaded" if the model held a context of its order — a fresh or
        # summarised context is a fraction of it, which is exactly the confusion being ruled out.
        floor = int(0.5 * manifest.get("context_tokens", 0))
        units: List[dict] = []
        for i, idx in enumerate(indices):
            stream = stream_dir / f"stream-{i:03d}.jsonl"
            check = acceptance_check(spoken_text(stream), cut)
            check["opening_turn"] = spoken_text(stream, before_first_tool=True)
            audit_path = stream_dir / f"prov-{i:03d}" / "context_audit.json"
            audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
            check.update(unit=i, task_idx=idx, context_audit=audit,
                         context_is_the_prefix=audit.get("max_context_tokens", 0) >= floor,
                         stayed_raw=bool(audit.get("stayed_raw")) or not cut.raw)
            check["pass"] = bool(check["pass"] and check["context_is_the_prefix"]
                                 and check["stayed_raw"])
            units.append(check)
        ok = bool(units) and all(u["pass"] for u in units)
        out["checkpoints"][cp] = {"pass": ok, "manifest": manifest, "units": units}
        out["pass"] = out["pass"] and ok
    return out


def audit_rows(run: str, arm: str, checkpoints: List[str]) -> dict:
    """Re-derive the context verdict for arms that have ALREADY run, off their persisted streams,
    and stamp a failing one onto the unit's row.

    A pass stamps its own rows as it runs, so this exists for the arms that ran BEFORE the row
    carried the verdict at all — the case the whole defect is about, where the audit lives only in
    ``context_audit.json`` and the row reads clean. Re-running the pass would stamp them too (the
    resume skip re-checks every scored unit), but that path wants a credential, images and a docker
    daemon to do nothing but read files; this needs none of them and spends nothing.

    Reads only the unit's own stream and writes only that unit's audit + row, so it is safe to point
    at a finished run: no container is started and no unit is replayed."""
    rd = config.RUNS_DIR / run / arm
    out: dict = {"run": run, "arm": arm, "checkpoints": {}, "flagged": 0}
    for cp in checkpoints:
        # The plain Cut is all a verdict needs (its name, and whether the segment is raw), so
        # nothing has to be materialized: a non-cut arm has no raw context to lose and is reported
        # as unaudited rather than silently passed.
        cut = CUTS.get(base_checkpoint(cp))
        stream_dir = rd / "heldout_eval" / cp
        provs = sorted(stream_dir.glob("prov-*")) if stream_dir.exists() else []
        flagged: List[dict] = []
        for prov in provs:
            i = int(prov.name.rsplit("-", 1)[-1])
            reason = audit_unit_context(cut, stream_dir / f"stream-{i:03d}.jsonl", prov)
            if reason is not None:
                flagged.append({"unit": i, "reason": reason})
        out["checkpoints"][cp] = {"units": len(provs), "audited": cut is not None,
                                  "raw_segment": bool(cut is not None and cut.raw),
                                  "flagged": flagged}
        out["flagged"] += len(flagged)
    return out


def run_eval(*, run: str, arm: str, checkpoints: List[str], n: int, concurrency: int,
             oauth: str, wandb_key: Optional[str], project: str,
             acceptance: bool = False) -> dict:
    """Resolve the run's checkpoints, run the held-out pass for each (bounded-parallel), roll up a
    per-checkpoint summary + the begin→end delta, and write it under ``<arm>/heldout_eval/``. The
    training self (workdir) AND memory home are asserted byte-identical before/after — this is a
    measurement, never a training signal.

    A checkpoint whose pass cannot account for every requested unit stops the eval: the scored units
    remain on disk (re-run the same command to retry only the failures) and the only thing written
    is ``summary.incomplete.json``, never a summary that could be mistaken for a finished arm.

    ``acceptance`` runs a cut arm as an ACCEPTANCE TEST rather than as data: the agent is asked, in
    its own words, what conversation it woke up in before it plays its task, and the answer is
    judged against the prefix that was mounted. Its units are written to a SEPARATE tree, so a probe
    that carries an extra question can never be picked up as a unit of the real arm."""
    root = config.RUNS_DIR / run
    rd = root / arm
    self_dir = rd / "self"
    self_home = rd / "self_home"
    prov = rd / "prov"
    if not self_dir.exists():
        raise SystemExit(f"BLOCKED: no training self at {self_dir} — is {run!r}/{arm} a real run?")

    # The context arm replays the FINAL self, resumed into the training conversation; a cut arm
    # replays a validated PREFIX of that conversation beside the memory of the same moment.
    try:
        cps, cut_sources = resolve_all(prov, self_dir, self_home, checkpoints)
    except (InvalidCut, PairMismatch) as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc
    session_id = read_session_id(root, rd / "stream.jsonl")
    server_name = detect_server_name(rd / "stream.jsonl")
    indices = list(public_split().heldout)[:n]
    # The acceptance probe asks an extra question, so it is NOT a unit of the arm; its tree is
    # separate and the scored arm never resumes from it.
    out_root = rd / "heldout_eval" / "acceptance" if acceptance else rd / "heldout_eval"
    # The probe rides in the USER TURN. An agent resumed 900k tokens into a task loop acts on the
    # turn in front of it; the same question appended to the standing system prompt is read as
    # background and stepped straight over into `get_task`.
    prompt = CUT_PROBE if acceptance else KICKOFF
    if acceptance:
        missing = [cp for cp in checkpoints if cp not in cut_sources]
        if missing:
            raise SystemExit(f"BLOCKED: --acceptance probes a transcript-cut arm; {missing} are "
                             f"not cut arms.")
        for cp in checkpoints:
            # The discriminator itself is checked first: a "cannot know this" term that is not
            # actually absent from the prefix would let a wrong conversation pass the test.
            assert_future_terms(read_records(cut_sources[cp].transcript), CUTS[cp])

    pre_self = content_hash(self_dir)
    pre_home = content_hash(self_home, skip=home_skip) if self_home.exists() else None

    # A distinct run-scope for THIS eval's containers so they never collide with a concurrently
    # running training study's brokers (which are scoped by the bare run id).
    scope = f"{run}-hoeval"
    summary: dict = {}
    incomplete: Optional[HeldoutIncomplete] = None
    for cp in checkpoints:
        info = cps[cp]
        cut: Optional[CutSource] = cut_sources.get(cp)
        nomem = withholds_memory(cp)
        nothink = strips_thinking(cp)
        withheld = (", memory withheld" if nomem else "") + (", thinking stripped" if nothink else "")
        how = ("CONTEXT-LOADED" + withheld + " (--resume)" if base_checkpoint(cp) == _CONTEXT_CP else
               f"CONTEXT-CUT at record {cut.cut.last_record}{withheld} (--resume a prefix)" if cut
               else "fresh context")
        print(f"==> held-out {cp}: probe self {info['self_hash']} + home {info['home_hash']} "
              f"[{how}] over {len(indices)} tasks (<= {concurrency} parallel)", flush=True)
        ctx = base_checkpoint(cp) == _CONTEXT_CP or cut is not None
        if ctx and not session_id:
            raise SystemExit(f"BLOCKED: {run!r} has no recoverable session id — "
                             f"cannot run the context-loaded probe.")
        try:
            agg = _heldout_pass(run_id=scope, arm=arm, cp=cp, indices=indices,
                                src=info["self"], home_src=info["home"],
                                stream_dir=out_root / cp, disallowed=_HELDOUT_OFF,
                                system=HELDOUT_PROMPT, prompt=prompt,
                                oauth=oauth, wandb_key=wandb_key,
                                project=project, concurrency=concurrency,
                                resume=session_id if ctx else None,
                                keep_context=base_checkpoint(cp) == _CONTEXT_CP,
                                drop_memory=nomem, strip_thinking=nothink,
                                server_name=server_name,
                                cut=cut)
        except HeldoutIncomplete as exc:
            # Stop here: an arm that cannot account for every requested unit gets NO summary.json.
            # The scored units stay on disk, so the same command re-run retries only the failures.
            incomplete = exc
            print(f"    {cp}: INCOMPLETE — {exc.scored}/{exc.requested} units scored", flush=True)
            break
        summary[cp] = {"n": agg["n"], "mean_reward": agg["mean_reward"],
                       "success_rate": agg["success_rate"],
                       "measured_self": info["self_hash"], "measured_home": info["home_hash"]}
        print(f"    {cp}: n={agg['n']} mean_reward={agg['mean_reward']:.4f}", flush=True)

    if acceptance:
        # Written whether or not the pass completed: the agent's own words about the conversation it
        # woke up in are the evidence, and they are just as informative about a unit that then died.
        verdict = acceptance_verdict(out_root, checkpoints, indices)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "acceptance.json").write_text(json.dumps(verdict, indent=2))
        for cp in verdict["checkpoints"]:
            for unit in verdict["checkpoints"][cp]["units"]:
                print(f"    [{cp} unit {unit['unit']:03d}] "
                      f"{'PASS' if unit['pass'] else 'FAIL'}: {unit['text'][:400]}", flush=True)
        if not verdict["pass"]:
            raise SystemExit(f"BLOCKED: acceptance FAILED — see {out_root / 'acceptance.json'}. "
                             f"The units ran, but not on the context they were supposed to.")

    assert content_hash(self_dir) == pre_self, "eval mutated the training self — isolation broken"
    if pre_home is not None:
        assert content_hash(self_home, skip=home_skip) == pre_home, \
            "eval mutated the training memory home — isolation broken"

    out_root.mkdir(parents=True, exist_ok=True)
    marker = out_root / "summary.incomplete.json"
    if incomplete is not None:
        # The only artifact a partial arm leaves: an explicitly-incomplete record naming the dead
        # units. No summary.json, so nothing here can be read as a finished measurement.
        marker.write_text(json.dumps(
            {"run": run, "arm": arm, "n": n, "complete": False,
             "failed_checkpoint": incomplete.cp, "requested": incomplete.requested,
             "scored": incomplete.scored, "failures": incomplete.failures,
             "completed_checkpoints": summary}, indent=2))
        raise SystemExit(f"BLOCKED: {incomplete}\n  wrote {marker} — NO summary.json for a partial "
                         f"arm; re-run the same command to finish it.")
    marker.unlink(missing_ok=True)  # a previous partial attempt was completed by this pass

    result: dict = {"run": run, "arm": arm, "n": n, "checkpoints": summary,
                    "self_unchanged": True}
    if "start" in summary and "end" in summary:
        result["delta_start_to_end"] = summary["end"]["mean_reward"] - summary["start"]["mean_reward"]
    (out_root / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="existing run id (e.g. sandbox-1785221622)")
    ap.add_argument("--arm", default="treatment", help="which arm's self to probe")
    ap.add_argument("--n", type=int, default=40, help="held-out tasks (first N of the pool)")
    ap.add_argument("--checkpoints", default="start,end",
                    help=f"comma list, subset of {','.join(CHECKPOINTS)} "
                         "(end-context = resume the training session: memory AND in-context "
                         "history; pre-/post-compact-N = resume a validated PREFIX of that session "
                         "beside the memory home of the same moment)")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--concurrency", type=int, default=4, help="max parallel (checkpoint x task)")
    ap.add_argument("--go", action="store_true", help="ACTUALLY run (real spend). Else: plan.")
    ap.add_argument("--acceptance", action="store_true",
                    help="run a transcript-cut arm as an ACCEPTANCE TEST: the agent is asked what "
                         "conversation it woke up in before it plays, and the answer is judged "
                         "against the mounted prefix. Separate output tree — never arm data.")
    ap.add_argument("--audit-rows", action="store_true",
                    help="re-audit the units of an ALREADY-RUN arm from their persisted streams and "
                         "stamp the verdict onto their rows (context_ok=false + reason). No spend, "
                         "no containers, no credential; exits non-zero if anything is flagged.")
    ap.add_argument("--build", action="store_true", help="(re)build the broker + agent images")
    ap.add_argument("--wandb", action="store_true",
                    help="stream broker-side metrics live to W&B (needs WANDB_API_KEY in env)")
    ap.add_argument("--project", default=config.WANDB_PROJECT)
    args = ap.parse_args()

    # Effort/model are read off ``config`` at call time by ``run_agent``; override the module
    # attributes so the CLI flags take effect without touching the training arms.
    config.MODEL = args.model
    config.EFFORT = args.effort

    checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    bad = [c for c in checkpoints if c not in CHECKPOINTS]
    if bad:
        raise SystemExit(f"BLOCKED: unknown checkpoints {bad} (want a subset of {list(CHECKPOINTS)})")

    if args.audit_rows:
        # Before anything that resolves a checkpoint: this reads finished units, so it must work on
        # a run whose transcript or self is no longer resolvable.
        verdict = audit_rows(args.run, args.arm, checkpoints)
        print(json.dumps({"audit": verdict}, indent=2))
        if verdict["flagged"]:
            raise SystemExit(f"BLOCKED: {verdict['flagged']} unit(s) did not measure the arm they "
                             f"ran for; their rows now carry context_ok=false — they are not usable "
                             f"in this arm's contrast.")
        return

    root = config.RUNS_DIR / args.run
    rd = root / args.arm
    try:
        cps, cut_sources = resolve_all(rd / "prov", rd / "self", rd / "self_home", checkpoints)
    except (InvalidCut, PairMismatch) as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc
    p = plan(args.run, args.arm, checkpoints, args.n, cps)
    if cut_sources:
        # Dry-run the cut too: validating the prefix (chain, unanswered tool calls, the facts the
        # arm pins) costs nothing and is the difference between a plan and a hope.
        p["cuts"] = {name: write_cut_manifest(
            src, rd / "heldout_eval" / name, home_src=cps[name]["home"],
            container=_dns_safe(f"selfopt-broker-{args.run}-hoeval-{args.arm}-ho-{name}-000"))
            for name, src in cut_sources.items()}
    print(json.dumps({"plan": p}, indent=2))
    if not args.go:
        print("\n[dry-run] plan only. Re-run with --go to spend on the held-out eval.")
        return

    oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if not oauth:
        raise SystemExit("BLOCKED: no CLAUDE_CODE_OAUTH_TOKEN in env (runtime-only credential).")
    wandb_key = os.environ.get("WANDB_API_KEY") if args.wandb else None
    if args.wandb and not wandb_key:
        print("[heldout_eval] --wandb set but WANDB_API_KEY absent — broker falls back to LocalSink.",
              file=sys.stderr)

    ensure_network()
    if args.build:
        build_images()
    out = run_eval(run=args.run, arm=args.arm, checkpoints=checkpoints, n=args.n,
                   concurrency=args.concurrency, oauth=oauth, wandb_key=wandb_key,
                   project=args.project, acceptance=args.acceptance)
    print(json.dumps({"result": out}, indent=2))


if __name__ == "__main__":
    main()
