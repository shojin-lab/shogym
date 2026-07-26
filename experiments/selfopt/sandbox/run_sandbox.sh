#!/bin/bash
# Containerized self-improvement TREATMENT-TRAIN demo (issue #57), modeled on the selfopt-poc
# poc_container topology. This is the single-arm smoke of the isolated topology; the FULL
# two-arm + held-out study (treatment + control + start/mid/end held-out, web-off held-out,
# broker-side W&B) runs under `study.py` in this same directory:
#     uv run python experiments/selfopt/sandbox/study.py --go --build --arm both --wandb
#
# The agent container (full tools + claude + web) connects to an
# ISOLATED broker container over the network. The current task's target end-state AND the
# held-out answers are unreachable from the agent's filesystem/process — integrity is the
# environment's job, not the allow-list's. "Get Better" is the only instruction; the workdir
# ("the self") persists across the whole stream on the agent volume and the broker snapshots it
# at every task boundary (a broker-only volume the agent cannot reach or forge).
#
# Topology:
#   selfopt-broker (container)  holds targets + the train/held-out split in-process;
#        |  http MCP :9000       provenance (task_idx + self-snapshots) on a broker-ONLY volume
#        |  on selfopt-net
#   selfopt-agent  (container)  claude -p "Get Better ...", full tools incl web, self-dir RW;
#                               NO broker mounts
#
# CREDS: runtime-only. Supply CLAUDE_CODE_OAUTH_TOKEN (or ANTHROPIC_API_KEY) in your shell env.
# On macOS the claude OAuth token lives in the Keychain (not ~/.claude*), so a read-only mount
# does NOT authenticate — inject the token at runtime:
#     export CLAUDE_CODE_OAUTH_TOKEN=$(security find-generic-password -s "Claude Code-credentials" -w | jq -r .claudeAiOauth.accessToken)
# (Never write it to a tracked file. This script passes it via -e at runtime only.)
set -euo pipefail
cd "$(dirname "$0")/../../.."                      # -> repo root
ROOT="$(pwd)"
SANDBOX="$ROOT/experiments/selfopt/sandbox"
RUNS="${HGYM_SELFOPT_RUNS:-$ROOT/experiments/selfopt/runs}"
RUN_ID="sandbox-$(date +%s)"
SELF="$RUNS/$RUN_ID/self"          # the persistent "self" (agent RW; broker RO for snapshots)
PROV="$RUNS/$RUN_ID/provenance"    # broker-only provenance (agent never mounts this)
QUEUE_SIZE="${SELFOPT_QUEUE_SIZE:-15}"
MODEL="${SELFOPT_MODEL:-claude-sonnet-5}"
EFFORT="${SELFOPT_EFFORT:-low}"

if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "BLOCKED: no credential in env. Set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY."
  echo "See this script's header for the Keychain extraction one-liner."
  exit 3
fi

mkdir -p "$SELF" "$PROV"
printf '# self\n' > "$SELF/CLAUDE.md"    # a minimal seed; the agent evolves it if it chooses to

echo "==> building images"
docker build -q -f "$SANDBOX/broker.Dockerfile" -t selfopt-ab-broker:latest "$ROOT" >/dev/null
docker build -q -f "$SANDBOX/agent.Dockerfile"  -t selfopt-ab-agent:latest  "$ROOT" >/dev/null

docker network inspect selfopt-net >/dev/null 2>&1 || docker network create selfopt-net
docker rm -f selfopt-broker selfopt-agent >/dev/null 2>&1 || true

# Broker-side W&B (opt-in, RUNTIME-only): the authoritative scorer streams the training-feedback
# metrics + self-snapshot artifacts live. Supply WANDB_API_KEY in your shell env (same pattern as
# the OAuth token — never baked into an image or committed); no key ⇒ the broker's LocalSink
# writes to the provenance volume, no crash, no network. The broker container gets network egress
# to wandb.ai; the AGENT container still gets NO broker mount — volume isolation is untouched.
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(-e SELFOPT_WANDB=1 -e WANDB_API_KEY="$WANDB_API_KEY" \
              -e SELFOPT_WANDB_PROJECT="${SELFOPT_WANDB_PROJECT:-hgym-selfopt-cell1}")
  echo "==> W&B live streaming ON (broker-side)"
fi

echo "==> starting isolated broker (train stream, size=$QUEUE_SIZE)"
docker run -d --name selfopt-broker --network selfopt-net \
  -e SELFOPT_QUEUE_SIZE="$QUEUE_SIZE" \
  -e SELFOPT_RUN_NAME="$RUN_ID-treatment-train" \
  "${WANDB_ARGS[@]}" \
  -v "$SELF":/self:ro \
  -v "$PROV":/provenance \
  selfopt-ab-broker:latest >/dev/null
sleep 6

cat > "$RUNS/$RUN_ID/.mcp.json" <<'JSON'
{"mcpServers": {"curriculum": {"type": "http", "url": "http://selfopt-broker:9000/mcp/"}}}
JSON

PROMPT='Get Better.

A curriculum will feed you a stream of tasks through an MCP server named `curriculum`. Nothing external loops for you — drive yourself: (1) call get_task to pull the next task; it returns the task instructions, its budget, and the exact set of tools available for that task; if it returns {done:true}, stop. (2) Carry the task out using the tools get_task listed for it, then call done to submit it for scoring. (3) Repeat until the stream is done. You cannot see targets or scores in advance; you are scored authoritatively. Begin now.'

echo "==> running agent (Get Better; full tools incl web; persistent self)"
docker run --rm --name selfopt-agent --network selfopt-net \
  -e CLAUDE_CODE_OAUTH_TOKEN="${CLAUDE_CODE_OAUTH_TOKEN:-}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  -v "$SELF":/work:rw \
  -v "$RUNS/$RUN_ID/.mcp.json":/work/.mcp.json:ro \
  -w /work \
  selfopt-ab-agent:latest \
  claude -p "$PROMPT" \
    --model "$MODEL" --effort "$EFFORT" \
    --mcp-config /work/.mcp.json --strict-mcp-config \
    --permission-mode dontAsk \
    --allowedTools "Bash,Read,Write,Edit,Glob,Grep,TodoWrite,ToolSearch,Skill,NotebookEdit,Agent,WebFetch,WebSearch,mcp__curriculum__get_task,mcp__curriculum__queue_info,mcp__curriculum__api_search,mcp__curriculum__api_fetch,mcp__curriculum__base64_encode,mcp__curriculum__done,mcp__curriculum__terminate" \
    --output-format stream-json --verbose --include-partial-messages \
  | tee "$RUNS/$RUN_ID/stream.jsonl"

echo "=== provenance (broker-side; authoritative, seal-scored) ==="
cat "$PROV/results.jsonl" 2>/dev/null || echo "(no results — agent produced no scored task)"
docker rm -f selfopt-broker >/dev/null 2>&1 || true
echo "run artifacts: $RUNS/$RUN_ID"
