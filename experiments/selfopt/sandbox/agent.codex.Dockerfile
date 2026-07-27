# Codex agent container (cell #2): the Codex CLI + full local tools + web. Connects to the
# broker over the network (streamable-HTTP MCP — Codex consumes it natively, no stdio shim). It
# has NO mount of the broker's filesystem/process, so neither the current task's target nor the
# held-out answers are reachable here — that is what makes "cheating is a finding" scientifically
# clean: the agent is free, the measurement is not. The mirror of agent.Dockerfile (Claude Code).
#
# The subscription credential (ChatGPT auth.json) and the per-run config.toml are provided at
# RUNTIME via an isolated CODEX_HOME mount — never baked into this image. The billed
# OPENAI_API_KEY is stripped from the run env (see study_codex.py / codex_arms.codex_run_env).
FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates ripgrep curl jq python3 python3-pip python3-venv procps \
    && rm -rf /var/lib/apt/lists/*
# The Codex CLI (creds injected at RUNTIME only). Pinned to the studied version for reproducibility.
RUN npm install -g @openai/codex@0.145.0
WORKDIR /work
CMD ["bash"]
