# Agent container: full local tools + the claude CLI + web. Connects to the broker over the
# network (streamable-HTTP MCP). It has NO mount of the broker's filesystem/process, so neither
# the current task's target nor the held-out answers are reachable here — that is what makes
# "cheating is a finding" scientifically clean: the agent is free, the measurement is not.
#
# CREDS are injected at RUNTIME only (docker run -e ...), never baked into this image.
FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates ripgrep curl jq python3 python3-pip python3-venv procps \
    && rm -rf /var/lib/apt/lists/*
# The claude CLI (creds injected at RUNTIME only).
RUN npm install -g @anthropic-ai/claude-code
WORKDIR /work
CMD ["bash"]
