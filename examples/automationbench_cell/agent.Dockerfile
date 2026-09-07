# The agent's domain: the Claude Code CLI and the tools it runs beside, and nothing this cell
# serves. It holds no benchmark source, no run directory and no grade, so what the agent can read
# about the tasks it is playing is what the server on the far side of the network hands it.
#
# The build the recorded cell reported is pinned here rather than taken from whatever npm calls
# latest, because a different build is a different agent: the system prompt, the built-in tools
# and the compaction belong to the CLI. The base is pinned by digest for the same reason one step
# down: a tag moves, and a rerun months from now on a moved tag would be a different shell, a
# different Node and a different set of OS packages under a record that still named this one.
#
# The OS packages are pinned the same way, and for the same reason. `apt-get install git curl jq`
# resolves against whatever the live Debian repository is serving on the day of the build, so an
# image rebuilt next month carries a different shell surface under the identity that says it does
# not. The archive is therefore read at one fixed moment of its history and every package is named
# with the exact version that moment holds, both handed in as build arguments so that the launch's
# recorded inputs and the packages the model reaches through Bash are one list rather than two.
#
# Credentials are passed at run time and never built in.
FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5
ARG APT_SNAPSHOT
ARG APT_PACKAGES
RUN set -eu; \
    rm -f /etc/apt/sources.list.d/*; \
    { \
      echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/${APT_SNAPSHOT}/ bookworm main"; \
      echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/${APT_SNAPSHOT}/ bookworm-security main"; \
      echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/${APT_SNAPSHOT}/ bookworm-updates main"; \
    } > /etc/apt/sources.list; \
    apt-get -o Acquire::Check-Valid-Until=false update; \
    apt-get install -y --no-install-recommends ${APT_PACKAGES}; \
    rm -rf /var/lib/apt/lists/*
ARG CLAUDE_CODE_VERSION
ARG CLAUDE_CODE_REGISTRY
RUN npm install -g --registry "${CLAUDE_CODE_REGISTRY}" \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"
WORKDIR /work
CMD ["bash"]
