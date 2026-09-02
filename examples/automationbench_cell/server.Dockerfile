# The measurement's domain: the generation, the durable history, the benchmark source, and every
# score the run commits. It publishes one thing, the gateway's MCP endpoint, and it publishes it
# on a private network rather than on the host, so the only way into this container from the agent
# side is through the protocol.
#
# The durable service the gateway runs on starts inside this container and binds this container's
# own loopback, which is why the history is not reachable from the agent even though the two share
# a network. Build context is the repo root, and the base is pinned by digest: this image carries
# the gateway and the grader, so which build served a run is part of what the run was.
#
# The distributions are the repository's own lock rather than whatever the index resolves today.
# A range resolved live gives the run a gateway, a validator and a benchmark loader nobody chose,
# under a build identity that says it did: the ranges in pyproject.toml admitted a FastMCP two
# major versions above the locked one. So the lock is exported to a fully pinned, hashed
# requirements file and installed from that, which makes uv.lock part of what this image is built
# from and part of what the launch compares.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
# `hle` is where the lock pins `openai`, and the pinned upstream AutomationBench source imports it
# at load (its Zapier tool modules do) while scoring stays offline and keyless. Taking it from the
# lock is what keeps that import a recorded version rather than a live resolution; the package
# only has to be importable. The exporter is uninstalled after the export so that what the image
# holds is the locked set and this repository's own source, and nothing that fetched them.
ARG UV_VERSION
RUN pip install --no-cache-dir "uv==${UV_VERSION}" \
    && uv export --frozen --no-dev --no-emit-project \
        --extra automationbench --extra hle \
        --format requirements.txt -o /tmp/requirements.txt \
    && pip uninstall -y uv \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt
COPY examples/automationbench_cell ./examples/automationbench_cell
ENV PYTHONUNBUFFERED=1
# The project is run from the source this image copied rather than built into a wheel here.
# Building one asks pip for a PEP 517 backend, which is resolved live from the index at whatever
# version it is serving that day: the one input outside both the hashed export above and the
# identity this image carries, and one that decides the wheel supplying the gateway and the
# grader. The path is what the export left out, since it was taken with the project not emitted.
ENV PYTHONPATH=/app/src
CMD ["python", "/app/examples/automationbench_cell/serve.py"]
