# Broker container: runs the AutomationBench curriculum broker over network (HTTP) MCP.
#
# It holds the task universe + the train/held-out split in-process and writes provenance
# (task_idx + self-snapshots) to a broker-ONLY volume. The agent container never mounts this
# filesystem, so the current task's target end-state — and the held-out answers — are
# unreachable from the agent even with full Bash + web. Integrity is the environment's job
# (container isolation), not an allow-list's. Build context = repo root.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
# The automationbench extra pulls `datasets` (the domain task loaders); the pinned upstream
# source is fetched lazily into ~/.cache/hgym on the first task (needs network once) or can be
# baked by setting AUTOMATIONBENCH_SRC to a mounted checkout.
RUN pip install --no-cache-dir ".[automationbench]"
# wandb is the OPT-IN live sink (SELFOPT_WANDB=1 + a runtime WANDB_API_KEY). Baked into the
# broker image only — the authoritative scorer streams the metrics; the agent image has no wandb.
# `openai` is an import-time dependency of the pinned upstream AutomationBench source (its Zapier
# tool modules `from openai import OpenAI` at load); AutomationBench SCORING stays keyless +
# offline — no OPENAI_API_KEY is ever needed or set — but the package must be importable.
RUN pip install --no-cache-dir wandb openai
# `experiments` is a namespace package (no __init__.py); PYTHONPATH=/app makes
# `python -m experiments.selfopt.broker` importable.
COPY experiments/selfopt ./experiments/selfopt
ENV SELFOPT_HTTP=1 \
    SELFOPT_HOST=0.0.0.0 \
    SELFOPT_PORT=9000 \
    SELFOPT_SPLIT=train \
    SELFOPT_SELF_DIR=/self \
    SELFOPT_PROV_DIR=/provenance \
    HGYM_SELFOPT_RUNS=/provenance \
    PYTHONPATH=/app
EXPOSE 9000
CMD ["python", "-m", "experiments.selfopt.broker"]
