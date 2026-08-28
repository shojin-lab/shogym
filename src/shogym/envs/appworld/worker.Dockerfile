# The image one episode's world runs in.
#
# It exists because the code an agent writes runs *as* the worker process, and a process is not a
# boundary: on one uid, everything the user can read agent-authored code can read. A container is,
# so the worker runs in one with only the served task tree bound in.
#
# Two things are baked here and nowhere else:
#
# - **the interpreter and the pinned release.** `appworld` pins `pydantic<2` and shogym's MCP layer
#   needs `pydantic>=2.7`, so the two cannot share an environment. Here they do not have to: this
#   image holds `appworld` and nothing of shogym.
# - **the app sources**, which the wheel ships packed and which `worker.py install` unpacks. Done
#   at build time so no episode pays for it, and written into site-packages and into
#   `APPWORLD_CACHE`, both outside any path a runtime mount lands on.
#
# **The corpus is deliberately not baked in.** It would save a download, and it carries every
# task's `ground_truth` next to every task's `specs.json`. Baking it would put the answers inside
# the container that runs the agent's code, which is the one thing this image exists to prevent.
# The served tree is mounted per episode instead, and it is a tree the answers are not in.
FROM python:3.12-slim-bookworm@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579 AS build

# `appworld` pins `psutil<6,>=5.9.8`, and psutil 5.9.8 publishes no cp312 wheel for linux/arm64,
# so on an arm64 host pip falls back to the sdist and the install fails in a slim image with no
# compiler. It is compiled here and only the installed tree is carried forward, so the toolchain
# is not in the image an episode runs in.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --prefix=/install "appworld==0.1.3.post1"


FROM python:3.12-slim-bookworm@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579

COPY --from=build /install /usr/local

# Not `/root`: `HOME` is pointed at a writable tmpfs at run time, and anything under it would be
# hidden by that mount.
ENV APPWORLD_CACHE=/opt/appworld/cache \
    PYTHONUNBUFFERED=1

COPY worker.py /opt/shogym/worker.py

# Unpack the nine app modules, byte-compile everything an episode will import, and make the lot
# readable and executable by any uid.
#
# The compile step is startup time rather than tidiness: the container is run read-only, so an
# import that found no `.pyc` would recompile on every episode and be unable to keep the result.
# The `chmod` is because the container is run with the host user's own uid, so that what it writes
# into the mounted output tree is owned by the run, and that uid is not one this image can know at
# build time.
RUN python /opt/shogym/worker.py install \
    && python -m compileall -q /usr/local/lib/python3.12/site-packages /opt/shogym \
    && chmod -R a+rX /opt/appworld /opt/shogym \
    && python -c "import appworld.apps.todoist.models"

ENTRYPOINT ["python", "/opt/shogym/worker.py"]
