# FROM registry.met.no/baseimg/ubuntu:24.04

# for local testing without registry.met.no access
FROM ubuntu:24.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install project into "/mlqc"
WORKDIR /mlqc

# Ensure output is not buffered
ENV PYTHONUNBUFFERED=1

# Enable compilation
ENV UV_COMPILE_BYTECODE=1

ENV UV_NO_CACHE=1

# Copy from cache
# ENV UV_LINK_MODE=copy

# Install dependencies
# RUN --mount=type=cache,target=/root/.cache/uv \
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Install project
COPY . /mlqc
# RUN --mount=type=cache,target=/root/.cache/uv \
RUN uv sync --locked --no-dev

# Ensure the installed binary is on the `PATH`
ENV PATH="/mlqc/.venv/bin/:$PATH"

# Reset entry point
ENTRYPOINT []

# Run application
CMD [ "mlqc", "training", "configs/config.toml" ]
