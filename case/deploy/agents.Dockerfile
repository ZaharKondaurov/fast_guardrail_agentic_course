FROM python:3.14-slim
ARG HERMES_REVISION=86fa9e6ca1de7ee71132d2be0103f3e37d1fa872
RUN apt-get update && apt-get install -y --no-install-recommends git curl ripgrep ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.8.13
WORKDIR /opt/hermes
RUN git init && git fetch --depth 1 https://github.com/NousResearch/hermes-agent.git "${HERMES_REVISION}" && git checkout --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "${HERMES_REVISION}" && uv sync --frozen --no-dev

# Hermes работает на Python 3.14, Arena и обучение AR-12 используют окружения Python 3.12.
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12
WORKDIR /opt/arena
COPY arena/pyproject.toml arena/uv.lock ./
RUN uv sync --frozen --python 3.12 --no-dev
WORKDIR /opt/ar12
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --python 3.12 --group boosting --group transformer
ENV PATH="/opt/ar12/.venv/bin:/opt/hermes/.venv/bin:${PATH}"
ENV HOME=/tmp/agent-home HERMES_HOME=/tmp/hermes-home
USER 10001:10001
WORKDIR /workspace
ENTRYPOINT ["/opt/hermes/.venv/bin/python", "-m", "case.agents.run"]
