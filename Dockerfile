# Use Python 3.13 slim for minimum footprint
FROM python:3.13-slim

# Install system dependencies only if strictly required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, efficient dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /app

# Copy lockfiles first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (exclude dev tools to save space/cost)
RUN uv sync --frozen --no-dev

# Copy project source
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY checkpoints/ ./checkpoints/
COPY main.py ./

# Required by Cloud Run
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Start the dashboard using your CLI entrypoint
CMD ["uv", "run", "python", "main.py", "dashboard"]
