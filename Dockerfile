# ============================================================================
# NeoDEM Training Worker Dockerfile
# Polls server for training jobs, runs SmolVLA LoRA fine-tuning
# ============================================================================

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml ./
COPY worker.py stats_worker.py callbacks.py storage.py config.py ./
COPY trainers/ ./trainers/
COPY scripts/ ./scripts/

# Install dependencies
RUN uv pip install --system -e .

# Non-root user
RUN useradd --system --uid 1001 neodem && \
    chown -R neodem:neodem /app
USER neodem

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "print('ok')"

CMD ["python", "worker.py"]
