# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Copy uv for faster package management
COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /bin/

WORKDIR /app

# Copy and install dependencies
COPY pyproject.toml ./
RUN uv pip install --system --no-cache-dir playwright

# Production stage
FROM python:3.11-slim

# Install runtime dependencies for Playwright
RUN apt-get update && apt-get install -y \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxss1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install Playwright browsers
RUN playwright install --with-deps chromium

WORKDIR /app

# Copy application code
COPY tools/scrape_website.py ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app
ENV PATH="/root/.local/bin/:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import playwright; print('OK')" || exit 1

CMD ["python", "scrape_website.py"]