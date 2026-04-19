FROM python:3.11-slim

WORKDIR /app

# System dependencies for PyMuPDF and ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY data/card_catalogs/ ./data/card_catalogs/

# Create required directories
RUN mkdir -p data/chroma_db data/cache data/rulebooks data/golden_rules \
             experiments checkpoints

# Default: run test suite
CMD ["pytest", "tests/", "-v", "--tb=short", "-q"]
