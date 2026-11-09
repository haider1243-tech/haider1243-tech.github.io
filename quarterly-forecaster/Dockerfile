# quarterly-forecaster: one image, run as four services (see docker-compose.yml).
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# libgomp1: OpenMP runtime LightGBM needs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# Optional FinBERT (adds ~1-2 GB): docker compose build --build-arg INSTALL_ML=true
ARG INSTALL_ML=false
COPY requirements-ml.txt .
RUN if [ "$INSTALL_ML" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers; fi

COPY pyproject.toml app.py watchlist.yaml portfolio.example.yaml ./
COPY .streamlit ./.streamlit
COPY static ./static
COPY src ./src
COPY results ./results
COPY pine ./pine
COPY scripts ./scripts
RUN pip install --no-deps -e . \
    && useradd --create-home --uid 10001 app \
    && mkdir -p data reports logs \
    && chown -R app:app /app
USER app

# Each service in docker-compose.yml sets its own command.
CMD ["python", "-m", "forecaster.monitor"]
