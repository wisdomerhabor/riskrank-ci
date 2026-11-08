# Scoring and explanation service (NFR14): one image, used as a GitHub Actions job service
# container and as a long-lived local service. It stores nothing and holds no secret (NFR4).
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
ENV PYTHONPATH=/app/src RISKRANK_MODELS=/models/models.json
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "riskrank.service:create_app()"]
