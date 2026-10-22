# AICA LLM server - production image (Cloud Run, GKE, or any container host).
#   docker build -t aica-llm-server .
#   docker run -p 8080:8080 --env-file .env aica-llm-server
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

COPY app ./app
RUN useradd --system --uid 10001 --home /srv aica
USER 10001

# Cloud Run injects PORT; behind Google's front end every hop is a trusted proxy.
ENV APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8080 \
    FORWARDED_ALLOW_IPS=*
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/health/live', timeout=2)"

# exec form: SIGTERM reaches the server directly and triggers the graceful drain
CMD ["python", "-m", "app"]
