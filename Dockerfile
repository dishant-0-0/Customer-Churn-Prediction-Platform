# Base Image
FROM python:3.13-slim

# Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working Directory
WORKDIR /app

# Install System Dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python Dependencies
COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy Application
COPY . /app

# Install Project
RUN pip install --no-cache-dir -e .

# Expose Fast API
EXPOSE 8000

# Health Check
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

# Start FastAPI Port
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
