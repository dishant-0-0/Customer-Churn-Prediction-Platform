# Base Image
FROM python:3.12-slim

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

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy Application
COPY . /app

# Expose Fast API
EXPOSE 8000

# Start FastAPI Port
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]