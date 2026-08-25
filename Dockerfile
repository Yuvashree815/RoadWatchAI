FROM python:3.11-slim

WORKDIR /app

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Set default port
ENV PORT=8000
EXPOSE 8000

# Start FastAPI server via python entrypoint (reads PORT dynamically inside main.py)
CMD ["python", "-m", "backend.main"]
