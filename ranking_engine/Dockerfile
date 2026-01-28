# LOCUS: Dockerfile
FROM python:3.9-slim

WORKDIR /app

# We will add requirements later
COPY . .

# Simple command to keep it alive for now
CMD ["python", "-c", "import time; print('Ranking Engine Online'); time.sleep(3600)"]