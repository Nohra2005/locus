FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Start the Ranking Engine on port 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]