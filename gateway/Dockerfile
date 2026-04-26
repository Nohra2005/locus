# Stage 1: Build React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Gateway
FROM python:3.11-slim
WORKDIR /app
COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gateway/ .
COPY --from=frontend-builder /frontend/dist ./frontend/dist
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
