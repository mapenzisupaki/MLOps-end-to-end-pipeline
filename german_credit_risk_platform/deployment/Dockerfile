# syntax=docker/dockerfile:1
FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY configs ./configs
COPY src ./src
COPY models ./models
COPY artifacts ./artifacts

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
