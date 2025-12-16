FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Cloud Run sets PORT; default is commonly 8080
ENV PORT=8080
EXPOSE 8080

# Production server for Cloud Run
# Use `sh -c` so $PORT gets expanded at runtime.
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker --bind :${PORT} --workers 1 --threads 8 --timeout 0 main:app"]

