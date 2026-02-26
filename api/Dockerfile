FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file from the root context
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the api code
COPY ./api /app/api

# Copy the frontend code
COPY ./frontend /app/frontend

EXPOSE 8000

ENV PYTHONPATH=/app
ENV LOG_DIR=/app/data/live_logs

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
