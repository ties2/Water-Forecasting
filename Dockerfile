FROM ubuntu:latest
LABEL authors="student"

ENTRYPOINT ["top", "-b"]

#the new code
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "src.deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]