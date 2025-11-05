FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY train.py .
COPY inference.py .

EXPOSE 8000

CMD ["python", "api.py"]
