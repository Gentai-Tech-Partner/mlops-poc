FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Training runs as one-shot command, not a long-running server
CMD ["python", "train.py"]
