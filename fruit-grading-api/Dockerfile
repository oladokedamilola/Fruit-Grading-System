FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both models (now both are local files)
COPY pretrained_mobilenetv2.keras .
COPY fruit_grading_mobilenet.keras ./model.keras

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]