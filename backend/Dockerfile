FROM python:3.11-slim

WORKDIR /app

# opencv-python dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

ENV EASYOCR_MODULE_PATH="/app/.EasyOCR"

EXPOSE 8000

CMD ["fastapi", "run", "main.py", "--port", "8000"]