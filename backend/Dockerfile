FROM python:3.11-slim

WORKDIR /app

# opencv-python dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 vim \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# patch triton util. ( read token value from config.yaml and set the header )
COPY ./triton_token.py /usr/local/lib/python3.11/site-packages/ultralytics/utils/triton.py


ENV EASYOCR_MODULE_PATH="/app/.EasyOCR"

EXPOSE 8000

CMD ["fastapi", "run", "main.py", "--port", "8000"]