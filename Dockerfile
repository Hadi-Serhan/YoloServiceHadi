FROM python:3.10-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY torch-requirements.txt requirements.txt ./


RUN pip install -r torch-requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
