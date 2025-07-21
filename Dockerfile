# 1) Базовый образ
FROM python:3.11-slim-bullseye

# Чтобы логи сразу шли в stdout
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 2) Системные библиотеки для Pillow/OpenCV/сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libjpeg-dev \
        zlib1g-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3) Копируем только requirements, чтобы кэш слоя работал
COPY requirements.txt .

# 4) Обновляем pip, ставим CPU-версию torch и остальные зависимости
RUN pip install --upgrade pip setuptools wheel \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# 5) Копируем код приложения и модель
COPY gradio_app ./gradio_app
COPY models       ./models
COPY data/test/images/ ./data/test/images/

# 6) Открываем порт Gradio
EXPOSE 7860

# 7) Запуск
CMD ["python", "-u", "gradio_app/app.py"]