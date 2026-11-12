FROM python:3.11-slim

# Reflex при сборке скачивает Bun/Node-зависимости: нужны curl и unzip
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Собираем фронтенд на этапе сборки образа: ошибки компиляции видны сразу, старт контейнера быстрее
RUN reflex export --frontend-only --no-zip

# В prod-режиме Reflex интерфейс и его WebSocket-бэкенд должны работать на одном порту
EXPOSE 3000

CMD ["reflex", "run", "--env", "prod", "--frontend-port", "3000", "--backend-port", "3000"]
