FROM python:3.11-slim

WORKDIR /app

# Копируем зависимости и ставим их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Открываем порты (3000 для UI, 8001 для WebSocket)
EXPOSE 3000 8001

# Запускаем в production режиме
CMD ["reflex", "run", "--env", "prod", "--backend-port", "8001"]