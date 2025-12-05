# Usa una imagen oficial de Python
FROM python:3.11-slim

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Copiamos solo requirements primero (para aprovechar la cache)
COPY backend/requirements.txt ./backend/requirements.txt

# Instalamos dependencias
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copiamos TODO el proyecto al contenedor
COPY . .

# Variables de entorno básicas
ENV PYTHONUNBUFFERED=1

# Comando de arranque: usar el puerto que da Railway
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
