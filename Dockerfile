FROM python:3.12-slim

WORKDIR /app

# Copiamos todo el proyecto dentro del contenedor
COPY . /app

# Instalamos dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponemos el puerto donde correrá MLflow
EXPOSE 5000

# Comando por defecto: levantar la interfaz de MLflow
CMD ["mlflow", "ui", "--host", "0.0.0.0"]