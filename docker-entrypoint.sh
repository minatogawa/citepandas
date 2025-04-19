#!/bin/bash

# Espera o PostgreSQL inicializar
echo "Waiting for PostgreSQL..."
until pg_isready -h db -U postgres; do
    sleep 1
done
echo "PostgreSQL started"

# Inicializa o banco de dados usando o comando Flask
echo "Initializing database..."
flask init-db

# Inicia a aplicação baseado no ambiente
echo "Starting application..."
if [ "$FLASK_ENV" = "production" ]; then
    exec gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
else
    # Usa flask run para desenvolvimento, garantindo que o app seja carregado
    exec flask run --host=0.0.0.0 --port=5000
fi