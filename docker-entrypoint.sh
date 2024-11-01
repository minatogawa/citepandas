#!/bin/bash

# Espera o PostgreSQL inicializar
echo "Waiting for PostgreSQL..."
until pg_isready -h db -U postgres; do
    sleep 1
done
echo "PostgreSQL started"

# Inicializa o banco de dados
python /app/migrations/init_db.py

# Inicia a aplicação baseado no ambiente
if [ "$FLASK_ENV" = "production" ]; then
    exec gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
else
    exec flask run --host=0.0.0.0
fi