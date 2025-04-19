from sqlalchemy import create_engine
from sqlalchemy.sql import text
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do .env.development
load_dotenv('.env.development')

# Pega a URL do banco de dados das variáveis de ambiente
DATABASE_URL = os.getenv('POSTGRES_URI')

def clean_migrations():
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
            conn.commit()
        print("Tabela alembic_version removida com sucesso!")
    except Exception as e:
        print(f"Erro ao remover tabela: {e}")

if __name__ == "__main__":
    clean_migrations()