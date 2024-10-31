import sys
sys.path.append('/app')

from app import app, db

def init_db():
    with app.app_context():
        # Cria todas as tabelas
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    init_db()