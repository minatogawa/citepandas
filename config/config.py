import os

class Config:
    """Base config."""
    TESTING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CACHE_TYPE = 'simple'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # Configurações comuns
    SQLALCHEMY_DATABASE_URI = os.getenv('POSTGRES_URI')
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY')

class ProductionConfig(Config):
    """Production config."""
    FLASK_ENV = 'production'
    DEBUG = False
    TESTING = False
    
    # Security configurations
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SECURE = True
    REMEMBER_COOKIE_HTTPONLY = True
    
    # Configurações do banco de dados para produção
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,        # Mais conexões para produção
        'max_overflow': 40,     # Permite mais conexões extras
        'pool_timeout': 60,     # Tempo maior de timeout
        'pool_recycle': 1800,   # Recicla conexões a cada 30 minutos
    }

class DevelopmentConfig(Config):
    """Development config."""
    FLASK_ENV = 'development'
    DEBUG = True
    # Configurações do banco de dados para desenvolvimento
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,        # Menos conexões para desenvolvimento
        'max_overflow': 20,     # Menos conexões extras
        'pool_timeout': 30,     # Timeout menor
        'pool_recycle': 1800,
    }