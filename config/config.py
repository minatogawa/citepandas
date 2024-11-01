import os
from datetime import timedelta

class Config:
    """Base config."""
    TESTING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CACHE_TYPE = 'simple'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # Configurações comuns
    SQLALCHEMY_DATABASE_URI = os.getenv('POSTGRES_URI')
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-key-please-change')  # Fallback para desenvolvimento
    
    # Configurações de Cache
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Configurações de Rate Limiting
    RATELIMIT_DEFAULT = "200 per day;50 per hour"
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', "memory://")

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
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)  # Sessão expira em 1 dia
    
    # Configurações do banco de dados para produção
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'max_overflow': 40,
        'pool_timeout': 60,
        'pool_recycle': 1800,
        'pool_pre_ping': True,  # Verifica conexões antes de usar
    }
    
    # Configurações SSL/TLS
    PREFERRED_URL_SCHEME = 'https'

class DevelopmentConfig(Config):
    """Development config."""
    FLASK_ENV = 'development'
    DEBUG = True
    
    # Configurações do banco de dados para desenvolvimento
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'max_overflow': 20,
        'pool_timeout': 30,
        'pool_recycle': 1800,
        'pool_pre_ping': True,
    }
    
    # Override do SQLALCHEMY_DATABASE_URI para desenvolvimento local
    SQLALCHEMY_DATABASE_URI = os.getenv('POSTGRES_URI', 'sqlite:///dev.db')