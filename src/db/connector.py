from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from src.config.settings import Settings

settings = Settings()
_engine = create_engine(settings.MYSQL_URI, pool_pre_ping=True)
db = SQLDatabase(_engine)

def get_db() -> SQLDatabase:
    return db

