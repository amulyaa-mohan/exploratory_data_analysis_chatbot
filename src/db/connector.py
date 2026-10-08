from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from src.config.settings import Settings

_db = None

def get_db() -> SQLDatabase:
    """Connect lazily so importing this module doesn't require MySQL to be running."""
    global _db
    if _db is None:
        engine = create_engine(Settings().MYSQL_URI, pool_pre_ping=True)
        _db = SQLDatabase(engine)
    return _db
