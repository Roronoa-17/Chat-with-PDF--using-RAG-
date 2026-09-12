from psycopg_pool import AsyncConnectionPool
from app.core.config import settings

pool = AsyncConnectionPool(
    conninfo=settings.DATABASE_URL,
    max_size=10,
    max_idle=120,  # Shrink pool if connections are idle for 2 minutes
    max_lifetime=300, # Forcefully replace connections every 5 minutes
    kwargs={
        "autocommit":True, 
        "prepare_threshold": 0,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5},
    open=False,
)