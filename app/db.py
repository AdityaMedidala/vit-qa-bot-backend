import os
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

# Create a Threaded pool (1 minimum connection, up to 20 concurrent connections)
db_pool = pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=20,
    dsn=os.getenv("SUPABASE_URL")
)

def get_db_connection():
    return db_pool.getconn()

def release_db_connection(conn):
    db_pool.putconn(conn)