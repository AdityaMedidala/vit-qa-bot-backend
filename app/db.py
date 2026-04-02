import os
import psycopg2
from psycopg2 import pool, OperationalError
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("SUPABASE_URL")

db_pool = pool.ThreadedConnectionPool(minconn=1, maxconn=20, dsn=DSN)


def get_db_connection():
    conn = db_pool.getconn()
    try:
        # Cheap ping — if connection is dead this raises OperationalError
        conn.cursor().execute("SELECT 1")
        return conn
    except OperationalError:
        # Dead connection — close it and open a fresh one directly
        try:
            conn.close()
        except Exception:
            pass
        return psycopg2.connect(DSN)


def release_db_connection(conn):
    try:
        db_pool.putconn(conn)
    except Exception:
        # conn was a direct fallback connection, just close it
        try:
            conn.close()
        except Exception:
            pass