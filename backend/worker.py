import json
from arq import create_pool
from config import REDIS_SETTINGS , MYSQL_PORT , MYSQL_HOST , MYSQL_USER , MYSQL_PASSWORD , MYSQL_DATABASE
import mysql.connector


def get_db_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        port=MYSQL_PORT,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        auth_plugin='mysql_native_password'
    )

async def log_query_job(ctx, payload: dict):
    conn = get_db_conn()
    cur = conn.cursor()

    if payload["is_authenticated"]:
        cur.execute(
            """
            INSERT INTO user_queries (anon_id, user_id, is_authenticated, query)
            VALUES (%s, %s, %s, %s)
            """,
            (
                None,          
                payload["user_id"],           
                payload["is_authenticated"],
                payload["query"],
            )
        )
    else:
        cur.execute(
            """
            INSERT INTO user_queries (anon_id, user_id, is_authenticated, query)
            VALUES (%s, %s, %s, %s)
            """,
            (
                payload["user_id"],          
                None,           
                payload["is_authenticated"],
                payload["query"],
            )
        ) 

    conn.commit()
    cur.close()
    conn.close()

class WorkerSettings:
    functions = [log_query_job]
    redis_settings = REDIS_SETTINGS
    max_jobs = 100          
    job_timeout = 30       
    retry_jobs = True
    max_tries = 5          