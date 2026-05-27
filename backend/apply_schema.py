import mysql.connector
from config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE , MYSQL_PORT

def initialize_schema():
    # 1. Connect to MySQL
    # Note: We use host=MYSQL_HOST which should be 'chatbot-mysql' in Docker
    db = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        port=MYSQL_PORT,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )
    cursor = db.cursor()

    # 2. Your SQL Schema
    schema = """
    CREATE TABLE IF NOT EXISTS user_queries (
        id               CHAR(36)       PRIMARY KEY DEFAULT (UUID()),
        anon_id          VARCHAR(255)   NULL,
        user_id          VARCHAR(255)   NULL,
        is_authenticated TINYINT(1)     NOT NULL DEFAULT 0,
        query            TEXT           NOT NULL,
        created_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_user_id    (user_id),
        INDEX idx_anon_id    (anon_id),
        INDEX idx_created_at (created_at)
    );
    """

    try:
        print("Applying schema...")
        cursor.execute(schema)
        db.commit()
        print("Schema applied successfully!")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        db.close()

if __name__ == "__main__":
    initialize_schema()