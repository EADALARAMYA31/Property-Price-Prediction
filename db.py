import psycopg2
import bcrypt
import streamlit as st

conn = psycopg2.connect(
    host=st.secrets["DB_HOST"],
    database=st.secrets["DB_NAME"],
    user=st.secrets["DB_USER"],
    password=st.secrets["DB_PASSWORD"],
    port=st.secrets["DB_PORT"]
)

cursor = conn.cursor()

def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id SERIAL PRIMARY KEY,
        username TEXT,
        model TEXT,
        prediction FLOAT
    )
    """)
    conn.commit()


def register_user(username, password):
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, hashed_pw.decode('utf-8'))
        )
        conn.commit()
        return True
    except Exception as e:
        print(e)
        return False


def login_user(username, password):
    cursor.execute(
        "SELECT password FROM users WHERE username=%s",
        (username,)
    )

    result = cursor.fetchone()

    if result:
        stored_hash = result[0]

        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return True

    return False


def save_prediction(username, model, prediction):
    cursor.execute(
        "INSERT INTO history (username, model, prediction) VALUES (%s, %s, %s)",
        (username, model, prediction)
    )
    conn.commit()


def get_history(username):
    cursor.execute(
        "SELECT model, prediction FROM history WHERE username=%s ORDER BY id DESC",
        (username,)
    )
    return cursor.fetchall()