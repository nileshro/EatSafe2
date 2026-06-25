import sqlite3
import bcrypt


def create_auth_table():

    conn = sqlite3.connect("eatsafe.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS auth_users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()


def register_user(
    name,
    email,
    password
):

    conn = sqlite3.connect("eatsafe.db")
    c = conn.cursor()

    try:

        hashed_password = bcrypt.hashpw(
            password.encode(),
            bcrypt.gensalt()
        )

        c.execute("""
        INSERT INTO auth_users
        (
            name,
            email,
            password
        )
        VALUES (?, ?, ?)
        """,
        (
            name,
            email,
            hashed_password.decode()
        ))

        conn.commit()

        return True

    except:

        return False

    finally:

        conn.close()


def login_user(
    email,
    password
):

    conn = sqlite3.connect("eatsafe.db")
    c = conn.cursor()

    c.execute("""
    SELECT
        id,
        name,
        email,
        password
    FROM auth_users
    WHERE email=?
    """, (email,))

    user = c.fetchone()

    conn.close()

    if not user:
        return None

    stored_hash = user[3]

    if bcrypt.checkpw(
        password.encode(),
        stored_hash.encode()
    ):

        return {
            "id": user[0],
            "name": user[1],
            "email": user[2]
        }

    return None

