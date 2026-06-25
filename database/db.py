
import sqlite3


def init_db():
    conn = sqlite3.connect("eatsafe.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        email TEXT PRIMARY KEY,
        diabetes INTEGER DEFAULT 0,
        hypertension INTEGER DEFAULT 0,
        obesity INTEGER DEFAULT 0
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS scans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        product_name TEXT,
        brand TEXT,
        score INTEGER,
        scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_scan(
    email,
    product_name,
    brand,
    score
):

    conn = sqlite3.connect("eatsafe.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO scans
    (
        email,
        product_name,
        brand,
        score
    )
    VALUES (?, ?, ?, ?)
    """,
    (
        email,
        product_name,
        brand,
        score
    ))

    conn.commit()
    conn.close()


def get_scan_history(email):

    conn = sqlite3.connect("eatsafe.db")
    c = conn.cursor()

    c.execute("""
    SELECT
        product_name,
        brand,
        score,
        scan_time
    FROM scans
    WHERE email=?
    ORDER BY id DESC
    """, (email,))

    rows = c.fetchall()

    conn.close()

    return rows


def save_user(email, diabetes, hypertension, obesity):

    conn = sqlite3.connect("eatsafe.db")
    c = conn.cursor()

    c.execute("""
    INSERT OR REPLACE INTO users
    (email, diabetes, hypertension, obesity)
    VALUES (?, ?, ?, ?)
    """, (email, diabetes, hypertension, obesity))

    conn.commit()
    conn.close()


def get_user(email):

    conn = sqlite3.connect("eatsafe.db")
    c = conn.cursor()

    c.execute("""
    SELECT * FROM users
    WHERE email=?
    """, (email,))

    user = c.fetchone()

    conn.close()

    return user


def get_profile_dict(email):

    user = get_user(email)

    if not user:
        return {
            "diabetes": False,
            "hypertension": False,
            "obesity": False
        }

    return {
        "diabetes": bool(user[1]),
        "hypertension": bool(user[2]),
        "obesity": bool(user[3])
    }


def get_dashboard_stats(email):

    conn = sqlite3.connect("eatsafe.db")
    c = conn.cursor()

    c.execute("""
    SELECT COUNT(*)
    FROM scans
    WHERE email=?
    """, (email,))
    total_scans = c.fetchone()[0]

    c.execute("""
    SELECT AVG(score)
    FROM scans
    WHERE email=?
    """, (email,))
    avg_score = c.fetchone()[0]

    c.execute("""
    SELECT product_name, score
    FROM scans
    WHERE email=?
    ORDER BY score DESC
    LIMIT 1
    """, (email,))
    best_product = c.fetchone()

    c.execute("""
    SELECT product_name, score
    FROM scans
    WHERE email=?
    ORDER BY score ASC
    LIMIT 1
    """, (email,))
    worst_product = c.fetchone()

    conn.close()

    return {
        "total_scans": total_scans,
        "avg_score": avg_score,
        "best_product": best_product,
        "worst_product": worst_product
    }

def get_all_scores(email):

    conn = sqlite3.connect("eatsafe.db")
    c = conn.cursor()

    c.execute("""
    SELECT
        product_name,
        score,
        scan_time
    FROM scans
    WHERE email=?
    ORDER BY id
    """, (email,))

    rows = c.fetchall()

    conn.close()

    return rows 