import sqlite3
import hashlib

DB_PATH = "users.db"

def create_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            fullname TEXT,
            email TEXT UNIQUE,
            phone TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password, fullname=None, email=None, phone=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users(username, password, fullname, email, phone)
            VALUES(?, ?, ?, ?, ?)
        """, (username, hash_password(password), fullname, email, phone))
        conn.commit()
        return True
    except sqlite3.IntegrityError as e:
        # Vérifie le type de doublon
        if "users.username" in str(e):
            print("⚠ Nom d'utilisateur déjà existant")
        elif "users.email" in str(e):
            print("⚠ Email déjà existant")
        return False
    finally:
        conn.close()



def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", 
                   (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user is not None
