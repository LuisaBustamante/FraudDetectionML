# database.py

import pymysql
import bcrypt

conn = pymysql.connect(
    host="localhost",
    port=3307,
    user="root",
    password="root",
    database="detectionfraud"
)

def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

def insert_user(username, password):
    hashed_password = hash_password(password)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO user (username, password) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
    except pymysql.IntegrityError as e:
        print("El usuario ya existe. Por favor, elige otro nombre de usuario.")
    except Exception as e:
        print(f"Error al insertar usuario: {e}")
    finally:
        cursor.close()

def verify_user_credentials(username, password):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password FROM user WHERE username=%s", (username,))
        hashed_password = cursor.fetchone()

        if hashed_password:
            hashed_password = hashed_password[0].encode('utf-8')  # Convertir a bytes
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                return True
            else:
                return False
        else:
            return False

    except Exception as e:
        print(f"Error al verificar credenciales: {e}")
        return False
    finally:
        cursor.close()
