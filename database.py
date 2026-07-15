import sqlite3
from datetime import datetime

connection = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = connection.cursor()


def init_db():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            image_name TEXT,
            predicted_class TEXT,
            confidence REAL,
            medical_report TEXT
        )
    """)
    connection.commit()


def save_prediction(image_name, predicted_class, confidence, medical_report):
    current_date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO predictions (date, image_name, predicted_class, confidence, medical_report)
        VALUES (?, ?, ?, ?, ?)
        """,
        (current_date, image_name, predicted_class, confidence, medical_report),
    )
    connection.commit()


def get_prediction_history():
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    return cursor.fetchall()


def get_total_predictions():
    cursor.execute("SELECT COUNT(*) FROM predictions")
    return cursor.fetchone()[0]


def get_latest_prediction():
    cursor.execute("SELECT predicted_class FROM predictions ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    return row[0] if row else None


def clear_history():
    cursor.execute("DELETE FROM predictions")
    connection.commit()


init_db()