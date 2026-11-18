
import sqlite3

from pathlib import Path
from datetime import datetime


DATABASE_PATH = Path(__file__).resolve().parent / "predictions.db"


def get_connection():

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT NOT NULL,
            prediction TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(job_title, prediction, fraud_probability):

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO predictions (
            job_title,
            prediction,
            fraud_probability,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            job_title,
            prediction,
            fraud_probability,
            datetime.now().strftime("%d %b %Y, %I:%M %p")
        )
    )

    connection.commit()
    connection.close()


def get_all_predictions():

    connection = get_connection()

    predictions = connection.execute(
        "SELECT * FROM predictions ORDER BY id DESC"
    ).fetchall()

    connection.close()

    return predictions


def clear_predictions():

    connection = get_connection()

    connection.execute("DELETE FROM predictions")

    connection.commit()
    connection.close()
