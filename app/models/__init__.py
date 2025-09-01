# ============================
# 📁 app/models/__init__.py
# (aus deiner bisherigen models.py)

from pydantic import BaseModel
import sqlite3
from datetime import datetime, timedelta

DB = "database.db"

class Article(BaseModel):
    title: str
    teaser: str
    url: str
    source: str
    topic: str
    published_at: datetime

def get_connection():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            teaser TEXT,
            url TEXT,
            source TEXT,
            topic TEXT,
            published_at TEXT
        )
    """)
    conn.commit()

def save_articles(articles):
    conn = get_connection()
    for art in articles:
        exists = conn.execute("SELECT 1 FROM articles WHERE url = ?", (art.url,)).fetchone()
        if not exists:
            conn.execute("""
                INSERT INTO articles (title, teaser, url, source, topic, published_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (art.title, art.teaser, art.url, art.source, art.topic, art.published_at))
    conn.commit()

def get_articles():
    conn = get_connection()
    cursor = conn.execute("SELECT * FROM articles")
    return [Article(**dict(row)) for row in cursor.fetchall()]

def get_articles_last_hours(hours):
    conn = get_connection()
    since = datetime.now() - timedelta(hours=hours)
    cursor = conn.execute("SELECT * FROM articles WHERE published_at >= ?", (since.isoformat(),))
    return [Article(**dict(row)) for row in cursor.fetchall()]

init_db()

