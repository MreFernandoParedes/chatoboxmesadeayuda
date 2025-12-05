# backend/db_logging.py

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional, List, Dict
import json

# Ruta a la base de datos: backend/data/chatbotx.db
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "chatbotx.db"


def get_connection() -> sqlite3.Connection:
    """
    Crea una conexión nueva a la base de datos SQLite.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Crea la tabla de logs si no existe.
    Llamar esta función al iniciar el backend (en el evento startup).
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                ip TEXT,
                user_agent TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                latency_ms INTEGER,
                retrieved_chunks TEXT,
                source_files TEXT,
                model_name TEXT,
                temperature REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER
            );
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interactions_timestamp
            ON interactions(timestamp);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interactions_session
            ON interactions(session_id);
            """
        )

        conn.commit()
    finally:
        conn.close()


def _to_json(value: Optional[Iterable[Any]]) -> str:
    """
    Convierte una lista/iterable a JSON (string).
    Si viene None, devuelve '[]'.
    Si ya es string, lo devuelve tal cual (asumimos que ya es JSON).
    """
    if value is None:
        return "[]"
    if isinstance(value, str):
        return value
    return json.dumps(list(value), ensure_ascii=False)


def save_interaction(
    *,
    timestamp: str,
    session_id: Optional[str],
    ip: Optional[str],
    user_agent: Optional[str],
    question: str,
    answer: str,
    latency_ms: Optional[int],
    retrieved_chunks: Optional[Iterable[Any]],
    source_files: Optional[Iterable[Any]],
    model_name: Optional[str],
    temperature: Optional[float],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
) -> None:
    """
    Inserta una fila en la tabla interactions.
    Llamar a esta función justo después de generar la respuesta.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO interactions (
                timestamp,
                session_id,
                ip,
                user_agent,
                question,
                answer,
                latency_ms,
                retrieved_chunks,
                source_files,
                model_name,
                temperature,
                input_tokens,
                output_tokens,
                total_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                timestamp,
                session_id,
                ip,
                user_agent,
                question,
                answer,
                latency_ms,
                _to_json(retrieved_chunks),
                _to_json(source_files),
                model_name,
                temperature,
                input_tokens,
                output_tokens,
                total_tokens,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_interactions(limit: Optional[int] = 500) -> List[Dict[str, Any]]:
    """
    Devuelve las últimas 'limit' interacciones como una lista de dicts.
    Si limit es None, devuelve todas (cuidado si la tabla es muy grande).
    """
    conn = get_connection()
    try:
        if limit is not None:
            cursor = conn.execute(
                """
                SELECT *
                FROM interactions
                ORDER BY id DESC
                LIMIT ?;
                """,
                (limit,),
            )
        else:
            cursor = conn.execute(
                """
                SELECT *
                FROM interactions
                ORDER BY id DESC;
                """
            )

        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
