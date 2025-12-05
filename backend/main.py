# backend/main.py

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, Dict
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel



from .rag import answer_question, build_or_load_index
from .db_logging import init_db, save_interaction, get_interactions


app = FastAPI(
    title="Chatbot RAG Consular",
    description="API para consultas de trámites consulares usando RAG sobre knowledge.txt",
)

# --- CORS (por si luego lo usas desde otro dominio, p.ej. GitHub Pages) ---
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.app\.github\.dev",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    # Opcional: el frontend puede enviar session_id
    session_id: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str


@app.on_event("startup")
async def startup_event():
    """
    Al iniciar el servidor:
    - Intentamos cargar (o construir) el índice RAG.
    - Inicializamos la base de datos SQLite para logs.
    """
    try:
        build_or_load_index()
        print("[API] Índice RAG cargado correctamente.")
    except Exception as e:
        print(f"[API] Error al construir/cargar índice: {e}")

    try:
        init_db()
        print("[API] Base de datos de logs inicializada correctamente.")
    except Exception as e:
        print(f"[API] Error al inicializar la base de datos: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest, request: Request):
    """
    Endpoint principal:
    - Valida la pregunta.
    - Llama a answer_question (RAG + OpenAI).
    - Mide latencia.
    - Extrae tokens de usage (si están disponibles).
    - Guarda todo en SQLite.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    try:
        # 1) Medir tiempo de inicio
        start_time = time.perf_counter()

        # 2) Llamar a la lógica RAG
        result: Any = answer_question(req.question)

        # 3) Soporte doble (string o dict)
        if isinstance(result, str):
            answer_text = result
            retrieved_chunks = None
            source_files = None
            model_name = None
            temperature = None
            usage: Dict[str, Any] = {}
        else:
            answer_text = result.get("answer", "")
            retrieved_chunks = result.get("retrieved_chunks")
            source_files = result.get("source_files")
            model_name = result.get("model_name")
            temperature = result.get("temperature")
            usage = result.get("usage") or {}

        # 4) Tiempo final + latencia
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)

        # 5) Timestamp ISO UTC
        timestamp = datetime.now(timezone.utc).isoformat()

        # 6) IP y User-Agent
        client_host = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        # 7) Tokens (si existen en usage)
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")

        # 8) Guardar en SQLite
        save_interaction(
            timestamp=timestamp,
            session_id=req.session_id,
            ip=client_host,
            user_agent=user_agent,
            question=req.question,
            answer=answer_text,
            latency_ms=latency_ms,
            retrieved_chunks=retrieved_chunks,
            source_files=source_files,
            model_name=model_name,
            temperature=temperature,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        # 9) Devolver respuesta
        return AnswerResponse(answer=answer_text)

    except Exception as e:
        print(f"[API] Error procesando pregunta: {e}")
        raise HTTPException(
            status_code=500,
            detail="Ocurrió un error al procesar la consulta en el asistente.",
        )


def _html_escape(text: str) -> str:
    """
    Escapa caracteres básicos para evitar romper el HTML.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@app.get("/logs", response_class=HTMLResponse)
async def view_logs(limit: int = 500):
    """
    Página HTML simple para ver el contenido de la tabla interactions.
    - Usa ?limit=N para cambiar cuántas filas se muestran.
    """
    rows = get_interactions(limit=limit)

    # Construir filas de la tabla
    table_rows_html = ""
    for row in rows:
        table_rows_html += "<tr>"
        table_rows_html += f"<td>{row.get('id')}</td>"
        table_rows_html += f"<td>{_html_escape(str(row.get('timestamp') or ''))}</td>"
        table_rows_html += f"<td>{_html_escape(str(row.get('session_id') or ''))}</td>"
        table_rows_html += f"<td>{_html_escape(str(row.get('ip') or ''))}</td>"
        table_rows_html += f"<td>{_html_escape(str(row.get('user_agent') or ''))}</td>"
        table_rows_html += f"<td>{_html_escape(str(row.get('question') or ''))}</td>"
        table_rows_html += f"<td>{_html_escape(str(row.get('answer') or ''))}</td>"
        table_rows_html += f"<td>{row.get('latency_ms') or ''}</td>"
        table_rows_html += f"<td>{_html_escape(str(row.get('retrieved_chunks') or ''))}</td>"
        table_rows_html += f"<td>{_html_escape(str(row.get('source_files') or ''))}</td>"
        table_rows_html += f"<td>{_html_escape(str(row.get('model_name') or ''))}</td>"
        table_rows_html += f"<td>{row.get('temperature') or ''}</td>"
        table_rows_html += f"<td>{row.get('input_tokens') or ''}</td>"
        table_rows_html += f"<td>{row.get('output_tokens') or ''}</td>"
        table_rows_html += f"<td>{row.get('total_tokens') or ''}</td>"
        table_rows_html += "</tr>\n"

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>Logs Chatbotx</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      padding: 20px;
      background: #f5f5f5;
    }}
    h1 {{
      margin-bottom: 0.2rem;
    }}
    .subtitle {{
      color: #555;
      margin-bottom: 1rem;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: #fff;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px;
      vertical-align: top;
      font-size: 0.85rem;
    }}
    th {{
      background-color: #f0f0f0;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr:nth-child(even) {{
      background-color: #fafafa;
    }}
    code {{
      font-family: "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      white-space: pre-wrap;
    }}
    .top-bar {{
      margin-bottom: 1rem;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 1rem;
      flex-wrap: wrap;
    }}
    .limit-info {{
      font-size: 0.9rem;
      color: #666;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      background: #e0e7ff;
      color: #1d4ed8;
      font-size: 0.75rem;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="top-bar">
    <div>
      <h1>Logs del Chatbot</h1>
      <div class="subtitle">
        Mostrando las últimas <strong>{limit}</strong> interacciones (ordenadas de la más reciente a la más antigua).
      </div>
    </div>
    <div class="limit-info">
      Cambia el límite en la URL: <code>?limit=100</code>, <code>?limit=1000</code>, etc.
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Timestamp</th>
        <th>Session ID</th>
        <th>IP</th>
        <th>User Agent</th>
        <th>Pregunta</th>
        <th>Respuesta</th>
        <th>Latency (ms)</th>
        <th>Retrieved Chunks</th>
        <th>Source Files</th>
        <th>Model</th>
        <th>Temp</th>
        <th>Input Tokens</th>
        <th>Output Tokens</th>
        <th>Total Tokens</th>
      </tr>
    </thead>
    <tbody>
      {table_rows_html}
    </tbody>
  </table>
</body>
</html>
"""
    return HTMLResponse(content=html)


# --- Servir el frontend estático desde FastAPI (mismo puerto 8001) ---

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Esto hace que / sirva index.html de frontend, y /css, /js, /assets, etc.
app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)
