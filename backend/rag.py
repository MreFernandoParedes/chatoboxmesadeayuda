# backend/rag.py

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

# Ajusta estos modelos si quieres usar otros
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Rutas de archivos
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_PATH = DATA_DIR / "knowledge.txt"
INDEX_PATH = DATA_DIR / "index.json"

client = OpenAI()

_index_cache: List[Dict[str, Any]] | None = None


@dataclass
class Chunk:
    id: int
    text: str
    embedding: List[float]


def _load_knowledge_text() -> str:
    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"No se encontró {KNOWLEDGE_PATH}")
    return KNOWLEDGE_PATH.read_text(encoding="utf-8")


def _split_text_into_chunks(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """
    Divide el texto en trozos de tamaño razonable para RAG.
    Muy sencillo: corta por palabras, respetando un máximo de caracteres.
    """
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []

    current_len = 0
    for word in words:
        # +1 por el espacio
        add_len = len(word) + (1 if current else 0)
        if current_len + add_len > max_chars and current:
            chunk = " ".join(current).strip()
            chunks.append(chunk)

            # solapamiento simple: tomamos las últimas palabras
            if overlap > 0:
                overlap_words = []
                while current and sum(len(w) + 1 for w in overlap_words) < overlap:
                    overlap_words.insert(0, current.pop())
                current = overlap_words
                current_len = sum(len(w) + 1 for w in current)
            else:
                current = []
                current_len = 0

        current.append(word)
        current_len += add_len

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def _embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_or_load_index(force_rebuild: bool = False) -> List[Chunk]:
    """
    Carga el índice si existe; si no, lo construye a partir de knowledge.txt.
    """
    global _index_cache

    if _index_cache is not None and not force_rebuild:
        return _index_cache

    if INDEX_PATH.exists() and not force_rebuild:
        raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        _index_cache = [
            Chunk(id=item["id"], text=item["text"], embedding=item["embedding"])
            for item in raw
        ]
        return _index_cache

    # Construir índice desde cero
    text = _load_knowledge_text()
    chunks_text = _split_text_into_chunks(text)

    index: List[Chunk] = []
    print(f"[RAG] Generando embeddings para {len(chunks_text)} trozos...")
    for i, chunk_text in enumerate(chunks_text):
        emb = _embed_text(chunk_text)
        index.append(Chunk(id=i, text=chunk_text, embedding=emb))

    # Guardar en JSON (embeddings como listas)
    serializable = [
        {"id": c.id, "text": c.text, "embedding": c.embedding} for c in index
    ]
    INDEX_PATH.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")

    _index_cache = index
    print("[RAG] Índice creado y guardado en index.json")
    return index


def _retrieve_relevant_chunks(question: str, k: int = 3) -> List[Chunk]:
    index = build_or_load_index()
    query_emb = _embed_text(question)

    scored = [
        (c, _cosine_similarity(query_emb, c.embedding)) for c in index
    ]
    # ordenar de mayor a menor similitud
    scored.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [c for c, _score in scored[:k]]
    return top_chunks


def answer_question(question: str, k: int = 3) -> Dict[str, Any]:
    """
    Punto de entrada principal: recibe la pregunta y devuelve un diccionario con:
      - answer: texto final
      - retrieved_chunks: lista de textos de los trozos usados
      - source_files: lista de fuentes (en este caso, knowledge.txt)
      - model_name: modelo de chat usado
      - temperature: temperatura usada
      - usage: dict con input_tokens, output_tokens, total_tokens
    """
    # 1) Recuperar los chunks más relevantes
    top_chunks = _retrieve_relevant_chunks(question, k=k)

    # 2) Preparar contexto y prompts
    context_text = "\n\n---\n\n".join(c.text for c in top_chunks)

    system_prompt = (
        "Eres un asistente virtual de trámites consulares del Ministerio de Relaciones Exteriores del Perú. "
        "Responde siempre en español claro y breve. "
        "Usa únicamente la información proporcionada en el CONTEXTO para responder. "
        "Si la información no está en el contexto o no es suficiente, indica amablemente "
        "que no puedes responder con certeza y sugiere contactar al consulado."
    )

    user_prompt = (
        f"CONTEXTO:\n{context_text}\n\n"
        f"PREGUNTA DEL CIUDADANO:\n{question}\n\n"
        "Responde de forma estructurada y fácil de entender."
    )

    # 3) Llamada al modelo de chat
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer_text = completion.choices[0].message.content.strip() if completion.choices else ""

    # 4) Extraer tokens (usage)
    usage: Dict[str, Any] = {}
    if hasattr(completion, "usage") and completion.usage is not None:
        usage = {
            "input_tokens": getattr(completion.usage, "prompt_tokens", None),
            "output_tokens": getattr(completion.usage, "completion_tokens", None),
            "total_tokens": getattr(completion.usage, "total_tokens", None),
        }

    # 5) Preparar datos de RAG para logging
    retrieved_chunks = [c.text for c in top_chunks]
    source_files = ["knowledge.txt" for _ in top_chunks]

    return {
        "answer": answer_text,
        "retrieved_chunks": retrieved_chunks,
        "source_files": source_files,
        "model_name": CHAT_MODEL,
        "temperature": 0.2,
        "usage": usage,
    }
