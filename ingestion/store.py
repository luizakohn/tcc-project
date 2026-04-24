import logging
import time

import psycopg2
import psycopg2.extras

import config
from ingestion.embedder import embed_texts

logger = logging.getLogger(__name__)


def _get_connection():
    return psycopg2.connect(config.POSTGRES_DSN)


def test_connection() -> bool:
    """Tests whether the PostgreSQL connection is working.

    Returns True if successful, False otherwise.
    """
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        logger.info("Conexão com o banco de dados bem-sucedida.")
        return True
    except Exception:
        logger.exception("Falha ao conectar ao banco de dados.")
        return False


def _ensure_table(cur, base_name: str, dimensions: int) -> None:
    """Cria a tabela e a extensão pgvector se não existirem."""
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {base_name} (
            id SERIAL PRIMARY KEY,
            passage_id TEXT,
            content TEXT NOT NULL,
            embedding vector({dimensions})
        );
        """
    )


def embed_and_store(
    base_name: str,
    dimensions: int,
    chunks: list[tuple[str, str]],
) -> None:
    """Embeda chunks e persiste no PostgreSQL com pgvector.

    Args:
        base_name: Nome da tabela no Postgres.
        dimensions: Dimensões do vetor de embedding.
        chunks: Lista de tuplas (passage_id, text) a embedar e salvar.
    """
    if not chunks:
        logger.warning("Lista de chunks vazia; nada a inserir.")
        return

    logger.info(
        "Iniciando ingestão: tabela=%s, dims=%d, chunks=%d",
        base_name,
        dimensions,
        len(chunks),
    )
    start = time.perf_counter()

    passage_ids, texts = zip(*chunks)
    embeddings = embed_texts(list(texts), dimensions)

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            _ensure_table(cur, base_name, dimensions)

            insert_sql = f"""
                INSERT INTO {base_name} (passage_id, content, embedding)
                VALUES (%s, %s, %s::vector)
            """
            data = [
                (pid, text, str(emb))
                for pid, text, emb in zip(passage_ids, texts, embeddings)
            ]
            psycopg2.extras.execute_batch(cur, insert_sql, data, page_size=500)

        conn.commit()
        elapsed = time.perf_counter() - start
        logger.info(
            "Ingestão concluída: %d registros em %.2fs na tabela %s",
            len(chunks),
            elapsed,
            base_name,
        )
    except Exception:
        conn.rollback()
        logger.exception("Erro durante a ingestão na tabela %s", base_name)
        raise
    finally:
        conn.close()
