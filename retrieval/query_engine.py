import logging
import time

import psycopg2

import config
from ingestion.embedder import embed_query
from retrieval.cache import clear_cache

logger = logging.getLogger(__name__)

METRIC_OPERATORS = {
    "cosine": "<=>",
    "euclidean": "<->",
    "dot_product": "<#>",
}

APPROX_METRICS = list(METRIC_OPERATORS.keys())


def _get_connection():
    return psycopg2.connect(config.POSTGRES_DSN)


def _run_query(
    cur,
    base_name: str,
    query_vec_str: str,
    operator: str,
    k: int,
) -> tuple[list[str], float]:
    """Executa uma consulta de similaridade e retorna passage_ids + tempo em ms."""
    sql = f"""
        SELECT passage_id
        FROM {base_name}
        ORDER BY embedding {operator} %s::vector
        LIMIT %s;
    """

    start = time.perf_counter()
    cur.execute(sql, (query_vec_str, k))
    elapsed_ms = (time.perf_counter() - start) * 1000

    ids = [row[0] for row in cur.fetchall()]
    return ids, elapsed_ms


def query_single_base(
    question_embedding: list[float],
    base_name: str,
    k: int,
) -> dict[str, dict]:
    """Executa as 3 consultas aproximadas em uma base e retorna passage_ids e tempos.

    Returns:
        {
            "cosine": {"ids": [...], "time_ms": float},
            "euclidean": {"ids": [...], "time_ms": float},
            "dot_product": {"ids": [...], "time_ms": float},
        }
    """
    vec_str = str(question_embedding)
    results: dict[str, dict] = {}

    conn = _get_connection()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for metric_name, operator in METRIC_OPERATORS.items():
                clear_cache(cur)
                ids, elapsed = _run_query(cur, base_name, vec_str, operator, k)
                results[metric_name] = {"ids": ids, "time_ms": elapsed}
                logger.info(
                    "%s | %s: %.2fms (%d resultados)",
                    base_name,
                    metric_name,
                    elapsed,
                    len(ids),
                )
    finally:
        conn.close()

    return results


def query_all_metrics(
    question: str,
    k: int,
    dimensions_list: list[int],
    base_names: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Executa queries com múltiplas métricas aproximadas em múltiplas bases.

    Args:
        question: Pergunta de teste.
        k: Número de resultados (top-k).
        dimensions_list: Lista de dimensões (ex: [768, 1024, 1536]).
        base_names: Nomes das tabelas correspondentes.

    Returns:
        {
            "768": {
                "cosine": [passage_id1, ...],
                "euclidean": [...],
                "dot_product": [...],
            },
            ...
        }
    """
    all_results: dict[str, dict[str, list[str]]] = {}
    timings: dict[str, dict[str, float]] = {}

    query_embeddings: dict[int, list[float]] = {}
    for dims in dimensions_list:
        query_embeddings[dims] = embed_query(question, dims)

    for dims, base_name in zip(dimensions_list, base_names):
        embedding = query_embeddings[dims]
        base_results = query_single_base(embedding, base_name, k)

        dim_key = str(dims)
        all_results[dim_key] = {
            metric: data["ids"] for metric, data in base_results.items()
        }
        timings[dim_key] = {
            metric: data["time_ms"] for metric, data in base_results.items()
        }

    return all_results, timings
