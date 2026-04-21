import logging
import time
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=config.OPENAI_API_KEY)


def embed_texts(texts: list[str], dimensions: int) -> list[list[float]]:
    """Embeda uma lista de textos usando a OpenAI Embedding API.

    Args:
        texts: Lista de textos para embedar.
        dimensions: Número de dimensões do vetor (768, 1024 ou 1536).

    Returns:
        Lista de vetores de embedding.
    """
    if not texts:
        return []

    logger.info("Embedando %d textos com %d dimensões...", len(texts), dimensions)
    start = time.perf_counter()

    batch_size = 512
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = _client.embeddings.create(
            input=batch,
            model=config.EMBEDDING_MODEL,
            dimensions=dimensions,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        logger.debug("Batch %d-%d concluído.", i, i + len(batch))

    elapsed = time.perf_counter() - start
    logger.info(
        "Embedding concluído: %d textos em %.2fs (%.1f textos/s)",
        len(texts),
        elapsed,
        len(texts) / elapsed if elapsed > 0 else 0,
    )
    return all_embeddings


def embed_query(question: str, dimensions: int) -> list[float]:
    """Embeda uma única query. Conveniência sobre embed_texts."""
    return embed_texts([question], dimensions)[0]
