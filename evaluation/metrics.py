import logging

logger = logging.getLogger(__name__)


def precision_at_k(returned_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@K: fração dos K retornados que são relevantes nos qrels.

    P@K = |retornados ∩ relevantes| / K
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for pid in returned_ids[:k] if pid in relevant_ids)
    return hits / k


def recall_at_k(returned_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K: fração dos relevantes nos qrels que foram recuperados nos top-K.

    R@K = |retornados ∩ relevantes| / |relevantes|
    Retorna 0.0 se não houver relevantes para a query.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    hits = sum(1 for pid in returned_ids[:k] if pid in relevant_ids)
    return hits / len(relevant_ids)
