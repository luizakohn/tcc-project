import logging

logger = logging.getLogger(__name__)


def clear_cache(cur) -> None:
    """Limpa caches do PostgreSQL para garantir isolamento entre consultas.

    Executa pg_prewarm para aquecer o buffer uniformemente e DISCARD ALL
    para resetar o estado da sessão.
    """
    try:
        cur.execute("SELECT pg_prewarm('pg_class');")
    except Exception:
        logger.debug("pg_prewarm não disponível; ignorando.")

    cur.execute("DISCARD ALL;")
    logger.debug("Cache do Postgres limpo (DISCARD ALL).")
