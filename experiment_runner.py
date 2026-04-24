"""Orquestrador principal do experimento RAG Benchmark.

Executa o pipeline completo:
1. Embeda e armazena chunks nas 3 bases (768d, 1024d, 1536d)
2. Para cada query de teste, consulta as 3 métricas aproximadas em cada base
3. Calcula Precision@K e Recall@K comparando com os qrels do Quati
4. Exporta CSVs de timings, resultados e summary
"""

import logging
import statistics
import sys

import config
from evaluation.exporter import export_results, export_summary, export_timings
from evaluation.metrics import precision_at_k, recall_at_k
from ingestion.store import embed_and_store
from quati_loader import load_passages, load_qrels, load_queries
from retrieval.query_engine import query_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

APPROX_METRICS = ["cosine", "euclidean", "dot_product"]


def run_ingestion(chunks: list[tuple[str, str]]) -> None:
    """Fase 1: Ingestão dos chunks nas 3 bases."""
    logger.info("=== FASE 1: INGESTÃO ===")
    for dims in config.DIMENSIONS:
        base_name = config.BASE_NAMES[dims]
        logger.info("Ingerindo na base %s (%d dims)...", base_name, dims)
        embed_and_store(base_name, dims, chunks)
    logger.info("Ingestão concluída para todas as bases.")


def run_experiment(
    queries: list[tuple[str, str]],
    qrels: dict[str, set[str]],
    k: int = None,
) -> None:
    """Fase 2+3: Consulta e avaliação via qrels.

    Args:
        queries: Lista de (query_id, query_text).
        qrels: Mapeamento query_id → set de passage_ids relevantes (score >= 1).
        k: Top-K (usa config.K se omitido).
    """
    if k is None:
        k = config.K

    dimensions_list = config.DIMENSIONS
    base_names = [config.BASE_NAMES[d] for d in dimensions_list]

    timings_per_dim: dict[int, list[dict[str, float]]] = {
        d: [] for d in dimensions_list
    }
    results_per_dim: dict[int, list[dict]] = {d: [] for d in dimensions_list}

    logger.info("=== FASE 2: CONSULTA E AVALIAÇÃO ===")
    logger.info("Queries: %d | K: %d | Bases: %s", len(queries), k, base_names)

    for q_idx, (query_id, question) in enumerate(queries):
        relevant_ids = qrels.get(query_id, set())
        logger.info(
            "Query %d/%d [%s]: %s (relevantes no qrel: %d)",
            q_idx + 1,
            len(queries),
            query_id,
            question[:70],
            len(relevant_ids),
        )

        all_results, all_timings = query_all_metrics(
            question, k, dimensions_list, base_names
        )

        for dims in dimensions_list:
            dim_key = str(dims)
            timings_per_dim[dims].append(all_timings[dim_key])

            for metric in APPROX_METRICS:
                returned_ids = all_results[dim_key][metric]
                prec = precision_at_k(returned_ids, relevant_ids, k)
                rec = recall_at_k(returned_ids, relevant_ids, k)

                results_per_dim[dims].append(
                    {
                        "question_id": q_idx,
                        "query_id": query_id,
                        "question": question,
                        "metric": metric,
                        "returned_ids": returned_ids,
                        "precision_at_k": prec,
                        "recall_at_k": rec,
                    }
                )

                logger.info(
                    "  [%dd] %s — P@%d=%.4f, R@%d=%.4f",
                    dims,
                    metric,
                    k,
                    prec,
                    k,
                    rec,
                )

    logger.info("=== FASE 3: EXPORTAÇÃO ===")

    for dims in dimensions_list:
        export_timings(timings_per_dim[dims], dims)
        export_results(results_per_dim[dims], dims)

    summary_rows = _build_summary(results_per_dim, timings_per_dim)
    export_summary(summary_rows)

    logger.info("=== EXPERIMENTO CONCLUÍDO ===")
    _print_summary(summary_rows)


def _build_summary(
    results_per_dim: dict[int, list[dict]],
    timings_per_dim: dict[int, list[dict[str, float]]],
) -> list[dict]:
    summary = []

    for dims in config.DIMENSIONS:
        for metric in APPROX_METRICS:
            metric_rows = [r for r in results_per_dim[dims] if r["metric"] == metric]
            if not metric_rows:
                continue

            avg_prec = statistics.mean(r["precision_at_k"] for r in metric_rows)
            avg_rec = statistics.mean(r["recall_at_k"] for r in metric_rows)
            avg_time = statistics.mean(
                t.get(metric, 0) for t in timings_per_dim[dims]
            )

            summary.append(
                {
                    "dimensions": dims,
                    "metric": metric,
                    "avg_precision_at_k": avg_prec,
                    "avg_recall_at_k": avg_rec,
                    "avg_time_ms": avg_time,
                }
            )

    return summary


def _print_summary(summary_rows: list[dict]) -> None:
    print("\n" + "=" * 76)
    print(f"{'Dims':>6} | {'Métrica':<14} | {'Avg P@K':>8} | {'Avg R@K':>8} | {'Avg ms':>10}")
    print("-" * 76)
    for r in summary_rows:
        print(
            f"{r['dimensions']:>6} | {r['metric']:<14} | "
            f"{r['avg_precision_at_k']:>8.4f} | {r['avg_recall_at_k']:>8.4f} | "
            f"{r['avg_time_ms']:>10.3f}"
        )
    print("=" * 76 + "\n")


if __name__ == "__main__":

    # Fase 1: Ingestão (executar apenas uma vez)
    # chunks = load_passages()
    # run_ingestion(chunks)

    # Fase 2+3: Experimento
    queries = load_queries()
    qrels = load_qrels()
    run_experiment(queries, qrels)
