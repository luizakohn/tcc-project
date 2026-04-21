"""Orquestrador principal do experimento RAG Benchmark.

Executa o pipeline completo:
1. Embeda e armazena chunks nas 3 bases (768d, 1024d, 1536d)
2. Para cada pergunta de teste, consulta as 4 métricas em cada base
3. Calcula Kendall's Tau e Overlap@K
4. Exporta CSVs de timings, resultados e summary
"""

import logging
import statistics
import sys
import random

import config
from evaluation.exporter import export_results, export_summary, export_timings
from evaluation.metrics import kendall_tau, overlap_at_k
from ingestion.store import embed_and_store
from quati_loader import load_quati, load_queries
from retrieval.query_engine import query_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

APPROX_METRICS = ["cosine", "euclidean", "dot_product"]


def run_ingestion(chunks: list[str]) -> None:
    """Fase 1: Ingestão dos chunks nas 3 bases."""
    logger.info("=== FASE 1: INGESTÃO ===")
    for dims in config.DIMENSIONS:
        base_name = config.BASE_NAMES[dims]
        logger.info("Ingerindo na base %s (%d dims)...", base_name, dims)
        embed_and_store(base_name, dims, chunks)
    logger.info("Ingestão concluída para todas as bases.")


def run_experiment(questions: list[str], k: int = None) -> None:
    """Fase 2+3: Consulta e avaliação."""
    if k is None:
        k = config.K

    dimensions_list = config.DIMENSIONS
    base_names = [config.BASE_NAMES[d] for d in dimensions_list]

    timings_per_dim: dict[int, list[dict[str, float]]] = {
        d: [] for d in dimensions_list
    }
    results_per_dim: dict[int, list[dict]] = {d: [] for d in dimensions_list}

    logger.info("=== FASE 2: CONSULTA E AVALIAÇÃO ===")
    logger.info("Perguntas: %d | K: %d | Bases: %s", len(questions), k, base_names)

    for q_idx, question in enumerate(questions):
        logger.info("Pergunta %d/%d: %s", q_idx + 1, len(questions), question[:80])

        all_results, all_timings = query_all_metrics(
            question, k, dimensions_list, base_names
        )

        for dims in dimensions_list:
            dim_key = str(dims)
            ground_truth = all_results[dim_key]["sequential"]

            timings_per_dim[dims].append(all_timings[dim_key])

            for metric in APPROX_METRICS:
                approx_ids = all_results[dim_key][metric]
                tau = kendall_tau(approx_ids, ground_truth)
                olap = overlap_at_k(approx_ids, ground_truth, k)

                results_per_dim[dims].append(
                    {
                        "question_id": q_idx,
                        "question": question,
                        "metric": metric,
                        "returned_ids": approx_ids,
                        "sequential_ids": ground_truth,
                        "kendall_tau": tau,
                        "overlap_at_k": olap,
                    }
                )

                logger.info(
                    "  [%dd] %s — tau=%.4f, overlap@%d=%.4f",
                    dims,
                    metric,
                    tau,
                    k,
                    olap,
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
    """Gera linhas de resumo agregado."""
    summary = []

    for dims in config.DIMENSIONS:
        for metric in APPROX_METRICS:
            metric_rows = [
                r for r in results_per_dim[dims] if r["metric"] == metric
            ]
            if not metric_rows:
                continue

            avg_tau = statistics.mean(r["kendall_tau"] for r in metric_rows)
            avg_olap = statistics.mean(r["overlap_at_k"] for r in metric_rows)
            avg_time = statistics.mean(
                t.get(metric, 0) for t in timings_per_dim[dims]
            )

            summary.append(
                {
                    "dimensions": dims,
                    "metric": metric,
                    "avg_kendall_tau": avg_tau,
                    "avg_overlap_at_k": avg_olap,
                    "avg_time_ms": avg_time,
                }
            )

    return summary


def _print_summary(summary_rows: list[dict]) -> None:
    """Imprime o resumo no console."""
    print("\n" + "=" * 70)
    print(f"{'Dims':>6} | {'Métrica':<14} | {'Avg τ':>8} | {'Avg O@K':>8} | {'Avg ms':>10}")
    print("-" * 70)
    for r in summary_rows:
        print(
            f"{r['dimensions']:>6} | {r['metric']:<14} | "
            f"{r['avg_kendall_tau']:>8.4f} | {r['avg_overlap_at_k']:>8.4f} | "
            f"{r['avg_time_ms']:>10.3f}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    
    
    questions = load_queries()

    # Fase 1: Ingestão (executar apenas uma vez)
    # run_ingestion(chunks)

    # Fase 2+3: Experimento
    run_experiment(questions)
