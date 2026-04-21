import csv
import logging
import os

logger = logging.getLogger(__name__)


def export_timings(
    timings_by_question: list[dict[str, float]],
    dimensions: int,
    output_dir: str = "results",
) -> str:
    """Exporta tempos de execução para CSV.

    Args:
        timings_by_question: Lista de dicts com chaves
            cosine_ms, euclidean_ms, dot_product_ms, sequential_ms.
        dimensions: Dimensão da base (para nome do arquivo).
        output_dir: Diretório de saída.

    Returns:
        Caminho do arquivo gerado.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"timings_{dimensions}d.csv")

    fieldnames = [
        "question_id",
        "cosine_ms",
        "euclidean_ms",
        "dot_product_ms",
        "sequential_ms",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(timings_by_question):
            writer.writerow(
                {
                    "question_id": idx,
                    "cosine_ms": f"{row.get('cosine', 0):.3f}",
                    "euclidean_ms": f"{row.get('euclidean', 0):.3f}",
                    "dot_product_ms": f"{row.get('dot_product', 0):.3f}",
                    "sequential_ms": f"{row.get('sequential', 0):.3f}",
                }
            )

    logger.info("Timings exportados: %s", filepath)
    return filepath


def export_results(
    results_by_question: list[dict],
    dimensions: int,
    output_dir: str = "results",
) -> str:
    """Exporta métricas de avaliação para CSV.

    Args:
        results_by_question: Lista de dicts com chaves
            question_id, metric, returned_ids, kendall_tau, overlap_at_k.
        dimensions: Dimensão da base (para nome do arquivo).
        output_dir: Diretório de saída.

    Returns:
        Caminho do arquivo gerado.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"results_{dimensions}d.csv")

    fieldnames = [
        "question_id",
        "question",
        "metric",
        "returned_ids",
        "sequential_ids",
        "kendall_tau",
        "overlap_at_k",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_by_question:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "metric": row["metric"],
                    "returned_ids": str(row["returned_ids"]),
                    "sequential_ids": str(row["sequential_ids"]),
                    "kendall_tau": f"{row['kendall_tau']:.4f}",
                    "overlap_at_k": f"{row['overlap_at_k']:.4f}",
                }
            )

    logger.info("Resultados exportados: %s", filepath)
    return filepath


def export_summary(
    summary_rows: list[dict],
    output_dir: str = "results",
) -> str:
    """Exporta resumo agregado do experimento.

    Args:
        summary_rows: Lista de dicts com chaves
            dimensions, metric, avg_kendall_tau, avg_overlap_at_k,
            avg_time_ms.
        output_dir: Diretório de saída.

    Returns:
        Caminho do arquivo gerado.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "summary.csv")

    fieldnames = [
        "dimensions",
        "metric",
        "avg_kendall_tau",
        "avg_overlap_at_k",
        "avg_time_ms",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "dimensions": row["dimensions"],
                    "metric": row["metric"],
                    "avg_kendall_tau": f"{row['avg_kendall_tau']:.4f}",
                    "avg_overlap_at_k": f"{row['avg_overlap_at_k']:.4f}",
                    "avg_time_ms": f"{row['avg_time_ms']:.3f}",
                }
            )

    logger.info("Summary exportado: %s", filepath)
    return filepath
