import csv
import logging
import os

logger = logging.getLogger(__name__)


def export_timings(
    timings_by_question: list[dict[str, float]],
    dimensions: int,
    output_dir: str = "results",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"timings_{dimensions}d.csv")

    fieldnames = [
        "question_id",
        "cosine_ms",
        "euclidean_ms",
        "dot_product_ms",
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
                }
            )

    logger.info("Timings exportados: %s", filepath)
    return filepath


def export_results(
    results_by_question: list[dict],
    dimensions: int,
    output_dir: str = "results",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"results_{dimensions}d.csv")

    fieldnames = [
        "question_id",
        "query_id",
        "question",
        "metric",
        "returned_ids",
        "precision_at_k",
        "recall_at_k",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_by_question:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "metric": row["metric"],
                    "returned_ids": str(row["returned_ids"]),
                    "precision_at_k": f"{row['precision_at_k']:.4f}",
                    "recall_at_k": f"{row['recall_at_k']:.4f}",
                }
            )

    logger.info("Resultados exportados: %s", filepath)
    return filepath


def export_summary(
    summary_rows: list[dict],
    output_dir: str = "results",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "summary.csv")

    fieldnames = [
        "dimensions",
        "metric",
        "avg_precision_at_k",
        "avg_recall_at_k",
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
                    "avg_precision_at_k": f"{row['avg_precision_at_k']:.4f}",
                    "avg_recall_at_k": f"{row['avg_recall_at_k']:.4f}",
                    "avg_time_ms": f"{row['avg_time_ms']:.3f}",
                }
            )

    logger.info("Summary exportado: %s", filepath)
    return filepath
