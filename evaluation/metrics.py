import logging

from scipy.stats import kendalltau

logger = logging.getLogger(__name__)


def kendall_tau(ranked_approx: list[int], ranked_ground_truth: list[int]) -> float:
    """Calcula Kendall's Tau entre um ranking aproximado e o ground truth.

    Alinha os IDs por posição nos dois rankings e computa a correlação.
    Retorna 0.0 se não houver elementos em comum.
    """
    if not ranked_approx or not ranked_ground_truth:
        return 0.0

    gt_set = set(ranked_ground_truth)
    common_ids = [x for x in ranked_approx if x in gt_set]

    if len(common_ids) < 2:
        return 0.0 if not common_ids else 1.0

    gt_rank = {id_: pos for pos, id_ in enumerate(ranked_ground_truth)}
    approx_positions = list(range(len(common_ids)))
    gt_positions = [gt_rank[id_] for id_ in common_ids]

    tau, _ = kendalltau(approx_positions, gt_positions)
    return float(tau) if tau == tau else 0.0  # NaN guard


def overlap_at_k(
    set_approx: list[int],
    set_ground_truth: list[int],
    k: int,
) -> float:
    """Calcula Overlap@K: fração de elementos em comum nos top-K.

    Args:
        set_approx: IDs retornados pela métrica aproximada.
        set_ground_truth: IDs retornados pelo sequential scan.
        k: Tamanho do corte.

    Returns:
        Proporção de interseção (0.0 a 1.0).
    """
    if k <= 0:
        return 0.0

    intersection = len(set(set_approx[:k]) & set(set_ground_truth[:k]))
    return intersection / k
