"""Evaluation metrics: Rank points, F-max, precision/recall."""
def rank_points(true_variant_rank: int | None) -> float:
    """100 pts if rank 1, partial credit within top 10, else 0."""
    if true_variant_rank is None:
        return 0.0
    if true_variant_rank == 1:
        return 100.0
    if 1 < true_variant_rank <= 10:
        # linear decay example; adapt to official scoring
        return 100 * (11 - true_variant_rank) / 10
    return 0.0

def fmax(precision_curve, recall_curve):
    raise NotImplementedError
