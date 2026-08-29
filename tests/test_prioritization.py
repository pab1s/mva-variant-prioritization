"""Tests for prioritization logic."""
def test_rank_points():
    from src.ranking.metrics import rank_points
    assert rank_points(1) == 100
    assert rank_points(11) == 0
