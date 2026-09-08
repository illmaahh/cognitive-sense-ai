from src.features import heuristic_load_score


def test_score_bounds():
    score = heuristic_load_score({"perf_reaction_time_s": 1.2, "perf_correct": 0.7})
    assert 0 <= score <= 100
