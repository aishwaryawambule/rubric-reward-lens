import numpy as np

from rubric_reward_lens.stats import (
    bootstrap_ci,
    cohen_kappa,
    paired_bootstrap_diff,
    quadratic_weighted_kappa,
    significant,
    spearman,
)


def test_bootstrap_ci_constant():
    pt, lo, hi = bootstrap_ci([0.5] * 100)
    assert abs(pt - 0.5) < 1e-9
    assert lo <= 0.5 <= hi


def test_bootstrap_ci_deterministic():
    a = bootstrap_ci([0.1, 0.2, 0.9, 0.4, 0.7], seed=0)
    b = bootstrap_ci([0.1, 0.2, 0.9, 0.4, 0.7], seed=0)
    assert a == b


def test_bootstrap_ci_empty_and_single():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([0.3]) == (0.3, 0.3, 0.3)


def test_paired_bootstrap_diff_positive_significant():
    a = [0.8, 0.9, 0.7, 0.85, 0.95]
    b = [0.2, 0.3, 0.1, 0.25, 0.35]
    pt, lo, hi = paired_bootstrap_diff(a, b)
    assert pt > 0
    assert significant(lo, hi)


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_spearman_no_variance():
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_cohen_kappa_perfect():
    assert cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == 1.0


def test_qwk_perfect_and_partial():
    assert quadratic_weighted_kappa([0, 1, 2], [0, 1, 2], 3) == 1.0
    # off-by-one is penalised less than a perfect mismatch
    near = quadratic_weighted_kappa([0, 1, 2, 0], [0, 1, 1, 0], 3)
    assert 0.0 < near < 1.0


def test_significant():
    assert significant(0.1, 0.3) is True
    assert significant(-0.3, -0.1) is True
    assert significant(-0.1, 0.2) is False
