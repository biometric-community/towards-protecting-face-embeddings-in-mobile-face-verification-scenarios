"""The paper's findings, reproduced with this repo's mapping on synthetic embeddings."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from polyprotect.protect.polyprotect import PolyProtect, sample_user_params

DIM, SUBJECTS, SAMPLES = 128, 40, 4


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture(scope="module")
def embeddings() -> np.ndarray:
    # The shared component makes impostors correlated, as real face embeddings are.
    rng = np.random.RandomState(0)
    shared = rng.normal(size=(1, 1, DIM))
    centers = rng.normal(size=(SUBJECTS, 1, DIM))
    return unit(shared + centers + 1.6 * rng.normal(size=(SUBJECTS, SAMPLES, DIM)))


def protect(pp, v, params):
    return unit(pp.transform_numpy(v, *params))


def score_sets(embeddings, overlap, stolen=False, seed=1):
    """Genuine and impostor scores for the N and SCE scenarios of Sec. 4.2."""
    pp = PolyProtect(5, overlap)
    rng = np.random.RandomState(seed)
    keys = [sample_user_params(5, rng) for _ in range(SUBJECTS)]
    genuine, impostor = [], []
    for s in range(SUBJECTS):
        reference = protect(pp, embeddings[s, 0], keys[s])
        for q in range(1, SAMPLES):
            genuine.append(reference @ protect(pp, embeddings[s, q], keys[s]))
        for other in range(SUBJECTS):
            if other != s:
                key = keys[s] if stolen else keys[other]
                impostor.append(reference @ protect(pp, embeddings[other, 1], key))
    return np.array(genuine), np.array(impostor)


def tmr(genuine: np.ndarray, impostor: np.ndarray, fmr: float = 0.01) -> float:
    return float((genuine >= np.quantile(impostor, 1.0 - fmr)).mean())


def test_accuracy_improves_with_overlap(embeddings):
    """Fig. 4 and Table 1: more overlap gives higher recognition accuracy."""
    assert tmr(*score_sets(embeddings, 4)) >= tmr(*score_sets(embeddings, 0))


def test_overlap_4_is_at_least_as_accurate_as_the_baseline(embeddings):
    """Table 1: user-specific parameters separate users, so overlap 4 matches the baseline."""
    genuine = np.array(
        [embeddings[s, 0] @ embeddings[s, q] for s in range(SUBJECTS) for q in range(1, SAMPLES)]
    )
    impostor = np.array(
        [embeddings[s, 0] @ embeddings[o, 1] for s in range(SUBJECTS) for o in range(SUBJECTS) if o != s]
    )
    assert tmr(*score_sets(embeddings, 4)) >= tmr(genuine, impostor)


def test_stolen_parameters_are_worse_than_normal(embeddings):
    """Sec. 4.2: the SCE scenario is the worst case."""
    _, normal = score_sets(embeddings, 4)
    _, stolen = score_sets(embeddings, 4, stolen=True)
    assert stolen.mean() > normal.mean()


def test_different_keys_give_unrelated_templates(embeddings):
    """Sec. 4.4: templates of one subject under different keys are effectively unlinkable."""
    pp = PolyProtect(5, 2)
    rng = np.random.RandomState(2)
    scores = [
        protect(pp, embeddings[s, 0], sample_user_params(5, rng))
        @ protect(pp, embeddings[s, 1], sample_user_params(5, rng))
        for s in range(SUBJECTS)
        for _ in range(10)
    ]
    assert abs(float(np.mean(scores))) <= 0.1


def invert(overlap: int, seed: int) -> float:
    """Sec. 4.3 attack: solve P -> V numerically with C and E known."""
    rng = np.random.RandomState(seed)
    pp = PolyProtect(5, overlap)
    v = unit(rng.normal(size=DIM))
    params = sample_user_params(5, rng)
    p = pp.transform_numpy(v, *params)
    solved = least_squares(lambda x: pp.transform_numpy(x, *params) - p, np.full(DIM, 0.01), max_nfev=60)
    return float(unit(solved.x) @ v)


def test_inversion_succeeds_at_overlap_4_and_fails_at_overlap_0():
    """Table 3: inversion success is about 0.95 at overlap 4 and 0.00 at overlap 0."""
    assert np.mean([invert(4, s) for s in range(3)]) >= 0.8
    assert np.mean([invert(0, s) for s in range(3)]) <= 0.6
