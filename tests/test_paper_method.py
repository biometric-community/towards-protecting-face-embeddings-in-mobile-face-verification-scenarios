"""The V -> P mapping and its parameters, checked against Sec. 3, Sec. 4.1 and Table 2."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from polyprotect.protect.polyprotect import PolyProtect, sample_user_params

C = np.array([3, -7, 11, -2, 5])
E = np.array([2, 4, 1, 5, 3])


def eq1(window: np.ndarray) -> float:
    return float(np.sum(C * window.astype(np.float64) ** E))


def test_eq1_on_positive_elements():
    v = np.array([0.5, 0.25, 0.75, 0.1, 0.9])
    assert PolyProtect(5, 0).transform_numpy(v, C, E)[0] == pytest.approx(eq1(v))


def test_eq1_on_negative_elements():
    """Eq. (1) raises v to e directly, so an even power of a negative v is positive."""
    v = np.array([-0.5, -0.25, 0.75, -0.1, 0.9])
    assert PolyProtect(5, 0).transform_numpy(v, C, E)[0] == pytest.approx(eq1(v))


def test_eq2_overlap_zero_uses_the_next_m_elements():
    v = np.linspace(0.1, 0.9, 10)
    assert PolyProtect(5, 0).transform_numpy(v, C, E)[1] == pytest.approx(eq1(v[5:10]))


def test_eq3_overlap_m_minus_1_slides_by_one():
    v = np.linspace(0.1, 0.9, 10)
    assert PolyProtect(5, 4).transform_numpy(v, C, E)[1] == pytest.approx(eq1(v[1:6]))


def test_incomplete_last_set_is_zero_padded():
    """Sec. 3: a 7-element V with overlap 0 pads the last set with zeros."""
    v = np.linspace(0.1, 0.7, 7)
    padded = np.concatenate([v[5:], np.zeros(3)])
    out = PolyProtect(5, 0).transform_numpy(v, C, E)
    assert len(out) == 2
    assert out[1] == pytest.approx(eq1(padded))


@pytest.mark.parametrize("overlap", [0, 1, 2, 3, 4])
def test_template_dimensionality_matches_table_2(overlap, targets):
    v = np.random.RandomState(0).normal(size=128)
    out = PolyProtect(5, overlap).transform_numpy(v, C, E)
    assert len(out) == targets["table_2_dims"][overlap]


def test_user_parameters_follow_the_paper():
    """C: unique non-zero integers in [-50, 50]. E: a permutation of 1..5."""
    rng = np.random.RandomState(0)
    for _ in range(500):
        c, e = sample_user_params(5, rng)
        assert len(set(c.tolist())) == 5 and 0 not in c
        assert c.min() >= -50 and c.max() <= 50
        assert sorted(e.tolist()) == [1, 2, 3, 4, 5]


def test_torch_and_numpy_paths_agree():
    pp = PolyProtect(5, 2)
    v = np.random.RandomState(1).normal(size=128)
    expected = pp.transform_numpy(v, C, E)
    expected /= np.linalg.norm(expected)
    got = pp.transform(torch.tensor(v, dtype=torch.float64), C, E).numpy()
    assert np.allclose(got, expected, atol=1e-6)
