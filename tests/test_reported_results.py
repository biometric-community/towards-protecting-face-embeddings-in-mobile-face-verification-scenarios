"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re

import pytest


def row(report: str, mode: str) -> tuple[float, ...]:
    match = re.search(rf"\| {mode} \| ([\d.]+)% \| ([\d.]+)% \| ([\d.]+)% \|", report)
    assert match, f"{mode} row missing from REPORT.md"
    return tuple(float(g) for g in match.groups())


@pytest.mark.parametrize("mode,key", [("baseline", "baseline"), ("polyprotect", "overlap_4")])
def test_tmr_at_fmr_1pct_matches_table_1(report, targets, mode, key):
    expected = targets["table_1_tmr_fmr_1pct"]["facenet"][key]
    assert row(report, mode)[2] == pytest.approx(expected, abs=targets["tolerance_pp"])


def test_polyprotect_with_overlap_4_does_not_lose_accuracy(report):
    """Table 1: at overlap 4 the protected system matches or beats the baseline."""
    assert row(report, "polyprotect")[2] >= row(report, "baseline")[2]


def test_stolen_parameter_scenario_is_reported(report):
    """Sec. 4.2 evaluates both the Normal and Stolen Coefficients and Exponents scenarios."""
    assert re.search(r"stolen|SCE", report, re.IGNORECASE)


def test_unlinkability_proxy_is_near_zero(report):
    match = re.search(r"Same-subject different-key mean cosine: (-?[\d.]+)", report)
    assert match and abs(float(match.group(1))) <= 0.1


def test_irreversibility_figure_reflects_overlap_4(report, default_cfg, targets):
    """Table 3: overlap 4 is inverted about 95% of the time, so recovery must be high."""
    overlap = default_cfg["polyprotect"]["overlap"]
    assert targets["table_3_inversion_fmr_0_1pct"]["facenet"][overlap] >= 0.9
    match = re.search(r"inversion mean recon cosine \(known C,E\): (-?[\d.]+)", report)
    assert match and float(match.group(1)) >= 0.8
