from __future__ import annotations

import numpy as np
import pytest

from sentinel.stats import (
    EWMA,
    EWMVariance,
    Latch,
    P2Quantile,
    RobustScale,
    benford_chi2,
    ledoit_wolf,
)


def test_ewma_halflife_semantics():
    ew = EWMA(halflife=10.0)
    for _ in range(400):
        ew.update(1.0)
    assert ew.value == pytest.approx(1.0, abs=1e-6)
    # after `halflife` steps of a step change, we are halfway there
    ew2 = EWMA(halflife=10.0)
    ew2.update(0.0)
    for _ in range(10):
        ew2.update(1.0)
    assert ew2.value == pytest.approx(0.5, abs=0.02)


def test_ewmvariance_matches_iid_sigma():
    rng = np.random.default_rng(0)
    ev = EWMVariance(halflife=500.0)
    for x in rng.normal(3.0, 2.0, size=20000):
        ev.update(x)
    assert ev.mean == pytest.approx(3.0, abs=0.15)
    assert ev.std == pytest.approx(2.0, rel=0.1)


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9, 0.995])
def test_p2_quantile_tracks_numpy(q):
    rng = np.random.default_rng(42)
    data = rng.lognormal(0.0, 1.0, size=50000)
    est = P2Quantile(q)
    for x in data:
        est.update(x)
    truth = float(np.quantile(data, q))
    assert est.value == pytest.approx(truth, rel=0.06)


def test_p2_quantile_is_constant_memory():
    est = P2Quantile(0.5)
    for x in range(100000):
        est.update(float(x))
    assert len(est.heights) == 5


def test_robust_scale_resists_contamination():
    """One 200-sigma print must not blind the estimator to the next one."""
    rng = np.random.default_rng(1)
    clean = RobustScale(halflife=100.0, warmup=50)
    dirty = RobustScale(halflife=100.0, warmup=50)
    data = rng.normal(0.0, 1.0, size=1000)
    for i, x in enumerate(data):
        clean.score(float(x))
        dirty.score(200.0 if i == 500 else float(x))
    assert dirty.scale == pytest.approx(clean.scale, rel=0.5)
    # the contaminated estimator still flags a 6-sigma move
    assert dirty.score(6.0) > 4.0


def test_ledoit_wolf_is_psd_and_shrinks_when_undersampled():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(15, 20))  # T < N: sample covariance is singular
    cov, shrink = ledoit_wolf(x)
    assert 0.0 < shrink <= 1.0
    eig = np.linalg.eigvalsh(cov)
    assert eig.min() > 0, "shrunk covariance must be invertible"
    sample_eig = np.linalg.eigvalsh(x.T @ x / len(x))
    assert sample_eig.min() < 1e-9  # the thing we fixed


def test_ledoit_wolf_shrinks_less_with_more_data():
    rng = np.random.default_rng(11)
    _, small = ledoit_wolf(rng.normal(size=(40, 10)))
    _, large = ledoit_wolf(rng.normal(size=(4000, 10)))
    assert large < small


def test_benford_flags_round_numbers_not_natural_ones():
    rng = np.random.default_rng(3)
    natural = rng.lognormal(10.0, 3.0, size=1000)
    chi2_nat, n = benford_chi2(natural)
    assert n == 1000
    assert chi2_nat < 26.1  # p=0.001, df=8

    tampered = rng.choice([1.0, 5.0], size=1000) * rng.choice([1e5, 1e6], size=1000)
    chi2_bad, _ = benford_chi2(tampered)
    assert chi2_bad > 100


def test_latch_fires_once_per_episode():
    latch = Latch(enter=5.0, exit_=3.0)
    fires = [latch.push(s) for s in [1, 6, 7, 6, 4, 8, 2, 9]]
    #                                    ^fire       stays latched  ^exit ^fire
    assert fires == [False, True, False, False, False, False, False, True]


def test_latch_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        Latch(enter=3.0, exit_=5.0)
