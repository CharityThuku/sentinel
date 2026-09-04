"""The test that matters most in this repo.

Every backtest of a detection system is a lie unless the detector's output at
time t is provably independent of data after t. A rolling z-score computed
with pandas over the full array, a covariance fitted on the whole sample, a
threshold picked from the full-sample 99th percentile - all of them look
excellent in a notebook and fail in production, and none of them would be
caught by a unit test of the maths.

So we test the property directly: truncate the tape at a random point, replay,
and assert the alerts up to that point are byte-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.detectors import default_stack
from sentinel.engine import Engine
from sentinel.synth import generate


def _alerts(snaps):
    return [a.to_row() for a in Engine(default_stack()).run(snaps)]


@pytest.mark.parametrize("cut", [1500, 3000, 4500, 5800])
def test_no_lookahead(cut):
    snaps, _ = generate(n_ticks=6000, seed=20260904)
    full = [a for a in _alerts(snaps) if a["idx"] < cut]
    truncated = [a for a in _alerts(snaps[:cut]) if a["idx"] < cut]
    assert full == truncated


def test_future_perturbation_cannot_change_the_past():
    """Corrupt the tail; the head's alerts must not move."""
    snaps, _ = generate(n_ticks=3000, seed=5)
    cut = 2000
    baseline = [a for a in _alerts(snaps) if a["idx"] < cut]

    tampered = list(snaps[:cut]) + list(generate(n_ticks=1000, seed=99)[0])
    tampered = [
        type(s)(idx=i, ts=s.ts, ticks=s.ticks) for i, s in enumerate(tampered)
    ]
    after = [a for a in _alerts(tampered) if a["idx"] < cut]
    assert baseline == after


def test_determinism_same_seed_same_alerts():
    a = _alerts(generate(n_ticks=2500, seed=1234)[0])
    b = _alerts(generate(n_ticks=2500, seed=1234)[0])
    assert a == b


def test_state_is_bounded():
    """Memory must not grow with stream length: detector state after 2,000
    snapshots and after 8,000 has the same shape."""
    import sys

    def footprint(n):
        snaps, _ = generate(n_ticks=n, seed=3)
        eng = Engine(default_stack())
        list(eng.run(snaps))
        return sum(
            sys.getsizeof(v)
            for det in eng.detectors
            for v in vars(det).values()
            if isinstance(v, (dict, list))
        )

    assert footprint(6000) <= footprint(2000) * 1.05


def test_generator_is_reproducible():
    s1, e1 = generate(n_ticks=800, seed=77)
    s2, e2 = generate(n_ticks=800, seed=77)
    assert e1 == e2
    assert np.allclose(
        [t.price for t in s1[-1].ticks.values()],
        [t.price for t in s2[-1].ticks.values()],
    )
