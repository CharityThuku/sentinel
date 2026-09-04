# sentinel

Streaming anomaly detection for financial time series. Point-in-time correct,
constant memory, and evaluated against injected faults rather than asserted to
work.

The premise: "alert when a price crosses a threshold" is a one-line problem.
Everything hard about it is what comes after — thresholds that mean something
in both a calm tape and a stressed one, scale estimates that survive the very
outliers they are hunting, anomalies that no per-tick rule can see, and an
alert channel a desk will still be reading in month two.

```bash
pip install -e ".[dev]"
sentinel demo          # run the labelled tape, print alerts
sentinel eval          # score every detector against the injected faults
sentinel run tape.csv  # run against a real tape
pytest                 # 43 tests, including the point-in-time property test
```

---

## Measured results

6,000 snapshots × 10 symbols, 14 injected faults across 9 fault types,
seeded and reproducible (`sentinel eval`):

```
recall                100.0%   (14/14 injected faults)
median latency        2 ticks
precision             74.5%
false alerts / 1k     2.33 snapshots
```

| fault | symbol | latency (ticks) | caught by |
|---|---|---|---|
| fat_finger | ALPH | 0 | robust_z, cross_sectional, correlation |
| fat_finger | JULT | 0 | robust_z, cross_sectional, correlation |
| idio_shock | DLTA | 0 | cross_sectional, robust_z, correlation |
| idio_shock | BRVO | 0 | cross_sectional, robust_z, correlation |
| stale_feed | BRVO | 19 | integrity |
| stale_feed | ALPH | 19 | integrity, cross_sectional |
| crossed_quote | ECHO | 0 | integrity |
| volume_spike | INDX | 0 | integrity |
| vol_regime | HTEL | 5 | robust_z, cross_sectional, correlation |
| decouple | FXTR | 0 | correlation, cross_sectional |
| decouple | ECHO | 28 | correlation, cross_sectional |
| directional_drift | CHRL | 192 | drift, correlation |
| directional_drift | DLTA | 100 | drift, correlation |
| digit_tamper | GOLF | 103 | digits |

Every threshold in this repo was set from that table — from measured recall
and false-alert rates — not from a textbook critical value. Where a tabulated
value would have been wrong, the code says why.

---

## The stack

Six layers, cheapest and most certain first. Each consumes `Snapshot` objects
in order and may never look forward.

**1. `integrity` — is the data real?**
Frozen feeds, crossed and locked quotes, non-positive prices, timestamp
regression, volume spikes. Deterministic, no warm-up, fires before the
statistical layer has anything to say. Most production incidents are not
exotic market events; they are a vendor feed that froze at 09:47 and kept
serving the same print. A model fed a frozen price reports zero volatility
and zero risk, which is exactly when it is most dangerous.

**2. `robust_z` — threshold breach, denominated correctly.**
"Alert above 3%" fails in both directions: it screams all day in March 2020
and stays silent when a normally-dead instrument moves 40bp. The threshold is
scaled to the instrument's current volatility regime, and the scale is an
online median plus MAD-equivalent (`1.4826 × EWMA|x − median|`) rather than a
rolling standard deviation. A single 20-sigma print inflates a rolling stdev
enough to mask the *next* one; it barely moves this. There is a unit test for
exactly that.

**3. `cross_sectional` — is the move idiosyncratic?**
A single-name detector cannot tell you anything a PM cares about. Down 4%
with the market is nothing; down 4% with the market flat means someone knows
something you don't. Strip the factor — cross-sectional **median** return
(the mean is contaminated by the very outlier we're hunting), online EWMA
beta per name — and score the residual against its own robust scale.

**4. `drift` — the anomaly no threshold can see.**
A quarter-sigma-per-tick drift. No single observation is unusual; the
cumulative displacement is. This is what a mismarked position or a
decalibrated pricing model looks like, and it is the expensive kind — a fat
finger gets busted in ten minutes, a drifting mark survives to month-end.
Page-Hinkley sequential test, with two things learned from the false-alert
numbers rather than from theory:

- *Aggregate first.* Tick-by-tick Page-Hinkley tests a quarter-sigma signal
  against unit-sigma noise; measured false alarm rate was ~1 per 1,000 ticks,
  unusable. Blocks of 20 divide the noise by √20 and leave the drift intact.
- *Scale by within-block tick variance, not by the variance of the block
  means.* The latter is circular — drift disperses the block means, which
  inflates the scale, which shrinks the statistic — and it makes a volatility
  regime shift read as a drift.

**5. `correlation` — did the structure break?**
Every leg moves within one sigma and the book still blows up, because legs
that always moved together stopped. Mahalanobis distance on the joint return
vector, with three pieces of hygiene a notebook version skips:

- **Ledoit–Wolf shrinkage.** A Mahalanobis distance inverts the covariance
  matrix. With T=250 and N=10 the smallest eigenvalues are mostly noise, and
  inverting noise manufactures enormous fake anomaly scores.
- **A learned threshold.** The 99.5th percentile of the detector's *own*
  score history, tracked with a P-square estimator, instead of a chi-square
  critical value that assumes Gaussian returns and is therefore wrong every
  day. Extreme observations are winsorised before they enter the baseline: a
  genuine 5,000-sigma print must not redefine what "extreme" means for the
  next month.
- **Quarantine.** The covariance refits on a rolling window, so a sustained
  break contaminates its own baseline within a few dozen ticks and the
  detector goes quietly blind mid-event. While an episode is latched, the
  estimator stops learning — but the *measurement* keeps running. Freezing
  both was the first version, and it deadlocked: the smoothed statistic could
  no longer fall back below its exit level, so the detector went permanently
  blind after its first alert. Quarantine also expires, because a real regime
  change must eventually become the new normal.

**6. `digits` — is the data still the same kind of data?**
Leading-digit forensics on trade notionals. Says nothing about prices; says
whether a stub service got plumbed into production, a broker started sending
round-lot placeholders, or a vendor silently switched units. None of those
move a price enough to trip a sigma threshold, and all change the digit
distribution immediately.

The obvious implementation is Benford's law, and it does not work here.
Benford holds for quantities spanning several orders of magnitude; one liquid
name's notionals span maybe one and a half. Measured on healthy data, a raw
Benford test scored chi² ≈ 200 against a p=0.001 critical value of 26.1 — it
would have fired constantly. So the detector tests each symbol against its
*own* long-run digit distribution, with the reference and the test window kept
strictly disjoint (a digit joins the reference only once it ages out of the
window), and reports the Benford divergence alongside as context.

---

## The parts that aren't detectors

**Point-in-time correctness, enforced by test.** Every backtest of a detection
system is a lie unless output at time *t* is provably independent of data
after *t*. A rolling z-score computed with pandas over the full array, a
covariance fitted on the whole sample, a threshold picked from the full-sample
99th percentile — all look excellent in a notebook, all fail in production,
and none would be caught by a unit test of the maths. `tests/test_pit.py`
truncates the tape at a random point, replays it, and asserts the alerts up to
that point are byte-identical. A second test corrupts the tail and asserts the
head is unmoved.

**Constant memory.** Nothing stores unbounded history. A detector that needs
the 99.5th percentile of its own score distribution over ten years of ticks
uses about 200 bytes, not ten years of ticks (Jain & Chlamtac's P-square
algorithm, five markers). There is a test asserting state after 6,000
snapshots is no larger than after 2,000.

**Alert governance.** Being right 200 times during one event is 199 pieces of
noise, and after two weeks of that nobody reads the channel. The engine owns
suppression explicitly rather than leaving it implicit in each detector's
thresholds: Schmitt-trigger hysteresis so an episode fires once on its rising
edge, per-(detector, symbol, kind) cooldown, severity derived from *ratio to
threshold* so a Mahalanobis distance of 40 and a z-score of 6 are comparable,
and storm collapse — if one detector lights up four or more symbols in a
single snapshot, that is a market event or a feed outage, not four anomalies,
and it escalates into one market-wide alert.

**A labelled benchmark.** You cannot tune an anomaly detector on real data:
nobody hands you the ground truth of which of 6,000 ticks were wrong. The
generator builds a market with a realistic factor structure and a two-state
volatility regime, injects a fixed schedule of known faults, and keeps the
labels. `sentinel eval` scores recall, detection latency, precision and
false-alerts-per-1k against them. It is seeded — the numbers above reproduce
exactly.

---

## Known limitations

Stated here because they are the first things I would ask about.

- **The drift channel has no clean null on a price series.** A driftless
  random walk contains stretches that look exactly like drift, so a
  change-point test on raw prices is testing a hypothesis you cannot reject.
  The reference mean is deliberately short (50 blocks) — long enough that a
  real 400-tick mismark survives it, short enough that the path's own
  wandering doesn't accumulate. Residual false alerts: ~2.5 per 1,000
  snapshots, measured. The right fix is to run this channel on a *stationary*
  series instead.
- **Precision of 74.5% is on synthetic data**, where an "unmatched" alert is
  often a genuine tail event the generator produced honestly. Treat it as a
  noise-budget indicator, not a quality score.
- **Every detector has a warm-up.** Roughly 300–500 snapshots before the
  statistical layers arm. Cold start after a process restart is a real
  operational concern and is not solved here — see below.
- **Cross-sectional and correlation layers assume a shared clock.** Ragged or
  asynchronous arrival needs a resampling stage in front.

## What I would do next

1. Move the drift test onto factor residuals, bases and hedge ratios — series
   that are actually stationary, so the null is real.
2. State snapshotting, so a restart resumes warm instead of blind for 500
   ticks. All estimators are small and serialisable by design.
3. Bayesian online change-point detection as a second drift channel, and
   compare it to Page-Hinkley on the same benchmark rather than on intuition.
4. Alert feedback: let a human mark an alert as noise and feed that back into
   the per-detector thresholds. Precision is the metric that decays in
   production, and it is the only one nobody measures.
5. Kafka/Redpanda source and a Prometheus exporter for the engine stats.

---

## Layout

```
src/sentinel/
  types.py            Tick, Snapshot, Alert, Severity
  stats.py            EWMA, EW variance, P-square quantiles, Ledoit-Wolf,
                      robust scale, Schmitt-trigger latch
  engine.py           orchestration + alert governance
  synth.py            labelled market generator with injected faults
  evaluate.py         recall / latency / precision / false-alert rate
  io.py               CSV ingestion
  cli.py              demo | eval | run | export
  detectors/          integrity, robust_z, cross_sectional, drift,
                      correlation, digits
tests/
  test_stats.py       estimator correctness, contamination resistance
  test_detectors.py   each detector catches its fault and stays quiet otherwise
  test_engine.py      cooldown, storm collapse, severity routing
  test_pit.py         no-lookahead, determinism, bounded state
```

## CSV input

```csv
ts,symbol,price,volume,bid,ask
1757000000,ALPH,101.24,14200,101.20,101.28
1757000000,BRVO,58.11,9800,58.09,58.13
```

Rows group into snapshots by `ts`. Any tape that flattens to this shape —
equity bars, bond marks, an internal position file — runs through the
identical stack. `sentinel export tape.csv` writes the benchmark tape in this
format if you want a starting point.

## License

MIT
