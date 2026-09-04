# Sentinel

**A watchdog for financial market data — it catches bad prices, broken feeds, and suspicious trading patterns in real time, before they turn into real losses.**

---

## The problem

Trading desks and risk teams watch a constant stream of prices, quotes, and trade volumes flowing in every second. Buried in that stream are things that can cost real money if nobody notices in time:

- A price feed that freezes and quietly keeps repeating the same stale number
- A "fat finger" trade — someone typing an extra zero
- A stock that suddenly moves on its own for no reason, while everything else is calm (a classic sign that someone knows something before the rest of the market does)
- A position or model that's slowly, quietly drifting away from reality — the kind of problem that doesn't look alarming on any single day, but is very expensive by the time someone notices
- Two things that normally move together suddenly stop moving together, which is often the first sign a portfolio's risk protections have quietly broken
- A data feed being switched, corrupted, or fed placeholder numbers by a broker or vendor system

Simple rules like "alert me if the price jumps by more than 3%" don't work well here. That threshold is too jumpy on a wild day and too blind on a quiet one. And even a good detector is useless if it also floods people with hundreds of false alarms — after a week of noise, everyone just starts ignoring the alerts.

## How Sentinel solves it

Sentinel watches the same stream of market data a trading desk sees, tick by tick, and runs it through six independent checks — cheapest and most certain first. If nothing looks wrong, it stays silent. If something does, it raises one clear, prioritized alert instead of a wall of noise.

It was built and tested the same way a lab tests a medical device: not just "here's the logic, trust it," but by deliberately planting 14 known problems into a realistic simulated market and measuring exactly how many Sentinel caught, how fast, and how many false alarms it raised along the way. Every alert threshold in the system was tuned from those measurements, not guessed.

### The six checks, in plain terms

1. **Is the data even real?** Before anything else, Sentinel checks for the boring-but-common failures: a feed that's frozen and repeating itself, a price that can't possibly be right, a data point that arrived out of order. Most real-world incidents start here, not with some exotic market event.

2. **Did this move more than it should have — for *this* stock, *right now*?** Instead of one fixed threshold for every stock on every day, Sentinel tracks each instrument's own normal "wobble" and adjusts as market conditions change, so it doesn't scream during ordinary volatility and doesn't stay silent when a normally-quiet stock suddenly moves.

3. **Did this move alone?** A stock dropping with the whole market is not surprising. A stock dropping 4% while everything else is flat is worth a second look — it's often the earliest visible sign of news, a leak, or a data error.

4. **Is something slowly drifting off course?** Some problems never produce one dramatic moment — they build up gradually, a little at a time, until the cumulative effect is large. This is the type of issue that a mismarked position or an out-of-date pricing model produces, and it's expensive precisely because nothing about it looks alarming day to day.

5. **Did relationships that are normally stable suddenly break?** Many portfolios rely on certain things moving together (a stock and its hedge, for example). Sentinel watches those relationships as a group and flags it when they stop behaving as expected — even if every individual price looks fine on its own.

6. **Does the data still look like the same kind of data?** This check looks at the "shape" of trade sizes rather than prices. It can catch a vendor quietly switching units, a broker sending placeholder numbers, or a broken system component — problems that don't move any price enough to trip the other checks, but do immediately change the pattern of the numbers.

### Keeping people's trust

A detector nobody trusts is a detector nobody reads. Sentinel is built to avoid alert fatigue on purpose:

- The same ongoing problem raises **one** alert, not one every second it continues.
- Alerts are ranked by severity so the most urgent ones stand out.
- If many things trip at once, Sentinel recognizes that's probably one big event (a market shock or a feed outage) and raises a single combined alert instead of dozens of small ones.

## Proof it works, not just a claim

Sentinel was tested against a simulated market with 14 known, deliberately hidden problems spread across 9 different problem types. Results:

| What we measured | Result |
|---|---|
| Problems caught | **14 out of 14 (100%)** |
| Typical time to catch a problem | **Almost instantly** (a couple of price updates) |
| Alerts that pointed to a real, known problem | **About 3 out of 4** |
| False alarms | **Roughly 2 per 1,000 data points** — low enough that a desk can act on every alert |

*(Note: these numbers come from a controlled test with known planted problems, which is the only honest way to measure a detector like this — real markets don't come with an answer key. Treat it as a strong sign the approach works, not a guaranteed real-world score.)*

## What's next

This is a solid working system, but it isn't finished. The honest next steps, in priority order:

1. **Make the "slow drift" check even more reliable.** It currently works well but can occasionally be tricked by an ordinary wandering price. The fix is to point it at more stable, purpose-built signals rather than raw prices.
2. **Let the system resume instantly after a restart**, instead of needing a short warm-up period to "re-learn" what normal looks like.
3. **Add a second way of detecting slow drift** and compare the two side by side to see which performs better in practice.
4. **Let a human mark an alert as a false alarm**, and have the system learn from that feedback over time — this is how the false-alarm rate should keep improving after launch.
5. **Connect it directly to live production data feeds** and add a monitoring dashboard so the team can watch it running continuously.

## Bottom line

Sentinel turns "watch the market for problems" from a vague, manual, tribal-knowledge task into a tested system with known, measured performance. It catches the boring, high-frequency failures (frozen feeds, fat fingers) and the subtle, expensive ones (slow drift, broken correlations) with the same tool, without burying the desk in false alarms — and every number in this document is something we measured, not something we assumed.

---

*For engineers: see [`ENGINEERING.md`](ENGINEERING.md) for the full technical design, detector internals, and how to run the code.*
