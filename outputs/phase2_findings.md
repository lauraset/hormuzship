# Phase 2 findings — AIS presence vs SAR activity (Q2)

Produced by `src/02_fetch_ais.py`, `src/05_activity_anomaly.py`,
`src/07_make_figures.py` on the real GFW data, for both AOIs
(`hormuz_core` = primary, `hormuz_narrow_strait` = nested sensitivity).

Sources read back for this document:
`outputs/tables/phase_summary.csv`, `outputs/tables/activity_anomaly_weekly.csv`,
`outputs/tables/ais_coverage.csv`, `outputs/tables/ais_vessel_type_summary.csv`,
`outputs/tables/ais_gear_type_summary.csv`,
`outputs/tables/ais_vessel_population_sensitivity.csv`,
`outputs/figures/fig_phase2_0*.png`.

---

## 1. What was actually downloaded

| item | value |
| --- | --- |
| dataset | `public-global-presence:v4.0` (pinned; never `latest`) |
| requests | 58 data requests = 2 AOIs × 29 months, **sequential**, ~3 s pause, 0 failures |
| cache | `data/raw/ais_presence/*.zip`, raw bytes, 125 MB |
| manifest | `outputs/logs/api_manifest.csv` (59 rows: 58 data requests + 1 auth probe) |

Two API facts had to be discovered live and are now encoded in the config and
tested: the report response is a **ZIP** (containing the activity CSV, a readme,
the GFW *"Considerations when using AIS data"* PDF and the reference geometry),
and the `date-range` **end date is exclusive** (`d,d` returns nothing; `d,d+1`
returns exactly one day). The client therefore requests `end + 1 day`, which
avoids silently dropping the last day of every chunk.

### Returned schema (never assumed)

One row per vessel × day × 0.01° presence cell:

```
Lat, Lon, Time Range, Vessel ID, Flag, Vessel Name, Entry Timestamp,
Exit Timestamp, Gear Type, Vessel Type, MMSI, IMO, CallSign,
First Transmission Date, Last Transmission Date, Vessel Presence Hours
```

### Vessel categories actually observed

`Vessel Type` (coarse, 9 levels) — share of presence hours in the primary AOI:
`OTHER_NON_FISHING` 42.4 %, `CARGO` 30.4 %, `GEAR` 21.4 %, `FISHING` 3.5 %,
`PASSENGER` 1.8 %, `BUNKER` 0.4 %, `INSUFFICIENT_DATA` 0.2 %, `CARRIER` 0.04 %,
`SEISMIC_VESSEL` 0.01 %.

`Gear Type` (finer, 18 levels) is the field that matters, because the coarse
field folds tankers away: **`TANKER` and `CARGO_OR_TANKER` exist only in
`Gear Type`**, where `TANKER` alone is the single largest category (30 % of
primary-AOI presence hours, ahead of `CARGO` at 30 % and `GEAR` at 21 %). The
commercial subset is therefore `TANKER + CARGO + CARGO_OR_TANKER`
(**64.5 %** of primary-AOI presence hours), defined in
`ais.vessel_population_sets` *after* inspecting the data.

Note that tankers — the dominant transiting hull in this strait — cannot be read
off the `Vessel Type` field at all; a study that classified vessels with the
coarse field would have silently mislabelled them.

### Coverage validation (missing days are never zero)

1 764 calendar days × 2 AOIs were checked day by day:

| AOI | days with an AIS row | days without |
| --- | --- | --- |
| primary `hormuz_core` | **882 / 882 (100 %)** | 0 |
| nested `hormuz_narrow_strait` | 866 / 882 (98.2 %) | **16** |

All 16 days without any AIS-reporting vessel in the narrow box fall inside the
disruption (2026-03-21 … 2026-05-17). They are not an API gap: the same months
return rows for every other day, and the enclosing primary AOI has rows on all 16
of those days. The correct reading is *"no AIS-reporting vessel was present in the
narrow transit box on those days"*, and `outputs/tables/ais_coverage.csv` records
them as `missing_no_row_returned` so no downstream step can read them as zero.

---

## 2. Answers to the seven decision questions

All anomalies are `log((2026 value + eps) / (median of the same ISO week in
2024 and 2025 + eps))`, computed weekly; the interval is a percentile bootstrap
over weeks. `AIS_SAR_days` uses only days that actually have a SAR acquisition,
and — because that changes how many days enter a week — it uses **mean hours per
observed acquisition day**, while `AIS_all_days` uses the weekly sum.

### Q1. How much did AIS activity decline relative to baseline?

| AOI | series | baseline | acute shock (Mar 1–27, 4 weeks) | persistent (Mar 28–May 31, 8–9 weeks) |
| --- | --- | --- | --- | --- |
| primary | `AIS_all_days` | +0.170 [+0.059, +0.326] | **−1.215** [−1.473, −0.934] | **−1.427** [−1.850, −0.971] |
| primary | `AIS_SAR_days` | +0.158 [+0.038, +0.274] | **−1.216** [−1.399, −1.050] | **−1.340** [−1.781, −0.899] |
| nested | `AIS_all_days` | +0.130 | **−4.603** [−7.649, −3.149] | **−3.773** [−4.517, −3.132] |
| nested | `AIS_SAR_days` | +0.142 | **−4.051** [−5.022, −3.081] | **−3.846** [−4.373, −2.841] |

In percentage terms: primary −70 % (acute) and −76 % (persistent); narrow
−99 % (acute) and −98 % (persistent). Pre-event weeks sit slightly *above* the
historical expectation (+0.13 to +0.17), so the baseline is not depressed.

### Q2. How much did SAR activity decline?

| AOI | baseline | acute shock | persistent |
| --- | --- | --- | --- |
| primary | +0.123 [+0.008, +0.191] | **−0.710** [−1.026, −0.464] | **−0.671** [−0.751, −0.584] |
| nested | −0.070 [−0.214, +0.079] | **−1.968** [−2.318, −1.658] | **−1.839** [−2.037, −1.627] |

i.e. primary −51 % / −49 %, narrow −86 % / −84 % in SAR density.

### Q3. Is the AIS decline materially larger than the SAR decline?

**Yes, in both AOIs**, and the formal gap
(`observability_assessment_gap = AIS anomaly − SAR anomaly`) is negative and
outside zero throughout the event:

| AOI | phase | gap, `AIS_all_days` | gap, `AIS_SAR_days` |
| --- | --- | --- | --- |
| primary | acute | **−0.504** | **−0.505** |
| primary | persistent | **−0.642** | **−0.669** |
| nested | acute | **−2.635** | **−2.083** |
| nested | persistent | **−2.020** | **−2.036** |

In the primary AOI the AIS measure fell about 1.6–1.9× more than the SAR measure
in log terms; in the narrow transit box, 7–13× more.

### Q4. Does the difference persist when AIS is restricted to SAR-acquisition days?

**Yes.** The two AIS series give the same answer: the gaps differ by ≤0.03 log
points in the primary AOI (−0.505 vs −0.504 acute; −0.669 vs −0.642 persistent)
and by ≤0.6 in the narrow AOI, which is well inside the bootstrap intervals. At
weekly resolution the two series track each other closely (e.g. primary
2026-05-11: −1.78 vs −1.81; narrow 2026-03-09: −3.96 vs −5.05).

Temporal sampling therefore does **not** explain the divergence: making AIS see
only the days SAR saw leaves the result intact.

### Q5. Does the result persist in the narrow-strait AOI?

**Yes — and it amplifies.** In the narrow AOI the median scene matching rate is
exactly 0.0 in **9 of the 12 post-event weeks** (the pooled rate is 0.000 in the
same 9), its AIS presence falls to 0–20 hours per acquisition day against
281–402 hours before, and four of twenty post-event acquisition days have zero
AIS hours — of which **three also have SAR detections > 0** (2026-03-24,
2026-05-06, 2026-05-15; the fourth, 2026-04-24, has no SAR detection either).
Phase 3's `ais_zero_missing_audit.csv` later confirmed all of these days are
`structural_zero` rather than `missing_unknown`.

The strongest single piece of evidence is the same-day, same-geography comparison
inside the narrow box (SAR acquisition days only):

| period | days | SAR detections / day | AIS presence hours / day | AIS vessels / day | days with zero AIS hours |
| --- | --- | --- | --- | --- | --- |
| Dec 15 2025 – Feb 23 2026 | 17 | 9–20 (median 14) | 281–402 (median 331) | 125–171 | **0** |
| Mar 4 – May 23 2026 | 20 | 0–3 (median 1) | 0–20 (median 5) | 0–10 | **4** |

### Q6. Does the AIS–SAR divergence coincide with the matching-rate decline?

**Yes, to within one week**, in both AOIs:

| signal | first affected week |
| --- | --- |
| SAR density anomaly | 2026-03-02 (−0.60 primary; −1.52 narrow) |
| AIS anomaly (both series) | 2026-03-02 (−0.90/−1.05 primary; −3.21/−3.16 narrow) |
| observability gap turns negative | 2026-03-02 |
| SAR–AIS matching rate collapses | 2026-03 (0.457→0.268 primary; 0.859→0.091 narrow) |

The pre-event divergence is centred on zero (weekly gaps span −0.41 to +0.75 over
224 weeks), so the turn is abrupt rather than a drift.

### Q7. Could coverage, pipeline version, vessel composition or sparse sampling explain it?

| candidate explanation | test performed | verdict |
| --- | --- | --- |
| SAR coverage collapse | coverage fraction stable (primary 0.32; narrow 0.91); daily AOI coverage stable | **ruled out** |
| AIS day gaps | 100 % of days in the primary AOI, 98.2 % in the narrow AOI; the 16 gaps are themselves the phenomenon | **ruled out as a data gap** |
| Sparse / unaligned sampling | SAR-day-restricted AIS series reproduces the all-days result | **ruled out** |
| Vessel composition | gap present in every subset: commercial gear types −1.06/−1.11, fishing gear types −3.01/−2.04, non-fishing −1.08/−1.16 (primary) | **not explained by one class** |
| GFW detection-pipeline switch (2026-03-01) | the AIS presence product is a *different* dataset, pinned at v4.0 and spanning 2012–2026, so the switch cannot produce the AIS-side decline; the SAR-side matching drop was already shown not to be archive-wide | **ruled out for the AIS leg** |
| Technical break timing | event 2026-02-28, pipeline switch 2026-03-01, i.e. one day apart | **cannot be fully separated for the SAR leg in Phase 2** |

---

## 3. Verdict

The screening rule set out in the brief was: *if AIS declines substantially more
than SAR, while the matching rate also declines and the historical placebo
remains stable, classify the project as a strong candidate for expansion to a
publication-scale multi-event analysis.*

| condition | status |
| --- | --- |
| AIS declines substantially more than SAR | **met** (primary gap −0.50 to −0.67 log; narrow −2.0 to −2.6) |
| SAR–AIS matching rate also declines | **met** (Phase 1: −15.3 pp primary, −72.8 pp narrow, both outside placebo) |
| historical placebo stable | **met** (2024/2025 pre-post deltas ≈0; baseline anomalies slightly positive) |
| survives the SAR-day alignment test | **met** |
| survives a pre-specified second AOI | **met** |
| not caused by coverage/volume artefacts | **met** |

**Classification: strong candidate for expansion to a publication-scale
multi-event analysis.**

What must still be carried forward rather than assumed:

1. **One-day separation** between the event onset and the `pipev3 → pipev4`
   switch. The AIS leg is unaffected, but the SAR leg's level shift cannot be
   fully attributed until GFW reprocessing (or an independent region-month
   check) is available.
2. **Mechanism is not identified.** These data cannot distinguish vessels not
   transmitting, AIS reception/interference problems, or position mismatches
   failing GFW's consistency test — and the Phase 1 finding that many residual
   detections carry an MMSI but a near-zero `matching_score` points at the third
   possibility. No claim about deliberate AIS shutdown is supported.
3. **Tankers are invisible in `Vessel Type`.** Any follow-up must use `Gear Type`
   (or a vessel-level classification), never the coarse field.
4. Phase 1 remains definition-sensitive (primary vs MMSI matching definition), so
   every downstream table should keep reporting both.

Figures: `outputs/figures/fig_phase2_01…07`. The gap figure shows two years of
noise around zero followed by a break exactly at the event; the combined figure
puts the matching-rate collapse and the gap on one timeline.
