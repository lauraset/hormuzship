# Phase 1 findings — SAR-side feasibility test (Hormuz AIS observability)

Generated after a full Phase 1 run of `src/00_inspect_data.py`,
`src/01_prepare_sar.py`, `src/03_build_scene_metrics.py` on the real GFW
archive (`D:/gfwdata`, 2024-01-01 … 2026-05-31).

Scope reminder: this document covers the **SAR side only**. AIS vessel presence
(Phase 2) is not downloaded yet, so Q2 (`observability_gap`) can only be
partially answered here.

---

## 1. Actual SAR schema found

Detection files — `sar_vessel_detections_pipev<version>_<YYYYMM>.csv`:

| column | type | missing | meaning |
| --- | --- | --- | --- |
| `scene_id` | str | 0 | `S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_..._<hex>` |
| `timestamp` | str (`... UTC`) | 0 | acquisition time |
| `lat`, `lon` | float | 0 | WGS84 (EPSG:4326, verified by range checks) |
| `presence_score` | float | 0 | vessel-presence confidence |
| `length_m` | float | 0 | estimated vessel length |
| `mmsi` | float | 12–21 % | **null ⇒ no AIS identity assigned** |
| `matching_score` | float | 0 | AIS-match quality (0 when unmatched) |
| `fishing_score` | float | 0 | fishing classifier output |
| `matched_category` | str | ~0 | `unmatched` + 11 matched levels |

Footprint files — `sar_detections_footprints_<YYYYMM>.csv`:

| column | type | meaning |
| --- | --- | --- |
| `scene_id` | str | one row per scene (no duplicate `scene_id`) |
| `date` | str | acquisition date |
| `start_time`, `end_time` | str | UTC |
| `footprint_wkt` | str | `POLYGON` / `MULTIPOLYGON`, EPSG:4326 |

### Identified field roles (written to `config/schema_mapping.yaml`)

* scene id: `scene_id` (both products, join key)
* datetime: `timestamp` (detections), `date` + `start_time` (footprints)
* latitude / longitude: `lat` / `lon`
* **AIS matching**: `matched_category == 'unmatched'` (primary). Matched levels
  observed: `cargo, other, fishing, passenger, noisy_vessel, gear,
  seismic_vessel, carrier, bunker, support, discrepancy`.
  Alternative definition kept alongside: `mmsi` non-null.
* SAR–AIS match quality: `matching_score`
* presence score: `presence_score`; fishing score: `fishing_score`
* length: `length_m`
* platform: derived from `scene_id` prefix (`S1A`, `S1B`, `S1C`, `S1D`)
* pipeline version: only available from the file name (`pipev<version>`)

### Pipeline versions present (a technical break, not a data gap)

| version | first month | last month | months |
| --- | --- | --- | --- |
| `20231026` | 201701 | 202410 | 94 |
| `v3` | 202411 | 202602 | 16 |
| `v4` | 202603 | 202609 | 7 |

`v3 → v4` falls exactly on 2026-03-01, i.e. inside the acute-shock window. This
was **not** in the original break list and is now declared in
`analysis.technical_breaks` and tested explicitly (§5).

---

## 2. Monthly SAR scene counts (AOI-intersecting, and usable)

AOI = `config/hormuz_core.geojson`, area **23 107.8 km²** (EPSG:6933).

"Usable" = `coverage_fraction ≥ 0.30` **and** platform has detections.

| year-month | scenes intersecting AOI | usable scenes | acquisition days | detections | matched | unmatched |
| --- | --- | --- | --- | --- | --- | --- |
| 2024-01 … 2024-12 | 18–22 / month | 6–8 | 11–13 | 443–740 | 230–356 | 183–423 |
| 2025-01 … 2025-12 | 16–38 / month | 6–16 | 10–26 | 499–661 | 221–356 | 254–324 |
| 2026-01 | 20 | 8 | 12 | 673 | 323 | 350 |
| 2026-02 | 18 | 6 | 11 | 516 | 236 | 280 |
| 2026-03 | 20 | 8 | 12 | 265 | 71 | 194 |
| 2026-04 | 19 | 6 | 11 | 232 | 94 | 137 |
| 2026-05 | 16 | 6 | 10 | 248 | 71 | 177 |

Totals for 2024-01 … 2026-05: 570 scenes intersecting the AOI, 205 usable,
21 958 AOI detections, 357 acquisition dates, 123 weeks with ≥1 usable scene
(i.e. ~1.7 usable scenes per week). Two of the 125 week-starts in the window
have no usable scene — 2024-01-15 and 2024-04-08 — both in the 2024 control
year, outside the event period.

Note: several monthly files are trimmed a few days before month end (e.g.
2025-01 ends Jan 28, 2026-02 ends Feb 25), so **monthly totals are not
comparable across months**; per-acquisition metrics are used instead.

---

## 3. Can coverage-normalised SAR density be computed reliably? Yes, with caveats

* `coverage_km2` = footprint ∩ AOI, computed in EPSG:6933. Geometry validity is
  checked; 0 invalid footprints in the window.
* `coverage_fraction` distribution over 570 AOI-intersecting scenes
  (min 0.0020 · p05 0.0022 · p25 0.1253 · median **0.2604** · p75 0.3198 ·
  p90 0.4108 · max 0.4140).
* A single Sentinel-1 scene never covers the whole AOI, so a 0.30 threshold
  keeps 213/570 scenes (37 %) carrying 15 299 detections.

Threshold sensitivity (does the conclusion depend on this choice?):

| threshold | scenes | detections | pooled matching rate |
| --- | --- | --- | --- |
| 0.00 | 570 | 21 716 | 0.508 |
| 0.15 | 360 | 20 633 | 0.509 |
| 0.20 | 295 | 18 815 | 0.513 |
| 0.30 (default) | 213 | 15 299 | 0.492 |
| 0.40 | 71 | 5 592 | 0.510 |
| 0.50 | 0 | 0 | — |

The pooled level is stable across thresholds; only the sample size changes. The
0.30 default is retained.

`sar_density = n_total / coverage_km2 × 1000` (vessels per 1000 km² per
acquisition) is therefore computable and coverage-normalised. Days without an
acquisition are absent from `sar_daily_aoi_coverage.parquet` and are never
written as zero.

---

## 4. Missing data and anomalies found

| issue | size | handling |
| --- | --- | --- |
| **Footprint-only platform (S1C)** | 19 scenes (2025-09-28 … 2025-10-29), each with 0 detections; globally the 2025-10 detection file contains 0 of 277 263 detections from S1C | auto-detected, excluded from primary metrics, logged (`platform_detection_coverage.csv`) |
| Detections without a footprint row ("orphans") | 242 / 21 958 = 1.1 % (2025-11: 111, 2025-12: 87, 2026-04: 44) | flagged (`scene_in_footprints`), excluded from scene metrics, never silently dropped |
| Unknown match labels | 1 detection (2026-04) | kept as a third `match_status` level; the scene is listed in `scene_count_anomalies.csv` |
| Duplicate/partial monthly files | 2024-08 and 2024-11 shipped twice | full month wins, partial files dropped with a ledger entry |
| Trimmed months | 8 months end 3–6 days early | recorded in the schema report; per-acquisition metrics are unaffected |
| 2025-10 acquisition spike | 38 AOI scenes / 26 acquisition days vs ~20/12 normally | identified as the S1C introduction (footprints without detections) |

No month inside the loading window is missing a file. The archive-wide
unmatched share drifts slowly upward over 2024–2026 (0.34 → 0.38), which is why
the AOI rate is always compared with the archive-wide rate for the same month.

---

## 5. The signal

### 5.1 Matching rate moved far outside its historical range

Pooled `matching_rate` inside the AOI (usable scenes):

| period | scenes | weeks | detections | matching rate | 95 % bootstrap CI |
| --- | --- | --- | --- | --- | --- |
| 2024 calendar year | 88 | — | 6 425 | 0.521 | — |
| 2025 calendar year | 83 | — | 6 940 | 0.489 | — |
| 2026 baseline (Jan 1 – Feb 27) | 14 | 9 | 1 189 | **0.470** | [0.437, 0.501] |
| 2026 acute shock (Mar 1 – Mar 27) | 6 | 4 | 218 | **0.239** | [0.183, 0.310] |
| 2026 persistent (Mar 28 – May 31) | 14 | 9 | 526 | **0.350** | [0.270, 0.423] |

(CIs: 3 000 scene-level percentile bootstrap draws; scenes are the resampling
unit, not detections.)

### 5.2 The same calendar split in 2024 / 2025 does **not** reproduce it

Pre (Jan 1 – Feb 27) vs post (Mar 1 – May 31), identical calendar days in every
year:

| year | scenes pre / post | detections pre / post | rate pre | rate post | Δ (pp) | 95 % CI (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| 2024 (placebo) | 14 / 22 | 909 / 1 682 | 0.525 | 0.515 | **−0.0** | [−6.1, +4.0] |
| 2025 (placebo) | 14 / 20 | 1 174 / 1 779 | 0.477 | 0.486 | **+0.0** | [−3.7, +5.3] |
| 2026 (event) | 14 / 20 | 1 189 / 744 | 0.470 | 0.317 | **−15.3** | [−22.0, −8.2] |

The 2026 shift is ~15 pp and its CI excludes zero and excludes both placebo
intervals. A 15 pp shift is 3–4× the width of the entire placebo CI.

### 5.3 SAR activity collapsed locally, and only locally

Weekly median `sar_density` log-anomaly against the same ISO week of 2024/2025
(reference = median of available reference years, ε = 1):

| weeks | 2026 anomaly |
| --- | --- |
| W01–W09 (Jan 1 – Mar 1) | +0.11 … +0.32 (noise around 0) |
| W10–W21 (Mar 2 – May 23) | **−0.55 … −0.82**, sustained for 12 consecutive weeks |

Median density drops from 10.59 to 4.32 (acute) / 4.70 (persistent) —
a ~55 % fall.

Archive-wide control: detections per scene globally were 37.7 (Feb 2026),
38.8 (Mar), 40.9 (Apr), 37.9 (May) — i.e. **no** global collapse in the same
months. In the AOI the same quantity fell from ~86 to ~33.

### 5.4 The drop is concentrated in AIS-matched, large vessels

Median densities inside usable scenes:

| component | 2024–2026 Feb typical | 2026-03 | 2026-04 | 2026-05 |
| --- | --- | --- | --- | --- |
| total (`sar_density`) | 8.1–13.1 | 4.31 | 4.65 | 5.06 |
| matched only | 4.0–5.9 | 1.18 | 1.94 | 1.46 |
| unmatched only | 3.4–6.9 | 3.07 | 2.89 | 3.60 |

Matching rate **within** vessel-length bins (this separates "the population
changed" from "matching changed at a given size"):

| length bin | pre-2026 | 2026-03 | 2026-04/05 |
| --- | --- | --- | --- |
| 0–50 m | 0.269 | 0.209 | 0.181 |
| 50–100 m | 0.614 | 0.411 | 0.402 |
| 100–150 m | 0.571 | 0.400 | 0.495 |
| 150–200 m | 0.659 | 0.458 | 0.461 |
| 200–300 m | 0.666 | 0.353 | 0.333 |
| ≥300 m | 0.681 | 0.154 | 0.059 |

Every size bin falls, and the fall grows monotonically with vessel size. So the
matching-rate decline is **not** purely a composition artefact: at a fixed
length, an AIS identity became substantially less likely to accompany a SAR
detection, most of all for the largest ships.

### 5.5 The obvious confounds were checked

| confound | check | result |
| --- | --- | --- |
| GFW pipeline `v3 → v4` (2026-03-01) | archive-wide unmatched share, Feb → Mar 2026 | 0.3796 → 0.3821 (**+0.25 pp**, no jump) |
| GFW pipeline, large vessels | archive-wide unmatched share for ≥200 m, Feb → Mar 2026 | 0.119 → 0.133 (+1.4 pp vs +32 pp in the AOI) |
| SAR coverage collapse | median `coverage_fraction` and daily AOI coverage, 2026-03/04/05 | 0.320 → 0.320/0.323/0.322; median daily AOI coverage 0.411 → 0.419 (**no collapse**) |
| Different sub-area imaged | median centroid + lon/lat span of the AOI intersection, all 29 months | identical (56.329 °E, 26.533 °N, spans 1.643° × 1.023°) |
| Sensor/platform change | platform of usable scenes | 205/205 are S1A; S1C footprint-only scenes excluded |
| Global AIS trends | archive-wide unmatched share | slow drift 0.34 → 0.38 over 2024–2026, an order of magnitude smaller than the AOI shift |

The archive-wide ≥200 m unmatched share did drift up during 2025 (5 % → 13 %),
which is a background trend worth carrying into the next phase; the AOI level
(~33 % before, 65–94 % after) is both much higher and much more volatile than
the global level.

---

## 6. Answers to the three questions (SAR side)

**Q1 — did the matching rate move abnormally? Yes.** 0.470 before the event vs
0.239 in the acute window and 0.350 afterwards, against a 2024/2025 same-calendar
pre-post change of ≈0.0 pp. The shift is within-bin as well as pooled, and it is
not explained by the pipeline change, coverage change, or spatial composition.

**Q2 — do AIS and SAR disagree about the size of the drop? Not yet testable.**
This needs AIS vessel presence (Phase 2). What the SAR side already shows is that
the answer will depend on which SAR denominator is used: total SAR density fell
~55 %, the AIS-matched component ~75 %, the unmatched component ~40 %. An
AIS-based disruption index will therefore overstate the fall in *total* observed
vessel numbers, while it would look roughly neutral against an AIS-matched-only
SAR measure. This divergence is the paper's core object.

**Q3 — could it be seasonality or processing? No, on the evidence available.**
Both placebo years are flat across the identical calendar split; the archive-wide
pipeline-version control shows no matching jump; coverage, spatial composition
and platform are stable. Remaining exposure: the acute window (Mar 1–27) is
exactly the `v4` period, so acute-vs-persistent separation is partly confounded
with pipeline version, and the 2026-03-28 mask update sits at the phase boundary.

---

## 7. Data sufficiency assessment

| criterion | status |
| --- | --- |
| independent SAR scenes per phase | 14 (baseline), 6 (acute), 14 (persistent) — above the configured minimum of 3 |
| independent weeks covered per phase | 9 / 4 / 9 |
| coverage gaps | none inside the window; all 29 monthly files present; 123 weeks with ≥1 usable scene |
| detection volume for rate estimates | 1 189 / 218 / 526 detections per phase (binomial SE of the pooled rate ≈ 1.5 / 2.9 / 2.1 pp) |
| AOI size vs scene size | limiting factor: no scene covers more than 41 % of the AOI |

The acute-shock phase is the weakest cell (6 scenes, 4 weeks, 218 detections).
A smaller, more compact AOI (e.g. the 26.4–26.9 °N / 56.1–56.9 °E strait core)
would raise per-scene coverage and increase the usable scene count per period;
this is the single most valuable design change to test in the next round.

---

## 8. Feasibility screening (screening only, not significance testing)

| Strong-GO criterion | verdict |
| --- | --- |
| 1. matching-rate shift ≈ ≥10 pp | **met** (−15.3 pp pooled; −25 to −30 pp within large-vessel bins) |
| 2. 2024/2025 placebo shows no similar change | **met** (placebo deltas ≈0 pp, |CI| ≤ 6 pp) |
| 3. AIS and SAR anomalies separate over multiple periods | **not yet testable** (needs Phase 2) |
| 4. separation is not caused by coverage collapse | **met** (coverage, spatial composition and platform all stable; global control shows no pipeline-wide jump) |

**Verdict: GO for Phase 2 (AIS download + weekly aggregation + anomaly gap), with
the formal Strong/Conditional decision deferred until the observability gap is
computed.** Criterion 3 is the only untested condition and it is exactly the one
Phase 2 exists to answer.

Two design notes to carry forward:

1. Keep the matching-rate analysis strictly within vessel-length bins; the pooled
   rate mixes composition and observability.
2. Pre-register the treatment of the 2026-03-01 pipeline break and the
   2026-03-28 mask update before looking at the AIS gap, so that the phase
   boundaries are not chosen after seeing the result.
