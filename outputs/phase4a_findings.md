# Phase 4A — multi-event feasibility screening

Scope: data feasibility and identification checks only. **No event effect is
estimated here**, and the frozen Hormuz specification was not touched (primary
`hormuz_core`, narrow AOI as a pre-specified sensitivity, event date 2026-02-28,
`matched_category != "unmatched"` as the primary matching definition, Phase 3
standardization model).

Deliverable table: [multievent_feasibility.csv](multievent_feasibility.csv)
(18 rows = 2 events × 3 year roles × 3 periods).

---

## 0. Correction of the Phase 3 reporting inconsistency

The two values that appeared to conflict are **different statistics, both
current**; the report mislabelled one of them.

| statistic | where it lives | window | value |
| --- | --- | --- | --- |
| pre/post **placebo split** | `placebo_results.csv`, `aoi_sensitivity.csv` (`row_type = pre_post_split`) | Jan 1 – Feb 27 vs Mar 1 – May 31, 2024 | **−4.12 pp** [−13.51, +5.67], 15 vs 23 scenes, 204+327 detections |
| **month-pair transition** | `pipeline_falsification.csv` (`comparison = placebo_transition`) | **February → March 2024 only** | **−15.03 pp** [−27.83, +4.11], 0.84 → 0.69, 111 vs 112 classified detections |

The −15 pp is therefore *not* stale and *not* wrong — it answers a different
question (month-to-month movement at the pipeline-break calendar, used by the
falsification test) and its window is two single months rather than the
three-month pre/post split. `phase3_findings.md` quoted it as "the 2024 placebo",
which implied the pre/post design; the sentence now names both statistics, their
tables and their windows so every number traces to one final table.

---

## 1. Event definitions (fixed before looking at any matching rate)

| | Suez 2021 | Red Sea 2023 |
| --- | --- | --- |
| event type | canal blockage | regional shipping disruption |
| event date | **2021-03-23** | **2023-12-15** |
| event window | 2021-03-23 … 2021-03-29 | 2023-12-15 … 2024-02-15 |
| documented chronology | Ever Given grounded 2021-03-23 ≈07:40 local in the southern single-lane section; refloated and canal reopened 2021-03-29; convoy backlog cleared in the following days | Galaxy Leader hijacked 2023-11-19; attacks on commercial shipping escalated Nov–Dec 2023; major carriers suspended Red Sea transits ≈2023-12-15; Operation Prosperity Guardian announced 2023-12-18; sustained Cape of Good Hope rerouting thereafter |
| AOI | `suez_canal_core`, bbox 32.30–32.65 E / 29.90–30.35 N — southern single-lane section, Suez entrance and the grounding site | `bab_el_mandeb_core`, bbox 43.10–43.65 E / 12.30–12.95 N — the strait plus the southern Red Sea approach |
| navigation rationale | single-lane section with no passing possibility | the one chokepoint every Red Sea transit must pass; rerouted traffic never crosses it |
| placebo years | 2019, 2020 | 2021, 2022 |

Both AOIs are marked `pre_specified: true`, `tuned_on_results: false`, and are
unit-tested to lie far from the Hormuz region.

---

## 2. Coverage denominators (a finding in itself)

GFW footprints are water-clipped polygons: their area tracks how much *water* a
swath contains, not the swath itself. Median footprint area across AOIs (same
product, same period):

| AOI | median footprint area | median scene ∩ AOI | box area |
| --- | --- | --- | --- |
| Gibraltar (open sea) | 27 958 km² | — | 1 501 km² |
| Hormuz primary | 24 421 km² | 7 400 km² | 23 108 km² |
| Bab el-Mandeb | 8 391 km² | 2 651 km² (union) | 4 297 km² |
| **Suez** | **3 501 km²** | **93.9 km²** | 1 682 km² |

The Suez AOI bounding box is ~94 % desert, so the Phase 1 box-relative coverage
fraction collapses to a median of **0.043–0.048** and **no** scene passes the 0.30
threshold — even though the canal water itself is imaged almost completely on
every pass.

| AOI | box area | imageable-water proxy | median scene intersection | scenes covering ≥ 30 % of the water |
| --- | --- | --- | --- | --- |
| `suez_canal_core` | 1 682.5 km² | **96.6 km²** | 93.9 km² | 552 of 622 (89 %) |
| `bab_el_mandeb_core` | 4 296.8 km² | 2 651.1 km² | — | 206 of 285 (72 %) by the box metric |

The screen therefore reports both `coverage_fraction` (box) and
`coverage_fraction_water` (intersection ÷ union of all scene intersections). The
water proxy is purely geometric and result-independent, and is applied identically
to both events. Hormuz is unaffected: it is not re-normalised here, and the frozen
Hormuz outputs are byte-identical to Phase 3.

---

## 3. SAR feasibility per event

### Suez 2021 — CONDITIONAL (SAR-only CONDITIONAL)

| period | window | usable scenes (water) | usable scenes (box) | acquisition days | weeks | detections | matched / unmatched | median water coverage | pipeline | platform |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre (8 w) | 2021-01-26 … 03-22 | 33 | 0 | 33 | 8 | 117 | 96 / 21 | 0.831 | 20231026 | S1A, S1B |
| **event** | **2021-03-23 … 03-29** | **4** | 0 | **4** | 1 | **57** | 49 / 8 | 0.756 | 20231026 | S1A, S1B |
| post (8 w) | 2021-03-30 … 05-24 | 36 | 0 | 36 | 9 | 159 | 140 / 19 | 0.833 | 20231026 | S1A, S1B |

Placebos in the same calendar window: 2019 → 5 scenes / **12** detections;
2020 → 5 scenes / **26** detections. The event week itself carries **57**
detections, i.e. 2.2–4.8× the placebo level, and the per-scene counts are
12/12/15/18 versus 0–6 (2019) and 3–8 (2020) — the signature of vessels queuing
inside the AOI (the box contains the Bitter Lakes waiting area).

Why only CONDITIONAL: the pre-specified `min_event_detections = 60` is missed by
three detections, and — decisive for identification — a 7-day event contains only
**one acquisition week**, so a within-event time profile is impossible. The
blockage also has no persistent phase: the canal reopened on day 7.

### Red Sea 2023 — SAR-only STRONG, combined CONDITIONAL

| period | window | usable scenes (water) | acquisition days | weeks | detections | matched / unmatched | median water coverage | pipeline | platform |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre (8 w) | 2023-10-20 … 12-14 | 11 | 7 | 6 | 99 | 86 / 13 | 0.611 | 20231026 | S1A |
| **event** | **2023-12-15 … 2024-02-15** | **14** | **10** | **8** | **90** | 72 / 18 | 0.898 | 20231026 | S1A |
| post (8 w) | 2024-02-16 … 04-11 | 14 | 10 | 9 | 84 | 71 / 13 | 0.897 | 20231026 | S1A |

Placebos in the same calendar window: 2021 → 12 scenes / 73 detections; 2022 → 13
scenes / 120 detections. Detection counts are of the same order as the event
period, so the *level* comparison is not the discriminating quantity — that is
expected, since the Red Sea disruption is about rerouting rather than about
vessels disappearing from the strait.

Every SAR-side criterion passes: ≥ 4 usable scenes (14), ≥ 3 acquisition days
(10), ≥ 60 detections (90), median water coverage 0.898 ≥ 0.30, two placebo years
available, and the same GFW pipeline in the pre and post periods.

---

## 4. Pipeline comparability (item 4 and item 8 of the brief)

| event | pipeline in pre | pipeline in event | pipeline in post | same pipeline pre/post |
| --- | --- | --- | --- | --- |
| Suez 2021 | 20231026 | 20231026 | 20231026 | **yes** |
| Red Sea 2023 | 20231026 | 20231026 | 20231026 | **yes** |
| Hormuz 2026 (frozen) | v3 | **v3 → v4 at 2026-03-01** | v4 | no |

Both candidate events sit entirely inside a single GFW pipeline generation, so the
collinearity that constrains Hormuz (event and pipeline switch one day apart) does
**not** arise. Conversely, this is exactly why item 7 of the brief matters:
Suez/Red Sea use `pipev20231026` while Hormuz uses `pipev3`/`pipev4`, so
**absolute matching-rate levels must never be compared across the three events**;
only within-event effects are comparable. The screen records the pipeline per
period for that reason, and the eventual model must be fitted per event (item 9:
never pool raw detections from different pipeline generations).

Data-field availability (item 4): `length_m` and `matched_category` are populated
for **100 %** of detections in every period of both events.

---

## 5. AIS presence availability (item 6) — partially blocked

The AIS side is the one incomplete item. The GFW report API accepted 7 reports for
the Suez AOI (2019-01 … 2020-02, all days observed) and then returned
**HTTP 429 "not currently enabled to perform more than one concurrent report"**
for every subsequent request, including a single merged 29-month request retried
four times with 90 s backoff. The manifest records each attempt with its status
and error text.

Consequence for the screen, by design rather than by accident: every day not
covered by a completed and parsed report is classified `missing_unknown`, never
`structural_zero`. The Suez AOI currently has **211 observed days** and 671
`missing_unknown` days inside 2019-01-01 … 2021-05-31; the *event window itself*
has `ais_valid_days = 0`, so no AIS-based statement is made about it.

The classification therefore reports two verdicts: `feasibility_status_sar_only`
and the combined `feasibility_status`. Red Sea 2023 is **SAR-only STRONG** and
combined CONDITIONAL *solely* because the AIS fetch is quota-blocked; Suez 2021 is
CONDITIONAL on both counts (event detections 57 < 60; and one acquisition week).

To complete the AIS side once the token's quota has reset:

```powershell
python src/02_fetch_ais.py --aois suez_canal_core  --start 2019-01-01 --end 2021-05-31 --chunk-months 29
python src/02_fetch_ais.py --aois bab_el_mandeb_core --start 2021-10-01 --end 2024-04-30 --chunk-months 32
python src/10_multievent_screening.py
```

The screen re-reads the AIS status table on every run, so the combined verdict
upgrades automatically once the reports land.

---

## 6. Frozen-Hormuz boundary convention (documented, deliberately not changed)

The frozen Hormuz phase windows are inclusive of the start date and stop at
**midnight** of the end date, so a detection late on a phase's final day falls
outside that phase. Exactly **one** usable scene per Hormuz AOI sits on such a
boundary — `S1A_IW_GRDH_1SDV_20240227T021527…`, 59 detections (primary) and 12
(narrow) in the 2024 placebo year, none in 2026. It affects only reference-year
phase estimates, never the 2026 event comparison. Because the specification is
frozen, this was documented rather than silently corrected; the new Phase 4A code
uses whole-day inclusive windows instead (unit-tested).

---

## 7. Screen verdicts

| event | SAR-side | combined | reason |
| --- | --- | --- | --- |
| **Suez 2021** | CONDITIONAL | CONDITIONAL | 57 event detections < the pre-specified 60; a 7-day event spans a single acquisition week, so no within-event profile is possible. Usable SAR exists (4 scenes, 4 acquisition days, 0.76 water coverage) and the pipeline is constant, so the event is **studiable with a broader window** (e.g. including the ±3-week convoy-backlog period) or as a short-window case study. |
| **Red Sea 2023** | **STRONG** | CONDITIONAL | All SAR and pipeline criteria pass (14 usable scenes, 10 acquisition days, 8 weeks, 90 detections, 0.90 coverage, two placebo years, single pipeline). Blocked only by the AIS report quota; expected to become STRONG once the AIS fetch completes. |

Interpretation guardrails for Phase 4B (not yet implemented):

1. the multi-event comparison must use **within-event** effects only (pre/post
   change, placebo-adjusted change, composition-standardized change, AIS and SAR
   activity anomalies, observability gap);
2. absolute matching-rate levels are not comparable across the three events
   because the GFW pipelines differ;
3. detections from different pipeline generations must never be pooled in one
   logistic model — fit per event, then compare the standardized effects;
4. the Suez event's short window means its estimate will be imprecise, and the
   Red Sea's "no decline in strait traffic" expectation means the discriminating
   quantity there is the rerouting/observability gap, not the activity level.
