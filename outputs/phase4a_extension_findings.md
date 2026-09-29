# Phase 4A-Extension — feasibility of two further candidate events

Scope: data feasibility and identification checks only. No composition-standardized
model, no final event inference and no pooled cross-event model was fitted. The
frozen Hormuz results and the already-screened Red Sea 2023 / Suez 2021 cases were
not modified.

Deliverables:
[phase4a_extension_sar_feasibility.csv](tables/phase4a_extension_sar_feasibility.csv) ·
[phase4a_extension_matching_metrics.csv](tables/phase4a_extension_matching_metrics.csv) ·
[phase4a_extension_activity_diagnostics.csv](tables/phase4a_extension_activity_diagnostics.csv) ·
[phase4a_extension_month_pair.csv](tables/phase4a_extension_month_pair.csv)

Case roles (fixed, never relaxed):

| event | case role | disruption type |
| --- | --- | --- |
| Hormuz 2026 | discovery case (frozen) | geopolitical / security |
| **Bab el-Mandeb 2026** | **concurrent spillover case** — same regional crisis, same GFW transition as Hormuz. Not an independent validation, never a control group | geopolitical / security |
| Red Sea Dec 2023 | independent security validation (screened earlier) | geopolitical / security |
| **Panama 2023–24** | **non-security contrast** — never a negative control | climate / capacity |
| Suez 2021 | short-window physical-blockage supplementary case (screened earlier) | physical blockage |

---

## 1. Event and AOI definitions (fixed before any result for these areas)

**Bab el-Mandeb 2026** — AOI `bab_el_mandeb_core`, the *same* pre-specified box used
for the Red Sea 2023 screen (43.10–43.65 E / 12.30–12.95 N), untouched. Onset
`2026-03-01`, taken from the external decision by major carriers to pause/reroute
Trans-Suez sailings through Bab el-Mandeb; phases are the frozen Hormuz calendar
phases (baseline 01-01…02-27, acute 03-01…03-27, persistent 03-28…05-31), so the
event year, technical environment and weekly calendar are identical to Hormuz.
Placebo years 2024 and 2025 with the same calendar windows. Screening stops at
2026-05-31.

**Panama 2023–24** — new AOI `panama_canal_core` (bbox −79.95…−79.50 E /
8.85…9.35 N, `pre_specified: true`, `tuned_on_results: false`), covering the canal
system from the Pacific locks through Gaillard Cut and Gatun Lake toward the
Atlantic locks. Onset `2023-11-03` = **onset of the stronger pre-specified
transit-restriction regime used for this event study** (not the absolute beginning
of the drought crisis; earlier restrictions predate it). Phases: pre 09-01…11-02,
restricted 11-03…02-29, adaptation 03-01…04-30 (anchored to the following calendar
year). Placebo windows, same calendar days: 2021-09-01…2022-04-30 and
2022-09-01…2023-04-30.

---

## 2. Coverage handling and the water-observable domain (Panama)

For open-water chokepoints (Hormuz, Bab el-Mandeb) the existing logic is retained.
For Panama both denominators are reported:

| AOI | bbox area | water-observable domain | median scene coverage (bbox) | median scene coverage (water) |
| --- | --- | --- | --- | --- |
| `panama_canal_core` | 2 735.6 km² | **85.7 km²** | 0.031 | **1.000** |
| `bab_el_mandeb_core` | 4 296.8 km² | 2 668.6 km² | 0.554 | 0.892 |

**How the water-observable domain is created:** it is the union of all
`scene ∩ AOI` intersection polygons over the event's SAR window, computed in
EPSG:6933 from the **GFW footprint geometry only**. GFW footprints are
water-clipped polygons, so this union is a geometric, reproducible proxy for the
AOI's observable water. **No vessel detection is used to build the mask.**

Robustness (recomputed year by year, as required):

| AOI | per-year domain | max relative deviation from the full window |
| --- | --- | --- |
| Panama | 85.7 / 85.7 / 85.7 / 85.7 km² (2021–2024) | **0.0 %** |
| Bab el-Mandeb | 2 651.1 / 2 651.1 / 2 668.6 km² (2024/2025/2026) | 0.7 % |

The footprint geometry therefore provides a robust, window-independent water
domain. It is decisive for Panama: the bounding-box denominator is useless
(0.031, no scene would pass), while the water-normalized denominator is 1.000 —
every usable scene images the entire observable water domain.

---

## 3. SAR feasibility metrics (Part 5 / Part 11)

### Bab el-Mandeb 2026

| year | phase | intersect. scenes | usable scenes | bbox-usable | acq. days | acq. weeks | detections | matched | unmatched | rate | det/scene | cover bbox | cover water | length_m | matched_cat | platform | pipeline | AIS days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | baseline (pre) | 17 | 13 | 13 | 9 | 8 | 95 | 84 | 11 | 0.884 | 7.31 | 0.553 | 0.890 | 100 % | 100 % | S1A | **v3** | 58/58 |
| 2026 | acute (event) | 9 | 7 | 7 | 5 | 5 | **50** | 42 | 8 | 0.840 | 7.14 | 0.554 | 0.892 | 100 % | 100 % | S1A | **v4** | 27/27 |
| 2026 | persistent (post) | 19 | 14 | 14 | 9 | 7 | 103 | 79 | 24 | 0.767 | 7.36 | 0.381 | 0.613 | 100 % | 100 % | S1A | v4 | 65/65 |
| 2024 | acute (placebo) | 8 | 6 | 6 | 4 | 3 | 40 | 31 | 9 | 0.775 | 6.67 | 0.465 | 0.748 | 100 % | 100 % | S1A | 20231026 | 27/27 |
| 2025 | acute (placebo) | 10 | 7 | 7 | 5 | 5 | 33 | 24 | 9 | 0.727 | 4.71 | 0.376 | 0.605 | 100 % | 100 % | S1A | v3 | 27/27 |

### Panama 2023–24

| year | phase | intersect. scenes | usable scenes | acq. days | acq. weeks | detections | matched | unmatched | rate | det/scene | cover water | length_m | matched_cat | platform | pipeline | AIS days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023/24 | pre | 7 | 7 | 4 | 4 | 96 | 89 | 7 | 0.927 | 13.71 | 1.000 | 100 % | 100 % | S1A | 20231026 | 63/63 |
| 2023/24 | restricted (event) | 19 | **19** | **10** | **10** | **205** | 199 | 6 | 0.971 | 10.79 | 1.000 | 100 % | 100 % | S1A | 20231026 | 119/119 |
| 2023/24 | adaptation | 10 | 10 | 5 | 5 | 123 | 112 | 11 | 0.911 | 12.30 | 1.000 | 100 % | 100 % | S1A | 20231026 | 61/61 |
| 2021/22 | restricted (placebo) | 17 | 17 | 9 | 9 | 234 | 200 | 34 | 0.855 | 13.76 | 1.000 | 100 % | 100 % | S1A | 20231026 | 118/118 |
| 2022/23 | restricted (placebo) | 16 | 16 | 8 | 8 | 220 | 205 | 15 | 0.932 | 13.75 | 1.000 | 100 % | 100 % | S1A | 20231026 | 118/118 |

Both events have the same satellite platform (S1A) in every phase, complete
`length_m` and `matched_category`, and unambiguous usable-scene counts under both
coverage denominators.

---

## 4. Pipeline comparability (Part 6)

| event | pre | event | post | same pipeline pre/event | onset coincides with a transition |
| --- | --- | --- | --- | --- | --- |
| Bab el-Mandeb 2026 | v3 | **v4** | v4 | **no** | **yes** (2026-03-01 v3→v4 — identical to Hormuz 2026) |
| Panama 2023–24 | 20231026 | 20231026 | 20231026 | **yes** | no |

Panama's single-generation span was **verified from the detection-file tokens**,
not assumed. Bab el-Mandeb shares the Hormuz transition on the same date, which is
analytically useful rather than disqualifying: a Hormuz-vs-Bab comparison at the
same calendar weeks is *not differentially* confounded by the pipeline change,
because both corridors switch generation on the same day. The within-event effect
for each still has to be read placebo-adjusted, exactly as in the frozen Hormuz
design. Absolute matching-rate levels are never compared across generations.

---

## 5. AIS feasibility (Part 7)

The earlier quota block cleared. The API protocol held throughout (sequential
requests, cached ZIPs, audited contents, exclusive end date respected, missing
days never converted to zero). One new constraint was discovered and encoded:
the report endpoint rejects ranges longer than **366 days**, so the chunker now
clamps every request (the first 29-month attempt returned HTTP 422, which the
manifest recorded).

| event | requests | days observed_positive | structural_zero | missing_unknown |
| --- | --- | --- | --- | --- |
| Bab el-Mandeb 2024-01-01…2026-05-31 | 3 (12-month chunks) | **882 / 882** | 0 | **0** |
| Panama 2021-09-01…2024-04-30 | 3 (12-month chunks) | **973 / 973** | 0 | **0** |

AIS coverage is complete for every phase of both events (Bab acute 27/27, Panama
restricted 119/119), so no AIS-based claim is blocked.

---

## 6. Screening-level matching metrics (Part 8)

Scene-level bootstrap (3 000 draws, percentile); no standardization, no pooling.

| event | year | phase | scenes | detections | rate | 95 % CI | pre → event change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bab 2026 | **2026** | acute | 7 | 50 | 0.840 | [0.659, 0.968] | **−4.42 pp** [−23.99, +11.70] |
| Bab 2026 | 2024 | acute | 6 | 40 | 0.775 | [0.676, 0.838] | −1.72 pp [−13.06, +7.34] |
| Bab 2026 | 2025 | acute | 7 | 33 | 0.727 | [0.613, 0.826] | −20.13 pp [−32.27, −9.96] |
| Panama | **2023/24** | restricted | 19 | 205 | 0.971 | [0.949, 0.990] | **+4.37 pp** [+1.21, +7.27] |
| Panama | 2021/22 | restricted | 17 | 234 | 0.855 | [0.802, 0.905] | +4.22 pp [−1.81, +9.56] |
| Panama | 2022/23 | restricted | 16 | 220 | 0.932 | [0.906, 0.960] | +4.02 pp [−1.81, +10.92] |

**Bab el-Mandeb February → March 2026 month-pair change: −3.50 pp [−24.23, +15.91]**
(56 vs 50 classified detections), built identically to the frozen falsification
rows. Side by side, descriptively only — this is **not** a difference-in-differences
test:

| unit | Feb → Mar 2026 |
| --- | --- |
| **Hormuz `hormuz_core`** | **−18.94 pp** [−26.16, −9.37] |
| **Bab el-Mandeb (this screen)** | **−3.50 pp** [−24.23, +15.91] |
| global archive | −0.25 pp [−2.62, +2.27] |
| Gibraltar | +1.56 pp [−5.92, +11.53] |
| Singapore | −0.38 pp [−3.55, +3.52] |
| Dover | −2.40 pp [−14.33, +7.62] |

Length-bin screening rates (only bins with ≥ 30 detections are read; all others are
flagged `small_sample` in the table): Bab baseline `[150,200)` 0.938 (n=32) →
persistent 0.912 (n=34); Panama restricted `[100,150)` 1.000 (n=54) and
`[150,200)` 1.000 (n=82), with the small-vessel `[0,50)` bin at 0.875 (n=32)
against 0.436 (2021 placebo) and 0.698 (2022 placebo).

---

## 7. Preliminary activity diagnostics (Part 9)

Weekly `log((current + ε) / median(same ISO weeks in the placebo years + ε))`,
same weekly calendar as the frozen Hormuz analysis; SAR normalized by covered
area; AIS as a weekly sum (all days) and as mean per observed acquisition day
(SAR-days). Screening-level, not final estimates.

| event | phase | weeks | SAR anomaly | AIS all-days | AIS SAR-days | gap (all) | gap (SAR-days) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Bab 2026** | baseline | 16 | +0.106 | +0.085 | +0.124 | −0.021 | +0.018 |
| **Bab 2026** | acute | 8 | **+0.202** | +0.093 | +0.166 | **−0.109** | −0.037 |
| **Bab 2026** | persistent | 14 | **+0.211** | +0.054 | +0.109 | **−0.158** | −0.102 |
| **Panama** | pre | 8 | +0.047 | +0.134 | +0.137 | +0.087 | +0.090 |
| **Panama** | restricted | 20 | **−0.301** | **+0.008** | **+0.009** | **+0.313** | **+0.310** |
| **Panama** | adaptation | 10 | +0.007 | −0.053 | −0.105 | −0.072 | −0.111 |

For comparison, the frozen Hormuz event-year phase means were: acute SAR −0.710,
AIS −1.215/−1.216, gap −0.504/−0.505; persistent SAR −0.671, AIS −1.427/−1.340,
gap −0.642/−0.669.

Two observations, stated without assuming a direction:

* **Bab el-Mandeb shows no Hormuz-scale observability gap.** Under the same
  regional shock and the same pipeline transition, its SAR activity anomaly is
  *positive* (+0.20/+0.21 — more detected vessels than in the same weeks of
  2024/2025) and its gap is small (−0.11/−0.16) against Hormuz's −0.50/−0.64.
* **Panama shows the opposite sign.** SAR-observed density in the canal corridor
  fell ≈26 % during the restrictions while AIS presence stayed flat, giving a gap
  of **+0.31** — the mirror image of Hormuz.

**Caveat that must travel with the Panama number:** AIS presence hours are a
*duration* measure while SAR detections are a *per-pass count*. Waiting/anchoring
during restrictions inflates presence hours (a vessel waiting three days
contributes 72 hours) while reducing transits, so part of the +0.31 gap may be
metric asymmetry rather than a pure observability difference. Phase 4B should add
a count-based AIS metric (distinct vessels) and, if possible, a transit-flow
measure, and the gap should be reported for both.

---

## 8. Feasibility classification (Part 10)

Classification uses observational support and technical comparability only — never
the size or significance of any matching change.

| event | SAR | AIS | combined | reason |
| --- | --- | --- | --- | --- |
| **Bab el-Mandeb 2026** | CONDITIONAL | STRONG | CONDITIONAL | 50 event-phase detections < the pre-specified 60. All other criteria pass: 7 usable scenes, 5 acquisition days, 5 acquisition weeks, 0.892 water coverage, complete AIS (27/27), placebos available, pipeline transition exactly at onset handled as in Hormuz. |
| **Panama 2023–24** | **STRONG** | **STRONG** | **STRONG** | 19 usable scenes, 10 acquisition days, 10 acquisition weeks, 205 detections, water coverage 1.000, both placebo years usable, single pipeline generation. |

No classification was influenced by whether a matching decline appeared: Bab has
essentially none and is still usable; Panama has none and is STRONG.

---

## 9. Answers to the required questions

### Bab el-Mandeb 2026

1. **Enough SAR Jan–May 2026?** Yes. 13 / 7 / 14 usable scenes in baseline / acute /
   persistent, 9 / 5 / 9 acquisition days and 8 / 5 / 7 acquisition weeks, with
   water-normalized coverage 0.890 / 0.892 / 0.613.
2. **Event-window sampling comparable with Hormuz?** Yes by construction — identical
   calendar phases, same onset date and same pipeline boundary. The scene counts are
   comparable (7 vs Hormuz's 6 in the acute phase); the detection count is ~4× lower
   (50 vs 218), because the Bab AOI is smaller and carries less traffic.
3. **Regional March-1 discontinuity or a Hormuz-specific pattern?** No regional
   discontinuity. Bab's February → March change is −3.50 pp with a CI spanning zero
   (−24.23, +15.91) against Hormuz's −18.94 pp, and Bab's acute rate (0.840) sits
   *above* both of its placebo acute rates (0.775, 0.727). The Hormuz matching
   collapse is therefore not shared by the other side of the same crisis.
4. **Are the 2024/2025 placebos stable?** Weaker than for Hormuz. Levels vary
   (0.727–0.929) and the 2025 placebo itself shows a −20.13 pp pre→acute change whose
   CI excludes zero, so Bab has larger natural variability. The 2026 change is
   *smaller* than that placebo change, so there is no evidence of an event effect —
   but the noise floor must be stated.
5. **AIS complete enough for an observability-gap analysis?** Yes. 882/882 days
   observed, 0 missing, 27/27 days in the acute phase.
6. **Should it enter Phase 4B as a concurrent spillover case?** Yes, with two
   conditions: it is a spillover case, never a validation or control; and its
   acute-phase detection count (50) plus larger placebo variability must be carried
   as precision limits. Its distinctive value is the shared calendar and shared
   pipeline boundary with Hormuz.

### Panama Canal 2023–24

1. **SAR frequency during the restriction period?** Yes: 10 acquisition days over
   10 distinct weeks across a 119-day restriction phase, 19 usable scenes and 205
   detections (~1 acquisition every 12 days).
2. **Does water normalization solve the land-heavy AOI problem?** Yes, decisively.
   Bounding-box coverage is 0.031 and would reject every scene; the
   footprint-derived water-observable domain (85.7 km²) gives 1.000 for every
   usable scene, and the domain is identical in all four years (0.0 % deviation).
3. **Is the whole restriction interval within one GFW pipeline generation?** Yes —
   `20231026` in pre, restricted and adaptation, verified from the detection-file
   tokens.
4. **Are the placebo windows adequate?** Yes: 17 usable scenes / 234 detections
   (2021/22) and 16 / 220 (2022/23) in the same calendar windows. No placebo year
   needed to be replaced.
5. **Is AIS coverage complete?** Yes: 973/973 days observed, 0 missing, 119/119 in
   the restricted phase.
6. **Suitable as the principal non-security contrast case?** Yes. It is STRONG on
   SAR, AIS and combined support, and it already shows the scientifically
   interesting configuration the brief anticipated: a substantial physical
   activity reduction with no AIS-side decline.

---

## 10. Recommended case matrix (Part 13)

Selected on observational support, technical comparability and interpretability
only — not on effect size.

* Bab el-Mandeb 2026: usable **CONDITIONAL** case (SAR CONDITIONAL, AIS STRONG).
* Panama 2023–24: **STRONG** on SAR, AIS and combined.

**Recommendation: Matrix A — Hormuz + Bab el-Mandeb 2026 + Red Sea 2023 + Panama
2023–24**, with Suez 2021 retained as the supplementary short-window
physical-blockage case (it remains AIS-blocked by quota and has only one
acquisition week inside its 7-day event).

What Matrix A would test, in the brief's own terms: whether *physical maritime
disruption* and *AIS observability disruption* are empirically distinct
dimensions. The screening already shows the two dimensions separating in both
directions — Hormuz (both dimensions move), Bab el-Mandeb (physical activity moves
up, observability does not move), Panama (physical activity moves down,
observability does not move) — which is exactly the variation a multi-event design
needs, and it is visible before any standardized model is fitted.
