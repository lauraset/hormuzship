# Phase 4B findings — publication-scale multi-event analysis

Framework: **Dimension A** = physical maritime activity disruption
(historical-normalized SAR activity anomaly). **Dimension B** = AIS observability
disruption (composition-standardized change in the probability that a SAR
detection carries a confirmed AIS match). The AIS activity metrics are a third,
supporting family. The two dimensions are never merged, no composite index is
built, events are not ranked, and absolute matching-rate levels are never compared
across GFW pipeline generations.

Frozen and untouched: the Hormuz AOIs, event date, phase definitions, matching
definition, composition-standardization specification and the Phase 3 estimates.
The frozen Hormuz row below is carried through from the Phase 3 tables.

Deliverables:
[cross_event_final.csv](tables/cross_event_final.csv) ·
[event_role_summary.csv](tables/event_role_summary.csv) ·
[redsea_standardized_effects.csv](tables/redsea_2023_standardized_effects.csv) ·
[panama_standardized_effects.csv](tables/panama_2023_standardized_effects.csv) ·
[panama_ais_metric_decomposition.csv](tables/panama_ais_metric_decomposition.csv) ·
[bab2026_standardized_effects.csv](tables/bab_mandeb_2026_standardized_effects.csv) ·
[panama_transit_core_sensitivity.csv](tables/panama_transit_core_sensitivity.csv) ·
figures `fig_phase4b_01…11`.

---

## 1. The cross-event result

| event | case role | SAR activity anomaly (A) | AIS presence-hours anomaly | AIS distinct-vessels anomaly | dwell change | raw matching change | **standardized matching change (B)** | 95 % CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Hormuz 2026** (frozen) | discovery | −0.707 | −1.461 | −1.358 | +20.5 % | −15.29 pp | **−11.54 pp** | [−17.58, −5.37] |
| **Red Sea 2023–24** | independent security validation | −0.204 | +0.052 | −0.552 | +81.1 % | −6.87 pp | **−10.34 pp** | [−21.67, −3.09] |
| **Panama 2023–24** | non-security contrast | −0.301 | −0.046 | −0.219 | −1.8 % | +4.37 pp | **+4.77 pp** | [+1.94, +7.59] |
| **Bab el-Mandeb 2026** | concurrent regional falsification | +0.171 | −0.249 | +0.097 | −4.1 % | −4.42 pp | **−4.23 pp** | [−19.83, +6.17] |
| **Suez 2021** | supplementary blockage | +2.134 | +1.377 | +0.580 | +4.9 % | +0.42 pp | not fitted (descriptive) | — |
| Panama transit-core (sensitivity AOI) | non-security, nested | −1.487 | +0.588 | +0.190 | +8.9 % | −7.05 pp | not fitted (insufficient support) | — |

Placebo benchmarks (same calendar split in the reference years):

| event | standardized placebo change | raw per-year placebo changes |
| --- | --- | --- |
| Red Sea 2023–24 | +5.39 pp [−2.01, +10.42] | +5.51, +7.89 pp |
| Panama 2023–24 | −1.64 pp [−5.55, +2.89] | +4.02, +4.22 pp |
| Bab el-Mandeb 2026 | +2.05 pp [−7.97, +6.36] | −20.13, −1.72 pp |

**The headline contrast is unambiguous.** Hormuz and the Red Sea — two security
disruptions in different years, on different GFW pipeline generations, with
different AOI sizes — both show a composition-standardized matching decline of
about −10 to −12 pp, while the same-calendar placebo years move the other way.
Panama — a climate/capacity disruption of comparable physical magnitude — shows a
physical decline with the matching probability *rising* slightly. Bab el-Mandeb,
under the same regional shock and the very same pipeline boundary as Hormuz, shows
no such collapse.

---

## 2. Red Sea 2023–24 (highest-priority validation)

AOI `bab_el_mandeb_core` (unchanged), onset 2023-12-15, event window
2023-12-15…2024-02-15, placebos 2021 and 2022, all phases on one pipeline
generation (`20231026`, verified from the detection file tokens).

| quantity | pre | event | post |
| --- | --- | --- | --- |
| usable SAR scenes / detections | 11 / 99 | 14 / 90 | 14 / 84 |
| weekly SAR density median | 3.76 | 2.86 | 2.48 |
| SAR activity anomaly | −0.070 | **−0.204** | −0.338 |
| AIS presence hours per day | 256.0 | 232.4 | 237.9 |
| AIS distinct vessels per day | 83.6 | **46.5** | 37.7 |
| presence hours per vessel | 3.33 | **6.04** | 7.08 |
| pooled matching rate | 0.869 | 0.800 | — |
| **standardized matching change** | — | **−10.34 pp** [−21.67, −3.09] | — |

Answer to the Red Sea question: **an independent security-driven disruption does
reproduce the Hormuz observability-decline direction**, and the magnitude is
substantial after standardization: −10.3 pp against a placebo change of +5.4 pp,
i.e. a placebo-adjusted change of roughly −15.7 pp — comparable to the frozen
Hormuz persistent-phase estimate of −11.5 pp. Note that the *raw* change (−6.9 pp)
understates it: composition moved in a direction that masked part of the decline,
which is exactly why the standardized estimate is the publication-level quantity.

The Red Sea case also carries a strong AIS-side signature that is *not* an
observability story: distinct vessels fell 44 % while presence hours were roughly
flat, so presence hours per vessel rose 81 %. A duration-based AIS activity metric
would therefore have missed most of the physical change.

---

## 3. Panama 2023–24 (non-security contrast) and the metric decomposition

AOI `panama_canal_core` (unchanged), regime onset 2023-11-03, phases
pre 09-01…11-02, restricted 11-03…02-29, adaptation 03-01…04-30, placebos
2021/2022, single pipeline generation.

| quantity | pre | restricted | adaptation |
| --- | --- | --- | --- |
| usable scenes / detections | 7 / 96 | 19 / 205 | 10 / 123 |
| SAR density median | 161.9 | **128.4** | 143.6 |
| SAR activity anomaly | +0.047 | **−0.301** | +0.007 |
| AIS presence hours per day | 793.7 | 723.6 | 605.5 |
| AIS distinct vessels per day | 78.3 | 72.1 | 67.9 |
| presence hours per vessel | 24.26 | 23.82 | 18.69 |
| pooled matching rate | 0.927 | 0.971 | 0.911 |
| **standardized matching change** | — | **+4.77 pp** [+1.94, +7.59] | — |

**Metric decomposition (the Part 4 question).** On the same water domain, from the
same phases:

| metric | pre | restricted | placebo years (pre → restricted) |
| --- | --- | --- | --- |
| presence hours / day | 793.7 | 723.6 (−9 %) | 727.0 → 750.6 (+3 %), 617.6 → 737.2 (+19 %) |
| distinct vessels / day | 78.3 | 72.1 (−8 %) | 82.8 → 82.2 (−1 %), 70.2 → 87.7 (+25 %) |
| distinct vessels on SAR days | 270.0 | 256.2 (−5 %) | 241.3 → 255.4 (+6 %), 263.3 → 287.0 (+9 %) |
| presence hours per vessel | 24.26 | 23.82 (−2 %) | 20.8 → 21.3, 20.3 → 19.0 |

So the restriction period shows **a real but modest physical decline in the AIS
record** (−8 % vessels, −9 % hours) against placebo seasons that *rose* by 3–25 %,
i.e. a placebo-adjusted decline of roughly 10–15 %. Dwell per vessel was flat
(−2 %). The presence-hours metric is *not* explained by dwell inflation here; the
reason presence hours and vessel counts diverge from the SAR signal is that hours
are a duration measure while SAR detects vessels per pass. On the same water
domain the vessel-count anomaly (−0.22) is far closer to the SAR anomaly (−0.30)
than the hours anomaly (−0.05) is.

**Answer to the Panama question: yes — a substantial physical/capacity disruption
occurred while AIS matching stayed stable (indeed rose slightly).** A stable or
improved matching probability is a valid and informative outcome, and it is what
the data show.

### 3.1 Spatial-domain finding that forced a correction (important)

GFW SAR footprints are water-clipped, and in the Panama box the observable water is
only **85.7 km²** (3.1 % of the 2,735.6 km² bounding box), concentrated at
**8.85–9.00 °N** — the Pacific locks, the southern Cut and the Pacific approach.
Between 9.00 °N and 9.30 °N (the canal's freshwater lake section) the footprints
contain **no** observable water, while that section carries **20 % of the AIS
presence hours** in the box.

Consequences, all of which are now handled explicitly:

* the Phase 4A "SAR fell, AIS stayed flat" contrast was partly a **spatial-domain
  artefact**: the AIS series included hours from water Sentinel-1 never images;
* Phase 4B therefore recomputes every AIS series **inside the same water domain**
  the footprints define (98.2 % of hours retained for Bab el-Mandeb, 15.5 % for
  Panama, 24.0 % for Suez — the shares are a property of geography, not of the
  event);
* the Panama SAR dimension is best described as measuring the **Pacific-side
  approach and anchorage water**, not the lock-and-lake transit path. Any statement
  about "the Panama Canal" must say which water it refers to.

### 3.2 Nested transit-core sensitivity AOI

`panama_transit_core` (bbox −79.72…−79.52 E / 8.88…9.10 N, `pre_specified: true`,
`tuned_on_results: false`) was added to capture the active transit corridor while
excluding waiting areas. Its SAR-observable water is **18.8 km²**, again at
8.85–9.00 °N, and its model frame contains only 126 detections (7 pre / 13
restricted / 12 adaptation) in 2 spatial cells. The standardized model is
therefore **not supported** and the AOI is reported descriptively: SAR density
anomaly −1.487, AIS presence-hours anomaly +0.588, AIS distinct-vessels anomaly
+0.190, raw matching change −7.05 pp.

Two honest readings: (i) the nested AOI confirms the *direction* of the physical
decline with a much larger amplitude, but on so few detections that it cannot carry
an inference; (ii) because SAR cannot see the canal's freshwater section, the
"transit core" as defined geographically cannot be isolated with this data source.
Neither AOI was selected on the basis of effect size, and the primary Panama AOI
remains the headline specification.

---

## 4. Bab el-Mandeb 2026 (concurrent regional falsification)

Same AOI as the Red Sea case, same calendar phases as Hormuz, same pipeline
transition (2026-03-01 v3→v4), placebos 2024/2025. Deliberately not over-modelled.

| phase | SAR anomaly | AIS hours anomaly | AIS vessels anomaly | pooled matching rate | standardized change |
| --- | --- | --- | --- | --- | --- |
| baseline (Jan 1 – Feb 27) | +0.149 | −0.21 | +0.11 | 0.884 | — |
| acute (Mar 1 – Mar 27) | +0.171 | −0.25 | +0.10 | 0.840 | −4.23 pp [−19.83, +6.17] |
| persistent (Mar 28 – May 31) | +0.211 | −0.12 | +0.09 | 0.767 | — |

February → March 2026 matching transition: **−3.50 pp [−24.23, +15.91]**
(descriptive comparison only, not a difference-in-differences estimate):

| unit | Feb → Mar 2026 |
| --- | --- |
| Hormuz | −18.94 pp [−26.16, −9.37] |
| **Bab el-Mandeb** | **−3.50 pp [−24.23, +15.91]** |
| global archive | −0.25 pp [−2.62, +2.27] |
| Gibraltar / Singapore / Dover | +1.56 / −0.38 / −2.40 pp |

**Bab el-Mandeb did not experience a Hormuz-like observability collapse**, and its
SAR activity was if anything *above* its placebo weeks. Its 2025 placebo is
noisy (raw pre→acute −20.1 pp), so the honest statement is "no evidence of a
Hormuz-scale effect at Bab", not "a precisely estimated null".

---

## 5. Suez 2021 (supplementary, descriptive)

Seven-day blockage window with pre and backlog/recovery phases, 2019/2020
placebos, no composition model (as instructed), inference strength
**descriptive**.

| quantity | pre | blockage | backlog |
| --- | --- | --- | --- |
| SAR scenes / detections | 33 / 117 | **4 / 57** | 36 / 159 |
| SAR density median | 39.8 | **182.3** | 40.1 |
| SAR activity anomaly | +0.228 | **+2.134** | −0.177 |
| AIS presence hours per day | 328.8 | **726.3** | 468.2 |
| AIS distinct vessels per day | 47.3 | 40.7 | 56.5 |
| pooled matching rate | 0.806 | 0.811 | — |

The sign is the opposite of a "traffic vanished" story: the AOI fills with waiting
vessels, so SAR density and presence hours both spike. **Matching was flat
(+0.42 pp) during a severe short physical blockage** — the Suez question is
answered descriptively as: yes, a severe short blockage can occur without a
comparable decline in AIS observability. With one acquisition week inside the
event window, no stronger claim is made.

---

## 6. What replaced the single-gap framework

The old `observability_gap = AIS anomaly − SAR anomaly` is retained only as a
descriptive quantity (it is still in
`phase4b_extension_activity_diagnostics.csv`). The publication framework is now:

* **Dimension A — physical activity disruption**: `SAR_activity_anomaly`,
  historical-normalized weekly SAR density, normalized by covered water area.
* **Dimension B — AIS observability disruption**:
  `standardized_matching_change_pp`, the composition-standardized change in the
  confirmed AIS–SAR matching probability with scene-clustered intervals.
* **Supporting AIS family** (never combined, never a composite index): presence
  hours, distinct vessels, distinct vessels on SAR days, and presence hours per
  vessel.

The reason the gap had to be demoted is visible in Panama and the Red Sea: the gap
moves for reasons that have nothing to do with observability (duration-versus-count
metric definitions, dwell, queueing), so it cannot serve as the publication-level
observability measure.

---

## 7. Statements evaluated

| # | statement | verdict |
| --- | --- | --- |
| 1 | Physical maritime disruption and AIS observability disruption are distinct dimensions | **supported** — Panama and Suez show large physical changes with stable matching; Hormuz and the Red Sea show both moving; Bab shows neither |
| 2 | A large decline in vessel activity does not necessarily imply a decline in AIS observability | **supported** — Panama: SAR density −26 % with matching +4.8 pp; Suez: density +2.13 (congestion) with matching +0.4 pp |
| 3 | AIS observability can deteriorate substantially during some maritime security disruptions | **supported but not universal** — Hormuz (−11.5 pp frozen, −18.4 acute) and Red Sea (−10.3 pp, placebo-adjusted ≈ −15.7) decline; Bab el-Mandeb does not (−4.2 pp, CI spanning zero) |
| 4 | AIS-based activity indicators can become biased for different reasons: loss of observability or changes in vessel dwell/traffic behaviour | **supported** — the Red Sea shows a −44 % vessel count with flat hours (+81 % dwell); Suez shows +121 % hours during queueing; Panama shows hours and counts diverging by a factor of ~4 in anomaly terms on the same water |
| 5 | AIS reliability is disruption-context dependent rather than constant | **supported with wording discipline** — what the data establish is that the *measured* matching probability and the *behaviour* of AIS activity metrics both depend on the disruption context; the mechanism (non-transmission vs reception interference vs position mismatch) is not identified |

---

## 8. Answers to the eleven required questions

1. **Does the Red Sea independently reproduce the Hormuz direction?** Yes:
   −10.34 pp [−21.67, −3.09] standardized, against a placebo change of +5.39 pp.
2. **Is the magnitude substantial after standardization?** Yes — comparable to the
   frozen Hormuz persistent estimate (−11.54 pp), and *larger* than its own raw
   change (−6.87 pp), because composition moved against the effect.
3. **Does Panama show real physical disruption with stable matching?** Yes:
   SAR density anomaly −0.301 (≈ −26 %) with standardized matching **+4.77 pp**
   [+1.94, +7.59], and AIS vessel counts −8 % against placebo seasons that rose.
4. **Does the distinct-vessel analysis explain the flat presence hours?** Partly.
   On the same water domain the vessel-count signal (−0.22 anomaly) tracks SAR
   (−0.30) closely, while presence hours (−0.05) do not; the dwell proxy is
   essentially flat (−2 %). The divergence is therefore mainly a
   duration-versus-count metric effect, not dwell inflation — and it disappears
   when the count metric is used.
5. **Does Bab-2026 show a Hormuz-like collapse?** No: −4.23 pp with a CI spanning
   zero, SAR activity above placebo weeks, and a February→March transition of
   −3.50 pp [−24.23, +15.91] against Hormuz's −18.94 pp.
6. **Does the same pipev3→v4 transition explain both regions similarly or
   differently?** Differently: the transition is common to both, yet Hormuz shows a
   large change and Bab el-Mandeb essentially none. A common technical break cannot
   by itself produce the Hormuz pattern, and the shared boundary means the
   Hormuz–Bab contrast is not differentially confounded by the pipeline.
7. **Are the two dimensions empirically separable?** Yes — the cross-event plane
   (figure 10) shows all four combinations except "physical up, observability up":
   Hormuz/Red Sea (both down), Panama (physical down, observability up), Bab
   (neither), Suez (physical up via congestion, observability flat).
8. **Is "AIS observability is disruption-context dependent" supported?** At the
   level of the measured quantity, yes. As a statement about vessel behaviour, no —
   the mechanism remains unidentified and no deliberate-shutdown claim is made.
9. **Which claims are high-confidence?** The Red Sea direction and magnitude; the
   Panama physical decline with stable/improved matching; the metric asymmetry
   between presence hours and vessel counts; the absence of a Bab el-Mandeb
   collapse; the frozen Hormuz estimates.
10. **Which remain descriptive?** Suez (one acquisition week); the transit-core
    sensitivity AOI (126 detections, model unsupported); the Bab magnitude (noisy
    placebo); the spatial attribution of the Panama SAR domain to the Pacific
    approach; every mechanistic statement.
11. **Breadth and robustness — Nature Communications-style general claim, or a
    strong Earth-observation / maritime-governance paper?** Assessed on breadth and
    robustness rather than on any acceptance calculus:

    *For the broader claim:* five cases spanning three disruption contexts; two
    dimensions measured independently; one frozen primary estimate with
    cluster-robust intervals; placebo checks in every case; a validated
    water-domain framework; an explicit metric-decomposition analysis; and a
    replication of the discovery direction in an independent security event on a
    different pipeline generation.

    *Against it:* only one case carries a full census-scale design (Hormuz); the
    Red Sea has 868 model detections; Panama's SAR dimension covers 85.7 km² of
    Pacific-approach water rather than the canal transit; Bab and Suez are
    descriptive; the mechanism behind the matching decline is not identified; and
    measurement is a matching-probability change, not observed non-transmission.

    Recommendation: the evidence supports a strong **Earth-observation /
    maritime-governance framing** now ("AIS-derived activity and observability are
    context-dependent; independent SAR must be used to validate AIS-based
    disruption metrics"), with the two-dimensional result as the central figure. A
    general, high-impact claim would additionally need (a) one more independent
    security event with Hormuz-scale sampling, (b) a mechanism test
    (reception-quality or position-consistency evidence), and (c) a harmonised
    cross-event metric protocol so that every case carries the same Dimension B
    estimate.

---

## 9. Verification and integrity notes

* Frozen Hormuz outputs were re-checked after every Phase 4B run: placebo deltas
  (−15.294 primary, −72.805 narrow), standardized changes (−11.538 / −63.787) and
  the falsification rows (−1.11 / −15.03) are unchanged.
* The Red Sea and Suez screening tables and the Panama/Red Sea/Bab AOIs were not
  modified; the Bab 2024–2026 window was prepared under a separate `--window-tag`
  so the already-screened 2021–2024 artifacts remain intact.
* Two data-handling defects were found and fixed during Phase 4B, both
  documented: overlapping cached AIS report windows were double-counted, and
  deduplicating on (date, hours, MMSI) deleted legitimate hours from different
  presence cells. The corrected series retain 98.2 % (Bab), 15.5 % (Panama) and
  24.0 % (Suez) of presence hours inside the footprint-defined water domain.
* The GFW report endpoint rejects ranges longer than 366 days; every request is
  now clamped, and each report is cached, audited and recorded in
  `outputs/logs/api_manifest.csv`.

---

## 10. Stop point

No further events were added, no machine learning was used, no composite
resilience index was built, and no search for stronger effects was performed.
The next step is manuscript design, unless a critical identification failure is
discovered on review.
