# Phase 3 findings — identification and robustness

Written after a full re-run of steps 00–09. Every number below is read back from
the generated tables:

`matching_phase_estimates.csv`, `placebo_results.csv`,
`matching_definition_crosstab.csv`, `ais_coverage.csv`,
`ais_zero_missing_audit.csv`, `pipeline_falsification.csv`,
`composition_shift.csv`, `formal_matching_models.csv`,
`standardized_matching_effects.csv`, `same_day_sar_ais.csv`,
`divergence_alignment.csv`, and `outputs/figures/fig_phase3_*.png`.

---

## 0. Terminology (fixed)

| term | definition | role |
| --- | --- | --- |
| **confirmed AIS–SAR matching** | `matched_category != "unmatched"` | **PRIMARY**, used in every model and headline |
| **MMSI-present diagnostic** | `mmsi is not null` | diagnostic only — **never** an alternative matching definition |

The cross-tab (`matching_definition_crosstab.csv`) shows why the distinction is
not cosmetic. Three groups exist:

| group | primary AOI, 2026 persistent | `matching_score` median | `matching_score` p95 | share with score < 1 |
| --- | --- | --- | --- | --- |
| matched + MMSI present | 184 (34.9 %) | 2.50 | 295.2 | 38.6 % |
| unmatched + MMSI present | 139 (26.4 %) | 0.001 | 0.008 | **99.3 %** |
| unmatched + MMSI absent | 204 (38.7 %) | 0.000 | 0.000 | 100 % |

For comparison, in the 2026 baseline the matched group has median
`matching_score` 4.82 (p95 397.6) while the unmatched + MMSI group has median
0.000 and p95 0.005. The MMSI-present-but-unmatched group is therefore a set of
**low-confidence candidate identities whose positional/time match was rejected**,
not a set of accepted matches. Its share of detections drifts gently upward
(14.0 % in 2024, 14.8 % in 2025, 18.8 % in 2026 in the primary AOI; 52.6 % in the
narrow AOI during the persistent phase). Nothing here identifies vessel
behaviour and no behavioural claim is made from it.

---

## 1. AIS zero versus missing (audit)

Every AOI-day was classified into exactly three states using the completed report
interval of each successfully parsed cache file:

| AOI | observed_positive | structural_zero | missing_unknown |
| --- | --- | --- | --- |
| primary `hormuz_core` | 882 | 0 | **0** |
| nested `hormuz_narrow_strait` | 866 | **16** | **0** |

All 16 structural zeros fall inside the disruption (2026-03-21 … 2026-05-17) and
all are corroborated: the enclosing primary AOI was requested with the *same*
dataset, resolution and date range and reports activity on every one of those 16
days (`parent_aoi_has_activity` in `ais_zero_missing_audit.csv`). A failed,
incomplete or malformed report would have been recorded as `missing_unknown`;
there are none. So the zeros are a property of the AOI — no AIS-reporting vessel
inside the narrow transit box — not of the data feed.

Weekly AIS metrics were recomputed with structural zeros counted as observed days
carrying zero hours. The effect is small but in the correct direction: the narrow
AOI's persistent-phase `AIS_SAR_days` anomaly moved from −3.85 to **−4.16** log
points.

---

## 2. Formal matching and placebo estimates

Scene-level bootstrap (3 000 draws, percentile), pooled detection-weighted rate.
Persistent-phase (Mar 28 – May 31) rates:

| AOI | 2024 placebo | 2025 placebo | 2026 event |
| --- | --- | --- | --- |
| primary | 0.515 [0.472, 0.560] (n=15) | 0.488 [0.462, 0.513] (n=14) | **0.350 [0.272, 0.421]** (n=14) |
| narrow | 0.807 [0.766, 0.844] (n=16) | 0.864 [0.827, 0.896] (n=14) | **0.105 [0.000, 0.278]** (n=14) |

Pre/post calendar split (identical days in every year):

| AOI | year | pooled pre → post | Δ (pp) | 95 % CI |
| --- | --- | --- | --- | --- |
| primary | 2024 | 0.525 → 0.515 | −0.99 | [−6.14, +4.04] |
| primary | 2025 | 0.477 → 0.486 | +0.92 | [−3.80, +5.26] |
| primary | **2026** | 0.470 → 0.317 | **−15.29** | [−22.48, −8.37] |
| narrow | 2024 | 0.809 → 0.768 | −4.12 | [−13.51, +5.67] |
| narrow | 2025 | 0.817 → 0.860 | +4.26 | [−1.46, +10.03] |
| narrow | **2026** | 0.843 → 0.115 | **−72.81** | [−84.11, −59.55] |

Raw counts are retained in every table (`n_scenes`, `n_detections`, `n_matched`,
`n_unmatched`, `n_unknown`), so no estimate has to be read without its sample.

---

## 3. Composition: what actually changed

Within-2026 pre (Jan 1 – Feb 27) vs post (Mar 1 – May 31) distributions
(`composition_shift.csv`); total-variation distance (TVD):

| unit | length bin TVD | spatial cell TVD | match-label TVD | median length pre → post |
| --- | --- | --- | --- | --- |
| primary | 0.248 | 0.632 | 0.166 | 132.5 → **67.9 m** |
| narrow | 0.302 | 0.838 | 0.756 | 167.7 → 105.1 m |
| Gibraltar (reference) | 0.136 | 0.172 | 0.032 | 131.7 → 150.9 m |
| Singapore (reference) | 0.049 | 0.075 | 0.044 | 159.6 → 155.9 m |
| Dover (reference) | 0.070 | 0.296 | 0.093 | 138.5 → 127.7 m |

Composition changed a lot in Hormuz — the median detected vessel halved in length
and detections concentrated spatially — while the reference corridors barely
moved. That is exactly why raw matching rates cannot be read as an observability
change on their own. Figures: `fig_phase3_length_distribution_*`,
`fig_phase3_spatial_distribution_*`.

---

## 4. Composition-standardized matching (the central result)

Detection-level logistic regression, standard errors **clustered by SAR scene**,
with `C(period) + C(length_bin) + C(cell)` on a 0.05° grid whose rare cells
(< 30 detections) are pooled. `matching_score` and MMSI presence are excluded as
predictors because both are produced by the matching process itself.

| AOI | period | detections | observed rate | **standardized probability** | standardized Δ vs baseline | 95 % CI (pp) | raw Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| primary | 2024 reference | 2 532 | 0.519 | 0.501 | +3.1 pp | [+0.1, +5.8] | — |
| primary | 2025 reference | 2 953 | 0.483 | 0.510 | +4.0 pp | [+1.2, +6.4] | — |
| primary | 2026 baseline | 1 189 | 0.470 | 0.470 | 0 (reference) | — | — |
| primary | 2026 acute | 218 | 0.239 | **0.286** | **−18.4** | [−27.9, −9.2] | −23.2 |
| primary | 2026 persistent | 527 | 0.349 | **0.355** | **−11.5** | [−17.6, −5.4] | −12.1 |
| narrow | 2026 baseline | 198 | 0.843 | 0.843 | 0 (reference) | — | — |
| narrow | 2026 acute | 7 | 0.143 | 0.240 | **−60.4** | [−77.9, −21.9] | −70.1 |
| narrow | 2026 persistent | 19 | 0.105 | 0.206 | **−63.8** | [−80.0, −20.6] | −73.8 |

Model details: 7 419 detections in 103 scenes (primary) and 1 259 in 103 scenes
(narrow); 153 kept spatial cells (+363 pooled) and 27 (+27); all models converged.
Confidence intervals come from 1 000 draws out of the cluster-robust covariance.

Period odds ratios (primary model, clustered by scene):
2026 acute **0.221** [0.111, 0.442], 2026 persistent **0.361** [0.231, 0.564],
2025 reference 1.059 [0.894, 1.256].
Vessel length dominates the baseline propensity: relative to the < 50 m bin the
odds ratios are 6.3 (50–100 m), 5.1 (100–150 m), 6.3 (150–200 m), 8.7 (200–300 m)
and 7.5 (≥ 300 m).

**Answer to the key question: yes — the decline remains after fixing vessel-size
and spatial composition**, in both AOIs and at the same order of magnitude.

---

## 5. Pipeline-transition falsification

### 5A. Global archive (February → March, all detections)

| transition | detections before / after | matching rate before → after | Δ (pp) | 95 % CI | length-standardized Δ |
| --- | --- | --- | --- | --- | --- |
| **2026-02 → 2026-03** (pipev4) | 185 481 / 243 126 | 0.620 → 0.618 | **−0.25** | [−2.62, +2.27] | +0.24 pp |
| 2024-02 → 2024-03 (placebo) | 191 043 / 244 407 | 0.649 → 0.653 | +0.35 | [−1.73, +2.42] | +2.05 pp |
| 2025-02 → 2025-03 (placebo) | 192 530 / 208 030 | 0.652 → 0.657 | +0.44 | [−1.81, +2.65] | +1.39 pp |

Eight fixed global regions for 2026-02 → 2026-03: Δ ranges from −1.77 pp (region
`other`) to +3.89 pp (west Africa); every interval spans zero
(`pipeline_falsification.csv`). Detections per scene are flat (v3 37.7 → v4 38.8).

### 5B. External technical reference corridors (pre-specified)

| unit | Feb 2026 rate | Mar 2026 rate | Δ (pp) | 95 % CI | length-standardized Δ |
| --- | --- | --- | --- | --- | --- |
| Gibraltar | 0.911 (n=146) | 0.927 (n=177) | +1.56 | [−5.92, +11.53] | −4.97 pp (thin bins) |
| Singapore | 0.957 (n=652) | 0.953 (n=513) | −0.38 | [−3.55, +3.52] | +0.01 pp |
| Dover | 0.881 (n=101) | 0.857 (n=182) | −2.40 | [−14.33, +7.62] | −1.75 pp |
| **`hormuz_core`** | 0.457 (n=516) | 0.268 (n=265) | **−18.94** | [−26.16, −9.37] | **−19.94 pp** |
| **`hormuz_narrow_strait`** | 0.859 (n=85) | 0.091 (n=11) | **−76.79** | [−89.47, −58.74] | n/a (post too small) |

The same corridors in the placebo transitions move by +2.8/+5.3 pp (Gibraltar),
0.0/+0.3 pp (Singapore) and +6.7/−2.3 pp (Dover), so their 2026 movement is
ordinary. The Hormuz AOIs' own February→March history is −1.11 pp (2024) and
−2.14 pp (2025) for the primary AOI.

**Conclusion: no evidence that `pipev3 → pipev4` produced a broad AIS–SAR
matching discontinuity.** The primary AOI's change is 4.9× the largest global
regional change and 7.9× the largest corridor change (the narrow AOI's is 19.7×
and 31.9×). It cannot be fully *proven* absent for Hormuz itself, because the
switch and the event are one day apart, but no measurable discontinuity exists in
three independent corridors, eight regions, the global archive, or the same AOI
in other years.

---

## 6. Joint AIS–SAR divergence and onset alignment

Phase-level historical-normalized estimates (mean log anomaly, 3 000-draw
bootstrap over weeks):

| AOI | series | baseline | acute | persistent | gap (acute / persistent) |
| --- | --- | --- | --- | --- | --- |
| primary | AIS all days | +0.170 | −1.215 | −1.427 | −0.504 / −0.642 |
| primary | AIS (SAR days) | +0.158 | −1.216 | −1.340 | −0.505 / −0.669 |
| primary | SAR density | +0.123 | −0.710 | −0.671 | — |
| narrow | AIS all days | +0.130 | −4.256 | −3.773 | −2.288 / −2.020 |
| narrow | AIS (SAR days) | +0.142 | −4.123 | −4.163 | −2.155 / −2.324 |
| narrow | SAR density | −0.070 | −1.968 | −1.839 | — |

Onset test (fixed rule: first event-year week below the pre-event 5th percentile
of that signal). **All four signals in both AOIs cross in the same week:
2026-03-02** (`divergence_alignment.csv`, `weeks_to_first_crossing = 0`).

Reading the three layers the brief asks to separate:

* **A. Real traffic decline** — SAR density falls 49 % (primary) and 84 % (narrow),
  is spatially concentrated in the transit corridor, and is not a coverage or
  spatial-composition artefact.
* **B. Cooperative-observation decline** — confirmed AIS–SAR matching falls
  −18.4 pp (primary acute) and −60.4 pp (narrow acute) *after* standardizing
  vessel size and space, from a period odds ratio of 0.22 (primary) and 0.03
  (narrow).
* **C. Assessment divergence** — AIS-based activity falls about 1.6–1.9× more than
  SAR in the primary AOI and 7–13× more in the narrow AOI, on the same days and in
  the same boxes.

The gap is an assessment divergence between two observation systems. It is not
the share of vessels that switched AIS off.

---

## 7. Same-day evidence

`same_day_sar_ais.csv` covers all 412 SAR acquisition days. Inside the narrow
transit box:

| period | acquisition days | SAR detections / day | AIS presence hours / day | AIS vessels / day | days with zero AIS |
| --- | --- | --- | --- | --- | --- |
| Dec 15 2025 – Feb 23 2026 | 17 | 9–20 (median 14) | 281–402 (median 331) | 125–171 | 0 |
| Mar 4 – May 23 2026 | 20 | 0–3 (median 1) | 0–20 (median 5) | 0–10 | 4 |

Of the four zero-AIS acquisition days, **three are structural zeros with
SAR-confirmed vessels** (2026-03-24, 2026-05-06, 2026-05-15; the fourth,
2026-04-24, has no SAR detection either). These are case observations, not a
causal estimate.

---

## 8. Answers to the seven Phase 3 questions

1. **Does the decline survive standardization?** Yes. Primary AOI: −18.4 pp
   (acute) and −11.5 pp (persistent) standardized, both intervals excluding zero.
   Narrow AOI: −60.4 pp and −63.8 pp.
2. **How much smaller is the standardized effect?** Primary acute 23.2 → 18.4 pp
   (79 % retained; composition accounts for ~20 %); primary persistent 12.1 →
   11.5 pp (95 % retained); narrow 70.1 → 60.4 pp and 73.8 → 63.8 pp (86 %
   retained).
3. **Is there a global or multi-region matching discontinuity at pipev3→pipev4?**
   No. Global Δ = −0.25 pp [−2.62, +2.27]; eight regions −1.8 to +3.9 pp, all
   intervals spanning zero; detections per scene flat.
4. **Can the pipeline transition explain the Hormuz magnitude?** No. Reference
   corridors move by −2.4 to +1.6 pp and the global archive by −0.25 pp, against
   −18.9 pp (primary, 7.9× the largest corridor move) and −76.8 pp (narrow,
   31.9×). The transition remains a one-day
   collinearity for the SAR leg and must be stated as a limitation, not as an
   explanation.
5. **Are the zero-AIS / positive-SAR days genuine structural zeros?** Yes: all 16
   zero days are covered by successfully completed and parsed reports, the
   enclosing AOI reports activity on all 16, and 3 of them coincide with positive
   SAR detections.
6. **Does AIS still show a larger anomaly than SAR after same-day alignment?**
   Yes. Restricting AIS to SAR acquisition days changes the gap by ≤0.03 log
   points in the primary AOI and leaves both intervals far from zero
   (−0.505 acute, −0.669 persistent).
7. **What is publication-grade and what is descriptive?**

   *Publication-grade (identified and robust):*
   - the corridor-specific traffic collapse visible in SAR density;
   - the confirmed AIS–SAR matching decline after composition standardization
     (−18.4 pp acute, −11.5 pp persistent, scene-clustered, odds ratio 0.22/0.36);
   - the AIS-vs-SAR assessment divergence surviving same-day alignment;
   - the absence of any comparable March-1 discontinuity in three pre-specified
     corridors, eight regions, or the global archive;
   - the structural-zero classification of the zero-AIS days.

   *Descriptive only:*
   - the narrow-AOI magnitudes (7 and 19 post-event detections; the point estimate
     is large but its interval is wide, and the 2024 placebo for that same AOI was
     itself −4.12 pp [−13.51, +5.67] in the pre/post design of
     `placebo_results.csv`. Note that the *different* statistic reported in
     `pipeline_falsification.csv` — the narrow AOI's February→March 2024
     month-pair transition, 0.84 → 0.69 over 111 vs 112 classified detections —
     is −15.03 pp [−27.83, +4.11]. It is a current value, not a stale one, but it
     answers a different question (month-to-month movement at the pipeline-break
     calendar, not the pre/post placebo split) and must not be quoted as "the
     2024 placebo".);
   - anything about mechanism — transmission behaviour, AIS reception,
     interference, or spoofing are all consistent with these data and cannot be
     separated here;
   - the ≥ 300 m bin (n = 4 and 11 post-event);
   - splitting the March-1 level shift between the disruption and the pipeline
     change for the SAR leg specifically.

---

## 9. Verdict

The stated criterion was: *if the standardized Hormuz effect remains substantial,
the AIS–SAR divergence survives same-day alignment, and global/external pipeline
diagnostics show no comparable March-1 discontinuity, classify the result as
publication-grade enough to justify the multi-event extension.*

| condition | status |
| --- | --- |
| standardized Hormuz effect substantial | **met** (−18.4 / −11.5 pp primary; −60.4 / −63.8 pp narrow) |
| divergence survives same-day alignment | **met** (gap −0.505 / −0.669; ≤0.03 log change) |
| no comparable March-1 discontinuity elsewhere | **met** (global −0.25 pp; corridors −2.4…+1.6 pp; regions all CIs span zero) |
| zero-AIS days verified as structural zeros | **met** (16/16, independently corroborated) |

**Classification: publication-grade for the Hormuz case, sufficient to justify the
multi-event extension** (Red Sea / Suez deliberately not started yet).
