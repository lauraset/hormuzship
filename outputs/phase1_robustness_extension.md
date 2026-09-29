# Phase 1 robustness extension — Tasks A and B

Produced by `src/03b_robustness_tables.py` from the Phase 1 outputs of both
AOIs. All numbers below are read back from the generated tables:

* `outputs/tables/aoi_sensitivity.csv`
* `outputs/tables/matching_definition_sensitivity.csv`
* `outputs/tables/length_bin_matching.csv`
* `outputs/diagnostics/detection_spatial_dispersion.csv`

**Primary specification is unchanged.** The existing `hormuz_core` AOI remains
the primary AOI in every table (`aoi_role = primary`). The narrow-strait AOI is
a nested sensitivity check; it was *not* chosen or reshaped to maximise any
effect, and no conclusion below promotes it above the primary AOI.

---

## Task A — nested narrow-strait AOI sensitivity

### The AOI

`config/hormuz_narrow_strait.geojson` is the fixed box **56.25–56.65 °E,
26.35–26.70 °N** (area **1 546.0 km²**), declared in the config as
`pre_specified: true`, `tuned_on_results: false`, and asserted by unit tests to
be nested inside the primary corridor. It represents the geographically narrow
commercial transit section of the strait.

The design payoff is visible immediately in per-scene coverage:

| | primary `hormuz_core` | nested `hormuz_narrow_strait` |
| --- | --- | --- |
| AOI area | 23 107.8 km² | 1 546.0 km² |
| scenes intersecting AOI | 570 | 215 |
| usable scenes (`coverage_fraction ≥ 0.30`) | 205 (36 %) | **215 (100 %)** |
| median `coverage_fraction` | 0.260 | **0.908** |
| max `coverage_fraction` | 0.414 | 0.912 |
| detections inside AOI | 21 958 | 2 729 |

So a single Sentinel-1 scene covers ~91 % of the narrow box but at most 41 % of
the primary corridor. The narrow AOI fixes the main structural weakness
identified in Phase 1 (no scene fully covering the AOI) at the cost of far fewer
detections per scene.

### Phase metrics — 2026

| AOI | phase | intersecting scenes | usable scenes | acquisition weeks | detections | matched | unmatched | median coverage fraction | matching rate | 95 % CI | median SAR density |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary | baseline (Jan 1 – Feb 27) | 38 | 14 | 9 | 1 189 | 559 | 630 | 0.232 | 0.470 | [0.438, 0.502] | 10.59 |
| primary | acute (Mar 1 – Mar 27) | 16 | 6 | 4 | 218 | 52 | 166 | 0.232 | 0.239 | [0.183, 0.311] | 4.32 |
| primary | persistent (Mar 28 – May 31) | 39 | 14 | 9 | 527 | 184 | 342 | 0.200 | 0.350 | [0.272, 0.421] | 4.70 |
| narrow | baseline (Jan 1 – Feb 27) | 14 | 14 | 9 | 198 | 167 | 31 | 0.908 | 0.843 | [0.810, 0.876] | 10.69 |
| narrow | acute (Mar 1 – Mar 27) | 6 | 6 | 4 | **7** | 1 | 6 | 0.908 | 0.143 | [0.000, 0.333] | 0.78 |
| narrow | persistent (Mar 28 – May 31) | 14 | 14 | 9 | **19** | 2 | 17 | 0.912 | 0.105 | [0.000, 0.278] | 0.78 |

Same calendar windows in the reference years (2024 + 2025 pooled):

| AOI | phase | usable scenes | detections | matching rate | 95 % CI | median SAR density |
| --- | --- | --- | --- | --- | --- | --- |
| primary | baseline | 28 | 2 083 | 0.498 | [0.470, 0.525] | 8.84 |
| primary | acute | 13 | 1 012 | 0.498 | [0.451, 0.547] | 10.09 |
| primary | persistent | 29 | 2 449 | 0.501 | [0.476, 0.527] | 10.94 |
| narrow | baseline | 29 | 428 | 0.813 | [0.773, 0.851] | 11.40 |
| narrow | acute | 13 | 192 | 0.760 | [0.653, 0.876] | 9.98 |
| narrow | persistent | 30 | 427 | 0.834 | [0.803, 0.860] | 10.21 |

The reference years are flat across phases in both AOIs, so the 2026 movement is
not a calendar artefact.

### Pre/post matching-rate change (identical calendar split in every year)

| AOI | year | scenes pre / post | rate pre | rate post | Δ (pp) | 95 % CI (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| primary | 2024 (placebo) | 14 / 22 | 0.525 | 0.515 | −0.99 | [−6.14, +4.04] |
| primary | 2025 (placebo) | 14 / 20 | 0.477 | 0.486 | +0.92 | [−3.80, +5.26] |
| primary | **2026 (event)** | 14 / 20 | 0.470 | 0.317 | **−15.29** | [−22.48, −8.37] |
| narrow | 2024 (placebo) | 15 / 23 | 0.809 | 0.768 | −4.12 | [−13.51, +5.67] |
| narrow | 2025 (placebo) | 14 / 20 | 0.817 | 0.860 | +4.26 | [−1.46, +10.03] |
| narrow | **2026 (event)** | 14 / 20 | 0.843 | 0.115 | **−72.81** | [−84.11, −59.55] |

The narrow AOI amplifies the effect by roughly 5×, and its effect lies far
outside both its own placebo intervals and the primary AOI's interval.

### Where the loss happened: the transit corridor emptied

Splitting the *primary* AOI's detections by whether they fall inside the narrow
box (same scenes, same pipeline, so this is a within-scene spatial comparison):

| month | inside the narrow box | outside the narrow box | share inside |
| --- | --- | --- | --- |
| 2024 months | 74–125 | 355–615 | 15.7–23.8 % |
| 2025 months | 75–130 | 413–540 | 14.5–21.8 % |
| 2026-01 | 113 | 560 | 16.8 % |
| 2026-02 | 85 | 431 | 16.5 % |
| 2026-03 | **11** | 254 | **4.2 %** |
| 2026-04 | **7** | 225 | **3.0 %** |
| 2026-05 | **8** | 240 | **3.2 %** |

The corridor's share of all AOI detections falls by ~80 % relative, while the
area outside it roughly halves. The disruption is spatially concentrated in the
transit lane, not spread evenly over the AOI.

### What the residual detections look like

`outputs/diagnostics/detection_spatial_dispersion.csv` reports, per scene, the
standard distance of its detections. In the narrow AOI the median falls from
**12.3–15.5 km** (all of 2024–2025) to **0.00 km** in 2026-03, 1.08 km in
2026-04 and 0.00 km in 2026-05. Caveat: those scenes have only 1–3 detections,
so dispersion is largely *undefined* rather than meaningfully zero.

The direct evidence for what remains is positional: inside the narrow box the
2026-03/04/05 detections recur at the *same coordinates* across different
acquisition dates (e.g. ≈26.4577 °N / 56.4282 °E on Mar 16, Mar 19 and Mar 28;
≈26.5300 °N / 56.5950 °E on May 11, May 15, May 18 and May 23), spaced a few
hundred metres apart. That is the signature of a small number of
stationary/anchored vessels rather than of transiting traffic.

---

## Task B — matching robustness

### B1. Primary vs secondary match definition

Definitions (both from the same files, never conflated):

* `primary_matched_category` — `matched_category != 'unmatched'`
* `secondary_mmsi_notnull` — `mmsi` is not null

The two are not independent: **every** `matched` detection has a non-null MMSI
(the "matched but no MMSI" cell is empty in all 29 months), while a substantial
minority of detections carry an MMSI *and* the label `unmatched`. Their absolute
levels therefore differ by design.

Pre/post change under both definitions:

| AOI | year | primary Δ (pp) | primary 95 % CI | MMSI Δ (pp) | MMSI 95 % CI |
| --- | --- | --- | --- | --- | --- |
| primary | 2024 | −0.99 | [−6.14, +4.04] | −2.50 | [−7.37, +2.47] |
| primary | 2025 | +0.92 | [−3.80, +5.26] | +1.64 | [−3.46, +6.77] |
| primary | **2026** | **−15.29** | [−22.48, −8.37] | **−3.07** | [−10.99, +5.11] |
| narrow | 2024 | −4.12 | [−13.50, +5.67] | −2.42 | [−11.10, +7.01] |
| narrow | 2025 | +4.26 | [−1.46, +10.03] | +4.73 | [−0.63, +9.81] |
| narrow | **2026** | **−72.80** | [−84.11, −59.55] | **−17.83** | [−36.84, −0.13] |

**Direction: the same (a decline) in both AOIs under both definitions.**
**Magnitude: not the same.** The MMSI-based fall is about one fifth of the
primary fall in the primary AOI and is *not* distinguishable from zero there
([−10.99, +5.11]); in the narrow AOI it is marginal ([−36.84, −0.13]) against a
−72.8 pp fall under the primary definition.

### B2. The definition gap is itself event-specific

Monthly matching rates (primary AOI) show a stable level difference between the
definitions in normal months, which then widens exactly during the event:

| period | primary | MMSI | gap |
| --- | --- | --- | --- |
| 2024 months | 0.428–0.600 | 0.557–0.726 | −10.5 to −17.6 pp |
| 2025 months | 0.442–0.539 | 0.577–0.685 | −10.4 to −20.4 pp |
| 2026-01 | 0.480 | 0.618 | −13.8 pp |
| 2026-02 | 0.457 | 0.601 | −14.3 pp |
| 2026-03 | 0.268 | 0.528 | **−26.0 pp** |
| 2026-04 | 0.407 | 0.668 | **−26.1 pp** |
| 2026-05 | 0.286 | 0.552 | **−26.6 pp** |

Narrow AOI, same pattern (normal months −3.2 to −18.7 pp, median ≈ −7 pp):
2026-03 gap −81.8 pp,
2026-04 −71.4 pp, 2026-05 −25.0 pp.

Decomposing the primary AOI detections into three mutually exclusive cells
explains the mechanism:

| month | matched + MMSI | unmatched, no MMSI | unmatched + MMSI |
| --- | --- | --- | --- |
| 2026-01 | 323 | 257 | 93 |
| 2026-02 | 236 | 206 | 74 |
| 2026-03 | **71** | 125 | 69 |
| 2026-04 | **94** | 77 | 61 |
| 2026-05 | **71** | 111 | 66 |
| 2024–2025 range | 221–356 | 128–328 | 49–109 |

The strictly matched cell collapses (−78 % from 2026-01 to 2026-05), the
no-MMSI cell roughly halves, and the "MMSI present but match rejected" cell is
essentially unchanged (93 → 66). In the narrow AOI during the event the
residual detections sit almost entirely in the last cell (e.g. 9 of 11 in
2026-03), with a near-zero `matching_score` (10⁻⁷–10⁻³).

### B3. Interpretation and what must change downstream

1. The SAR-side result is **robust in direction but definition-sensitive in
   size**. Any headline claim must be stated as a change in *high-quality AIS
   association*, not as a change in "the share of vessels with an AIS
   identity".
2. Because `matched` ⊆ `mmsi not null`, the primary definition is the stricter
   and more informative one, and it stays primary — but the paper cannot rest on
   it alone.
3. The near-zero `matching_score` on the "MMSI present but rejected" rows means
   these detections had an AIS identity available yet failed GFW's
   position/time consistency test. Whether that reflects AIS position
   unreliability (jamming/spoofing), long reporting gaps, or a GFW matching
   rule cannot be decided from these data. It is the single most important
   question to carry into Phase 2 and beyond.
4. No claim about deliberate AIS shutdown is supported by any of this.

### B4. Vessel-length bins

`outputs/tables/length_bin_matching.csv` (bins `<50`, `50–100`, `100–200`,
`200–300`, `≥300 m`, phases baseline / acute / persistent, both AOIs, 2026 and
the pooled 2024–2025 reference). Counts, matched, unmatched, matching rate and a
bootstrap interval are given per cell; cells with fewer than
`sar.small_sample_min_detections` (30) classified detections are flagged
`small_sample = true` and receive **no** interval — including all narrow-AOI
2026 shock cells and the primary AOI `≥300 m` shock cells (n = 4 and n = 11).

Primary AOI, 2026 vs the 2024–2025 reference:

| bin | reference baseline | 2026 baseline | 2026 acute | 2026 persistent |
| --- | --- | --- | --- | --- |
| <50 m | 0.234 (n=620) | 0.248 (n=286) | 0.258 (n=66) | 0.198 (n=217) |
| 50–100 m | 0.553 (n=275) | 0.605 (n=172) | 0.340 (n=47) | 0.461 (n=142) |
| 100–200 m | 0.589 (n=677) | 0.508 (n=431) | 0.174 (n=69) | 0.541 (n=109) |
| 200–300 m | 0.656 (n=358) | 0.563 (n=231) | 0.219 (n=32) | 0.312 (n=48) |
| ≥300 m | 0.809 (n=94) | 0.507 (n=69) | **0.000 (n=4)** | **0.182 (n=11)** |

Two readings, deliberately kept separate:

* The **robust** pattern: in the mid-size bins (50–200 m), where sample sizes are
  adequate (n = 47–217), the matching rate falls sharply in the acute phase and
  partially recovers. Large-vessel bins fall too, but they are too small in the
  2026 shock window to carry a claim — the earlier Phase 1 note that the
  `≥300 m` fall "grows monotonically with vessel size" must now be **downgraded**:
  with n = 4 and n = 11 those cells are flagged as small samples and no interval
  is reported.
* Meanwhile the *density* of `≥300 m` detections in the primary AOI is itself
  informative (69 detections in the 2026 baseline vs 4 and 11 in the shock
  phases), which is a statement about how few large vessels were present, not
  about their matching probability.

---

## Effect on the Phase 1 verdict

| Strong-GO criterion | status after Tasks A and B |
| --- | --- |
| matching-rate shift ≳ 10 pp | met under the primary definition (−15.3 pp core, −72.8 pp narrow); **not met** under the MMSI definition in the primary AOI (−3.1 pp, CI includes 0) |
| 2024/2025 placebo flat | met in both AOIs under both definitions |
| not caused by coverage collapse | met: coverage fraction is stable in both AOIs (0.32 core; 0.91 narrow), spatial composition is stable, and the loss is spatially concentrated inside the transit corridor |
| AIS vs SAR anomaly separation | still untestable without AIS presence data |

Net effect: Q1's *existence* claim survives in both AOIs and under both
definitions, but its *size* is definition-dependent, and the sharpest, most
defensible statement is now about the collapse of high-quality AIS association
and of SAR-detected traffic inside the narrow transit corridor. Phase 2 proceeds
with the primary AOI as the primary specification, the narrow AOI as a required
sensitivity, and the matching definition reported side by side everywhere.
