# Hormuz AIS–SAR observability feasibility test

**Research question.** During a major geopolitical maritime disruption, does AIS
remain a representative measure of real vessel activity? If it does not, can an
AIS-only measurement over- or under-state the intensity of a shipping
disruption?

This repository is a *feasibility test*, not a finished paper. It asks whether
the publicly available data contain a signal strong enough to justify a full
study, using the **2026 Strait of Hormuz disruption** as the case
(`event_date = 2026-02-28`).

Three questions drive the test:

| | Question | Core metric |
| --- | --- | --- |
| Q1 | Did the AIS–SAR matching rate shift abnormally? | `matching_rate = matched_SAR_detections / total_SAR_detections` |
| Q2 | Do AIS and SAR disagree about the size of the activity drop? | `observability_gap = AIS_anomaly − SAR_anomaly` |
| Q3 | Could the shift be seasonality or a data-processing change? | same-location historical controls, 2024/2025 placebo splits, explicit technical-break flags |

Everything here uses **public data only**: GFW Sentinel-1 SAR vessel
detections, GFW SAR detection footprints, and GFW AIS vessel presence. No
commercial AIS, no proprietary data.

---

## 1. Current status: Phase 1 complete

Implementation is deliberately staged (see §7). Phase 1 — scaffold, schema
discovery, SAR ingestion, footprint coverage, scene-level metrics — is
implemented and has been run end-to-end on the real archive.

| Script | Status |
| --- | --- |
| `src/00_inspect_data.py` | implemented and run |
| `src/01_prepare_sar.py` | implemented and run |
| `src/03_build_scene_metrics.py` | implemented and run |
| `src/02_fetch_ais.py` | implemented and run (58 sequential GFW requests) |
| `src/03b_robustness_tables.py` | implemented and run (AOI + matching-definition + size robustness) |
| `src/04_matching_analysis.py` | implemented and run (formal matching + placebo + definition cross-tab) |
| `src/05_activity_anomaly.py` | implemented and run (AIS vs SAR anomalies, observability gap) |
| `src/06_statistical_tests.py` | implemented and run (scene-clustered logit + composition standardization) |
| `src/07_make_figures.py` | implemented and run (7 Phase 2 figures) |
| `src/09_identification_checks.py` | implemented and run (composition + pipeline falsification) |
| `src/10_multievent_screening.py` | implemented and run (Phase 4A multi-event feasibility screen) |
| `src/11_phase4a_extension_screen.py` | implemented and run (Phase 4A-extension: Bab el-Mandeb 2026, Panama 2023–24) |
| `src/hormuz_event_model.py` | shared Phase 4B machinery (event windows, SAR/AIS activity, matching model) |
| `src/12_phase4b_events.py` | implemented and run (Phase 4B per-event analysis) |
| `src/13_phase4b_cross_event.py` | implemented and run (cross-event synthesis, Panama decomposition) |
| `src/14_phase4b_figures.py` | implemented and run (12 Phase 4B figures) |
| `src/08_feasibility_report.py` | stub — reserved for the publication-scale multi-event report |

The one remaining stub exits with a clear message instead of pretending to work.
Findings are written up in `outputs/phase1_findings.md`,
`outputs/phase1_robustness_extension.md`, `outputs/phase2_findings.md` and
`outputs/phase3_findings.md`, plus `outputs/phase4a_findings.md` for the
multi-event feasibility screen and `outputs/phase4a_extension_findings.md` for the
two additional candidate events, and `outputs/phase4b_findings.md` for the
publication-scale multi-event analysis.

---

## 2. Python environment

Any environment with the packages in `requirements.txt` works. This project was
developed and verified with CPython 3.12.

```powershell
# option A: create a dedicated environment
conda create -n hormuz python=3.12 pip -y
conda activate hormuz
python -m pip install -r requirements.txt

# option B: reuse an existing environment that already has geopandas/statsmodels
# (this machine: the "goci" environment satisfies everything except python-dotenv)
```

Check the installation:

```powershell
python -c "import pandas, geopandas, shapely, pyarrow, statsmodels; print('ok')"
python -m pytest tests -q
```

`python-dotenv` is listed in `requirements.txt` and preferred when present. If it
is missing, `.env` is still read by a small built-in parser (the fact is logged),
so the pipeline runs in either case.

---

## 3. GFW API token

AIS vessel presence is the only component that needs the network and a token.

1. Create a free token: <https://globalfishingwatch.org/our-apis/tokens>
2. Copy the template and fill it in:

   ```powershell
   Copy-Item .env.example .env
   # then edit .env
   ```

   ```dotenv
   GFW_API_TOKEN=your_token_here
   ```

3. `.env` is never committed and the token never appears in code, logs, or the
   API manifest. Only the key name is read (`ais.api_token_env`).

Optional environment overrides (see `config/config.yaml`, `paths:` block):
`HORMUZ_SAR_ARCHIVE_ROOT`, `HORMUZ_AIS_PRESENCE_DIR`, and so on.

---

## 4. Where the raw data live

### SAR detections — `data/raw/sar_detections/` (or the external archive)

Expected filenames:

```
sar_vessel_detections_pipev<version>_<YYYYMM>.csv        # full month
sar_vessel_detections_pipev<version>_<YYYYMMDD>.csv      # partial re-download
```

On this machine the archive is `D:/gfwdata/<year>/…`. The external root is
configured as `paths.sar_archive_root` and is treated as **read-only**. Local
copies in `data/raw/sar_detections/` take precedence when the same month exists
in both places.

### SAR footprints — `data/raw/sar_footprints/` (or the external archive)

```
sar_detections_footprints_<YYYYMM>.csv
```

Same search order. A full-month file always beats a partial one; duplicates are
logged in the discovery ledger (see §8).

### AIS vessel presence — `data/raw/ais_presence/`

Written by `src/02_fetch_ais.py`, one cached raw **ZIP** response per request
(the report endpoint returns a ZIP containing the activity CSV, a readme, the
GFW AIS caveats PDF and the reference geometry). The dataset version is pinned in
`config/config.yaml` (`ais.dataset`, default `public-global-presence:v4.0`);
`latest` is only a logged fallback and the version actually returned is always
recorded.

Three API behaviours are encoded in the client because they were verified live
and each one silently corrupts data if assumed otherwise:

1. `geojson` must go in the **JSON body**; as a query parameter the API rejects it.
2. `format` must be upper case (`CSV`).
3. the `date-range` **end date is exclusive**, so an inclusive window is sent as
   `end + 1 day` (otherwise the last day of every chunk disappears).

---

## 5. AOI configuration

All AOI coordinates come from **`config/hormuz_core.geojson`**. No coordinate is
ever hardcoded in Python.

The shipped `hormuz_core` polygon is a hand-drawn corridor in EPSG:4326 covering
the Gulf-side approach (Bandar Abbas / Qeshm), the narrowest section near
56.5°E, and the Gulf of Oman approach toward Musandam. It is ~23 100 km² and
deliberately includes some land; the area denominator is the full polygon.

To use your own AOI, replace the geometry (keep the `name` property matching
`aoi.primary_feature`, or change that key in the config). `src/00_inspect_data.py`
records the AOI area for provenance; steps 01/03 re-read the file every run.

---

## 6. Run order

```powershell
# Phase 1 (implemented, one pass per AOI)
python src/00_inspect_data.py          # schema discovery -> config/schema_mapping.yaml
python src/01_prepare_sar.py --aoi hormuz_core
python src/01_prepare_sar.py --aoi hormuz_narrow_strait
python src/03_build_scene_metrics.py --aoi hormuz_core
python src/03_build_scene_metrics.py --aoi hormuz_narrow_strait
python src/03b_robustness_tables.py    # aoi_sensitivity + matching/length robustness

# Phase 2 (implemented)
python src/02_fetch_ais.py --check-auth    # validate the token with one request
python src/02_fetch_ais.py                 # sequential download + processing
python src/05_activity_anomaly.py          # weekly AIS/SAR anomalies + gap
python src/07_make_figures.py              # 7 figures

# Phase 3 (implemented)
python src/04_matching_analysis.py         # formal matching + placebo + cross-tab
python src/06_statistical_tests.py         # scene-clustered logit + standardization
python src/09_identification_checks.py     # composition + pipeline falsification
python src/07_make_figures.py              # also writes the Phase 3 figures

# Phase 3 identification needs the corridors to have been ingested first
for aoi in gibraltar_strait singapore_strait english_channel_dover:
    python src/01_prepare_sar.py --aoi $aoi   # PowerShell: foreach, see below
    python src/03_build_scene_metrics.py --aoi $aoi

# Phase 4A (implemented) - multi-event feasibility screening
python src/01_prepare_sar.py --aoi suez_canal_core --start 2019-01-01 --end 2021-05-31
python src/03_build_scene_metrics.py --aoi suez_canal_core --event-date 2021-03-23
python src/01_prepare_sar.py --aoi bab_el_mandeb_core --start 2021-10-01 --end 2024-04-30
python src/03_build_scene_metrics.py --aoi bab_el_mandeb_core --event-date 2023-12-15
python src/10_multievent_screening.py   # -> outputs/tables/multievent_feasibility.csv

# Phase 4A-extension (implemented) - two further candidate events
python src/01_prepare_sar.py --aoi bab_el_mandeb_core --start 2024-01-01 --end 2026-05-31 --window-tag w2024_2026
python src/03_build_scene_metrics.py --aoi bab_el_mandeb_core --window-tag w2024_2026
python src/01_prepare_sar.py --aoi panama_canal_core --start 2021-09-01 --end 2024-04-30
python src/03_build_scene_metrics.py --aoi panama_canal_core --event-date 2023-11-03
python src/02_fetch_ais.py --aois bab_el_mandeb_core panama_canal_core --chunk-months 12 --start 2021-09-01 --end 2026-05-31
python src/11_phase4a_extension_screen.py
```

`--window-tag` (or the single config key `paths.window_tag`) lets the **same AOI**
be prepared for a second calendar window without overwriting the first run's
artifacts — that is how Bab el-Mandeb 2026 was screened without touching the
already-screened Red Sea 2023 window.

```powershell
# Phase 4B (implemented) - publication-scale multi-event analysis
python src/12_phase4b_events.py        # per-event activity + standardized matching
python src/13_phase4b_cross_event.py   # cross_event_final.csv, roles, Panama decomposition
python src/14_phase4b_figures.py       # 12 figures
```

Phase 4B runs on the frozen case matrix in `phase4b.events`: Hormuz 2026
(discovery; its estimates are carried through from Phase 3, never recomputed), Red
Sea 2023–24 (independent security validation), Panama 2023–24 (non-security
contrast, plus a nested pre-specified transit-core sensitivity AOI), Bab el-Mandeb
2026 (concurrent regional falsification) and Suez 2021 (supplementary,
descriptive). It reports two separate dimensions — the SAR activity anomaly and
the composition-standardized matching change — and never combines them into an
index.

**AIS series are restricted to the SAR-observable water domain** before any
comparison. GFW footprints are water-clipped: in the Panama box only 85.7 km² of
the 2,735.6 km² bounding box is ever imaged (the Pacific-side approach), while the
canal's freshwater lake section carries about 20 % of the AOI's AIS presence hours
and no SAR coverage at all. Comparing an AOI-wide AIS series with a
footprint-limited SAR series would be a spatial-domain artefact, so
`load_ais_daily_in_domain` and `load_vessel_days(inside_geom=...)` recompute both
AIS families inside the same water polygon (98.2 % of hours retained for Bab
el-Mandeb, 15.5 % for Panama, 24.0 % for Suez).

PowerShell equivalent for the corridors:

```powershell
foreach ($aoi in "gibraltar_strait","singapore_strait","english_channel_dover") {
  python src/01_prepare_sar.py --aoi $aoi
  python src/03_build_scene_metrics.py --aoi $aoi
}
```

The primary AOI keeps the plain output filenames (`data_coverage.csv`,
`sar_scene_metrics.parquet`, …); every other AOI gets a
`__<aoi_key>` suffix (`data_coverage__hormuz_narrow_strait.csv`,
`sar_scene_metrics__hormuz_narrow_strait.parquet`, …). Nothing in the primary
specification changes when a sensitivity AOI is added.

`src/02_fetch_ais.py` needs network access and a token. It supports
`--dry-run` (print the exact requests), `--from-cache` / `--inspect-only`
(rebuild everything from cached ZIPs with no network) and `--check-auth`.
Downloads are resumable: an existing cache file is never re-requested.

Steps 01 and 03 refuse to run if the schema mapping is missing or if a required
role could not be resolved. Run 00 first, and read
`outputs/diagnostics/schema_report.md`.

Helper flags: `python src/00_inspect_data.py --all-years` (exact row counts for
every year in the archive), `--confirm` (stamp the mapping as manually
confirmed), `python src/01_prepare_sar.py --limit-months 2` (quick smoke test).

---

## 7. Outputs

### Processed data

| Path | Contents |
| --- | --- |
| `data/interim/sar_scenes.parquet` | one row per AOI-intersecting SAR scene: acquisition time, platform, pipeline, `coverage_km2`, `coverage_fraction`, intersection WKT |
| `data/interim/sar_detections_aoi.parquet` | every detection inside the AOI with `match_status` ∈ {matched, unmatched, unknown} |
| `data/interim/sar_daily_aoi_coverage.parquet` | per acquisition date: how much of the AOI was imaged, by how many scenes |
| `data/processed/sar_scene_metrics.parquet` | scene-level counts, `matching_rate`, `sar_density`, flags |
| `data/processed/sar_weekly.parquet` | weekly SAR panel (median density over scenes) |

### Tables

| Path | Contents |
| --- | --- |
| `outputs/tables/data_coverage.csv` | year × month: scenes, detections, matched, unmatched, observed area, acquisition days, AIS valid days |
| `outputs/tables/scene_metrics_preview.csv` | readable preview of the scene table |
| `outputs/tables/detection_composition_by_month.csv` | GFW match-label composition per month |
| `outputs/tables/matching_rate_by_length_bin.csv` | matching rate by vessel-length bin — separates composition from observability |
| `outputs/tables/aoi_sensitivity.csv` | primary vs nested AOI, per phase: scenes, weeks, detections, matching rate + CI, pre/post change, SAR density |
| `outputs/tables/matching_definition_sensitivity.csv` | matching rate under `matched_category` vs `mmsi` definitions, by period and month |
| `outputs/tables/length_bin_matching.csv` | counts and matching rate by length bin × phase, with small-sample flags |
| `outputs/tables/ais_vessel_type_summary.csv` | GFW `Vessel Type` totals and relative contribution |
| `outputs/tables/ais_gear_type_summary.csv` | GFW `Gear Type` totals — the field that actually contains `TANKER` |
| `outputs/tables/ais_coverage.csv` | per-day AIS coverage validation (missing days flagged, never zero) |
| `outputs/tables/activity_anomaly_weekly.csv` | weekly AIS and SAR log anomalies and the observability gap |
| `outputs/tables/phase_summary.csv` | baseline / acute / persistent anomalies with bootstrap intervals |
| `outputs/tables/ais_vessel_population_sensitivity.csv` | anomaly by vessel population (all / commercial / non-fishing / fishing) |
| `outputs/tables/matching_phase_estimates.csv` | formal per-phase matching rates with scene-bootstrap CIs, per AOI and year group |
| `outputs/tables/placebo_results.csv` | formal pre/post split table for 2024 / 2025 / 2026 |
| `outputs/tables/matching_definition_crosstab.csv` | `matched_category` x MMSI presence x `matching_score` quantiles |
| `outputs/tables/ais_zero_missing_audit.csv` | per AOI-day `observed_positive` / `structural_zero` / `missing_unknown` with the same-day SAR record |
| `outputs/tables/pipeline_falsification.csv` | Feb→Mar matching changes for the global archive, 8 regions, 3 reference corridors and both Hormuz AOIs |
| `outputs/tables/composition_shift.csv` | pre/post length, spatial-cell, score and label distributions with total-variation distances |
| `outputs/tables/formal_matching_models.csv` | scene-clustered logistic coefficients (log-odds, odds ratios, CIs) for all three specifications |
| `outputs/tables/standardized_matching_effects.csv` | observed vs composition-standardized matching probabilities and differences |
| `outputs/tables/same_day_sar_ais.csv` | same-day SAR vs AIS record on every acquisition day |
| `outputs/tables/divergence_alignment.csv` | onset week of each signal against its pre-event threshold |
| `outputs/tables/multievent_feasibility.csv` | Phase 4A screen: one row per event × year role × period, with SAR usability, pipeline comparability, AIS availability and the STRONG / CONDITIONAL / NO-GO verdict |
| `outputs/tables/phase4a_extension_sar_feasibility.csv` | Phase 4A-extension: same screen for Bab el-Mandeb 2026 and Panama 2023–24, with separate SAR / AIS / combined verdicts |
| `outputs/tables/phase4a_extension_matching_metrics.csv` | Screening-level phase matching rates, scene-bootstrap CIs, pre→event changes and length-bin rates |
| `outputs/tables/phase4a_extension_activity_diagnostics.csv` | Weekly SAR and AIS anomalies with the preliminary observability gap, same calendar weeks |
| `outputs/tables/phase4a_extension_month_pair.csv` | Bab el-Mandeb February→March 2026 change, built identically to the frozen falsification rows |
| `outputs/tables/cross_event_final.csv` | Phase 4B cross-event table: Dimension A (SAR activity anomaly), Dimension B (standardized matching change + CI), supporting AIS family, inference strength and interpretation |
| `outputs/tables/event_role_summary.csv` | Case roles and their statistical status (the cases are not interchangeable replicas) |
| `outputs/tables/redsea_2023_standardized_effects.csv` | Red Sea standardized matching probabilities by period, with placebo benchmark |
| `outputs/tables/panama_2023_standardized_effects.csv` | Panama standardized matching probabilities by period |
| `outputs/tables/bab_mandeb_2026_standardized_effects.csv` | Bab el-Mandeb 2026 standardized matching probabilities |
| `outputs/tables/panama_ais_metric_decomposition.csv` | Panama presence hours vs distinct vessels vs dwell proxy on the same water domain |
| `outputs/tables/panama_transit_core_sensitivity.csv` | Primary vs nested transit-core Panama AOI |
| `outputs/tables/phase4b_sar_activity.csv` · `phase4b_sar_weekly.csv` | Phase-level and weekly SAR activity with historical-normalized anomalies |
| `outputs/tables/phase4b_ais_metrics.csv` · `phase4b_ais_phase_summary.csv` | AIS metric family (hours, distinct vessels, SAR-day variants, dwell proxy) |
| `outputs/tables/phase4b_matching_coefficients.csv` | Cluster-robust logistic coefficients for every fitted case |

### Diagnostics

| Path | Contents |
| --- | --- |
| `outputs/diagnostics/schema_report.json` / `.md` | full schema discovery report |
| `outputs/diagnostics/sar_scene_coverage_distribution.csv` | coverage-fraction quantiles used to calibrate the scene threshold |
| `outputs/diagnostics/coverage_threshold_sensitivity.csv` | sample size and pooled matching rate at several coverage thresholds |
| `outputs/diagnostics/prepare_sar_ledger.csv` / `.json` | every filter with before / after / removed / reason |
| `outputs/diagnostics/scene_count_anomalies.csv` | scenes where `n_total ≠ n_matched + n_unmatched` |
| `outputs/diagnostics/orphan_detections_by_file.csv` | detections whose `scene_id` is absent from the footprint product |
| `outputs/diagnostics/platform_detection_coverage.csv` | footprint-only platforms (no detections at all) |
| `outputs/diagnostics/global_archive_label_share.csv` | archive-wide match-label share per month — the pipeline-artifact test |
| `outputs/diagnostics/detection_spatial_dispersion.csv` | per-scene spatial dispersion: separates "fewer vessels" from "one anchorage" |
| `outputs/diagnostics/ais_schema_report.json` | the AIS schema actually returned, with `vessel_type`/`gear_type` value counts |
| `outputs/diagnostics/figure_manifest.json` | which figures were produced and which breaks were annotated |

### Figures, logs, report

`outputs/figures/` (Phase 3, matplotlib only, one figure per file),
`outputs/logs/` (`run.log`, API manifest), `outputs/feasibility_report.md`.

---

## 8. Configuration rules

`config/config.yaml` is the single source of truth for **all** dates,
thresholds, dataset versions, and schema assumptions. Hardcoding any of them in
analysis code is a bug.

Key entries:

* `analysis.event_date`, `analysis.phases` (`MM-DD` windows so the same calendar
  window applies to every year), `analysis.pre_post`, `analysis.technical_breaks`;
* `sar.min_scene_coverage_fraction` (default 0.30), `sar.min_scene_coverage_km2`,
  `sar.sar_density_scale_km2`, `sar.matching_rate_length_bins_m`;
* `ais.dataset` (pinned version), `ais.group_by`, retry/pause settings;
* `stats.bootstrap_iterations` (3000), `stats.bootstrap_unit` (scene),
  `models.primary_formula`, `models.cluster_variable`;
* `identification.spatial_cell_degrees` (0.05, pre-specified),
  `identification.min_detections_per_cell`, `identification.length_bins_m`,
  `identification.standardize_to_phase` (the pre-event 2026 baseline) and
  `identification.falsification` (break month, placebo pairs, fixed regions).

**Terminology.** `matched_category != "unmatched"` is the PRIMARY AIS–SAR matching
definition. `mmsi is not null` is the **MMSI-present diagnostic** and is never
treated as an equivalent match: in the cross-tab its `matching_score` is
effectively zero (p95 < 0.01) while confirmed matches have median 2.5–4.8, so it
is a set of low-confidence candidate identities whose match was rejected.

**Zero versus missing.** Every AOI-day is classified `observed_positive`,
`structural_zero` (a completed, parsed report covered the date and contained no
activity entry) or `missing_unknown` (anything else, including failed or
unparseable reports). Structural zeros count as observed days carrying zero hours;
`missing_unknown` days are excluded from both numerator and denominator.

`config/schema_mapping.yaml` is **generated** by step 00 and lists the verified
column for every analysis role. Later steps read it and fail loudly if a
required role is unresolved.

Every filter in steps 01/03 is reported as
`before_rows / after_rows / removed_rows / reason` (see the ledgers). Missing
values are never silently dropped or filled.

---

## 9. How to read the matching rate

`matching_rate = matched_SAR_detections / total_SAR_detections`, where "matched"
means GFW attached an AIS identity to the SAR detection
(`matched_category != 'unmatched'`, the primary definition pinned in
`schema_mapping.yaml`; the alternative MMSI-based definition is stored next to it
as `matching_rate_mmsi`).

Three cautions:

1. **A falling matching rate is not automatically an observability failure.** If
   large AIS-carrying ships leave an area, the remaining population is dominated
   by small craft that never carried AIS, and the pooled rate falls for purely
   compositional reasons. That is why the rate must also be read *within*
   vessel-length bins — see
   `outputs/tables/matching_rate_by_length_bin.csv`.
2. **It is a ratio, so a shrinking denominator matters.** Report the scene
   distribution (median, mean, bootstrap CI), not only the pooled ratio over all
   detections.
3. **It has a global trend.** The archive-wide unmatched share drifts slowly
   upward over 2024–2026, so the AOI rate must be compared with the archive-wide
   rate for the same month (`global_archive_label_share.csv`) before any of the
   movement is attributed to local events.

`sar_density` is the activity measure: `n_total / coverage_km2 × 1000`, i.e.
vessels per 1000 km² per SAR acquisition. It uses the **covered** area, not the
whole AOI, so partially imaged scenes stay comparable.

---

## 10. Why unmatched SAR detections are not "dark vessels"

`unmatched` means only *"no AIS identity was attached to this SAR detection"*. It
can arise from many causes that are not deliberate AIS non-transmission:

* genuine absence of AIS equipment (small craft, dhows, wooden hulls);
* AIS reception gaps — the SAR footprint and the terrestrial/satellite AIS
  receiver network do not cover the same area equally;
* position/time mismatch beyond the matching tolerance (the `matching_score`
  field) simply because the two observations are minutes apart and the vessel
  moved;
* AIS interference, jamming or spoofing, which is a documented issue in this
  region;
* pipeline effects — a change of the GFW matching version.

Calling these "dark vessels" would assert intent that the data cannot support.
The feasible claim is weaker and still interesting: *the probability that a SAR
detection carries an AIS identity changed*, which is a statement about AIS
observability, not about vessel behaviour.

---

## 11. Why a date without a SAR acquisition is never "0 vessels"

Sentinel-1 images the AOI only every ~2–3 days, and a single scene covers only
part of the AOI. A day without an acquisition carries **no information** about
vessel activity; coding it as zero invents a huge fake drop and destroys the
time series.

The pipeline therefore:

* keeps acquisition dates only (`sar_daily_aoi_coverage.parquet`), and a week
  without any usable scene is simply **absent** from `sar_weekly.parquet`;
* distinguishes "imaged and found nothing" (`n_total = 0` on a real scene) from
  "not imaged" (no row at all);
* normalises density by the covered area, and reports the covered area so the
  reader can see any coverage change.

The same rule applies to AIS: the weekly AIS panel is built on a full calendar
and missing days stay NA.

---

## 12. Why historical placebo splits are required

A change after `2026-02-28` is only interesting if the *same calendar split* in
quiet years does not produce it. Both 2024 and 2025 are split at the same
month-day boundaries (`01-01 … 02-27` vs `03-01 … 05-31`), and the resulting
pre/post shifts are compared with 2026 in `outputs/tables/placebo_results.csv`
(Phase 3). Placebo splits catch:

* seasonality (spring/summer traffic, monsoon, fishing seasons);
* slow drift such as the archive-wide AIS-matching trend;
* one-off months (e.g. 2024-12 and 2025-11 look different from their neighbours,
  so a single-year comparison would be fragile);
* data-release artefacts.

Seasonality is also why the anomaly is computed against the *same calendar
week* of the reference years rather than against a neighbouring month.

---

## 13. Technical breaks

Every break inside the estimation window is declared in
`analysis.technical_breaks` and used by the analysis and the figures.

| Date | Break | Phase-1 treatment |
| --- | --- | --- |
| 2026-03-01 | GFW detection pipeline `pipev3` → `pipev4` (visible in the file names) | flagged; treated as a possible confounder and tested with the archive-wide label check |
| 2026-03-28 | GFW SAR ocean–land mask update | start of `persistent_disruption`; the period after it is reported separately |
| 2026-06-30 | Sentinel-1C/1D platform transition | out of scope: Phase 1 stops at 2026-05-31 (`analysis.phase1_end`) |

How they are handled:

* `global_archive_label_share.csv` computes the archive-wide unmatched share per
  month. A pipeline change moves the global share in the same month; a local
  disruption does not. This is a measurement-validity diagnostic, **not** a
  causal control region.
* `sar_weekly.parquet` and the scene table carry `pipeline` and
  `satellite_platform`, so any result can be re-cut by pipeline or platform.
* **Footprint-only platforms are excluded.** In 2025-10 the footprint product
  contains 19 S1C scenes over the AOI, and the detection product contains no S1C
  detections at all (globally, for that month, 0 of 277 263 detections come from
  S1C). Those scenes are acquisitions without a detection product, not
  observations of zero vessels; they are detected automatically and excluded
  (`platform_detection_coverage.csv`, and a ledger entry).
* Phase 1 deliberately stops before the S1C/S1D transition so that
  sensor/platform effects do not have to be modelled yet. If the signal holds,
  extend to 2026-09 and model the platform effect explicitly.

---

## 14. Repository layout

```
├── README.md
├── requirements.txt
├── .env.example                  # copy to .env and add GFW_API_TOKEN
├── config/
│   ├── config.yaml               # single source of truth
│   ├── hormuz_core.geojson       # PRIMARY AOI (all coordinates come from here)
│   ├── hormuz_narrow_strait.geojson   # pre-specified nested sensitivity AOI
│   ├── ref_corridor_gibraltar.geojson # pipeline-break test corridor (not a control)
│   ├── ref_corridor_singapore.geojson
│   ├── ref_corridor_dover.geojson
│   ├── schema_mapping.yaml       # generated by step 00, read by later steps
│   └── ais_schema_mapping.yaml   # generated by step 02 from the returned AIS schema
├── data/
│   ├── raw/{sar_detections,sar_footprints,ais_presence}/
│   ├── interim/
│   └── processed/
├── src/
│   ├── hormuz_common.py          # shared config/logging/geometry/ledger helpers
│   ├── hormuz_stats.py           # scene-level bootstrap estimators
│   ├── 00_inspect_data.py
│   ├── 01_prepare_sar.py
│   ├── 02_fetch_ais.py
│   ├── 03_build_scene_metrics.py
│   ├── 03b_robustness_tables.py
│   ├── 04_matching_analysis.py
│   ├── 05_activity_anomaly.py
│   ├── 06_statistical_tests.py
│   ├── 07_make_figures.py
│   ├── 08_feasibility_report.py
│   └── 09_identification_checks.py
├── outputs/{figures,tables,diagnostics,logs}/
└── tests/
```

Additions to the requested tree, all documented above: the shared modules
`src/hormuz_common.py` and `src/hormuz_stats.py` (avoid duplicating config,
logging, geometry, ledger and bootstrap logic across scripts), the nested
sensitivity AOI, the robustness-tables step, and the two generated schema
mappings that later steps are required to use.

---

## 15. Data-quality facts worth knowing before interpreting anything

These were verified on the real archive during Phase 1:

1. **Column schema is stable; the pipeline version is not.** Detections always
   have `scene_id, timestamp, lat, lon, presence_score, length_m, mmsi,
   matching_score, fishing_score, matched_category`. The file-name token changes
   `pipev20231026` → `pipev3` (2024-11) → `pipev4` (2026-03).
2. **Some monthly files are trimmed.** Several months end a few days early
   (e.g. 2025-01 ends Jan 28, 2026-02 ends Feb 25). Monthly *totals* are
   therefore not comparable across months; per-acquisition metrics are.
3. **A single scene never covers the whole AOI.** Median `coverage_fraction` is
   ≈0.26 and the maximum observed is ≈0.41, so the 0.30 threshold keeps ~36 % of
   scenes. `coverage_threshold_sensitivity.csv` shows the sample at other
   thresholds.
4. **Footprint-only acquisitions exist** (S1C, 2025-10) and are excluded.
5. **A small number of detections have no footprint row** (242 of 21 958 ≈ 1.1 %,
   concentrated in 2025-11/12 and 2026-04). They are flagged and excluded from
   scene metrics, never silently dropped.
6. **Unknown match labels are rare** (151 detections archive-wide in 2026-03)
   and are kept as a third `match_status` level so that
   `n_total = n_matched + n_unmatched` can be checked rather than assumed.
