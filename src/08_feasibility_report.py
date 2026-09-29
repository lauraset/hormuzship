"""STEP 8 - automatic feasibility report (PHASE 3, NOT IMPLEMENTED YET).

Planned behaviour
-----------------
Assemble ``outputs/feasibility_report.md`` from the processed tables, answering:

* Data sufficiency - independent SAR scenes per phase, weeks covered, coverage
  gaps;
* Q1 - did the 2026 matching rate deviate from the historical same-period level?
* Q2 - did the AIS and SAR activity anomalies separate?
* Q3 - did the 2024 / 2025 placebo splits show the same movement?
* Technical caveats - the 2026-03-28 mask update, coverage change, and detection
  composition change.

It then applies the Strong GO / Conditional GO / No-Go screening rules as
feasibility screening only. Those thresholds are not formal significance tests,
and the report must say so.

Inputs : outputs/tables/*.csv, data/processed/*.parquet
Outputs: outputs/feasibility_report.md
"""

from __future__ import annotations

import sys

MESSAGE = (
    "src/08_feasibility_report.py is a Phase 3 step and is not implemented yet.\n"
    "Complete the Phase 1 review and Phase 2 (src/02_fetch_ais.py) first.\n"
    "See README.md section 6 for the run order."
)


def main() -> int:
    """Refuse to run until this phase is implemented."""
    print(MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
