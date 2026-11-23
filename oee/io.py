"""Loading, schema documentation and validation for the MES-style tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .catalog import reason_catalog

# column -> description. These are the columns the dashboard needs.
SCHEMA: dict[str, dict[str, str]] = {
    "state_events": {
        "line_id": "Line identifier, e.g. L1",
        "machine_id": "Machine identifier, unique across the plant, e.g. L1-CTN",
        "start_ts": "Interval start, 'YYYY-MM-DD HH:MM:SS' (local plant time)",
        "end_ts": "Interval end (exclusive), same format",
        "state": "RUNNING | PLANNED_STOP | UNPLANNED_STOP",
        "reason_code": "Stop reason (blank for RUNNING), see reason_codes",
    },
    "production_counts": {
        "line_id": "Line identifier",
        "machine_id": "Machine identifier",
        "interval_start": "Start of the counting interval (e.g. hourly)",
        "ideal_cycle_time_s": "Ideal (nameplate / validated) cycle time per unit, seconds",
        "total_count": "All units produced in the interval",
        "good_count": "Units that passed first time (good = total - rejects)",
    },
}
OPTIONAL: dict[str, dict[str, str]] = {
    "production_counts": {
        "reject_count": "Rejected units (derived as total - good if missing)",
        "startup_reject_count": "Rejects made while warming up after a restart (default 0)",
        "interval_end": "End of the counting interval (informational)",
    },
    "reject_events": {
        "line_id": "Line identifier", "machine_id": "Machine identifier",
        "interval_start": "Counting interval", "reject_reason": "Reject reason code",
        "phase": "startup | production", "qty": "Number of rejected units",
    },
}


def validate(name: str, df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty when the table is usable)."""
    problems = []
    required = SCHEMA.get(name, OPTIONAL.get(name, {}))
    missing = [c for c in required if c not in df.columns]
    if name == "production_counts" and "good_count" in missing and "reject_count" in df.columns:
        missing.remove("good_count")
    if missing:
        problems.append(f"{name}: missing column(s) {', '.join(missing)}")
        return problems
    if df.empty:
        problems.append(f"{name}: table is empty")
    if name == "state_events":
        for col in ("start_ts", "end_ts"):
            if pd.to_datetime(df[col], errors="coerce").isna().any():
                problems.append(f"{name}: some {col} values are not valid timestamps")
        bad_state = set(df["state"].astype(str).str.upper().str.strip()) - {"RUNNING", "PLANNED_STOP", "UNPLANNED_STOP"}
        if bad_state:
            problems.append(f"{name}: unknown state value(s) {sorted(bad_state)}")
    if name == "production_counts":
        for col in ("total_count", "ideal_cycle_time_s"):
            if pd.to_numeric(df[col], errors="coerce").isna().any():
                problems.append(f"{name}: non-numeric values in {col}")
    return problems


def load_folder(folder: str | Path) -> dict[str, pd.DataFrame]:
    folder = Path(folder)
    tables = {}
    for name in ("state_events", "production_counts", "reject_events", "reason_codes", "machines"):
        p = folder / f"{name}.csv"
        if p.exists():
            tables[name] = pd.read_csv(p)
    tables.setdefault("reason_codes", reason_catalog())
    return tables


def load_sqlite(db_path: str | Path) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_path) as con:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {n: pd.read_sql(f"SELECT * FROM {n}", con) for n in names}


def schema_markdown() -> str:
    lines = []
    for name, cols in {**SCHEMA}.items():
        lines.append(f"**{name}.csv** (required)\n\n| column | meaning |\n|---|---|")
        lines += [f"| `{c}` | {d} |" for c, d in cols.items()]
        for c, d in OPTIONAL.get(name, {}).items():
            lines.append(f"| `{c}` *(optional)* | {d} |")
        lines.append("")
    lines.append("**reject_events.csv** (optional)\n\n| column | meaning |\n|---|---|")
    lines += [f"| `{c}` | {d} |" for c, d in OPTIONAL["reject_events"].items()]
    return "\n".join(lines)
