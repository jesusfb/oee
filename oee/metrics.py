"""OEE, Six Big Losses, downtime and reliability metrics.

Definitions (all times in seconds unless the column name says otherwise)::

    All time                = sum of all state-event durations in scope
    Planned production time = All time - schedule loss (breaks, no orders, planned PM)
    Run time                = Planned production time - setup (changeover/cleaning)
                              - unplanned stops >= minor-stop threshold
    Net run time            = ideal cycle time x total count
    Fully productive time   = ideal cycle time x good count

    Availability = Run time / Planned production time
    Performance  = Net run time / Run time       (= ideal CT x total / run time)
    Quality      = Fully productive / Net run    (= good / total for a single machine)
    OEE          = A x P x Q = Fully productive time / Planned production time

When several machines with different ideal cycle times are aggregated, Quality
is weighted by ideal time so that A x P x Q stays exactly equal to
fully-productive / planned time (the standard "sum the times, then divide" rule).
Any ratio whose denominator is zero is returned as NaN, never 0 or 100 %.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .catalog import (
    BREAKDOWN,
    DEFAULT_MINOR_STOP_THRESHOLD_MIN,
    DEFAULT_SHIFTS,
    PLANNED_STOP,
    RUN,
    RUNNING,
    SCHEDULE_LOSS,
    SETUP,
    SMALL_STOP,
    UNPLANNED_STOP,
    ShiftPattern,
    reason_catalog,
)

SIX_BIG_LOSSES = (
    ("setup_s", "Setup & adjustments (planned stops)", "Availability"),
    ("breakdown_s", "Equipment failure (unplanned stops)", "Availability"),
    ("small_stop_s", "Idling & minor stops", "Performance"),
    ("reduced_speed_s", "Reduced speed (slow cycles)", "Performance"),
    ("production_reject_s", "Process defects (production rejects)", "Quality"),
    ("startup_reject_s", "Reduced yield (start-up rejects)", "Quality"),
)


# --------------------------------------------------------------------------- core math
def safe_div(num, den):
    """Element-wise num/den that returns NaN where den == 0 (works for scalars and arrays)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den == 0, np.nan, num / np.where(den == 0, 1, den))
    return out.item() if out.ndim == 0 else out


@dataclass(frozen=True)
class OEEResult:
    availability: float
    performance: float
    quality: float
    oee: float


def oee_factors(planned_time_s: float, run_time_s: float, ideal_cycle_time_s: float,
                total_count: float, good_count: float) -> OEEResult:
    """OEE for a single machine/period from its five base quantities.

    >>> r = oee_factors(28800 - 3600, 25200 - 2820, 1.0, 19271, 18848)
    """
    if min(planned_time_s, run_time_s, ideal_cycle_time_s, total_count, good_count) < 0:
        raise ValueError("inputs must be non-negative")
    if run_time_s > planned_time_s:
        raise ValueError("run time cannot exceed planned production time")
    if good_count > total_count:
        raise ValueError("good count cannot exceed total count")
    a = safe_div(run_time_s, planned_time_s)
    p = safe_div(ideal_cycle_time_s * total_count, run_time_s)
    q = safe_div(good_count, total_count)
    oee = safe_div(ideal_cycle_time_s * good_count, planned_time_s)
    return OEEResult(a, p, q, oee)


# --------------------------------------------------------------------------- shifts
def assign_shift(ts: pd.Series, shifts: ShiftPattern = DEFAULT_SHIFTS) -> pd.DataFrame:
    """Return shift_date (production day the shift *started* on), shift name and shift start."""
    ts = pd.to_datetime(ts)
    t0 = ts - pd.Timedelta(hours=shifts.first_start_hour)
    shift_date = t0.dt.normalize()
    idx = ((t0 - shift_date) // pd.Timedelta(hours=shifts.shift_hours)).astype(int)
    names = np.array(shifts.names, dtype=object)
    shift_start = shift_date + pd.Timedelta(hours=shifts.first_start_hour) + pd.to_timedelta(idx * shifts.shift_hours, unit="h")
    return pd.DataFrame({"shift_date": shift_date, "shift": names[idx.to_numpy()], "shift_start": shift_start},
                        index=ts.index)


def split_at_shift_boundaries(events: pd.DataFrame, shifts: ShiftPattern = DEFAULT_SHIFTS) -> pd.DataFrame:
    """Split any event that spans a shift boundary into one piece per shift."""
    ev = events.reset_index(drop=True)
    ss = assign_shift(ev["start_ts"], shifts)["shift_start"]
    boundary = ss + pd.Timedelta(hours=shifts.shift_hours)
    crossing = ev["end_ts"] > boundary
    if not crossing.any():
        return ev
    keep = [ev[~crossing]]
    pieces = []
    step = pd.Timedelta(hours=shifts.shift_hours)
    for (_, row), b in zip(ev[crossing].iterrows(), boundary[crossing]):
        start, end = row["start_ts"], row["end_ts"]
        while start < end:
            piece = row.copy()
            piece["start_ts"], piece["end_ts"] = start, min(end, b)
            pieces.append(piece)
            start, b = b, b + step
    keep.append(pd.DataFrame(pieces))
    return pd.concat(keep, ignore_index=True).sort_values(["machine_id", "start_ts"], ignore_index=True)


# --------------------------------------------------------------------------- preparation
def prepare_events(events: pd.DataFrame, catalog: pd.DataFrame | None = None,
                   minor_stop_threshold_min: float = DEFAULT_MINOR_STOP_THRESHOLD_MIN,
                   shifts: ShiftPattern = DEFAULT_SHIFTS) -> pd.DataFrame:
    """Normalise a state-event table and tag each event with its loss bucket and shift.

    Unknown planned reasons are treated as *setup* (they stay inside planned time,
    the conservative choice); unknown unplanned reasons are treated as stops.
    """
    catalog = reason_catalog() if catalog is None else catalog
    ev = events.reset_index(drop=True)
    ev["start_ts"] = pd.to_datetime(ev["start_ts"])
    ev["end_ts"] = pd.to_datetime(ev["end_ts"])
    ev["state"] = ev["state"].astype(str).str.strip().str.upper()
    ev["reason_code"] = ev["reason_code"].fillna("").astype(str).str.strip().str.upper()
    bad = ev["end_ts"] < ev["start_ts"]
    if bad.any():
        raise ValueError(f"{int(bad.sum())} events have end_ts before start_ts")
    unknown_states = set(ev["state"]) - {RUNNING, PLANNED_STOP, UNPLANNED_STOP}
    if unknown_states:
        raise ValueError(f"unknown state values: {sorted(unknown_states)}")

    ev = split_at_shift_boundaries(ev, shifts)
    ev["duration_s"] = (ev["end_ts"] - ev["start_ts"]).dt.total_seconds()
    ev = ev.join(assign_shift(ev["start_ts"], shifts))

    cat = catalog.set_index("reason_code")
    bucket = ev["reason_code"].map(cat["loss_bucket"])
    ev["is_failure"] = ev["reason_code"].map(cat["is_failure"]).fillna(False).astype(bool)
    ev["loss_bucket"] = np.select(
        [ev["state"] == RUNNING,
         ev["state"] == PLANNED_STOP,
         (ev["state"] == UNPLANNED_STOP) & (ev["duration_s"] < minor_stop_threshold_min * 60)],
        [RUN, bucket.fillna(SETUP), SMALL_STOP],
        default=BREAKDOWN,
    )
    # a planned code logged as unplanned (or vice-versa) still follows its state for planned-ness
    planned_bucket_on_unplanned = (ev["state"] == UNPLANNED_STOP) & ev["loss_bucket"].isin([SCHEDULE_LOSS, SETUP])
    ev.loc[planned_bucket_on_unplanned, "loss_bucket"] = BREAKDOWN
    return ev


def prepare_counts(counts: pd.DataFrame, shifts: ShiftPattern = DEFAULT_SHIFTS) -> pd.DataFrame:
    """Normalise a production-count table and add ideal-time columns."""
    c = counts.reset_index(drop=True)
    c["interval_start"] = pd.to_datetime(c["interval_start"])
    if "good_count" not in c and "reject_count" in c:
        c["good_count"] = c["total_count"] - c["reject_count"]
    if "reject_count" not in c:
        c["reject_count"] = c["total_count"] - c["good_count"]
    if "startup_reject_count" not in c:
        c["startup_reject_count"] = 0
    for col in ("total_count", "good_count", "reject_count", "startup_reject_count", "ideal_cycle_time_s"):
        c[col] = pd.to_numeric(c[col], errors="raise").fillna(0)
    if (c["good_count"] > c["total_count"]).any():
        raise ValueError("good_count exceeds total_count in some rows")
    c = c.join(assign_shift(c["interval_start"], shifts))
    ict = c["ideal_cycle_time_s"]
    c["net_run_s"] = ict * c["total_count"]
    c["fully_productive_s"] = ict * c["good_count"]
    c["startup_reject_s"] = ict * c["startup_reject_count"]
    c["production_reject_s"] = ict * (c["reject_count"] - c["startup_reject_count"])
    return c


# --------------------------------------------------------------------------- aggregation
def _keys(by: Sequence[str] | None) -> list[str]:
    return list(by) if by else []


def loss_table(ev: pd.DataFrame, cnt: pd.DataFrame, by: Sequence[str] | None = None) -> pd.DataFrame:
    """Time-loss accounting and OEE factors per group (prepared inputs required).

    Returns one row per group with the full waterfall (seconds), counts and the
    four ratios. ``by=None`` returns a single total row.
    """
    keys = _keys(by)
    e, c = ev.assign(_all=1), cnt.assign(_all=1)
    gk = keys or ["_all"]
    t = e.pivot_table(index=gk, columns="loss_bucket", values="duration_s", aggfunc="sum", fill_value=0.0)
    for b in (RUN, SCHEDULE_LOSS, SETUP, BREAKDOWN, SMALL_STOP):
        if b not in t:
            t[b] = 0.0
    t = t[[RUN, SCHEDULE_LOSS, SETUP, BREAKDOWN, SMALL_STOP]].rename(columns=lambda b: f"{b}_s")
    n_stops = e[e["loss_bucket"].isin([BREAKDOWN, SMALL_STOP])].groupby(gk).size().rename("unplanned_stop_count")
    cc = c.groupby(gk)[["total_count", "good_count", "reject_count", "startup_reject_count", "net_run_s",
                        "fully_productive_s", "startup_reject_s", "production_reject_s"]].sum()
    out = t.join(n_stops, how="outer").join(cc, how="outer").fillna(0.0)

    out["all_time_s"] = out[["running_s", "schedule_loss_s", "setup_s", "breakdown_s", "small_stop_s"]].sum(axis=1)
    out["planned_time_s"] = out["all_time_s"] - out["schedule_loss_s"]
    out["run_time_s"] = out["planned_time_s"] - out["setup_s"] - out["breakdown_s"]
    out["reduced_speed_s"] = out["run_time_s"] - out["small_stop_s"] - out["net_run_s"]
    out["availability"] = safe_div(out["run_time_s"], out["planned_time_s"])
    out["performance"] = safe_div(out["net_run_s"], out["run_time_s"])
    out["quality"] = safe_div(out["fully_productive_s"], out["net_run_s"])
    out["oee"] = safe_div(out["fully_productive_s"], out["planned_time_s"])
    out = out.reset_index()
    return out.drop(columns="_all") if not keys else out


def six_big_losses(row: pd.Series) -> pd.DataFrame:
    """Waterfall steps (hours) from planned production time down to fully productive time."""
    rows = [dict(step="Planned production time", hours=row["planned_time_s"] / 3600, kind="total", factor="")]
    for col, name, factor in SIX_BIG_LOSSES:
        rows.append(dict(step=name, hours=-row[col] / 3600, kind="loss", factor=factor))
    rows.append(dict(step="Fully productive time", hours=row["fully_productive_s"] / 3600, kind="total", factor=""))
    return pd.DataFrame(rows)


def downtime_pareto(ev: pd.DataFrame, by: str = "reason_code",
                    buckets: Iterable[str] = (SETUP, BREAKDOWN, SMALL_STOP)) -> pd.DataFrame:
    """Downtime minutes and occurrences by reason (or any column), sorted, with cumulative %."""
    d = ev[ev["loss_bucket"].isin(list(buckets))]
    p = d.groupby(by).agg(minutes=("duration_s", lambda s: s.sum() / 60), occurrences=("duration_s", "size"))
    p = p.sort_values("minutes", ascending=False).reset_index()
    total = p["minutes"].sum()
    p["pct"] = safe_div(p["minutes"], total) if len(p) else []
    p["cum_pct"] = p["pct"].cumsum() if len(p) else []
    return p


def mtbf_mttr(ev: pd.DataFrame, by: Sequence[str] | None = ("machine_id",)) -> pd.DataFrame:
    """MTBF (run hours per failure) and MTTR (minutes per failure).

    A *failure* is an unplanned stop whose reason is flagged ``is_failure`` and whose
    duration is at or above the minor-stop threshold (shorter ones are small stops).
    """
    gk = _keys(by) or ["_all"]
    e = ev.assign(_all=1)
    run = e.groupby(gk).apply(
        lambda g: g.loc[g["loss_bucket"].isin([RUN, SMALL_STOP]), "duration_s"].sum(), include_groups=False
    ).rename("run_time_s")
    f = e[(e["loss_bucket"] == BREAKDOWN) & e["is_failure"]]
    fail = f.groupby(gk).agg(failures=("duration_s", "size"), repair_s=("duration_s", "sum"))
    out = pd.DataFrame(run).join(fail, how="left").fillna({"failures": 0, "repair_s": 0.0})
    out["failures"] = out["failures"].astype(int)
    out["mtbf_h"] = safe_div(out["run_time_s"] / 3600, out["failures"])
    out["mttr_min"] = safe_div(out["repair_s"] / 60, out["failures"])
    out = out.reset_index()
    return out.drop(columns="_all") if gk == ["_all"] else out


def reject_pareto(rejects: pd.DataFrame) -> pd.DataFrame:
    p = rejects.groupby("reject_reason", as_index=False)["qty"].sum().sort_values("qty", ascending=False)
    p["pct"] = safe_div(p["qty"], p["qty"].sum()) if len(p) else []
    p["cum_pct"] = p["pct"].cumsum() if len(p) else []
    return p.reset_index(drop=True)


def top_loss(row: pd.Series) -> str:
    vals = {name: row[col] for col, name, _ in SIX_BIG_LOSSES}
    return max(vals, key=vals.get)


def worst_offenders(ev: pd.DataFrame, cnt: pd.DataFrame, n: int = 15,
                    by: Sequence[str] = ("line_id", "machine_id", "shift_date", "shift")) -> pd.DataFrame:
    """The n lowest-OEE scheduled groups with their single biggest loss."""
    lt = loss_table(ev, cnt, by)
    lt = lt[lt["planned_time_s"] > 0].sort_values("oee").head(n).copy()
    lt["biggest_loss"] = lt.apply(top_loss, axis=1)
    lt["biggest_loss_min"] = lt.apply(lambda r: max(r[c] for c, _, _ in SIX_BIG_LOSSES) / 60, axis=1)
    return lt


# --------------------------------------------------------------------------- what-if
@dataclass(frozen=True)
class WhatIfResult:
    baseline: OEEResult
    scenario: OEEResult
    reclaimed_hours: float
    extra_good_units: float


def what_if(row: pd.Series, setup_cut: float = 0.0, breakdown_cut: float = 0.0,
            small_stop_cut: float = 0.0, scrap_cut: float = 0.0) -> WhatIfResult:
    """Re-compute OEE after cutting losses by the given fractions (0..1).

    Assumptions (kept deliberately transparent):
    * time removed from setup or breakdowns becomes run time; it contains small stops
      at today's absolute level and runs at today's *speed efficiency while running*
      (net run time / (run time - small stops));
    * cutting small stops converts that time into running at the same efficiency;
    * cutting scrap reduces the reject fraction proportionally.
    Planned production time is unchanged, so OEE gains translate to extra output.
    """
    for v in (setup_cut, breakdown_cut, small_stop_cut, scrap_cut):
        if not 0 <= v <= 1:
            raise ValueError("cuts must be fractions between 0 and 1")
    ppt, rt, small, nrt = row["planned_time_s"], row["run_time_s"], row["small_stop_s"], row["net_run_s"]
    a0, p0, q0, oee0 = row["availability"], row["performance"], row["quality"], row["oee"]
    base = OEEResult(a0, p0, q0, oee0)
    if ppt <= 0 or rt <= 0:
        return WhatIfResult(base, base, 0.0, 0.0)

    speed_eff = safe_div(nrt, rt - small)
    speed_eff = 0.0 if np.isnan(speed_eff) else speed_eff
    reclaimed = row["setup_s"] * setup_cut + row["breakdown_s"] * breakdown_cut
    rt1 = rt + reclaimed
    small1 = small * (1 - small_stop_cut)
    nrt1 = speed_eff * (rt1 - small1)
    q0_ = 0.0 if np.isnan(q0) else q0
    q1 = 1 - (1 - q0_) * (1 - scrap_cut) if nrt > 0 else q0
    a1, p1 = safe_div(rt1, ppt), safe_div(nrt1, rt1)
    oee1 = a1 * p1 * q1
    fpt0 = row["fully_productive_s"]
    avg_ict = safe_div(row["net_run_s"], row["total_count"])
    extra = safe_div(oee1 * ppt - fpt0, avg_ict)
    return WhatIfResult(base, OEEResult(a1, p1, q1, oee1), reclaimed / 3600 + small * small_stop_cut / 3600,
                        0.0 if np.isnan(extra) else float(extra))
