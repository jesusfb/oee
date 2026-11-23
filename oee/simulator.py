"""Seeded, reproducible factory data simulator.

Produces MES-style event-log tables:

* ``machines``          - master data (line, machine, ideal cycle time)
* ``state_events``      - one row per machine state interval (RUNNING / PLANNED_STOP /
                          UNPLANNED_STOP) with a reason code; intervals never cross a
                          shift boundary and tile each shift exactly
* ``production_counts`` - hourly counter snapshot per machine (total, good, rejects)
* ``reject_events``     - hourly reject quantities per machine by reject reason

The model is deliberately simple but has realistic "personality": fixed breaks,
random changeovers whose length depends on the line, competing-risk unplanned
stops with lognormal durations, slower cycles and more jams on night shift,
start-up scrap after changeovers / long stops, and one machine whose breakdown
rate drifts upwards over the period (a wearing bearing) so trends have a story.

Run ``python -m oee.simulator --help`` to generate CSV and/or SQLite output.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import (
    DEFAULT_SHIFTS,
    PLANNED_STOP,
    REJECT_REASONS,
    RUNNING,
    UNPLANNED_STOP,
    ShiftPattern,
    reason_catalog,
)

HOUR = 3600
MIN = 60


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class StopProfile:
    """Unplanned stop behaviour: mean running minutes between stops, lognormal duration."""

    mean_minutes_between: float
    median_duration_min: float
    sigma: float


BASE_STOPS: dict[str, StopProfile] = {
    "JAM": StopProfile(45, 2.0, 0.55),
    "STARVED_BLOCKED": StopProfile(140, 3.0, 0.8),
    "SENSOR_FAULT": StopProfile(180, 3.5, 0.7),
    "BREAKDOWN": StopProfile(720, 32, 0.6),
    "MATERIAL_SHORTAGE": StopProfile(540, 16, 0.6),
    "NO_OPERATOR": StopProfile(1100, 9, 0.5),
    "QUALITY_HOLD": StopProfile(1500, 28, 0.5),
}

# multipliers on stop *frequency* by shift index (0=A day, 1=B swing, 2=C night)
SHIFT_STOP_MULT: dict[int, dict[str, float]] = {
    0: {},
    1: {"MATERIAL_SHORTAGE": 1.3},
    2: {"JAM": 1.35, "NO_OPERATOR": 1.8, "SENSOR_FAULT": 1.2},
}
SHIFT_SPEED_PENALTY = {0: 0.0, 1: 0.01, 2: 0.035}  # added to the slow-cycle factor
SHIFT_DEFECT_MULT = {0: 1.0, 1: 1.05, 2: 1.25}


@dataclass(frozen=True)
class MachineSpec:
    machine_id: str
    machine_name: str
    ideal_cycle_time_s: float
    slow_factor: float = 1.06  # actual cycle = ideal * slow_factor (>= 1)
    defect_rate: float = 0.012
    stop_mult: dict[str, float] = field(default_factory=dict)
    breakdown_drift: float = 0.0  # extra breakdown-frequency multiplier reached by the last day
    reject_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LineSpec:
    line_id: str
    line_name: str
    machines: tuple[MachineSpec, ...]
    changeovers_per_shift: float = 0.6
    changeover_median_min: float = 35
    cleaning_prob_day_shift: float = 0.5


DEFAULT_PLANT: tuple[LineSpec, ...] = (
    LineSpec(
        "L1",
        "Blister Packaging",
        (
            MachineSpec("L1-THF", "Thermoformer", 0.60, 1.05, 0.010,
                        reject_weights={"SEAL_DEFECT": 5, "SURFACE_DEFECT": 2, "MISSING_COMPONENT": 3}),
            MachineSpec("L1-CTN", "Cartoner", 0.62, 1.08, 0.008, {"JAM": 1.6},
                        reject_weights={"LABEL_ERROR": 3, "MISSING_COMPONENT": 4, "DIMENSIONAL": 1}),
            MachineSpec("L1-CSP", "Case Packer", 9.0, 1.04, 0.004,
                        reject_weights={"LABEL_ERROR": 4, "DIMENSIONAL": 2}),
        ),
        changeovers_per_shift=0.7,
        changeover_median_min=38,
    ),
    LineSpec(
        "L2",
        "Bottle Filling",
        (
            MachineSpec("L2-FIL", "Filler", 0.80, 1.07, 0.009, breakdown_drift=2.5,
                        reject_weights={"DIMENSIONAL": 3, "CONTAMINATION": 2, "SEAL_DEFECT": 1}),
            MachineSpec("L2-CAP", "Capper", 0.75, 1.05, 0.011, {"SENSOR_FAULT": 1.5},
                        reject_weights={"SEAL_DEFECT": 5, "DIMENSIONAL": 1}),
            MachineSpec("L2-LBL", "Labeler", 0.70, 1.10, 0.015,
                        reject_weights={"LABEL_ERROR": 6, "SURFACE_DEFECT": 1}),
        ),
        changeovers_per_shift=0.5,
        changeover_median_min=25,
    ),
    LineSpec(
        "L3",
        "Device Assembly",
        (
            MachineSpec("L3-PRS", "Press", 4.0, 1.04, 0.014, {"BREAKDOWN": 1.4},
                        reject_weights={"DIMENSIONAL": 5, "SURFACE_DEFECT": 2}),
            MachineSpec("L3-ASM", "Assembly Cell", 6.0, 1.12, 0.018, {"MATERIAL_SHORTAGE": 1.8},
                        reject_weights={"MISSING_COMPONENT": 5, "DIMENSIONAL": 2}),
            MachineSpec("L3-LKT", "Leak Tester", 5.0, 1.03, 0.022,
                        reject_weights={"SEAL_DEFECT": 6, "CONTAMINATION": 1}),
        ),
        changeovers_per_shift=0.6,
        changeover_median_min=62,  # long changeovers: the SMED opportunity
    ),
)

# Fixed breaks: (offset from shift start in minutes, duration in minutes)
BREAKS = ((120, 10), (240, 30), (360, 10))
STARTUP_WINDOW_S = 8 * MIN          # running time after a restart that produces start-up scrap
STARTUP_SCRAP_RATE = 0.10
LONG_STOP_RESTART_MIN = 15          # unplanned stops at least this long trigger a start-up window
PM_DURATION_MIN = 120


@dataclass(frozen=True)
class SimConfig:
    start_date: str = "2026-08-31"  # a Monday
    days: int = 28
    seed: int = 42
    shifts: ShiftPattern = DEFAULT_SHIFTS
    plant: tuple[LineSpec, ...] = DEFAULT_PLANT


# --------------------------------------------------------------------------- helpers
def _lognormal_seconds(rng: np.random.Generator, median_min: float, sigma: float, floor_s: int = 20) -> int:
    return max(floor_s, int(round(rng.lognormal(np.log(median_min * MIN), sigma))))


def _completions(seg_start: int, cycle_s: float, lo: float, hi: float) -> int:
    """Parts completed in (lo, hi] by a segment that started at seg_start with cycle_s."""
    if hi <= lo:
        return 0
    return int(np.floor((hi - seg_start) / cycle_s + 1e-9)) - int(np.floor((lo - seg_start) / cycle_s + 1e-9))


def _shift_is_scheduled(weekday: int, shift_idx: int) -> bool:
    """Plant runs 24/5 plus Saturday day & swing shifts; Saturday night and Sunday are off."""
    if weekday == 6:
        return False
    if weekday == 5 and shift_idx == 2:
        return False
    return True


# --------------------------------------------------------------------------- simulator
class _MachineSim:
    def __init__(self, line: LineSpec, m: MachineSpec, cfg: SimConfig, rng: np.random.Generator, origin: pd.Timestamp):
        self.line, self.m, self.cfg, self.rng, self.origin = line, m, cfg, rng, origin
        self.events: list[tuple] = []
        n_hours = cfg.days * 24
        self.totals = np.zeros(n_hours, dtype=np.int64)
        self.startup_rej = np.zeros(n_hours, dtype=np.int64)
        self.prod_rej = np.zeros(n_hours, dtype=np.int64)
        self.rejects: dict[tuple[int, str], int] = {}
        self.startup_left = STARTUP_WINDOW_S
        w = m.reject_weights or {"DIMENSIONAL": 1}
        self.rej_codes = list(w)
        p = np.array([w[c] for c in self.rej_codes], dtype=float)
        self.rej_p = p / p.sum()

    # -- event recording ----------------------------------------------------
    def _add(self, start: int, end: int, state: str, reason: str | None) -> None:
        if end > start:
            self.events.append((start, end, state, reason))

    def _run(self, start: int, end: int, shift_idx: int, day_frac: float) -> None:
        """Record a RUNNING interval and the parts it produces, bucketed per hour."""
        if end <= start:
            return
        self._add(start, end, RUNNING, None)
        slow = self.m.slow_factor + SHIFT_SPEED_PENALTY[shift_idx] + abs(self.rng.normal(0, 0.015))
        cycle = self.m.ideal_cycle_time_s * max(1.0, slow)
        su_end = start + min(self.startup_left, end - start)
        self.startup_left -= su_end - start
        p_def = min(0.5, self.m.defect_rate * SHIFT_DEFECT_MULT[shift_idx] * (1 + 0.3 * day_frac))
        h0, h1 = start // HOUR, (end - 1) // HOUR
        for h in range(h0, h1 + 1):
            lo, hi = max(start, h * HOUR), min(end, (h + 1) * HOUR)
            n_su = _completions(start, cycle, lo, min(hi, su_end)) if lo < su_end else 0
            n_all = _completions(start, cycle, lo, hi)
            n_pr = n_all - n_su
            r_su = int(self.rng.binomial(n_su, STARTUP_SCRAP_RATE)) if n_su else 0
            r_pr = int(self.rng.binomial(n_pr, p_def)) if n_pr else 0
            self.totals[h] += n_all
            self.startup_rej[h] += r_su
            self.prod_rej[h] += r_pr
            if r_su:
                key = (h, "STARTUP_SCRAP")
                self.rejects[key] = self.rejects.get(key, 0) + r_su
            if r_pr:
                for code, q in zip(self.rej_codes, self.rng.multinomial(r_pr, self.rej_p)):
                    if q:
                        self.rejects[(h, code)] = self.rejects.get((h, code), 0) + int(q)

    # -- one shift ----------------------------------------------------------
    def shift(self, s0: int, s1: int, shift_idx: int, weekday: int, day_idx: int, pm_today: bool) -> None:
        rng, line = self.rng, self.line
        if not _shift_is_scheduled(weekday, shift_idx):
            self._add(s0, s1, PLANNED_STOP, "NO_ORDERS")
            self.startup_left = STARTUP_WINDOW_S
            return

        # planned blocks: breaks, optional PM, cleaning, changeovers
        blocks: list[tuple[int, int, str]] = [(s0 + o * MIN, s0 + (o + d) * MIN, "BREAK") for o, d in BREAKS]
        cursor = s0
        if pm_today:
            blocks.append((s0, s0 + PM_DURATION_MIN * MIN, "PLANNED_MAINT"))
        if shift_idx == 0 and rng.random() < line.cleaning_prob_day_shift:
            blocks.append((s0, s0 + _lognormal_seconds(rng, 18, 0.3), "CLEANING"))
        for _ in range(rng.poisson(line.changeovers_per_shift)):
            start = s0 + int(rng.uniform(20 * MIN, (s1 - s0) - 60 * MIN))
            blocks.append((start, start + _lognormal_seconds(rng, line.changeover_median_min, 0.35), "CHANGEOVER"))
        # resolve overlaps by pushing later blocks back; clip at shift end
        blocks.sort()
        placed: list[tuple[int, int, str]] = []
        for b0, b1, code in blocks:
            dur = b1 - b0
            b0 = max(b0, placed[-1][1] if placed else s0)
            b1 = min(b0 + dur, s1)
            if b1 > b0:
                placed.append((b0, b1, code))

        day_frac = day_idx / max(1, self.cfg.days - 1)
        for b0, b1, code in placed:
            self._fill(cursor, b0, shift_idx, day_frac)
            self._add(b0, b1, PLANNED_STOP, code)
            if code in ("CHANGEOVER", "CLEANING", "PLANNED_MAINT"):
                self.startup_left = STARTUP_WINDOW_S
            cursor = b1
        self._fill(cursor, s1, shift_idx, day_frac)

    def _stop_rates(self, shift_idx: int, day_frac: float) -> tuple[list[str], np.ndarray]:
        codes, rates = [], []
        for code, prof in BASE_STOPS.items():
            r = 1.0 / (prof.mean_minutes_between * MIN)
            r *= self.m.stop_mult.get(code, 1.0) * SHIFT_STOP_MULT[shift_idx].get(code, 1.0)
            if code == "BREAKDOWN":
                r *= 1.0 + self.m.breakdown_drift * day_frac
            codes.append(code)
            rates.append(r)
        return codes, np.array(rates)

    def _fill(self, a: int, b: int, shift_idx: int, day_frac: float) -> None:
        """Fill [a, b) with alternating RUNNING and unplanned stops (competing risks)."""
        codes, rates = self._stop_rates(shift_idx, day_frac)
        total_rate, p = rates.sum(), rates / rates.sum()
        t = a
        while t < b:
            gap = max(1, int(round(self.rng.exponential(1.0 / total_rate))))
            if t + gap >= b:
                self._run(t, b, shift_idx, day_frac)
                return
            self._run(t, t + gap, shift_idx, day_frac)
            code = codes[self.rng.choice(len(codes), p=p)]
            prof = BASE_STOPS[code]
            stop_end = min(b, t + gap + _lognormal_seconds(self.rng, prof.median_duration_min, prof.sigma))
            self._add(t + gap, stop_end, UNPLANNED_STOP, code)
            if stop_end - (t + gap) >= LONG_STOP_RESTART_MIN * MIN:
                self.startup_left = STARTUP_WINDOW_S
            t = stop_end


def simulate(cfg: SimConfig | None = None) -> dict[str, pd.DataFrame]:
    """Run the simulation and return the MES-style tables as DataFrames."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    sp = cfg.shifts
    origin = pd.Timestamp(cfg.start_date) + pd.Timedelta(hours=sp.first_start_hour)

    machines_rows, ev_frames, cnt_frames, rej_frames = [], [], [], []
    hour_index = origin + pd.to_timedelta(np.arange(cfg.days * 24), unit="h")
    pm_rotation = 0
    for line in cfg.plant:
        for m in line.machines:
            machines_rows.append(dict(line_id=line.line_id, line_name=line.line_name, machine_id=m.machine_id,
                                      machine_name=m.machine_name, ideal_cycle_time_s=m.ideal_cycle_time_s))
            sim = _MachineSim(line, m, cfg, rng, origin)
            for d in range(cfg.days):
                weekday = (origin + pd.Timedelta(days=d)).weekday()
                for k in range(sp.n_shifts):
                    s0 = (d * 24 + k * sp.shift_hours) * HOUR
                    # weekly PM on Saturday day shift; each machine every 3rd week (rotating)
                    week = d // 7
                    pm = weekday == 5 and k == 0 and (week + pm_rotation) % 3 == 0
                    sim.shift(s0, s0 + sp.shift_hours * HOUR, k, weekday, d, pm)
            pm_rotation += 1

            ev = pd.DataFrame(sim.events, columns=["start_s", "end_s", "state", "reason_code"])
            ev.insert(0, "machine_id", m.machine_id)
            ev.insert(0, "line_id", line.line_id)
            ev_frames.append(ev)

            good = sim.totals - sim.startup_rej - sim.prod_rej
            cnt_frames.append(pd.DataFrame(dict(
                line_id=line.line_id, machine_id=m.machine_id, interval_start=hour_index,
                interval_end=hour_index + pd.Timedelta(hours=1), ideal_cycle_time_s=m.ideal_cycle_time_s,
                total_count=sim.totals, good_count=good, reject_count=sim.startup_rej + sim.prod_rej,
                startup_reject_count=sim.startup_rej,
            )))
            if sim.rejects:
                rj = pd.DataFrame([(h, c, q) for (h, c), q in sorted(sim.rejects.items())],
                                  columns=["h", "reject_reason", "qty"])
                rj.insert(0, "machine_id", m.machine_id)
                rj.insert(0, "line_id", line.line_id)
                rej_frames.append(rj)

    events = pd.concat(ev_frames, ignore_index=True)
    events["start_ts"] = origin + pd.to_timedelta(events.pop("start_s"), unit="s")
    events["end_ts"] = origin + pd.to_timedelta(events.pop("end_s"), unit="s")
    events["duration_min"] = ((events["end_ts"] - events["start_ts"]).dt.total_seconds() / 60).round(4)
    events["reason_code"] = events["reason_code"].fillna("")
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    events = events[["event_id", "line_id", "machine_id", "start_ts", "end_ts", "duration_min", "state", "reason_code"]]

    rejects = pd.concat(rej_frames, ignore_index=True)
    rejects["interval_start"] = origin + pd.to_timedelta(rejects.pop("h"), unit="h")
    rejects["phase"] = np.where(rejects["reject_reason"] == "STARTUP_SCRAP", "startup", "production")
    rejects = rejects[["line_id", "machine_id", "interval_start", "reject_reason", "phase", "qty"]]

    return dict(
        machines=pd.DataFrame(machines_rows),
        state_events=events,
        production_counts=pd.concat(cnt_frames, ignore_index=True),
        reject_events=rejects,
        reason_codes=reason_catalog(),
        reject_reasons=pd.DataFrame(list(REJECT_REASONS.items()), columns=["reject_reason", "description"]),
    )


# --------------------------------------------------------------------------- output
def write_csv(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")


def write_sqlite(tables: dict[str, pd.DataFrame], db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        for name, df in tables.items():
            df.to_sql(name, con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_events_machine ON state_events(machine_id, start_ts)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_counts_machine ON production_counts(machine_id, interval_start)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate a seeded MES-style OEE dataset.")
    ap.add_argument("--days", type=int, default=28, help="number of days to simulate (default 28)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    ap.add_argument("--start", default="2026-08-31", help="first day, YYYY-MM-DD (default 2026-08-31)")
    ap.add_argument("--out", default="data/sample", help="CSV output folder (default data/sample)")
    ap.add_argument("--sqlite", default=None, help="optional SQLite file to also write, e.g. data/mes.db")
    args = ap.parse_args(argv)

    tables = simulate(SimConfig(start_date=args.start, days=args.days, seed=args.seed))
    write_csv(tables, args.out)
    print(f"Wrote CSVs to {args.out}: " + ", ".join(f"{k}({len(v)})" for k, v in tables.items()))
    if args.sqlite:
        write_sqlite(tables, args.sqlite)
        print(f"Wrote SQLite database {args.sqlite}")


if __name__ == "__main__":
    main()
