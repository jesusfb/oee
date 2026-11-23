"""Reason-code catalog, shift pattern and loss-bucket definitions.

Every stop event carries a ``state`` and a ``reason_code``. The catalog maps a
reason code to the *loss bucket* it belongs to in the Six Big Losses model:

==================  ===========================================  ==============
loss bucket         Six Big Loss (classic / modern name)          OEE factor
==================  ===========================================  ==============
schedule_loss       not a loss: excluded from planned time        (none)
setup               Setup & adjustments / Planned stops           Availability
breakdown           Equipment failure / Unplanned stops           Availability
small_stop          Idling & minor stops / Small stops            Performance
(derived)           Reduced speed / Slow cycles                   Performance
(derived)           Process defects / Production rejects          Quality
(derived)           Reduced yield / Startup rejects               Quality
==================  ===========================================  ==============

Unplanned stops shorter than the *minor-stop threshold* (default 5 min) are
re-classified from ``breakdown`` to ``small_stop`` at analysis time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RUNNING = "RUNNING"
PLANNED_STOP = "PLANNED_STOP"
UNPLANNED_STOP = "UNPLANNED_STOP"
STATES = (RUNNING, PLANNED_STOP, UNPLANNED_STOP)

# Loss buckets
SCHEDULE_LOSS = "schedule_loss"
SETUP = "setup"
BREAKDOWN = "breakdown"
SMALL_STOP = "small_stop"
RUN = "running"

DEFAULT_MINOR_STOP_THRESHOLD_MIN = 5.0


@dataclass(frozen=True)
class Reason:
    code: str
    description: str
    state: str
    loss_bucket: str
    is_failure: bool = False  # counts towards MTBF / MTTR


REASONS: tuple[Reason, ...] = (
    # Planned, excluded from planned production time
    Reason("BREAK", "Scheduled break / meal", PLANNED_STOP, SCHEDULE_LOSS),
    Reason("NO_ORDERS", "Not scheduled / no orders", PLANNED_STOP, SCHEDULE_LOSS),
    Reason("PLANNED_MAINT", "Planned preventive maintenance", PLANNED_STOP, SCHEDULE_LOSS),
    # Planned, inside planned production time (availability loss)
    Reason("CHANGEOVER", "Product / format changeover", PLANNED_STOP, SETUP),
    Reason("CLEANING", "Line clearance & cleaning", PLANNED_STOP, SETUP),
    # Unplanned stops
    Reason("BREAKDOWN", "Mechanical / electrical breakdown", UNPLANNED_STOP, BREAKDOWN, True),
    Reason("JAM", "Product jam / misfeed", UNPLANNED_STOP, BREAKDOWN, True),
    Reason("SENSOR_FAULT", "Sensor / vision-system fault", UNPLANNED_STOP, BREAKDOWN, True),
    Reason("MATERIAL_SHORTAGE", "Material / component shortage", UNPLANNED_STOP, BREAKDOWN),
    Reason("QUALITY_HOLD", "Quality hold / investigation", UNPLANNED_STOP, BREAKDOWN),
    Reason("NO_OPERATOR", "Operator unavailable", UNPLANNED_STOP, BREAKDOWN),
    Reason("STARVED_BLOCKED", "Starved upstream / blocked downstream", UNPLANNED_STOP, BREAKDOWN),
)

REJECT_REASONS: dict[str, str] = {
    "STARTUP_SCRAP": "Start-up / warm-up scrap",
    "DIMENSIONAL": "Out-of-tolerance dimension",
    "SEAL_DEFECT": "Seal / weld defect",
    "SURFACE_DEFECT": "Surface / cosmetic defect",
    "LABEL_ERROR": "Label / print / code error",
    "MISSING_COMPONENT": "Missing component",
    "CONTAMINATION": "Foreign matter / contamination",
}


def reason_catalog() -> pd.DataFrame:
    """Return the reason-code catalog as a DataFrame."""
    return pd.DataFrame([r.__dict__ for r in REASONS]).rename(columns={"code": "reason_code"})


@dataclass(frozen=True)
class ShiftPattern:
    """A repeating pattern of equal-length shifts starting at ``first_start_hour``."""

    first_start_hour: int = 6
    shift_hours: int = 8
    names: tuple[str, ...] = ("A (Day)", "B (Swing)", "C (Night)")

    @property
    def n_shifts(self) -> int:
        return len(self.names)

    def __post_init__(self) -> None:
        if self.shift_hours * self.n_shifts != 24:
            raise ValueError("shift_hours * number of shifts must equal 24")


DEFAULT_SHIFTS = ShiftPattern()
