"""OEE & Downtime Dashboard - Streamlit UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from oee import charts as C
from oee import io as oio
from oee import metrics as M
from oee.catalog import DEFAULT_MINOR_STOP_THRESHOLD_MIN, REJECT_REASONS, reason_catalog
from oee.simulator import SimConfig, simulate

SAMPLE_DIR = Path(__file__).parent / "data" / "sample"

st.set_page_config(page_title="OEE & Downtime Dashboard", page_icon="🏭", layout="wide")
st.markdown(
    """<style>
    [data-testid="stMetricValue"] {font-size: 2.0rem; color: #ff8a3d;}
    [data-testid="stMetric"] {background:#12141a; border:1px solid #262a33; border-radius:10px; padding:14px 16px;}
    .block-container {padding-top: 2.2rem;}
    h1 span.accent {color:#ff8a3d;}
    </style>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- data loading
@st.cache_data(show_spinner=False)
def load_sample() -> dict[str, pd.DataFrame]:
    if (SAMPLE_DIR / "state_events.csv").exists():
        return oio.load_folder(SAMPLE_DIR)
    return simulate(SimConfig())


@st.cache_data(show_spinner="Simulating factory...")
def load_simulated(days: int, seed: int) -> dict[str, pd.DataFrame]:
    return simulate(SimConfig(days=days, seed=seed))


@st.cache_data(show_spinner="Preparing data...")
def prepare(events: pd.DataFrame, counts: pd.DataFrame, rejects: pd.DataFrame | None,
            catalog: pd.DataFrame, threshold: float):
    ev = M.prepare_events(events, catalog, threshold)
    cn = M.prepare_counts(counts)
    rj = None
    if rejects is not None and len(rejects):
        rj = rejects.copy()
        rj = rj.join(M.assign_shift(rj["interval_start"]))
    return ev, cn, rj


def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


st.sidebar.title("🏭 OEE & Downtime")
source = st.sidebar.radio("Data source", ["Sample plant (28 days)", "Simulate a new plant", "Upload my CSVs"])
tables: dict[str, pd.DataFrame] | None = None
if source == "Sample plant (28 days)":
    tables = load_sample()
elif source == "Simulate a new plant":
    c1, c2 = st.sidebar.columns(2)
    days = c1.number_input("Days", 3, 90, 28)
    seed = c2.number_input("Seed", 0, 10_000, 7)
    tables = load_simulated(int(days), int(seed))
else:
    st.sidebar.caption("Use the schema on the *Data & schema* tab. Reject log is optional.")
    f_ev = st.sidebar.file_uploader("state_events.csv", type="csv")
    f_cn = st.sidebar.file_uploader("production_counts.csv", type="csv")
    f_rj = st.sidebar.file_uploader("reject_events.csv (optional)", type="csv")
    if f_ev and f_cn:
        tables = {"state_events": pd.read_csv(f_ev), "production_counts": pd.read_csv(f_cn)}
        if f_rj:
            tables["reject_events"] = pd.read_csv(f_rj)
        problems = oio.validate("state_events", tables["state_events"]) + \
            oio.validate("production_counts", tables["production_counts"])
        if problems:
            for p in problems:
                st.sidebar.error(p)
            tables = None

threshold = st.sidebar.slider("Minor-stop threshold (min)", 1.0, 15.0, DEFAULT_MINOR_STOP_THRESHOLD_MIN, 0.5,
                              help="Unplanned stops shorter than this count as *small stops* (Performance loss); "
                                   "longer ones are *unplanned stops* (Availability loss).")

st.markdown("<h1>OEE &amp; <span class='accent'>Downtime</span> Dashboard</h1>", unsafe_allow_html=True)

if tables is None:
    st.info("Upload **state_events.csv** and **production_counts.csv** in the sidebar, or pick the sample plant. "
            "Templates are on the *Data & schema* tab.")
    st.stop()

catalog = tables.get("reason_codes", reason_catalog())
try:
    ev_all, cn_all, rj_all = prepare(tables["state_events"], tables["production_counts"],
                                     tables.get("reject_events"), catalog, threshold)
except (ValueError, KeyError) as exc:
    st.error(f"Could not process the data: {exc}")
    st.stop()

# --------------------------------------------------------------------------- filters
dmin, dmax = ev_all["shift_date"].min().date(), ev_all["shift_date"].max().date()
dr = st.sidebar.date_input("Production days", (dmin, dmax), min_value=dmin, max_value=dmax)
d0, d1 = (dr if isinstance(dr, (tuple, list)) and len(dr) == 2 else (dmin, dmax))
lines = sorted(ev_all["line_id"].unique())
sel_lines = st.sidebar.multiselect("Lines", lines, default=lines)
machines = sorted(ev_all.loc[ev_all["line_id"].isin(sel_lines), "machine_id"].unique())
sel_machines = st.sidebar.multiselect("Machines", machines, default=machines)
shifts = list(dict.fromkeys(sorted(ev_all["shift"].unique())))
sel_shifts = st.sidebar.multiselect("Shifts", shifts, default=shifts)


def _filt(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["shift_date"].between(pd.Timestamp(d0), pd.Timestamp(d1)) & df["machine_id"].isin(sel_machines)
              & df["shift"].isin(sel_shifts)]


ev, cn = _filt(ev_all), _filt(cn_all)
rj = _filt(rj_all) if rj_all is not None else None
if ev.empty:
    st.warning("No data for the current filters.")
    st.stop()

total = M.loss_table(ev, cn).iloc[0]
st.caption(f"{len(sel_machines)} machines · {d0:%d %b %Y} – {d1:%d %b %Y} · {len(sel_shifts)} shift(s) · "
           f"{len(ev):,} state events · {int(cn['total_count'].sum()):,} units")

k = st.columns(5)
k[0].metric("OEE", pct(total["oee"]), help="Fully productive time ÷ planned production time")
k[1].metric("Availability", pct(total["availability"]), help="Run time ÷ planned production time")
k[2].metric("Performance", pct(total["performance"]), help="(Ideal cycle time × total count) ÷ run time")
k[3].metric("Quality", pct(total["quality"]), help="Good count ÷ total count")
k[4].metric("Good units", f"{int(total['good_count']):,}", f"{int(total['reject_count']):,} rejects",
            delta_color="off", delta_arrow="off")
if total["performance"] > 1.0001:
    st.warning("Performance is above 100 % - the ideal cycle times in the data are probably too slow.")

tabs = st.tabs(["📈 Overview", "🌊 Six Big Losses", "⏱️ Downtime", "🔁 Shifts", "✅ Quality",
                "🚩 Worst offenders", "🧪 What-if", "🗂️ Data & schema"])

# --------------------------------------------------------------------------- overview
with tabs[0]:
    level = st.radio("Trend by", ["line_id", "machine_id"], horizontal=True,
                     format_func=lambda s: "Line" if s == "line_id" else "Machine")
    daily = M.loss_table(ev, cn, ["shift_date", level])
    daily_total = M.loss_table(ev, cn, ["shift_date"])
    st.plotly_chart(C.oee_trend(daily.dropna(subset=["oee"]), level, daily_total.dropna(subset=["oee"]),
                                "Daily OEE (unscheduled days have no OEE)"), width="stretch")
    c1, c2 = st.columns(2)
    by_m = M.loss_table(ev, cn, ["line_id", "machine_id"])
    c1.plotly_chart(C.oee_by(by_m, "machine_id", "line_id", "OEE by machine"), width="stretch")
    by_l = M.loss_table(ev, cn, ["line_id"])
    c2.plotly_chart(C.apq_grouped(by_l, "line_id", "Availability · Performance · Quality by line"),
                    width="stretch")
    cal = M.loss_table(ev, cn, ["machine_id", "shift_date"]).dropna(subset=["oee"])
    piv = cal.pivot(index="machine_id", columns="shift_date", values="oee")
    piv.columns = [c.strftime("%a %d %b") for c in piv.columns]
    st.plotly_chart(C.heatmap(piv, "OEE calendar (machine × day)", pct=True), width="stretch")

# --------------------------------------------------------------------------- six big losses
with tabs[1]:
    scope = st.selectbox("Scope", ["All selected"] + sel_machines)
    row = total if scope == "All selected" else M.loss_table(ev[ev["machine_id"] == scope],
                                                              cn[cn["machine_id"] == scope]).iloc[0]
    wf = M.six_big_losses(row)
    st.plotly_chart(C.waterfall(wf), width="stretch")
    st.caption("All time → minus schedule loss (breaks, no orders, planned PM) = planned production time. "
               "Availability losses turn it into run time, performance losses into net run time, "
               "quality losses into fully productive time. OEE = fully productive ÷ planned.")
    tbl = wf.copy()
    tbl["share of planned"] = (tbl["hours"].abs() / max(row["planned_time_s"] / 3600, 1e-9)).map("{:.1%}".format)
    tbl["hours"] = tbl["hours"].abs().round(1)
    st.dataframe(tbl[["step", "factor", "hours", "share of planned"]], hide_index=True, width="stretch")
    st.caption(f"All time {row['all_time_s']/3600:,.0f} h · schedule loss (not counted against OEE) "
               f"{row['schedule_loss_s']/3600:,.0f} h")

# --------------------------------------------------------------------------- downtime
with tabs[2]:
    p = M.downtime_pareto(ev)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(C.pareto(p, "reason_code", "minutes", "Downtime Pareto by reason (setup + stops)"),
                    width="stretch")
    pp = p.merge(catalog[["reason_code", "description"]], on="reason_code", how="left")
    pp["hours"] = (pp["minutes"] / 60).round(1)
    pp["avg_min"] = (pp["minutes"] / pp["occurrences"]).round(1)
    c2.dataframe(pp[["reason_code", "description", "hours", "occurrences", "avg_min", "cum_pct"]],
                 hide_index=True, width="stretch",
                 column_config={"cum_pct": st.column_config.ProgressColumn("cum %", format="percent",
                                                                           min_value=0, max_value=1)})
    rel = M.mtbf_mttr(ev, ["line_id", "machine_id"])
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(C.mtbf_mttr_scatter(rel), width="stretch")
    c2.dataframe(rel[["machine_id", "failures", "mtbf_h", "mttr_min"]].round(2), hide_index=True,
                 width="stretch")
    c2.caption("Failure = unplanned stop with an equipment-failure reason (breakdown, jam, sensor fault) "
               "lasting at least the minor-stop threshold. MTBF = run time ÷ failures; MTTR = repair time ÷ failures.")
    hm = ev[ev["loss_bucket"].isin(["setup", "breakdown", "small_stop"])]
    piv = hm.pivot_table(index="reason_code", columns="machine_id", values="duration_s", aggfunc="sum",
                         fill_value=0) / 3600
    st.plotly_chart(C.heatmap(piv, "Downtime hours: reason × machine", fmt=",.1f"), width="stretch")

# --------------------------------------------------------------------------- shifts
with tabs[3]:
    by_s = M.loss_table(ev, cn, ["shift"])
    st.plotly_chart(C.apq_grouped(by_s, "shift", "Shift comparison"), width="stretch")
    c1, c2 = st.columns(2)
    per = M.loss_table(ev, cn, ["shift_date", "shift"]).dropna(subset=["oee"])
    c1.plotly_chart(C.box_by(per, "shift", "oee", "Distribution of shift OEE (one dot per shift worked)"),
                    width="stretch")
    st_tbl = by_s.assign(
        unplanned_stops=by_s["unplanned_stop_count"].astype(int),
        setup_h=(by_s["setup_s"] / 3600).round(1), breakdown_h=(by_s["breakdown_s"] / 3600).round(1),
        small_stop_h=(by_s["small_stop_s"] / 3600).round(1), reject_rate=1 - by_s["good_count"] / by_s["total_count"],
    )
    c2.dataframe(st_tbl[["shift", "oee", "availability", "performance", "quality", "unplanned_stops", "setup_h",
                         "breakdown_h", "small_stop_h", "reject_rate"]], hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent")
                                for c in ("oee", "availability", "performance", "quality", "reject_rate")})

# --------------------------------------------------------------------------- quality
with tabs[4]:
    if rj is None or rj.empty:
        st.info("No reject log loaded - upload reject_events.csv to see reject reasons.")
    else:
        qp = M.reject_pareto(rj)
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(C.pareto(qp, "reject_reason", "qty", "Reject Pareto by reason", unit="units"),
                        width="stretch")
        qp["description"] = qp["reject_reason"].map(REJECT_REASONS)
        c2.dataframe(qp[["reject_reason", "description", "qty", "cum_pct"]], hide_index=True, width="stretch",
                     column_config={"cum_pct": st.column_config.ProgressColumn("cum %", format="percent",
                                                                               min_value=0, max_value=1)})
        piv = rj.pivot_table(index="reject_reason", columns="machine_id", values="qty", aggfunc="sum", fill_value=0)
        st.plotly_chart(C.heatmap(piv, "Rejects: reason × machine"), width="stretch")
    yq = M.loss_table(ev, cn, ["line_id", "machine_id"])
    yq["first_pass_yield"] = yq["good_count"] / yq["total_count"]
    yq["startup_share_of_rejects"] = yq["startup_reject_count"] / yq["reject_count"]
    st.dataframe(yq[["machine_id", "total_count", "good_count", "reject_count", "first_pass_yield",
                     "startup_share_of_rejects"]], hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent")
                                for c in ("first_pass_yield", "startup_share_of_rejects")})

# --------------------------------------------------------------------------- worst offenders
with tabs[5]:
    gran = st.radio("Granularity", ["machine × shift", "machine × day", "line × shift"], horizontal=True)
    by = {"machine × shift": ["line_id", "machine_id", "shift_date", "shift"],
          "machine × day": ["line_id", "machine_id", "shift_date"],
          "line × shift": ["line_id", "shift_date", "shift"]}[gran]
    n = st.slider("Rows", 5, 50, 15)
    wo = M.worst_offenders(ev, cn, n=n, by=by)
    wo["shift_date"] = wo["shift_date"].dt.date
    wo["downtime_h"] = ((wo["setup_s"] + wo["breakdown_s"]) / 3600).round(2)
    st.dataframe(wo[by + ["oee", "availability", "performance", "quality", "downtime_h", "biggest_loss",
                          "biggest_loss_min"]].round({"biggest_loss_min": 0}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent")
                                for c in ("oee", "availability", "performance", "quality")})
    st.download_button("Download as CSV", wo.to_csv(index=False), "worst_offenders.csv", "text/csv")

# --------------------------------------------------------------------------- what-if
with tabs[6]:
    st.markdown("Move the sliders to see what reducing each loss would do to OEE for the current selection. "
                "Planned production time stays the same, so every point of OEE is extra good output.")
    c1, c2, c3, c4 = st.columns(4)
    setup_cut = c1.slider("Cut changeover / setup time by", 0, 100, 30, 5, format="%d%%") / 100
    bd_cut = c2.slider("Cut unplanned stops by", 0, 100, 0, 5, format="%d%%") / 100
    ss_cut = c3.slider("Cut minor stops by", 0, 100, 0, 5, format="%d%%") / 100
    scrap_cut = c4.slider("Cut scrap by", 0, 100, 0, 5, format="%d%%") / 100
    res = M.what_if(total, setup_cut, bd_cut, ss_cut, scrap_cut)
    m = st.columns(4)
    m[0].metric("Scenario OEE", pct(res.scenario.oee),
                f"{(res.scenario.oee - res.baseline.oee) * 100:+.1f} pts" if not np.isnan(res.baseline.oee) else None)
    m[1].metric("Hours reclaimed", f"{res.reclaimed_hours:,.0f} h")
    m[2].metric("Extra good units", f"{res.extra_good_units:,.0f}")
    days_n = max(1, ev["shift_date"].nunique())
    m[3].metric("Extra good units / day", f"{res.extra_good_units / days_n:,.0f}")
    st.plotly_chart(C.what_if_bars(res.baseline, res.scenario), width="stretch")
    st.caption("Assumptions: reclaimed setup / stop time runs at today's speed efficiency and today's "
               "minor-stop level; cutting scrap shrinks the reject fraction proportionally. "
               "A SMED project on changeovers is the classic first lever.")

# --------------------------------------------------------------------------- data
with tabs[7]:
    st.markdown("#### Upload schema")
    st.markdown(oio.schema_markdown())
    st.markdown("#### Reason-code catalog (how each stop is classified)")
    st.dataframe(catalog, hide_index=True, width="stretch")
    st.markdown("#### Preview of the loaded data")
    st.dataframe(tables["state_events"].head(200), hide_index=True, width="stretch")
    st.dataframe(tables["production_counts"].head(200), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.download_button("Download state_events template (first 500 rows)",
                       tables["state_events"].head(500).to_csv(index=False), "state_events.csv", "text/csv")
    c2.download_button("Download production_counts template (first 500 rows)",
                       tables["production_counts"].head(500).to_csv(index=False), "production_counts.csv", "text/csv")
