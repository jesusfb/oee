"""Render README charts, app screenshots and 1920x1080 showcase images.

Requires the dev extras:  pip install -r requirements-dev.txt  &&  python -m playwright install chromium

    python scripts/make_media.py                 # charts + hero card only
    python scripts/make_media.py --app-url http://localhost:8501   # also screenshot the running app
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oee import charts as C  # noqa: E402
from oee import io as oio  # noqa: E402
from oee import metrics as M  # noqa: E402

CHARTS = ROOT / "docs" / "charts"
SHOTS = ROOT / "docs" / "screenshots"
MEDIA = ROOT / "docs" / "media"


def build_figures():
    t = oio.load_folder(ROOT / "data" / "sample")
    ev = M.prepare_events(t["state_events"])
    cn = M.prepare_counts(t["production_counts"])
    total = M.loss_table(ev, cn).iloc[0]
    daily = M.loss_table(ev, cn, ["shift_date", "line_id"]).dropna(subset=["oee"])
    daily_t = M.loss_table(ev, cn, ["shift_date"]).dropna(subset=["oee"])
    figs = {
        "oee_trend": C.oee_trend(daily, "line_id", daily_t, "Daily OEE by line - 4 weeks of simulated MES data"),
        "six_big_losses": C.waterfall(M.six_big_losses(total), "Six Big Losses - planned time to fully productive time"),
        "downtime_pareto": C.pareto(M.downtime_pareto(ev), "reason_code", "minutes", "Downtime Pareto by reason code"),
        "shift_comparison": C.apq_grouped(M.loss_table(ev, cn, ["shift"]), "shift", "Shift comparison - A · P · Q · OEE"),
        "mtbf_mttr": C.mtbf_mttr_scatter(M.mtbf_mttr(ev, ["line_id", "machine_id"])),
    }
    return figs, total, daily_t


def _shoot_html(page, html: str, out: Path, w: int, h: int, scale: float = 1.0):
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
    page.set_viewport_size({"width": w, "height": h})
    page.goto(Path(f.name).as_uri())
    page.wait_for_timeout(1200)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(out))
    print("wrote", out.relative_to(ROOT))


def fig_page(fig, w, h, pad=24) -> str:
    fig = fig.update_layout(width=w - 2 * pad, height=h - 2 * pad)
    body = fig.to_html(full_html=False, include_plotlyjs=True, config={"displayModeBar": False})
    return (f"<html><body style='margin:0;padding:{pad}px;background:{C.BG}'>{body}</body></html>")


def hero_page(total: pd.Series, daily_t: pd.DataFrame, fig) -> str:
    fig = fig.update_layout(width=1060, height=560, title=None, margin=dict(l=50, r=20, t=30, b=40))
    chart = fig.to_html(full_html=False, include_plotlyjs=True, config={"displayModeBar": False})
    tiles = "".join(
        f"<div class='tile'><div class='lab'>{lab}</div><div class='val'>{v:.1%}</div></div>"
        for lab, v in (("OEE", total["oee"]), ("Availability", total["availability"]),
                       ("Performance", total["performance"]), ("Quality", total["quality"])))
    return f"""<html><head><style>
    body{{margin:0;background:{C.BG};color:{C.INK};font-family:{C.FONT};width:1920px;height:1080px;overflow:hidden}}
    .wrap{{display:flex;height:100%;}}
    .left{{width:760px;padding:110px 40px 80px 110px;box-sizing:border-box;display:flex;flex-direction:column}}
    .kicker{{color:{C.ACCENT};font-weight:700;letter-spacing:.18em;font-size:20px;text-transform:uppercase}}
    h1{{font-size:78px;line-height:1.02;margin:22px 0 26px;font-weight:800}}
    h1 span{{color:{C.ACCENT}}}
    p{{color:{C.INK_2};font-size:26px;line-height:1.45;margin:0 0 36px}}
    .tiles{{display:grid;grid-template-columns:1fr 1fr;gap:18px;max-width:560px}}
    .tile{{background:{C.PANEL};border:1px solid {C.GRID};border-radius:14px;padding:18px 22px}}
    .lab{{color:{C.INK_2};font-size:18px}} .val{{color:{C.ACCENT};font-size:44px;font-weight:800;margin-top:4px}}
    .right{{flex:1;display:flex;align-items:center;justify-content:center;padding-right:70px}}
    .card{{background:{C.BG};border:1px solid {C.GRID};border-radius:18px;padding:24px}}
    .foot{{margin-top:auto;color:{C.INK_2};font-size:19px}}
    </style></head><body><div class='wrap'><div class='left'>
    <div class='kicker'>Manufacturing analytics</div>
    <h1>OEE &amp; <span>Downtime</span><br>Dashboard</h1>
    <p>Simulated MES event logs → Availability × Performance × Quality, the Six Big Losses, downtime Pareto,
    MTBF/MTTR and a what-if panel.</p>
    <div class='tiles'>{tiles}</div>
    <div class='foot'>Piyush Bhamare · github.com/PiyushBhamare2111/oee-downtime-dashboard</div>
    </div><div class='right'><div class='card'>{chart}</div></div></div></body></html>"""


def explained_page(total: pd.Series) -> str:
    """Time-bucket breakdown card: all time -> fully productive time, to scale."""
    at = total["all_time_s"]
    rows = [
        ("All time", total["all_time_s"], "Schedule loss: breaks, no orders, planned PM", total["schedule_loss_s"], ""),
        ("Planned production time", total["planned_time_s"], "Availability loss: changeovers + unplanned stops",
         total["setup_s"] + total["breakdown_s"], f"A = {total['availability']:.1%}"),
        ("Run time", total["run_time_s"], "Performance loss: minor stops + slow cycles",
         total["small_stop_s"] + total["reduced_speed_s"], f"P = {total['performance']:.1%}"),
        ("Net run time", total["net_run_s"], "Quality loss: start-up + production rejects",
         total["startup_reject_s"] + total["production_reject_s"], f"Q = {total['quality']:.1%}"),
        ("Fully productive time", total["fully_productive_s"], "", 0, f"OEE = {total['oee']:.1%}"),
    ]
    colours = ["#5b616e", "#8a909c", C.FACTOR_COLOURS["Availability"], C.FACTOR_COLOURS["Performance"],
               C.FACTOR_COLOURS["Quality"]]
    bars = ""
    for i, (name, val, loss_name, loss, ratio) in enumerate(rows):
        w = 100 * (val - loss) / at  # solid = what survives to the next level
        lw = 100 * loss / at
        bars += f"""<div class='row'><div class='name'>{name}<span>{val/3600:,.0f} h</span></div>
        <div class='track'><div class='bar' style='width:{w:.2f}%;background:{C.ACCENT}'></div>
        <div class='loss' style='width:{lw:.2f}%;border-color:{colours[min(i+1,4)]}'></div></div>
        <div class='ratio'>{ratio}</div></div>
        {f"<div class='lossname' style='color:{colours[min(i+1,4)]}'>dashed = {loss_name} ({loss/3600:,.0f} h)</div>" if loss_name else ""}"""
    return f"""<html><head><style>
    body{{margin:0;background:{C.BG};color:{C.INK};font-family:{C.FONT};width:1920px;height:1080px;overflow:hidden}}
    .wrap{{padding:90px 120px}}
    .kicker{{color:{C.ACCENT};font-weight:700;letter-spacing:.18em;font-size:20px;text-transform:uppercase}}
    h1{{font-size:60px;margin:14px 0 8px;font-weight:800}} h1 span{{color:{C.ACCENT}}}
    .sub{{color:{C.INK_2};font-size:24px;margin-bottom:46px}}
    .row{{display:grid;grid-template-columns:400px 1fr 210px;align-items:center;gap:24px;margin-top:22px}}
    .name{{font-size:26px;font-weight:700}} .name span{{display:block;color:{C.INK_2};font-weight:400;font-size:20px}}
    .track{{display:flex;height:58px;background:{C.PANEL};border-radius:8px;overflow:hidden}}
    .bar{{height:100%;border-radius:8px 0 0 8px}}
    .loss{{height:100%;box-sizing:border-box;border:3px dashed;border-left:none;border-radius:0 8px 8px 0}}
    .ratio{{font-size:30px;font-weight:800;color:{C.ACCENT}}}
    .lossname{{margin:8px 0 0 424px;font-size:20px}}
    </style></head><body><div class='wrap'>
    <div class='kicker'>How OEE is calculated</div>
    <h1>From <span>planned time</span> to fully productive time</h1>
    <div class='sub'>4 weeks · 9 machines · 3 lines of simulated MES data. OEE = Availability × Performance × Quality.</div>
    {bars}</div></body></html>"""


def shoot_app(page, url: str):
    page.set_viewport_size({"width": 1920, "height": 1080})
    page.goto(url)
    page.wait_for_selector('[data-testid="stPlotlyChart"]', timeout=60_000)
    page.wait_for_timeout(4000)
    plan = [("Overview", "01_overview"), ("Six Big Losses", "02_six_big_losses"),
            ("Downtime", "03_downtime"), ("What-if", "04_what_if"), ("Worst offenders", "05_worst_offenders")]
    for tab, name in plan:
        page.get_by_role("tab", name=tab).click()
        page.wait_for_timeout(2500)
        for d in (SHOTS, MEDIA):
            d.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / f"{name}.png"))
        print("wrote", (SHOTS / f"{name}.png").relative_to(ROOT))
        if name in ("01_overview", "02_six_big_losses", "04_what_if"):
            page.screenshot(path=str(MEDIA / f"app_{name}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-url", default=None)
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    figs, total, daily_t = build_figures()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=1)
        for name, fig in figs.items():
            _shoot_html(page, fig_page(fig, 1400, 640), CHARTS / f"{name}.png", 1400, 640)
        _shoot_html(page, hero_page(total, daily_t, figs["oee_trend"]), MEDIA / "00_hero.png", 1920, 1080)
        _shoot_html(page, explained_page(total), MEDIA / "01_oee_explained.png", 1920, 1080)
        _shoot_html(page, fig_page(figs["six_big_losses"], 1920, 1080, pad=60),
                    MEDIA / "chart_six_big_losses.png", 1920, 1080)
        if args.app_url:
            shoot_app(page, args.app_url)
        browser.close()


if __name__ == "__main__":
    main()
