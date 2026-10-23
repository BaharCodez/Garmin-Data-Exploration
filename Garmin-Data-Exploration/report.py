"""Statistical analysis + static HTML research dashboard from the Spark ETL outputs.

Usage: python report.py <data dir> <output html>
"""
import html
import json
import os
import re
import sys
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm

# --public: portfolio-safe version (no calendar dates of periods, no per-day data points)
PUBLIC = "--public" in sys.argv
DATA, OUT_HTML = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
RACE = pd.Timestamp("2026-05-24")
BASE_START, BLOCK_START = pd.Timestamp("2025-08-25"), RACE - pd.Timedelta(weeks=10)

# Palette (validated reference palette, light mode): categorical slots in fixed order.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND = "rgba(137,135,129,0.10)"
SEQ = [[0,"#cde2fb"],[0.25,"#86b6ef"],[0.5,"#3987e5"],[0.75,"#1c5cab"],[1,"#0d366b"]]
DIV = [[0,"#c0392b"],[0.25,"#ec835a"],[0.5,"#f0efec"],[0.75,"#5598e7"],[1,"#184f95"]]
PHASES = ["Menstrual", "Follicular", "Ovulatory", "Luteal"]

d = pd.read_parquet(f"{DATA}/daily.parquet")
r = pd.read_parquet(f"{DATA}/runs.parquet")
for df in (d, r):
    df["date"] = pd.to_datetime(df["date"])
r["week"] = pd.to_datetime(r["week"])
d = d.sort_values("date").set_index("date").asfreq("D").reset_index()


def detrend(s, window=35):
    """Deviation from a centred rolling median: removes slow fitness/season drift
    while keeping within-cycle (~32 d) variation."""
    return s - s.rolling(window, center=True, min_periods=15).median()


d["hrv_dev"] = detrend(d["hrv"])
d["rhr_dev"] = detrend(d["sleep_rhr"])
d["score_dev"] = detrend(d["sleep_score"])
d["low_hrv"] = d["hrv"] < d["hrv_base_lo"]
d["wake_h"] = pd.to_datetime(d["waketime"]).dt.hour + pd.to_datetime(d["waketime"]).dt.minute / 60


def spear(x, y):
    m = pd.concat([x, y], axis=1).dropna()
    n = len(m)
    if n < 5:
        return dict(rho=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=n)
    rho, p = stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1])
    z, se = np.arctanh(rho), 1 / np.sqrt(n - 3)
    return dict(rho=rho, lo=np.tanh(z - 1.96 * se), hi=np.tanh(z + 1.96 * se), p=p, n=n)


def fmt_p(p):
    return "—" if pd.isna(p) else ("&lt;0.001" if p < 0.001 else f"{p:.3f}")


# ---------------------------------------------------------------- figure helpers
FIG_N = [0]
sections = []


def style(fig, height=380, ytitle=None, xtitle=None, legend=True):
    fig.update_layout(
        height=height, template="simple_white", paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color=INK2),
        margin=dict(l=60, r=20, t=30, b=50), showlegend=legend,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="#fcfcfb", font_color=INK, bordercolor=AXIS))
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickcolor=AXIS, tickfont=dict(color=MUTED),
                     title_text=xtitle, title_font=dict(color=INK2))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=AXIS, tickcolor=AXIS,
                     tickfont=dict(color=MUTED), title_text=ytitle, title_font=dict(color=INK2), zeroline=False)
    return fig


def shade_training(fig, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    for x0, x1, label in [(BASE_START, BLOCK_START, "Base build"), (BLOCK_START, RACE, "10-week block")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=BAND, line_width=0, layer="below",
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=11, color=MUTED), **kw)
    fig.add_vline(x=RACE, line=dict(color=MUTED, width=1, dash="dot"), **kw)


MD_FIGS, MD_TBLS = {}, []
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(OUT_HTML)), "images")


def figure(fig, caption):
    FIG_N[0] += 1
    n = FIG_N[0]
    div = plotly.offline.plot(fig, include_plotlyjs=False, output_type="div",
                              config={"displaylogo": False, "responsive": True})
    if PUBLIC:  # static copy for the Markdown version
        os.makedirs(IMG_DIR, exist_ok=True)
        fig.write_image(f"{IMG_DIR}/figure-{n:02d}.png", width=1000, height=fig.layout.height, scale=2)
        MD_FIGS[n] = f"![Figure {n}](images/figure-{n:02d}.png)\n\n*Figure {n}. {md_text(caption)}*\n\n"
    return f'<figure><!--FIG:{n}-->{div}<figcaption><b>Figure {n}.</b> {caption}</figcaption></figure>'


def md_text(t):
    return html.unescape(re.sub(r"<[^>]+>", "", t))


def table(df, caption=None):
    MD_TBLS.append(((f"*{md_text(caption)}*\n\n" if caption else "") +
                    df.astype(str).replace({"nan": "—", "None": "—"}).to_markdown(index=False).replace("&lt;", "<")))
    cap = f"<!--TBL:{len(MD_TBLS) - 1}--><caption>{caption}</caption>" if caption else f"<!--TBL:{len(MD_TBLS) - 1}-->"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in df.astype(str).replace({"nan": "—", "None": "—"}).values)
    return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def corr_table(pairs, caption):
    rows = []
    for label_x, label_y, x, y in pairs:
        s = spear(x, y)
        rows.append([label_x, label_y, f"{s['rho']:+.2f}", f"[{s['lo']:+.2f}, {s['hi']:+.2f}]", fmt_p(s["p"]), s["n"]])
    return table(pd.DataFrame(rows, columns=["Variable X", "Variable Y", "Spearman ρ", "95% CI", "p", "n"]), caption)


def forest(pairs, title_x="Spearman ρ (95% CI)", height=None):
    labels, rho, lo, hi, txt = [], [], [], [], []
    for label, x, y in pairs:
        s = spear(x, y)
        labels.append(label); rho.append(s["rho"]); lo.append(s["rho"] - s["lo"]); hi.append(s["hi"] - s["rho"])
        txt.append(f"ρ = {s['rho']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]<br>p = {fmt_p(s['p']).replace('&lt;', '<')}, n = {s['n']}")
    fig = go.Figure(go.Scatter(x=rho, y=labels, mode="markers", marker=dict(size=10, color=S1),
                               error_x=dict(type="data", array=hi, arrayminus=lo, color=S1, thickness=2, width=0),
                               hovertext=txt, hoverinfo="text"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    style(fig, height=height or 60 + 42 * len(pairs), xtitle=title_x, legend=False)
    fig.update_xaxes(range=[-1, 1], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


# ================================================================ 0. Overview
nights = d["sleep_score"].notna().sum()
hrv_n = d["hrv"].notna().sum()
completed_cycles = d.dropna(subset=["cycle_len"])["cycle_id"].nunique()
overlap_cycles = d.dropna(subset=["hrv", "cycle_len"])["cycle_id"].nunique()
tiles = [
    ("Nights of sleep data", f"{nights}", f"{d.loc[d.sleep_score.notna(), 'date'].min():%b %Y} – {d.loc[d.sleep_score.notna(), 'date'].max():%b %Y}"),
    ("Mean sleep / night", f"{d.total_sleep_h.mean():.1f} h", f"SD {d.total_sleep_h.std():.1f} h"),
    ("Mean overnight HRV", f"{d.hrv.mean():.0f} ms", f"{hrv_n} nights, from {d.loc[d.hrv.notna(), 'date'].min():%b %Y}"),
    ("Menstrual cycles with HRV", f"{overlap_cycles}", "completed cycles overlapping HRV data"),
    ("Runs analysed", f"{len(r)}", f"{r.distance_km.sum():.0f} km total"),
]
tiles_html = "".join(f'<div class="tile"><div class="tl">{a}</div><div class="tv">{b}</div><div class="ts">{c}</div></div>' for a, b, c in tiles)

# ================================================================ 1. Sleep since getting the Garmin
s = d.dropna(subset=["sleep_score"]).copy()
fig = go.Figure()
fig.add_scatter(x=s.date, y=s.total_sleep_h, mode="markers", name="Nightly", marker=dict(size=4, color=S1, opacity=0.25),
                hovertemplate="%{x|%a %d %b %Y}<br>%{y:.1f} h<extra></extra>")
fig.add_scatter(x=d.date, y=d.total_sleep_h.rolling(14, min_periods=7, center=True).mean(), mode="lines",
                name="14-day mean", line=dict(color=S1, width=2), hovertemplate="%{x|%d %b %Y}<br>14-d mean %{y:.2f} h<extra></extra>")
shade_training(fig)
style(fig, ytitle="Total sleep (h)")
f1 = figure(fig, "Total sleep time per night (light dots) and centred 14-day rolling mean. Grey bands mark the marathon base build and 10-week block; dotted line is race day.")

m = s.groupby(s.date.dt.to_period("M")).agg(Deep=("deep_h", "mean"), Light=("light_h", "mean"), REM=("rem_h", "mean"),
                                             Awake=("awake_h", "mean"), score=("sleep_score", "mean"), n=("date", "size"))
m = m[m.n >= 10]
mx = m.index.to_timestamp()
fig = go.Figure()
for col, c in zip(["Deep", "Light", "REM", "Awake"], [S1, S2, S3, S4]):
    fig.add_bar(x=mx, y=m[col], name=col, marker=dict(color=c, line=dict(color="#fcfcfb", width=2)),
                hovertemplate="%{x|%b %Y}<br>" + col + ": %{y:.2f} h<extra></extra>")
fig.update_layout(barmode="stack", bargap=0.25)
style(fig, ytitle="Mean hours per night")
f2 = figure(fig, "Monthly mean sleep architecture (hours per night by stage). Months with &lt;10 recorded nights are excluded.")

stage_tbl = m.copy()
stage_tbl.index = stage_tbl.index.strftime("%b %Y")
stage_tbl = stage_tbl.assign(Total=stage_tbl[["Deep", "Light", "REM"]].sum(axis=1)).round(2).reset_index(names="Month")
stage_tbl["score"] = stage_tbl["score"].round(0).astype(int)
stage_tbl = stage_tbl.rename(columns={"score": "Sleep score", "n": "Nights"})[["Month", "Nights", "Total", "Deep", "Light", "REM", "Awake", "Sleep score"]]

# weekday pattern
wd_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
night_of = (s.date - pd.Timedelta(days=1)).dt.strftime("%a")  # the night *starting* on this weekday
wd = s.assign(night=night_of).groupby("night").agg(h=("total_sleep_h", "mean"), sc=("sleep_score", "mean"),
                                                    bed=("bedtime_h", "mean"), n=("date", "size")).reindex(wd_order)
fig = go.Figure(go.Bar(x=wd.index, y=wd.h, marker=dict(color=S1, cornerradius=4),
                       customdata=np.c_[wd.sc, wd.bed % 24, wd.n],
                       hovertemplate="Night of %{x}<br>%{y:.2f} h<br>Sleep score %{customdata[0]:.0f}<br>Mean bedtime %{customdata[1]:.2f} h<br>n = %{customdata[2]}<extra></extra>"))
style(fig, height=300, ytitle="Mean total sleep (h)", xtitle="Night beginning on", legend=False)
fig.update_yaxes(range=[0, max(wd.h) * 1.15])
f3 = figure(fig, "Mean total sleep by the weekday on which the night begins.")

# calendar heatmap of sleep score
cal = s.assign(wk=s.date.dt.to_period("W-SUN").dt.start_time, dow=s.date.dt.dayofweek)
fig = go.Figure(go.Heatmap(x=cal.wk, y=cal.dow, z=cal.sleep_score, colorscale=SEQ, zmin=50, zmax=100, xgap=2, ygap=2,
                           customdata=np.c_[cal.date.dt.strftime("%a %d %b %Y"), cal.total_sleep_h.round(1)],
                           hovertemplate="%{customdata[0]}<br>Sleep score %{z}<br>%{customdata[1]} h<extra></extra>",
                           colorbar=dict(title=dict(text="Score", font=dict(color=INK2)), thickness=12, tickfont=dict(color=MUTED))))
style(fig, height=280, legend=False)
fig.update_yaxes(tickvals=list(range(7)), ticktext=wd_order, autorange="reversed", showgrid=False, title_text="Wake-up day")
fcal = figure(fig, "Every night since May 2025, coloured by Garmin sleep score (darker = better). Pale streaks show runs of poor nights.")

# HRV vs sleep (same night)
hrv_sleep_pairs = [("Total sleep (h)", d.hrv, d.total_sleep_h), ("Sleep score", d.hrv, d.sleep_score),
                   ("Quality score", d.hrv, d.quality_score), ("Deep sleep %", d.hrv, d.deep_pct),
                   ("REM sleep %", d.hrv, d.rem_pct), ("Awake time (h)", d.hrv, d.awake_h),
                   ("Awakenings (count)", d.hrv, d.awakeCount), ("Sleep stress", d.hrv, d.avgSleepStress),
                   ("Wake-up time", d.hrv, d.wake_h), ("Bedtime", d.hrv, d.bedtime_h)]
f4 = figure(forest(hrv_sleep_pairs), "Spearman correlation between overnight HRV and each sleep metric from the same night. Whiskers are 95% CIs (Fisher z). Hover for p and n.")

lo_cmp = []
for lbl, col, fmt in [("Total sleep (h)", "total_sleep_h", "{:.2f}"), ("Sleep score", "sleep_score", "{:.1f}"),
                      ("Deep (h)", "deep_h", "{:.2f}"), ("REM (h)", "rem_h", "{:.2f}"), ("Awake (h)", "awake_h", "{:.2f}"),
                      ("Wake-up time (h)", "wake_h", "{:.2f}"), ("Sleep stress", "avgSleepStress", "{:.1f}"),
                      ("Resting HR overnight", "sleep_rhr", "{:.1f}"), ("Body Battery peak", "bb_high", "{:.1f}")]:
    a = d.loc[d.low_hrv == True, col].dropna(); b = d.loc[(d.low_hrv == False) & d.hrv.notna(), col].dropna()
    u = stats.mannwhitneyu(a, b) if len(a) > 3 else None
    lo_cmp.append([lbl, fmt.format(a.mean()), fmt.format(b.mean()), fmt_p(u.pvalue if u else np.nan), f"{len(a)} / {len(b)}"])
low_tbl = pd.DataFrame(lo_cmp, columns=["Metric", "Low-HRV nights", "Other nights", "Mann–Whitney p", "n (low / other)"])

# ================================================================ 2. Menstrual cycle & HRV
cyc_starts = d.loc[d.cycle_day == 1, "date"]
fig = go.Figure()
h = d.dropna(subset=["hrv"])
fig.add_scatter(x=h.date, y=h.hrv, mode="markers", name="Nightly HRV", marker=dict(size=4, color=S1, opacity=0.3),
                hovertemplate="%{x|%a %d %b %Y}<br>HRV %{y:.0f} ms<extra></extra>")
fig.add_scatter(x=d.date, y=d.hrv.rolling(7, min_periods=4, center=True).mean(), mode="lines", name="7-day mean",
                line=dict(color=S1, width=2), hovertemplate="%{x|%d %b %Y}<br>7-d mean %{y:.1f} ms<extra></extra>")
for i, cs in enumerate(cyc_starts[cyc_starts >= h.date.min()]):
    fig.add_vline(x=cs, line=dict(color=S2, width=1.5))
if True:
    fig.add_scatter(x=[None], y=[None], mode="lines", name="Period start (Clover)", line=dict(color=S2, width=1.5))
shade_training(fig)
style(fig, height=400, ytitle="Overnight HRV (ms)")
fig.update_xaxes(range=[h.date.min() - pd.Timedelta(days=10), h.date.max() + pd.Timedelta(days=10)])
f5 = figure(fig, "Overnight HRV with period start dates overlaid. Repeating dips just before a period (vertical lines) show as a saw-tooth pattern in the 7-day mean.")

ph = d.dropna(subset=["phase", "hrv_dev"])
fig = go.Figure()
for p in PHASES:
    v = ph.loc[ph.phase == p, "hrv_dev"]
    fig.add_box(y=v, name=f"{p}<br><sub>n={len(v)}</sub>", marker=dict(color=S1, size=4), line=dict(color=S1, width=2),
                fillcolor="rgba(42,120,214,0.15)", boxmean=True, boxpoints="all", jitter=0.4, pointpos=0,
                hovertemplate="%{y:+.1f} ms<extra>" + p + "</extra>")
fig.add_hline(y=0, line=dict(color=AXIS, width=1))
style(fig, height=380, ytitle="HRV vs personal 35-day median (ms)", legend=False)
f6 = figure(fig, "Detrended HRV by estimated cycle phase. Box = IQR, line = median, dashed = mean. Ovulation is estimated as 14 days before the next period. Detrending removes slow changes from training and season.")

kw_rows = []
for lbl, col in [("HRV (detrended, ms)", "hrv_dev"), ("HRV (raw, ms)", "hrv"), ("Overnight RHR (detrended, bpm)", "rhr_dev"),
                 ("Sleep score", "sleep_score"), ("Total sleep (h)", "total_sleep_h"), ("Deep sleep %", "deep_pct"),
                 ("REM sleep %", "rem_pct"), ("Awake time (h)", "awake_h"), ("Sleep stress", "avgSleepStress"),
                 ("Daytime stress", "stress_awake"), ("Respiration (brpm)", "averageRespiration")]:
    groups = [d.loc[d.phase == p, col].dropna() for p in PHASES]
    k = stats.kruskal(*groups)
    kw_rows.append([lbl] + [f"{g.mean():.2f}" for g in groups] + [fmt_p(k.pvalue)])
phase_tbl = pd.DataFrame(kw_rows, columns=["Metric"] + PHASES + ["Kruskal–Wallis p"])

# aligned to period start
al = d.dropna(subset=["cycle_day"]).copy()
pre = al.dropna(subset=["days_to_next_period"]).assign(rel=lambda x: x.days_to_next_period)
pre = pre[pre.rel >= -21]
post = al[al.cycle_day <= 10].assign(rel=lambda x: x.cycle_day - 1)
aligned = pd.concat([pre, post])


def ci_band(fig, g, col, color, rowcol):
    agg = g.groupby("rel")[col].agg(["mean", "sem", "count"])
    agg = agg[agg["count"] >= 5]
    x = agg.index
    fig.add_scatter(x=list(x) + list(x[::-1]), y=list(agg["mean"] + 1.96 * agg["sem"]) + list((agg["mean"] - 1.96 * agg["sem"])[::-1]),
                    fill="toself", fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba"), line=dict(width=0),
                    hoverinfo="skip", showlegend=False, **rowcol)
    fig.add_scatter(x=x, y=agg["mean"], mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=6, color=color),
                    customdata=agg["count"], showlegend=False,
                    hovertemplate="Day %{x:+d}<br>%{y:+.1f}<br>n = %{customdata}<extra></extra>", **rowcol)


fig = make_subplots(rows=1, cols=2, subplot_titles=("HRV (detrended, ms)", "Overnight resting HR (detrended, bpm)"), horizontal_spacing=0.1)
ci_band(fig, aligned, "hrv_dev", "rgb(42,120,214)", dict(row=1, col=1))
ci_band(fig, aligned, "rhr_dev", "rgb(42,120,214)", dict(row=1, col=2))
for c in (1, 2):
    fig.add_vline(x=0, line=dict(color=S2, width=1.5), row=1, col=c)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=1, col=c)
style(fig, height=380, xtitle="Days relative to period start (0 = day 1)", legend=False)
fig.update_annotations(font=dict(size=13, color=INK2))
f7 = figure(fig, "HRV and overnight resting HR aligned to period start (mean ± 95% CI across cycles). Negative days = days before the period. Orange line = first day of bleeding.")


# every cycle as a row: HRV deviation by days before next period
hm = d.dropna(subset=["days_to_next_period", "hrv_dev"])
hm = hm[hm.days_to_next_period >= -30]
piv = hm.pivot_table(index="cycle_start", columns="days_to_next_period", values="hrv_dev")
piv.index = pd.to_datetime(piv.index).strftime("%d %b %Y")
cl = d.dropna(subset=["cycle_len"]).groupby(pd.to_datetime(d.cycle_start).dt.strftime("%d %b %Y")).cycle_len.first()
fig = go.Figure(go.Heatmap(x=piv.columns, y=[(f"Cycle {k + 1}" if PUBLIC else i) + f" ({int(cl[i])} d)" for k, i in enumerate(piv.index)], z=piv.values, colorscale=DIV, zmid=0, zmin=-20, zmax=20,
                           xgap=2, ygap=3, hovertemplate="Cycle %{y}<br>%{x} days before period<br>HRV %{z:+.1f} ms vs baseline<extra></extra>",
                           colorbar=dict(title=dict(text="ms", font=dict(color=INK2)), thickness=12, tickfont=dict(color=MUTED))))
fig.add_vline(x=-5.5, line=dict(color=INK2, width=1, dash="dot"))
style(fig, height=60 + 34 * len(piv), xtitle="Days before next period starts", legend=False)
fig.update_yaxes(showgrid=False, title_text="Cycle (length)" if PUBLIC else "Cycle start (length)")
fheat = figure(fig, "One row per completed cycle, lined up on the next period. Blue = HRV above my 35-day baseline, red = below. The red band to the right of the dotted line (last ~5 days) repeats in most cycles, whatever their length. That points to the cycle driving HRV.")

# Direction-of-effect: does follicular-phase HRV/stress predict a later (longer) cycle?
cyc = (d.dropna(subset=["cycle_len"]).groupby("cycle_id")
       .agg(start=("cycle_start", "first"), length=("cycle_len", "first"))
       .join(d[d.phase.isin(["Menstrual", "Follicular"])].groupby("cycle_id")
             .agg(fol_hrv=("hrv", "mean"), fol_hrv_dev=("hrv_dev", "mean"), fol_stress=("stress_awake", "mean"),
                  fol_sleep=("sleep_score", "mean"), fol_n=("hrv", "count")))
       .join(d[d.phase == "Luteal"].groupby("cycle_id").agg(lut_hrv=("hrv", "mean"), lut_n=("hrv", "count")))
       .join(d[(d.days_to_next_period >= -7)].groupby("cycle_id").agg(late_lut_hrv_dev=("hrv_dev", "mean"))))
# 'previous cycle' HRV -> this cycle length, a lagged test
cyc["prev_lut_hrv"] = cyc["lut_hrv"].shift(1)
cyc_h = cyc[cyc.fol_n >= 5]
g_cyc_rhr = d.dropna(subset=['cycle_len']).groupby('cycle_id').agg(length=('cycle_len', 'first'), rhr=('rhr', 'mean'), srhr=('sleep_rhr', 'mean'))
cyc["start"] = pd.to_datetime(cyc["start"])
fig = make_subplots(rows=1, cols=2, subplot_titles=("Follicular-phase HRV (deviation, ms)", "Follicular-phase daytime stress"), horizontal_spacing=0.1)
for c, col in [(1, "fol_hrv_dev"), (2, "fol_stress")]:
    sub = cyc_h.dropna(subset=[col])
    fig.add_scatter(x=sub[col], y=sub.length, mode="markers", marker=dict(size=10, color=S1, line=dict(color="#fcfcfb", width=2)),
                    customdata=[f"#{k + 1}" for k in range(len(sub))] if PUBLIC else pd.to_datetime(sub.start).dt.strftime("%d %b %Y"),
                    hovertemplate="Cycle starting %{customdata}<br>x = %{x:.1f}<br>length = %{y} d<extra></extra>", row=1, col=c)
    if len(sub) > 3:
        b = np.polyfit(sub[col], sub.length, 1); xs = np.linspace(sub[col].min(), sub[col].max(), 20)
        fig.add_scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color=MUTED, width=2, dash="dash"), hoverinfo="skip", row=1, col=c)
style(fig, height=360, ytitle="Cycle length (days)", legend=False)
fig.update_annotations(font=dict(size=13, color=INK2))
f8 = figure(fig, "Each dot is one completed cycle (cycles with ≥5 HRV nights in the first half). If low HRV or high stress <i>before ovulation</i> delays ovulation, lower HRV / higher stress should go with longer cycles (a negative slope on the left, a positive slope on the right). Dashed line = least-squares fit.")
dir_tbl_html = corr_table([
    ("Follicular HRV deviation", "Cycle length", cyc_h.fol_hrv_dev, cyc_h.length),
    ("Follicular HRV (raw)", "Cycle length", cyc_h.fol_hrv, cyc_h.length),
    ("Follicular daytime stress", "Cycle length", cyc_h.fol_stress, cyc_h.length),
    ("Follicular sleep score", "Cycle length", cyc_h.fol_sleep, cyc_h.length),
    ("Previous cycle's luteal HRV", "Cycle length", cyc.prev_lut_hrv, cyc.length),
], "Cycle-level correlations (direction-of-effect test). n is the number of cycles, so power is low.")
cyc_tbl = cyc_h.assign(start=[f"Cycle {k + 1}" for k in range(len(cyc_h))] if PUBLIC else pd.to_datetime(cyc_h.start).dt.strftime("%d %b %Y"))[["start", "length", "fol_hrv", "lut_hrv", "late_lut_hrv_dev", "fol_stress", "fol_sleep"]].round(1)
cyc_tbl.columns = ["Cycle start", "Length (d)", "Follicular HRV", "Luteal HRV", "Last-7-day HRV dev.", "Follicular stress", "Follicular sleep score"]


# ---------- full cycle history from Clover (2023 onwards)
cy_all = pd.read_parquet(f"{DATA}/cycles.parquet")
ch = (cy_all.groupby("cycle_id").agg(start=("cycle_start", "first"), length=("cycle_len", "first"), plen=("period_len", "first"))
      .assign(start=lambda x: pd.to_datetime(x.start)))
done = ch.dropna(subset=["length"])
is_long = done.length > 35
fig = go.Figure()
fig.add_hrect(y0=21, y1=35, fillcolor=BAND, line_width=0, layer="below")
for mask, name, c in [(~is_long, "≤35 days", S1), (is_long, "Long (>35 days)", S2)]:
    sub = done[mask]
    fig.add_bar(x=(done.reset_index().index[mask.values] + 1) if PUBLIC else sub.start, y=sub.length, name=name, width=0.7 if PUBLIC else 1000 * 3600 * 24 * 18, marker=dict(color=c, cornerradius=4),
                text=sub.length.astype(int).astype(str) if name.startswith("Long") else None, textposition="outside",
                textfont=dict(color=INK2, size=11), customdata=sub.plen,
                hovertemplate=("Cycle %{x}" if PUBLIC else "Cycle starting %{x|%d %b %Y}") + "<br>%{y:.0f} days<br>Period %{customdata} days<extra></extra>")
if not PUBLIC:
    shade_training(fig)
style(fig, height=400, ytitle="Cycle length (days)", xtitle="Cycle number (chronological)" if PUBLIC else None)
if PUBLIC:
    fig.update_xaxes(dtick=2)
fig.update_yaxes(range=[0, done.length.max() * 1.15])
fhist = figure(fig, f"Length of every logged cycle in Clover ({len(done)} completed cycles, {done.start.min():%Y} – {done.start.max():%Y}), in chronological order. Grey band = typical 21–35-day range; orange = longer than 35 days." if PUBLIC else f"Length of every logged cycle in Clover ({len(done)} completed cycles, {done.start.min():%b %Y} – {done.start.max():%b %Y}), plotted at its start date. Grey band = typical 21–35-day range; orange = longer than 35 days; labels give the length.")

# months with / without a period starting
starts = ch.start
months = pd.period_range(starts.min().to_period("M"), pd.Timestamp("2026-10-01").to_period("M"), freq="M")
cnt = starts.dt.to_period("M").value_counts().reindex(months, fill_value=0)
years = sorted({m.year for m in months}); mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
z = np.full((len(years), 12), np.nan); txt = np.full((len(years), 12), "", dtype=object)
for m, v in cnt.items():
    if m == pd.Period("2026-10", "M"):
        continue  # month in progress
    z[years.index(m.year), m.month - 1] = v; txt[years.index(m.year), m.month - 1] = "✕" if v == 0 else ""
fig = go.Figure(go.Heatmap(x=mnames, y=[str(y) for y in years], z=z, text=txt, texttemplate="%{text}", textfont=dict(color=S2, size=16),
                           colorscale=[[0, "#f0efec"], [0.5, "#86b6ef"], [1, "#1c5cab"]], zmin=0, zmax=2, xgap=4, ygap=4, showscale=False,
                           hovertemplate="%{x} %{y}<br>%{z} period start(s)<extra></extra>"))
style(fig, height=250, legend=False)
fig.update_yaxes(autorange="reversed", showgrid=False); fig.update_xaxes(side="top")
fmonth = "" if PUBLIC else figure(fig, "Calendar months in which a period started (blue; darker = two starts) and months with no period at all (✕). Blank = outside the tracking window or month still in progress.")

no_months = [m.strftime("%b %Y") for m, v in cnt.items() if v == 0 and m != pd.Period("2026-10", "M")]
yr = done.assign(year=done.start.dt.year).groupby("year").agg(cycles=("length", "size"), mean_len=("length", "mean"),
                                                             median_len=("length", "median"), longest=("length", "max"),
                                                             long_n=("length", lambda x: (x > 35).sum()))
yr["% long"] = (100 * yr.long_n / yr.cycles).round(0).astype(int)
yr = yr.round(1).reset_index().rename(columns={"year": "Year (cycle start)", "cycles": "Cycles", "mean_len": "Mean length (d)",
                                              "median_len": "Median (d)", "longest": "Longest (d)", "long_n": "Cycles >35 d"})
plen_note = f"Period length is logged as 7 days in {int((ch.plen == 7).sum())} of {len(ch)} cycles (likely the app default), so it carries little information; the exception is a {int(ch.plen.max())}-day period" + ("." if PUBLIC else f" starting {ch.loc[ch.plen.idxmax(), 'start']:%d %b %Y}.")

# ================================================================ 3. Stress & sleep stages
stress_pairs = [("Daytime stress (day before) → sleep score", d.stress_prev_day, d.sleep_score),
                ("Daytime stress → deep sleep %", d.stress_prev_day, d.deep_pct),
                ("Daytime stress → REM sleep %", d.stress_prev_day, d.rem_pct),
                ("Daytime stress → light sleep (h)", d.stress_prev_day, d.light_h),
                ("Daytime stress → awake time (h)", d.stress_prev_day, d.awake_h),
                ("Daytime stress → total sleep (h)", d.stress_prev_day, d.total_sleep_h),
                ("Daytime stress → overnight HRV", d.stress_prev_day, d.hrv),
                ("Sleep stress → deep sleep %", d.avgSleepStress, d.deep_pct),
                ("Sleep stress → REM sleep %", d.avgSleepStress, d.rem_pct),
                ("Sleep stress → awake time (h)", d.avgSleepStress, d.awake_h),
                ("Sleep stress → sleep score", d.avgSleepStress, d.sleep_score)]
f9 = figure(forest(stress_pairs), "Spearman correlations between Garmin stress and sleep stages. ‘Daytime stress’ = mean awake-time stress on the day before the night. ‘Sleep stress’ = mean stress during that night (largely the inverse of HRV, so expect strong links).")

q = d.dropna(subset=["stress_prev_day", "sleep_score"]).copy()
q["bin"] = pd.qcut(q.stress_prev_day, 5)
qb = q.groupby("bin", observed=True).agg(x=("stress_prev_day", "mean"), deep=("deep_h", "mean"), rem=("rem_h", "mean"),
                                         light=("light_h", "mean"), awake=("awake_h", "mean"), n=("date", "size"))
labels = [f"Q{i+1}<br><sub>{iv.left:.0f}–{iv.right:.0f}</sub>" for i, iv in enumerate(qb.index)]
fig = go.Figure()
for col, name, c in [("deep", "Deep", S1), ("light", "Light", S2), ("rem", "REM", S3), ("awake", "Awake", S4)]:
    fig.add_bar(x=labels, y=qb[col], name=name, marker=dict(color=c, line=dict(color="#fcfcfb", width=2)),
                hovertemplate=name + ": %{y:.2f} h<extra></extra>")
fig.update_layout(barmode="stack", bargap=0.3)
style(fig, height=360, ytitle="Mean hours that night", xtitle="Quintile of previous-day stress (score range)")
f10 = figure(fig, "Sleep architecture by quintile of the previous day's mean daytime stress (each quintile ≈ 100 nights).")

# ================================================================ 4. Marathon training block
rt = r[(r.date >= BASE_START) & (r.date <= RACE) & ~r.is_race].copy()
wk = rt.groupby("week").agg(km=("distance_km", "sum"), runs=("distance_km", "size"), hr=("avg_hr", "mean"),
                            long_km=("distance_km", "max"))
easy = rt[~rt.is_long_run]
wk = wk.join(easy.groupby("week").agg(pace=("pace_min_km", "mean"), pace_n=("pace_min_km", "size"), hr_easy=("avg_hr", "mean")))
dd = d.assign(week=d.date.dt.to_period("W-SUN").dt.start_time)
wk = wk.join(dd.groupby("week").agg(hrv=("hrv", "mean"), sleep_h=("total_sleep_h", "mean"), score=("sleep_score", "mean"),
                                    deep=("deep_h", "mean"), rem=("rem_h", "mean"), stress=("stress_awake", "mean"), rhr=("rhr", "mean")))
wk.index = pd.to_datetime(wk.index)
wk_pace = wk[wk.pace_n >= 2]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08,
                    subplot_titles=("Mean running heart rate (bpm)", "Weekly running volume (km)"))
fig.add_scatter(x=wk.index, y=wk.hr, mode="lines+markers", line=dict(color=S1, width=2), marker=dict(size=8, color=S1),
                customdata=wk.runs, hovertemplate="Week of %{x|%d %b}<br>%{y:.0f} bpm<br>%{customdata} runs<extra></extra>", row=1, col=1)
fig.add_bar(x=wk.index, y=wk.km, marker=dict(color=S1, cornerradius=4), hovertemplate="Week of %{x|%d %b}<br>%{y:.1f} km<extra></extra>", row=2, col=1)
shade_training(fig, row=1, col=1); shade_training(fig, row=2, col=1)
style(fig, height=520, legend=False)
fig.update_annotations(font=dict(size=13, color=INK2))
f11 = figure(fig, "Weekly mean of per-run average heart rate (top) and weekly distance (bottom) from 25 Aug 2025 to race week. The race itself is excluded.")

fig = make_subplots(rows=1, cols=2, subplot_titles=("Weekly HRV vs running HR", "Weekly HRV vs easy-run pace"), horizontal_spacing=0.1)
for c, col, ylab in [(1, "hr", "bpm"), (2, "pace", "min/km")]:
    sub = wk.dropna(subset=["hrv", col]) if col == "hr" else wk_pace.dropna(subset=["hrv", col])
    fig.add_scatter(x=sub.hrv, y=sub[col], mode="markers", marker=dict(size=9, color=S1, line=dict(color="#fcfcfb", width=2)),
                    customdata=sub.index.strftime("%d %b %Y"), hovertemplate="Week of %{customdata}<br>HRV %{x:.0f} ms<br>%{y:.2f} " + ylab + "<extra></extra>", row=1, col=c)
    b = np.polyfit(sub.hrv, sub[col], 1); xs = np.linspace(sub.hrv.min(), sub.hrv.max(), 20)
    fig.add_scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color=MUTED, width=2, dash="dash"), hoverinfo="skip", row=1, col=c)
fig.update_xaxes(title_text="Weekly mean HRV (ms)")
fig.update_yaxes(title_text="Mean run HR (bpm)", row=1, col=1); fig.update_yaxes(title_text="Pace (min/km)", autorange="reversed", row=1, col=2)
style(fig, height=380, legend=False)
fig.update_xaxes(title_text="Weekly mean HRV (ms)")
fig.update_annotations(font=dict(size=13, color=INK2))
f12 = figure(fig, "Each dot is one training week. Pace axis is reversed so ‘faster’ is up. Dashed line = least-squares fit.")

fig = go.Figure()
fig.add_scatter(x=wk_pace.index, y=wk_pace.pace, mode="lines+markers", line=dict(color=S1, width=2), marker=dict(size=8, color=S1),
                customdata=np.c_[wk_pace.pace_n, wk_pace.score], name="Pace",
                hovertemplate="Week of %{x|%d %b}<br>%{y:.2f} min/km<br>%{customdata[0]} runs<br>Sleep score %{customdata[1]:.0f}<extra></extra>")
shade_training(fig)
style(fig, height=340, ytitle="Mean pace excl. long run (min/km)", legend=False)
fig.update_yaxes(autorange="reversed")
f13 = figure(fig, "Weekly mean pace of all runs except that week's long run (the longest run of the week, if ≥10 km). Only weeks with ≥2 such runs. Axis reversed: higher = faster.")


# every run as a bubble
rb = r[(r.date >= pd.Timestamp("2025-06-01"))].dropna(subset=["avg_hr"])
fig = go.Figure(go.Scatter(x=rb.date, y=rb.pace_min_km, mode="markers",
                           marker=dict(size=np.clip(rb.distance_km, 3, 45) * 1.1 + 4, color=rb.avg_hr, colorscale=SEQ, cmin=150, cmax=185,
                                       line=dict(color="#fcfcfb", width=1.5), opacity=0.85,
                                       colorbar=dict(title=dict(text="Avg HR", font=dict(color=INK2)), thickness=12, tickfont=dict(color=MUTED))),
                           customdata=np.c_[rb.distance_km.round(1), rb.avg_hr, rb.training_phase],
                           hovertemplate="%{x|%a %d %b %Y}<br>%{customdata[0]} km @ %{y:.2f} min/km<br>Avg HR %{customdata[1]:.0f}<br>%{customdata[2]}<extra></extra>"))
shade_training(fig)
style(fig, height=440, ytitle="Pace (min/km)", legend=False)
fig.update_yaxes(autorange="reversed")
fbub = figure(fig, "Every run: bubble size = distance, colour = average HR (darker = harder). Long runs grow through the block and bubbles get lighter: HR falls, but easy pace also slows slightly.")

# ---------- resting HR & HRV across training: is the body under strain?
wk2 = dd[(dd.week >= BASE_START - pd.Timedelta(weeks=2))].groupby("week").agg(rhr=("rhr", "mean"), srhr=("sleep_rhr", "mean"), hrv=("hrv", "mean"))
wk2.index = pd.to_datetime(wk2.index)
wk2 = wk2.join(r.groupby("week").agg(run_hr=("avg_hr", "mean"), km=("distance_km", "sum")))
long_cyc = done[done.length > 35]
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("Mean running HR (bpm)", "Resting HR (bpm)", "Overnight HRV (ms)"))
fig.add_scatter(x=wk2.index, y=wk2.run_hr, mode="lines+markers", name="Running HR", line=dict(color=S1, width=2), marker=dict(size=6, color=S1),
                hovertemplate="Week of %{x|%d %b %Y}<br>%{y:.0f} bpm<extra>Running HR</extra>", showlegend=False, row=1, col=1)
fig.add_scatter(x=wk2.index, y=wk2.rhr, mode="lines", name="Daytime resting HR (Garmin daily)", line=dict(color=S1, width=2),
                hovertemplate="Week of %{x|%d %b %Y}<br>%{y:.1f} bpm<extra>Daily RHR</extra>", row=2, col=1)
fig.add_scatter(x=wk2.index, y=wk2.srhr, mode="lines", name="Overnight resting HR", line=dict(color=S2, width=2),
                hovertemplate="Week of %{x|%d %b %Y}<br>%{y:.1f} bpm<extra>Overnight RHR</extra>", row=2, col=1)
fig.add_scatter(x=wk2.index, y=wk2.hrv, mode="lines+markers", name="HRV", line=dict(color=S1, width=2), marker=dict(size=6, color=S1),
                hovertemplate="Week of %{x|%d %b %Y}<br>%{y:.0f} ms<extra>HRV</extra>", showlegend=False, row=3, col=1)
for rw in (1, 2, 3):
    shade_training(fig, row=rw, col=1)
    for _, c in ([] if PUBLIC else long_cyc[long_cyc.start >= wk2.index.min()].iterrows()):
        fig.add_vrect(x0=c.start, x1=c.start + pd.Timedelta(days=int(c.length)), fillcolor="rgba(235,104,52,0.15)", line_width=0, layer="below", row=rw, col=1)
if not PUBLIC:
    fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", size=12, color="rgba(235,104,52,0.35)"), name="Long cycle (>35 d)", row=1, col=1)
style(fig, height=660)
fig.update_layout(margin=dict(t=80), legend=dict(y=1.12))
fig.update_annotations(font=dict(size=13, color=INK2))
fstrain = figure(fig, "Weekly running HR, resting HR and HRV from the start of training. Grey = training phases" + ("." if PUBLIC else "; orange = cycles longer than 35 days.") + " Through the block, running HR falls while resting HR rises and HRV falls. That pattern is consistent with accumulated fatigue, and it peaks in the same window as the 50-day cycle. Note that Jul–Sep 2026 shows similar resting values without a training block.")
strain_tbl = corr_table([
    ("Weekly volume (km)", "Weekly daytime resting HR", wk2.km, wk2.rhr),
    ("Weekly volume (km)", "Weekly overnight resting HR", wk2.km, wk2.srhr),
    ("Weekly volume (km)", "Weekly HRV", wk2.km, wk2.hrv),
    ("Cycle mean daytime resting HR", "Cycle length", g_cyc_rhr.rhr, g_cyc_rhr.length),
    ("Cycle mean overnight resting HR", "Cycle length", g_cyc_rhr.srhr, g_cyc_rhr.length),
], "Training load vs resting physiology (weekly, from Aug 2025), and cycle-level resting HR vs cycle length.")
ph_rest = rt_phase_tbl = (d[(d.date >= BASE_START - pd.Timedelta(weeks=12))]
    .assign(tp=lambda x: np.select([x.date < BASE_START, x.date < BLOCK_START, x.date <= RACE, x.date <= RACE + pd.Timedelta(weeks=8)],
                                   ["Pre-training (Jun–Aug 2025)", "Base build", "10-week block", "8 weeks post-race"], "Later (Jul–Sep 2026)"))
    .groupby("tp", sort=False).agg(**{"Daytime RHR": ("rhr", "mean"), "Overnight RHR": ("sleep_rhr", "mean"), "HRV": ("hrv", "mean"),
                                      "Sleep score": ("sleep_score", "mean"), "Nights": ("date", "size")}).round(1).reset_index().rename(columns={"tp": "Period"}))

wk_corr = corr_table([
    ("Weekly HRV", "Mean run HR", wk.hrv, wk.hr), ("Weekly HRV", "Easy-run HR", wk.hrv, wk.hr_easy),
    ("Weekly resting HR", "Mean run HR", wk.rhr, wk.hr),
    ("Weekly sleep score", "Pace excl. long run", wk_pace.score, wk_pace.pace),
    ("Weekly sleep hours", "Pace excl. long run", wk_pace.sleep_h, wk_pace.pace),
    ("Weekly deep sleep (h)", "Pace excl. long run", wk_pace.deep, wk_pace.pace),
    ("Weekly HRV", "Pace excl. long run", wk_pace.hrv, wk_pace.pace),
    ("Weekly daytime stress", "Pace excl. long run", wk_pace.stress, wk_pace.pace),
    ("Weekly volume (km)", "Mean run HR", wk.km, wk.hr),
], "Week-level correlations during training (25 Aug 2025 – 23 May 2026). For pace, a negative ρ means more of X goes with faster running.")

# run-level: prior-night sleep -> efficiency & effort
er = r[~r.is_long_run & ~r.is_race & r.avg_hr.notna()].copy()
er["speed_m_min"] = 1000 / er.pace_min_km
er["efficiency"] = er.speed_m_min / er.avg_hr  # metres per heartbeat (higher = more economical)
er = er.merge(d[["date", "hrv_dev", "score_dev"]], on="date", how="left")
run_corr = corr_table([
    ("Prior-night sleep score", "Efficiency (m per beat)", er.sleep_score, er.efficiency),
    ("Prior-night HRV", "Efficiency (m per beat)", er.hrv, er.efficiency),
    ("Prior-night HRV (detrended)", "Efficiency (m per beat)", er.hrv_dev, er.efficiency),
    ("Prior-night total sleep", "Efficiency (m per beat)", er.total_sleep_h, er.efficiency),
    ("Prior-night sleep score", "RPE", er.sleep_score, er.rpe),
    ("Prior-night HRV", "RPE", er.hrv, er.rpe),
    ("Previous-day stress", "RPE", er.stress_prev_day, er.rpe),
    ("Previous-day stress", "Efficiency (m per beat)", er.stress_prev_day, er.efficiency),
], "Run-level correlations (non-long runs, all dates). Efficiency = speed ÷ heart rate, so it adjusts pace for effort.")

# ================================================================ 5. Predictive model
feat = pd.DataFrame({
    "HRV previous night": d.hrv.shift(1), "Daytime stress (previous day)": d.stress_prev_day,
    "Run km (previous day)": d.run_km_prev_day, "Bedtime (h)": d.bedtime_h,
    "Luteal phase": (d.phase == "Luteal").astype(float), "Menstrual phase": (d.phase == "Menstrual").astype(float),
    "Weekend night": d.is_weekend_night.astype(float)})
models = {}
for target_name, target in [("Sleep score", d.sleep_score), ("Overnight HRV", d.hrv)]:
    df = pd.concat([feat, target.rename("y"), d.date], axis=1).dropna()
    X = df[feat.columns]; Xs = (X - X.mean()) / X.std()
    ys = (df.y - df.y.mean()) / df.y.std()
    split = int(len(df) * 0.75)
    fit = sm.OLS(ys, sm.add_constant(Xs)).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    tr = sm.OLS(df.y.iloc[:split], sm.add_constant(X.iloc[:split])).fit()
    pred = tr.predict(sm.add_constant(X.iloc[split:], has_constant="add"))
    yt = df.y.iloc[split:]
    r2_test = 1 - ((yt - pred) ** 2).sum() / ((yt - df.y.iloc[:split].mean()) ** 2).sum()
    mae = (yt - pred).abs().mean(); mae_naive = (yt - df.y.iloc[:split].mean()).abs().mean()
    models[target_name] = dict(fit=fit, n=len(df), r2=fit.rsquared, r2_test=r2_test, mae=mae, mae_naive=mae_naive,
                               test_from=df.date.iloc[split])

fig = make_subplots(rows=1, cols=2, subplot_titles=tuple(models), horizontal_spacing=0.28, shared_yaxes=False)
for c, (name, mdl) in enumerate(models.items(), start=1):
    fit = mdl["fit"]; params = fit.params.drop("const"); ci = fit.conf_int().drop("const")
    fig.add_scatter(x=params, y=params.index, mode="markers", marker=dict(size=10, color=S1),
                    error_x=dict(type="data", array=ci[1] - params, arrayminus=params - ci[0], color=S1, thickness=2, width=0),
                    hovertext=[f"β = {b:+.2f} [{l:+.2f}, {h_:+.2f}]<br>p = {fmt_p(p).replace('&lt;', '<')}" for b, l, h_, p in zip(params, ci[0], ci[1], fit.pvalues.drop('const'))],
                    hoverinfo="text", row=1, col=c)
    fig.add_vline(x=0, line=dict(color=AXIS, width=1), row=1, col=c)
fig.update_yaxes(autorange="reversed", showgrid=False)
style(fig, height=380, legend=False)
fig.update_xaxes(title_text="Standardised β (95% CI, HAC)", showgrid=True, gridcolor=GRID)
fig.update_annotations(font=dict(size=13, color=INK2))
f14 = figure(fig, "Standardised regression coefficients predicting tonight's sleep score (left) and overnight HRV (right) from information available beforehand. Units are SDs of the outcome per SD of the predictor (phase/weekend: present vs absent). Newey–West (HAC, 7-day) errors account for day-to-day autocorrelation.")
model_tbl = pd.DataFrame([[k, v["n"], f"{v['r2']:.2f}", f"{v['r2_test']:.2f}", f"{v['mae']:.1f}", f"{v['mae_naive']:.1f}", f"{v['test_from']:%d %b %Y}"]
                          for k, v in models.items()],
                         columns=["Outcome", "n nights", "In-sample R²", "Out-of-sample R²", "Test MAE", "Naive MAE (mean)", "Test period from"])

# ================================================================ key numbers for narrative
K = {}
K["phase_means"] = d.groupby("phase").hrv.mean().round(1).to_dict()
K["phase_dev"] = d.groupby("phase").hrv_dev.mean().round(1).to_dict()
K["phase_kw_p"] = stats.kruskal(*[d.loc[d.phase == p, "hrv_dev"].dropna() for p in PHASES]).pvalue
fl = d.loc[d.phase == "Follicular", "hrv_dev"].dropna(); lu = d.loc[d.phase == "Luteal", "hrv_dev"].dropna()
K["fol_vs_lut_p"] = stats.mannwhitneyu(fl, lu).pvalue
K["hrv_sleep"] = {lbl: spear(x, y) for lbl, x, y in hrv_sleep_pairs}
K["stress"] = {lbl: spear(x, y) for lbl, x, y in stress_pairs}
K["dir"] = {"fol_hrv_dev": spear(cyc_h.fol_hrv_dev, cyc_h.length), "fol_stress": spear(cyc_h.fol_stress, cyc_h.length),
            "prev_lut": spear(cyc.prev_lut_hrv, cyc.length)}
K["aligned"] = aligned.groupby("rel")[["hrv_dev", "rhr_dev"]].mean().round(1).loc[-10:6].to_dict()
K["wk"] = {"hrv_hr": spear(wk.hrv, wk.hr), "score_pace": spear(wk_pace.score, wk_pace.pace), "sleep_pace": spear(wk_pace.sleep_h, wk_pace.pace),
           "hrv_pace": spear(wk_pace.hrv, wk_pace.pace), "stress_pace": spear(wk_pace.stress, wk_pace.pace)}
K["phase_train"] = rt.groupby("training_phase").agg(hr=("avg_hr", "mean"), pace=("pace_min_km", "mean")).round(2).to_dict()
K["hr_first_last"] = (wk.hr.iloc[:4].mean(), wk.hr.iloc[-6:-2].mean())
K["pace_first_last"] = (wk_pace.pace.iloc[:4].mean(), wk_pace.pace.iloc[-6:-2].mean())
K["low_hrv"] = low_tbl.to_dict("records")
K["models"] = {k: {kk: (vv if not hasattr(vv, "params") else vv.params.round(2).to_dict()) for kk, vv in v.items()} for k, v in models.items()}
K["sleep_month"] = stage_tbl[["Month", "Total", "Deep", "REM", "Sleep score"]].to_dict("records")
K["weekday"] = wd.round(2).to_dict()
K["no_months"] = no_months
K["yr"] = yr.to_dict("records")
K["ph_rest"] = ph_rest.to_dict("records")
K["strain"] = {"km_rhr": spear(wk2.km, wk2.rhr), "km_srhr": spear(wk2.km, wk2.srhr), "km_hrv": spear(wk2.km, wk2.hrv)}
K["race"] = r[r.is_race][["distance_km", "pace_min_km", "avg_hr"]].round(2).to_dict("records")
if not PUBLIC:
    json.dump(K, open(f"{DATA}/key_numbers.json", "w"), indent=1, default=str)

# ================================================================ HTML
# private intro lives in the git-ignored data dir; falls back to the public text
_ip = f"{DATA}/intro_private.html"
INTRO_PUBLIC = """<p>My overnight heart rate variability (HRV) varies a lot across the month, and low-HRV nights often coincide with poor sleep. These changes appear linked to the menstrual cycle, but the direction is unclear: does the cycle lower HRV, or does a stretch of low HRV (stress, training load, poor sleep) delay the cycle? Resting heart rate also fluctuates noticeably.</p>"""
def fn(f):
    return re.search(r"<!--FIG:(\d+)-->", f).group(1)


findings_path = f"{DATA}/findings_public.html" if PUBLIC else f"{DATA}/findings.html"
try:
    findings = open(findings_path).read()
    for name, val in [("f7", f7), ("fheat", fheat), ("f10", f10), ("f8", f8)]:
        findings = findings.replace("[[" + name + "]]", fn(val))
except FileNotFoundError:
    findings = "<p><i>Findings summary not yet written.</i></p>"

body = f"""
<header><h1>Sleep, HRV, menstrual cycle and marathon training: a single-subject study</h1>
<p class="sub">Garmin Connect export + Clover period tracker · {d.date.min():%B %Y} – {d.date.max():%B %Y} · generated {date.today():%d %b %Y}</p></header>
<section class="tiles">{tiles_html}</section>

<section><h2>Key findings</h2>{findings}</section>

<section><h2>1. Introduction</h2>
{INTRO_PUBLIC if PUBLIC or not os.path.exists(_ip) else open(_ip).read()}
<p>The study uses my own wearable data to (i) describe how my sleep changed over time, (ii) test how HRV, sleep and stress vary across the menstrual cycle, and test the direction of the HRV–cycle link, (iii) relate daytime and overnight stress to sleep stages, (iv) look at how heart rate and pace changed through marathon training and whether sleep/HRV explain good and bad weeks, and (v) build a simple model to predict the next night's sleep quality and HRV.</p></section>

<section><h2>2. Methods</h2>
<ul>
<li><b>Data.</b> Garmin Connect GDPR export: nightly sleep (stages, sleep score, sleep stress, respiration), overnight HRV and resting HR from the Health Status feature (HRV for May–Sep 2025, which that feed lacks, is recovered from the watch's raw FIT files: <code>hrv_status_summary</code> messages, which match Health Status on 99.7% of overlapping nights), daily summaries (stress, Body Battery, steps), and 203 runs. Period dates come from the Clover app ({completed_cycles} completed cycles in the study window).</li>
<li><b>Processing.</b> Raw JSON is parsed and joined in <b>Apache Spark (PySpark)</b> into one row per day (<code>daily</code>) and one row per run (<code>runs</code>). Garmin dates sleep by the wake-up morning, so the previous day's stress and training are joined as predictors of each night. Statistics use pandas / SciPy / statsmodels.</li>
<li><b>Cycle phases.</b> Menstrual = logged bleeding days. Ovulation is estimated as 14 days before the next period (a standard luteal-length assumption, since no ovulation tests were logged). Ovulatory = ovulation ±1 day. Follicular = between bleeding and the ovulatory window. Luteal = after the ovulatory window. The current, unfinished cycle has no phase beyond bleeding.</li>
<li><b>Detrending.</b> HRV and resting HR also drift with fitness and season. For cycle analyses each night is expressed as a deviation from a centred 35-day rolling median, so only within-month variation is left.</li>
<li><b>Statistics.</b> Spearman rank correlations (robust to outliers and non-linearity) with Fisher-z 95% CIs; Kruskal–Wallis / Mann–Whitney for group differences; OLS with Newey–West errors for the predictive model, evaluated on the last 25% of nights as a hold-out set. Nightly data are autocorrelated, so p-values for daily correlations are optimistic. Treat ρ and its CI as the main result.</li>
<li><b>Training.</b> Base build: 25 Aug 2025 → 15 Mar 2026. 10-week block: 15 Mar → 24 May 2026 (race). Long run = the longest run of each week if ≥10 km. Running efficiency = speed (m/min) ÷ average HR.</li>
</ul></section>

<section><h2>3. Sleep since getting the Garmin</h2>
{f1}{f2}
<details><summary>Monthly sleep table</summary>{table(stage_tbl)}</details>
{f3}{fcal}</section>

<section><h2>4. Sleep and HRV</h2>
{f4}
{table(low_tbl, "Nights where HRV fell below Garmin's personal baseline range vs all other nights.")}
</section>

<section><h2>5. Menstrual cycle, HRV and sleep</h2>
{f5}{f6}
{table(phase_tbl, "Mean of each metric by cycle phase.")}
{f7}{fheat}
<h3>Which way round? Cycle → HRV, or HRV → delayed period</h3>
<p>If the cycle drives HRV, HRV should follow the same within-cycle pattern every month regardless of cycle length (Figures {fn(f7)}–{fn(fheat)}). If low HRV (stress, load) delays the period, it would have to act <i>before ovulation</i>, because a late period almost always means late ovulation (the luteal phase is fairly fixed). So follicular-phase HRV/stress should predict cycle length.</p>
{f8}
{dir_tbl_html}
<details><summary>Per-cycle table</summary>{table(cyc_tbl)}</details>
<h3>Cycle history, 2023–2026 (Clover)</h3>{fhist}{fmonth}<p>{f"{len(no_months)} calendar months in the tracking window had no period at all." if PUBLIC else f"Months with no period: <b>{', '.join(no_months) or 'none'}</b>."} {plen_note}</p>{table(yr, "Cycle length by year (by cycle start).")}
</section>

<section><h2>6. Stress and sleep stages</h2>
{f9}{f10}</section>

<section><h2>7. Marathon training block</h2>
{f11}{f12}{f13}{fbub}
<h3>Resting heart rate through the training block</h3>{fstrain}{table(ph_rest, 'Mean resting physiology by training period.')}{strain_tbl}
{wk_corr}
{run_corr}
</section>

<section><h2>8. Predicting sleep</h2>
{f14}
{table(model_tbl, "Model performance. Out-of-sample R² compares hold-out error with predicting the training mean; values near 0 mean little predictive power beyond the average.")}
</section>

<section><h2>9. Limitations</h2>
<ul>
<li>Single subject (n = 1), observational data: correlations, not causal effects. Training, study stress, season and cycle all overlap in time.</li>
<li>Only {overlap_cycles} completed cycles overlap with HRV data, so the cycle-level direction test (Figure {fn(f8)}) has very low power. A null result there is not evidence of no effect.</li>
<li>Ovulation is estimated, not measured. Irregular cycles (25–76 days here) make the 14-day luteal assumption less certain. Basal body temperature or LH tests would sharpen the phase boundaries.</li>
<li>Garmin sleep staging from wrist PPG/accelerometry agrees only moderately with polysomnography, especially for deep vs light sleep. Garmin “stress” is itself derived from HRV, so stress–HRV links are partly built in.</li>
<li>Weekly pace mixes easy runs, workouts and races. Efficiency (speed ÷ HR) partly adjusts for this, but terrain, heat and workout type are not controlled.</li>
{"<li>Life events such as exam periods were not labelled. Adding them as a variable is a natural next step.</li>" if PUBLIC else "<li>Study/exam periods were not labelled in this version. Adding them as a variable is the obvious next step.</li>"}
</ul></section>
<footer>{"Pipeline: raw Garmin + Clover exports → PySpark ETL → SciPy / statsmodels → Plotly. Personal dates have been removed from this public version." if PUBLIC else "Data tables: <code>data/daily.csv</code>, <code>data/runs.csv</code>, <code>data/cycles.csv</code>. Pipeline: <code>etl.py</code> (PySpark) → <code>report.py</code>."}</footer>
"""

css = """
body{margin:0;background:#f9f9f7;color:#0b0b0b;font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:32px 28px 64px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:20px;margin:40px 0 12px;padding-top:12px;border-top:1px solid #e1e0d9}
h3{font-size:16px;margin:24px 0 8px} .sub{color:#52514e;margin:0}
.tiles{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:24px 0}
.tile{background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:8px;padding:14px}
.tl{font-size:12px;color:#52514e}.tv{font-size:26px;font-weight:600;margin:2px 0}.ts{font-size:12px;color:#898781}
figure{margin:20px 0;background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:8px;padding:8px 8px 4px}
figcaption{font-size:13px;color:#52514e;padding:6px 8px 8px}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13px;background:#fcfcfb;font-variant-numeric:tabular-nums}
caption{text-align:left;font-size:13px;color:#52514e;padding:6px 0;caption-side:top}
th,td{padding:6px 10px;border-bottom:1px solid #e1e0d9;text-align:left} th{color:#52514e;font-weight:600}
details{margin:8px 0} summary{cursor:pointer;color:#2a78d6} code{font-size:13px;background:#f0efec;padding:1px 4px;border-radius:4px}
.findings li{margin:6px 0} footer{margin-top:40px;color:#898781;font-size:13px}
@media print{figure,table{break-inside:avoid}}
"""
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Sleep &amp; HRV study</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{css}</style>
<script>{plotly.offline.get_plotlyjs()}</script></head><body><main>{body}</main></body></html>"""
open(OUT_HTML, "w").write(page)

if PUBLIC:
    from markdownify import markdownify
    md = body
    md = re.sub(r'<section class="tiles">.*?</section>', "TILESTOKEN", md, flags=re.S)
    md = re.sub(r"<figure><!--FIG:(\d+)-->.*?</figure>", lambda m: f"\n\nFIGTOKEN{m.group(1)}\n\n", md, flags=re.S)
    md = re.sub(r"<table><!--TBL:(\d+)-->.*?</table>", lambda m: f"\n\nTBLTOKEN{m.group(1)}\n\n", md, flags=re.S)
    md = re.sub(r"<details><summary>(.*?)</summary>", r"<h4>\1</h4>", md).replace("</details>", "")
    md = markdownify(md, heading_style="ATX", bullets="-")
    md = md.replace("TILESTOKEN", pd.DataFrame([[b, a, c] for a, b, c in tiles], columns=["Value", "Metric", "Detail"]).to_markdown(index=False))
    md = re.sub(r"FIGTOKEN(\d+)", lambda m: MD_FIGS[int(m.group(1))], md)
    md = re.sub(r"TBLTOKEN(\d+)", lambda m: MD_TBLS[int(m.group(1))], md)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    md_path = os.path.splitext(OUT_HTML)[0] + ".md"
    open(md_path, "w").write(md)
    print("wrote", md_path)
print("wrote", OUT_HTML, "figures:", FIG_N[0])
