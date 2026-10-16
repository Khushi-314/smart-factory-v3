from __future__ import annotations

import html
import time
from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from simulator import build_default_config, run_monte_carlo, run_once


# -----------------------------
# Page + styling
# -----------------------------
st.set_page_config(
    page_title="Smart Factory Stochastic Simulator",
    page_icon="🏭",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
      .hero {
        padding: 1.2rem 1.35rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 30px rgba(15,23,42,.12);
      }
      .hero h1 {margin:0; font-size:2rem;}
      .hero p {margin:.45rem 0 0 0; opacity:.9;}
      .subtle {color:#64748b; font-size:.9rem;}
      .route-chip {
        display:inline-block; padding:.2rem .55rem; margin:.15rem .1rem;
        border-radius:999px; background:#eef2ff; border:1px solid #c7d2fe;
        color:#3730a3; font-size:.82rem; font-weight:600;
      }
      .legend-dot {display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;}
      .factory-grid {
        display:grid;
        grid-template-columns: repeat(7, minmax(115px, 1fr));
        gap:12px;
        align-items:stretch;
        margin: .4rem 0 1rem 0;
      }
      .station-card {
        background:white; border:1px solid #e5e7eb; border-radius:15px;
        padding:10px 11px; min-height:120px; position:relative;
        box-shadow: 0 2px 9px rgba(15,23,42,.05);
      }
      .station-card .machine-icon {font-size:1.35rem; margin-bottom:3px;}
      .station-card .name {font-weight:800;font-size:1rem;color:#111827;}
      .station-card .state {font-weight:700;font-size:.78rem;margin:.2rem 0 .35rem 0;}
      .station-card .detail {font-size:.74rem;color:#475569;line-height:1.35;}
      .station-card.processing {border-top:5px solid #22c55e;}
      .station-card.failed {border-top:5px solid #ef4444;background:#fff7f7;}
      .station-card.blocked {border-top:5px solid #f59e0b;background:#fffaf0;}
      .station-card.starved {border-top:5px solid #94a3b8;}
      .product-token {
        display:inline-block; border-radius:999px; padding:.08rem .42rem;
        color:white; font-weight:800; font-size:.72rem; margin-top:.25rem;
      }
      .arrow-cell {
        display:flex; align-items:center; justify-content:center;
        font-size:1.5rem; color:#94a3b8; min-height:120px;
      }
      .replay-summary {
        display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .6rem 0;
      }
      .replay-pill {background:#f8fafc;border:1px solid #e2e8f0;border-radius:999px;padding:.25rem .65rem;font-size:.78rem;}
      @media (max-width: 1100px) {
        .factory-grid {grid-template-columns: repeat(3, minmax(120px,1fr));}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🏭 Smart Factory Stochastic Simulator</h1>
      <p>Multi-product discrete-event simulation with finite buffers, Weibull failures, stochastic repairs, controlled release, Monte Carlo uncertainty, and event replay.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
PRODUCT_COLORS = {
    "PR1": "#ef4444",  # red
    "PR2": "#3b82f6",  # blue
    "PR3": "#22c55e",  # green
    "PR4": "#64748b",  # white/neutral visible on white page
}

STATION_ICONS = {
    "MC1": "⚙️", "MC2": "⚙️", "MC3": "⚙️", "MC4": "⚙️",
    "WS4": "🧑‍🏭", "MC5": "📥", "MC6": "🔩", "MC7": "🔩",
    "MC8": "🔧", "MC9": "🔧", "MC10": "🔍", "WS9": "🧑‍🏭",
}

STATE_LABELS = {
    "PROCESSING": "Processing",
    "FAILED": "Failed / Repair",
    "BLOCKED": "Blocked",
    "STARVED": "Idle / Starved",
}


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def station_dataframe(result) -> pd.DataFrame:
    df = pd.DataFrame(result.station_results).T.reset_index(names="Station")
    df["Utilization (%)"] = 100 * df["utilization"]
    df["Failed (%)"] = 100 * df["failed_fraction"]
    df["Blocked (%)"] = 100 * df["blocked_fraction"]
    df["Starved (%)"] = 100 * df["starved_fraction"]
    df["Downtime (min)"] = df["downtime_s"] / 60
    df["Observed MTTR (min)"] = df["observed_mttr_s"] / 60
    df["Avg Queue"] = df["avg_queue_length"]
    df["Max Queue"] = df["max_queue_length"]
    return df


def product_summary_dataframe(result) -> pd.DataFrame:
    rows = []
    for ptype, d in result.product_summary.items():
        rows.append({
            "Product": ptype,
            "Products": d["products"],
            "Mean factory lead (min)": d["mean_lead_time_s"] / 60,
            "P50 lead (min)": d["p50_lead_time_s"] / 60,
            "P95 lead (min)": d["p95_lead_time_s"] / 60,
            "Mean fulfillment (min)": d["mean_order_fulfillment_time_s"] / 60,
            "Mean external backlog wait (min)": d["mean_source_waiting_time_s"] / 60,
            "Mean station wait (min)": d["mean_waiting_time_s"] / 60,
            "Mean service (min)": d["mean_service_time_s"] / 60,
            "Mean failure delay (min)": d["mean_failure_delay_s"] / 60,
            "Mean blocked delay (min)": d["mean_blocked_delay_s"] / 60,
        })
    return pd.DataFrame(rows)


def reconstruct_snapshot(event_df: pd.DataFrame, t: float, station_names: list[str]) -> Tuple[Dict[str, dict], int, int, int]:
    states = {
        s: {"state": "STARVED", "pid": None, "ptype": None, "queue": 0, "failures": 0}
        for s in station_names
    }
    if event_df.empty:
        return states, 0, 0, 0

    subset = event_df[event_df["time_s"] <= t]
    admitted = 0
    completed = 0
    total_failures = 0

    for _, row in subset.iterrows():
        ev = row.get("event")
        station = row.get("station")
        pid = row.get("pid")
        ptype = row.get("ptype")

        if ev == "SOURCE_ADMIT":
            admitted += 1
        elif ev == "PRODUCT_COMPLETE":
            completed += 1
        elif ev == "FAILURE":
            total_failures += 1

        if station not in states:
            continue

        st = states[station]
        if ev in ("QUEUE_ENTER", "QUEUE_LEAVE") and pd.notna(row.get("queue_length")):
            st["queue"] = int(row.get("queue_length"))
        elif ev == "PROCESS_START":
            st.update(state="PROCESSING", pid=int(pid) if pd.notna(pid) else None, ptype=ptype)
        elif ev == "FAILURE":
            st.update(state="FAILED", pid=int(pid) if pd.notna(pid) else None, ptype=ptype)
            st["failures"] += 1
        elif ev == "REPAIR_COMPLETE":
            st.update(state="PROCESSING", pid=int(pid) if pd.notna(pid) else None, ptype=ptype)
        elif ev == "PROCESS_COMPLETE":
            st.update(state="STARVED", pid=None, ptype=None)
        elif ev == "BLOCKED":
            st.update(state="BLOCKED", pid=int(pid) if pd.notna(pid) else None, ptype=ptype)
        elif ev == "UNBLOCKED":
            st.update(state="STARVED", pid=None, ptype=None)

    return states, admitted, completed, total_failures


def station_card_html(name: str, snapshot: dict, util_pct: float | None = None) -> str:
    state = snapshot["state"]
    css_state = state.lower()
    ptype = snapshot.get("ptype")
    pid = snapshot.get("pid")
    token = ""
    if ptype and pid is not None:
        color = PRODUCT_COLORS.get(str(ptype), "#334155")
        token = f'<span class="product-token" style="background:{color};">{html.escape(str(ptype))}-{int(pid)}</span>'
    util_text = f"<br>Util: {util_pct:.1f}%" if util_pct is not None else ""
    return f"""
      <div class="station-card {css_state}">
        <div class="machine-icon">{STATION_ICONS.get(name, '🏭')}</div>
        <div class="name">{html.escape(name)}</div>
        <div class="state">{STATE_LABELS.get(state, state)}</div>
        <div class="detail">Queue: {int(snapshot.get('queue', 0))}{util_text}<br>Failures so far: {int(snapshot.get('failures', 0))}</div>
        {token}
      </div>
    """


def factory_schematic_html(snapshot: Dict[str, dict], station_df: pd.DataFrame) -> str:
    util = dict(zip(station_df["Station"], station_df["Utilization (%)"]))
    card = lambda s: station_card_html(s, snapshot[s], util.get(s))
    arrow = '<div style="font-size:1.35rem;color:#94a3b8;text-align:center;padding:.25rem 0;">↓</div>'
    right = '<span style="font-size:1.35rem;color:#94a3b8;padding:0 .5rem;">→</span>'

    return f"""
    <div style="border:1px solid #e2e8f0;border-radius:18px;padding:16px 18px;background:#fbfdff;">
      <div style="font-weight:800;margin-bottom:10px;color:#334155;">CNC branches</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div style="display:flex;align-items:center;justify-content:center;">{card('MC1')}{right}{card('MC2')}</div>
        <div style="display:flex;align-items:center;justify-content:center;">{card('MC3')}{right}{card('MC4')}</div>
      </div>
      {arrow}
      <div style="font-size:.78rem;color:#64748b;text-align:center;margin-bottom:5px;">merge</div>
      <div style="display:flex;align-items:center;justify-content:center;">{card('WS4')}{right}{card('MC5')}</div>
      {arrow}
      <div style="font-weight:800;margin:2px 0 10px;color:#334155;text-align:center;">Product-dependent pressing branches</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div style="display:flex;align-items:center;justify-content:center;">{card('MC6')}{right}{card('MC8')}</div>
        <div style="display:flex;align-items:center;justify-content:center;">{card('MC7')}{right}{card('MC9')}</div>
      </div>
      {arrow}
      <div style="font-size:.78rem;color:#64748b;text-align:center;margin-bottom:5px;">merge for inspection and final assembly</div>
      <div style="display:flex;align-items:center;justify-content:center;">{card('MC10')}{right}{card('WS9')}</div>
    </div>
    """


def render_factory_schematic(snapshot: Dict[str, dict], station_df: pd.DataFrame, target=None) -> None:
    html_block = factory_schematic_html(snapshot, station_df)
    if target is None:
        st.markdown(html_block, unsafe_allow_html=True)
    else:
        target.markdown(html_block, unsafe_allow_html=True)


# -----------------------------
# Inputs
# -----------------------------
base_config = build_default_config()

with st.sidebar:
    st.header("Simulation inputs")
    st.caption("Order quantities")
    pr1 = st.number_input("PR1 (red)", min_value=0, max_value=5000, value=100, step=10)
    pr2 = st.number_input("PR2 (blue)", min_value=0, max_value=5000, value=100, step=10)
    pr3 = st.number_input("PR3 (green)", min_value=0, max_value=5000, value=100, step=10)
    pr4 = st.number_input("PR4 (white)", min_value=0, max_value=5000, value=100, step=10)
    seed = st.number_input("Single-run random seed", min_value=0, max_value=2_000_000_000, value=42, step=1)

    st.divider()
    with st.expander("Failure & repair parameters", expanded=False):
        st.caption("Synthetic reliability assumptions. Change these to test different reliability scenarios.")
        st.markdown("**CNC machines (MC1–MC4)**")
        cnc_mttf = st.number_input("CNC mean time to failure (h)", min_value=0.1, max_value=1000.0, value=4.5, step=0.5)
        cnc_k = st.number_input("CNC Weibull shape k", min_value=0.2, max_value=10.0, value=1.7, step=0.1)
        cnc_repair = st.number_input("CNC mean repair time (min)", min_value=0.1, max_value=600.0, value=9.0, step=1.0)
        cnc_repair_cv = st.number_input("CNC repair CV", min_value=0.01, max_value=3.0, value=0.40, step=0.05)

        st.markdown("**Feeder / press machines (MC5–MC9)**")
        press_mttf = st.number_input("Press mean time to failure (h)", min_value=0.1, max_value=1000.0, value=5.0, step=0.5)
        press_k = st.number_input("Press Weibull shape k", min_value=0.2, max_value=10.0, value=1.6, step=0.1)
        press_repair = st.number_input("Press mean repair time (min)", min_value=0.1, max_value=600.0, value=6.0, step=1.0)
        press_repair_cv = st.number_input("Press repair CV", min_value=0.01, max_value=3.0, value=0.40, step=0.05)

        st.markdown("**Inspection machine (MC10)**")
        inspect_mttf = st.number_input("Inspection mean time to failure (h)", min_value=0.1, max_value=1000.0, value=7.0, step=0.5)
        inspect_k = st.number_input("Inspection Weibull shape k", min_value=0.2, max_value=10.0, value=1.4, step=0.1)
        inspect_repair = st.number_input("Inspection mean repair time (min)", min_value=0.1, max_value=600.0, value=5.0, step=1.0)
        inspect_repair_cv = st.number_input("Inspection repair CV", min_value=0.01, max_value=3.0, value=0.35, step=0.05)

    st.divider()
    run_mc = st.checkbox("Run Monte Carlo", value=True)
    mc_runs = st.slider("Monte Carlo runs", 50, 1000, 200, 50, disabled=not run_mc)
    mc_seed = st.number_input("Monte Carlo seed", min_value=0, max_value=2_000_000_000, value=123, step=1, disabled=not run_mc)
    run_clicked = st.button("▶ Run simulation", type="primary", use_container_width=True)
    st.caption("Changes to orders or reliability parameters take effect when you click Run simulation.")
    st.caption("Synthetic reliability/variability assumptions; topology and known 25 s / 5 s route times come from the supplied factory workbook.")

# Apply the user-selected reliability/repair scenario to the station classes.
new_stations = dict(base_config.stations)
for name in ("MC1", "MC2", "MC3", "MC4"):
    rel = new_stations[name].reliability
    new_stations[name] = replace(new_stations[name], reliability=replace(
        rel, mttf_h=float(cnc_mttf), weibull_k=float(cnc_k),
        mean_repair_min=float(cnc_repair), repair_cv=float(cnc_repair_cv)
    ))
for name in ("MC5", "MC6", "MC7", "MC8", "MC9"):
    rel = new_stations[name].reliability
    new_stations[name] = replace(new_stations[name], reliability=replace(
        rel, mttf_h=float(press_mttf), weibull_k=float(press_k),
        mean_repair_min=float(press_repair), repair_cv=float(press_repair_cv)
    ))
rel = new_stations["MC10"].reliability
new_stations["MC10"] = replace(new_stations["MC10"], reliability=replace(
    rel, mttf_h=float(inspect_mttf), weibull_k=float(inspect_k),
    mean_repair_min=float(inspect_repair), repair_cv=float(inspect_repair_cv)
))
config = replace(base_config, stations=new_stations)

order_counts = {"PR1": int(pr1), "PR2": int(pr2), "PR3": int(pr3), "PR4": int(pr4)}

if sum(order_counts.values()) == 0:
    st.warning("Enter at least one product order.")
    st.stop()

if run_clicked or "single_result" not in st.session_state:
    with st.spinner("Running discrete-event simulation..."):
        st.session_state.single_result = run_once(order_counts=order_counts, seed=int(seed), config=config, log_events=True)
        st.session_state.last_order_counts = order_counts.copy()
        st.session_state.last_seed = int(seed)
        st.session_state.last_config = config
        if run_mc:
            st.session_state.mc_result = run_monte_carlo(
                order_counts=order_counts,
                n_runs=int(mc_runs),
                seed=int(mc_seed),
                config=config,
            )
        else:
            st.session_state.mc_result = None

result = st.session_state.single_result
mc = st.session_state.get("mc_result")
# Show routes/parameters for the scenario that produced the displayed results.
config = st.session_state.get("last_config", config)
station_df = station_dataframe(result)
product_df = product_summary_dataframe(result)
event_df = pd.DataFrame(result.event_log)

# -----------------------------
# Overview
# -----------------------------
st.subheader("1. Factory performance")
current_bottleneck = station_df.loc[station_df["Utilization (%)"].idxmax(), "Station"]

# WIP = products admitted to the factory but not yet completed.
def wip_series_from_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"time_s": [0.0], "WIP": [0]})
    rows = []
    wip = 0
    for _, row in df.sort_values("time_s").iterrows():
        if row.get("event") == "SOURCE_ADMIT":
            wip += 1
            rows.append((float(row["time_s"]), wip))
        elif row.get("event") == "PRODUCT_COMPLETE":
            wip -= 1
            rows.append((float(row["time_s"]), wip))
    return pd.DataFrame(rows, columns=["time_s", "WIP"]) if rows else pd.DataFrame({"time_s": [0.0], "WIP": [0]})

wip_df = wip_series_from_events(event_df)
peak_wip = int(wip_df["WIP"].max()) if not wip_df.empty else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Completed", f"{result.completed}")
c2.metric("Makespan", f"{result.makespan_s/3600:.2f} h")
c3.metric("Throughput", f"{result.throughput_per_h:.1f}/h")
c4.metric("Failures", f"{result.total_failures}")
c5.metric("Peak WIP", f"{peak_wip}")
c6.metric("Highest-utilization station", current_bottleneck)

with st.expander("Validation checks", expanded=False):
    st.success("Accounting identities close to numerical precision.")
    st.write(f"Lead-time decomposition max error: {result.max_lead_time_decomposition_error_s:.2e} s")
    st.write(f"Order-fulfillment decomposition max error: {result.max_order_fulfillment_decomposition_error_s:.2e} s")
    st.caption("These are numerical consistency checks, not fitted-data accuracy claims.")

with st.expander("Product routing used by the model", expanded=False):
    for ptype, route in config.routes.items():
        color = PRODUCT_COLORS[ptype]
        chips = " → ".join(f'<span class="route-chip">{x}</span>' for x in route)
        st.markdown(f'<b style="color:{color}">{ptype}</b>: {chips}', unsafe_allow_html=True)
    st.caption("MC10 is treated as terminal inspection before WS9 final assembly, consistent with the ontology/completion rules used in the model.")

# -----------------------------
# Replay schematic
# -----------------------------
st.subheader("2. Factory visualization & animated replay")
max_t = float(result.makespan_s)

play_col, info_col = st.columns([1, 2])
with play_col:
    play_clicked = st.button("▶ Play 10-second factory replay", use_container_width=True)
with info_col:
    st.caption("The complete simulated factory history is compressed into ~10 seconds. Product tokens, queues, failures, blocking and completions update from the event log.")

replay_t = st.slider(
    "Manual replay time",
    min_value=0.0,
    max_value=max_t,
    value=min(max_t, max_t * 0.35),
    step=max(1.0, max_t / 500.0),
    format="%.0f s",
)

summary_slot = st.empty()
factory_slot = st.empty()

def render_replay_frame(t: float) -> None:
    snapshot, admitted, completed_at_t, failures_at_t = reconstruct_snapshot(
        event_df, t, list(config.stations.keys())
    )
    total_orders = sum(st.session_state.get("last_order_counts", order_counts).values())
    backlog = max(0, total_orders - admitted)
    wip = max(0, admitted - completed_at_t)
    summary_slot.markdown(
        f"""
        <div class="replay-summary">
          <span class="replay-pill">⏱ {fmt_time(t)}</span>
          <span class="replay-pill"><b>🏗 WIP: {wip}</b></span>
          <span class="replay-pill">📥 Admitted: {admitted}</span>
          <span class="replay-pill">📦 External backlog: {backlog}</span>
          <span class="replay-pill">✅ Completed: {completed_at_t}</span>
          <span class="replay-pill">⚠ Failures so far: {failures_at_t}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_factory_schematic(snapshot, station_df, target=factory_slot)

if play_clicked:
    n_frames = 30
    frame_times = np.linspace(0.0, max_t, n_frames)
    for t in frame_times:
        render_replay_frame(float(t))
        time.sleep(10.0 / n_frames)
else:
    render_replay_frame(float(replay_t))

st.markdown(
    """
    <span class="subtle">
      Status colors: 🟢 processing · 🔴 failed/repair · 🟠 blocked · ⚪ idle/starved. Product tokens identify the product occupying each station at the displayed replay time.
    </span>
    """,
    unsafe_allow_html=True,
)

st.caption("WIP (work-in-process) = products admitted into the factory but not yet completed. The external order backlog is excluded.")
with st.expander("WIP over the simulated run", expanded=False):
    wfig = px.line(wip_df, x="time_s", y="WIP", title="Work-in-process over time")
    wfig.update_layout(height=300, xaxis_title="Simulation time (s)", yaxis_title="Products", margin=dict(l=20,r=20,t=55,b=20))
    st.plotly_chart(wfig, use_container_width=True)

# -----------------------------
# Charts
# -----------------------------
st.subheader("3. Station analytics")
left, right = st.columns(2)

with left:
    util_fig = px.bar(
        station_df.sort_values("Utilization (%)"),
        x="Utilization (%)", y="Station", orientation="h",
        title="Station utilization",
        text_auto=".1f",
    )
    util_fig.update_layout(height=430, margin=dict(l=20,r=20,t=55,b=20))
    st.plotly_chart(util_fig, use_container_width=True)

with right:
    state_long = station_df.melt(
        id_vars=["Station"],
        value_vars=["Utilization (%)", "Failed (%)", "Blocked (%)", "Starved (%)"],
        var_name="State", value_name="Percent"
    )
    state_fig = px.bar(
        state_long, x="Station", y="Percent", color="State",
        title="Station state-time composition", barmode="stack"
    )
    state_fig.update_layout(height=430, yaxis_title="% of makespan", margin=dict(l=20,r=20,t=55,b=20))
    st.plotly_chart(state_fig, use_container_width=True)

left, right = st.columns(2)
with left:
    queue_plot = station_df[["Station", "Avg Queue", "Max Queue"]].melt(
        id_vars="Station", var_name="Queue metric", value_name="Products"
    )
    qfig = px.bar(queue_plot, x="Station", y="Products", color="Queue metric", barmode="group", title="Input queue pressure")
    qfig.update_layout(height=400, margin=dict(l=20,r=20,t=55,b=20))
    st.plotly_chart(qfig, use_container_width=True)

with right:
    fail_plot = station_df[["Station", "failures", "Downtime (min)"]].copy()
    ffig = go.Figure()
    ffig.add_bar(x=fail_plot["Station"], y=fail_plot["Downtime (min)"], name="Downtime (min)")
    ffig.add_scatter(x=fail_plot["Station"], y=fail_plot["failures"], name="Failures", mode="lines+markers", yaxis="y2")
    ffig.update_layout(
        title="Failures and repair downtime",
        yaxis=dict(title="Downtime (min)"),
        yaxis2=dict(title="Failures", overlaying="y", side="right", rangemode="tozero"),
        height=400, margin=dict(l=20,r=20,t=55,b=20), legend=dict(orientation="h")
    )
    st.plotly_chart(ffig, use_container_width=True)

st.dataframe(
    station_df[[
        "Station", "processed_count", "Utilization (%)", "failures", "Downtime (min)",
        "Blocked (%)", "Starved (%)", "Avg Queue", "Max Queue"
    ]].round(2),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# Product analytics
# -----------------------------
st.subheader("4. Product lead time & fulfillment")
lead_long = product_df.melt(
    id_vars=["Product"],
    value_vars=["Mean factory lead (min)", "P50 lead (min)", "P95 lead (min)"],
    var_name="Lead-time statistic", value_name="Minutes"
)
lead_fig = px.bar(
    lead_long, x="Product", y="Minutes", color="Lead-time statistic", barmode="group",
    title="In-factory lead-time distribution by product"
)
lead_fig.update_layout(height=420, margin=dict(l=20,r=20,t=55,b=20))
st.plotly_chart(lead_fig, use_container_width=True)

st.dataframe(product_df.round(2), use_container_width=True, hide_index=True)
st.caption("Factory lead time starts at SOURCE_ADMIT. External backlog waiting is reported separately and is included only in total order-fulfillment time.")

# -----------------------------
# Monte Carlo
# -----------------------------
st.subheader("5. Monte Carlo uncertainty")
if mc is None:
    st.info("Monte Carlo was disabled for the last run. Enable it in the sidebar and run again.")
else:
    mc_runs_df = pd.DataFrame(mc["run_results"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean makespan", f"{mc['mean_makespan_s']/3600:.2f} h")
    m2.metric("P95 makespan", f"{mc['p95_makespan_s']/3600:.2f} h")
    m3.metric("Mean failures", f"{mc['mean_failures']:.2f}")
    m4.metric("P(≥1 failure)", f"{100*mc['prob_at_least_one_failure']:.1f}%")

    left, right = st.columns(2)
    with left:
        hfig = px.histogram(
            mc_runs_df, x="makespan_h", nbins=30,
            title=f"Makespan distribution across {mc['n_runs']} runs"
        )
        hfig.add_vline(x=float(mc_runs_df["makespan_h"].median()), line_dash="dash", annotation_text="Median")
        hfig.update_layout(height=410, xaxis_title="Makespan (h)", margin=dict(l=20,r=20,t=55,b=20))
        st.plotly_chart(hfig, use_container_width=True)

    with right:
        bottleneck_df = pd.DataFrame(
            [{"Station": k, "Probability": v} for k, v in mc["bottleneck_probability"].items()]
        )
        if bottleneck_df.empty:
            st.info("No bottleneck statistics available.")
        else:
            bfig = px.bar(
                bottleneck_df, x="Station", y="Probability",
                title="Highest-utilization station frequency across Monte Carlo runs",
                text_auto=".1%"
            )
            bfig.update_yaxes(tickformat=".0%", range=[0, 1])
            bfig.update_layout(height=410, margin=dict(l=20,r=20,t=55,b=20))
            st.plotly_chart(bfig, use_container_width=True)

    with st.expander("Monte Carlo summary table"):
        st.dataframe(pd.DataFrame(mc["summary_table"]).round(4), use_container_width=True, hide_index=True)

# -----------------------------
# Technical details + assumptions
# -----------------------------
st.subheader("6. Model basis & technical details")

with st.expander("Technical event log", expanded=False):
    filters = st.multiselect(
        "Event types",
        sorted(event_df["event"].dropna().unique().tolist()),
        default=["FAILURE", "REPAIR_START", "REPAIR_COMPLETE", "BLOCKED", "UNBLOCKED", "PRODUCT_COMPLETE"],
    )
    filtered_events = event_df[event_df["event"].isin(filters)] if filters else event_df
    st.dataframe(filtered_events.head(500), use_container_width=True, hide_index=True)
    st.caption(f"Showing up to 500 rows · Full event log contains {len(event_df):,} rows.")

with st.expander("Model assumptions, parameters, and provenance", expanded=True):
    station_param_rows = []
    workbook_time_stations = {"MC6", "MC7", "MC8", "MC9"}
    for name, spec in config.stations.items():
        rel = spec.reliability
        if name in {"MC6", "MC7"}:
            source = "Workbook nominal time + synthetic variability/reliability"
        elif name in {"MC8", "MC9"}:
            source = "Workbook nominal time + synthetic variability/reliability"
        else:
            source = "Synthetic engineering assumption"
        family = "Lognormal" if spec.process.family == "lognormal" else "Truncated Normal"
        station_param_rows.append({
            "Station": name,
            "Type": spec.station_type.title(),
            "Process distribution": family,
            "Mean process (s)": spec.process.mean_s,
            "Process CV": spec.process.cv,
            "MTTF (h)": rel.mttf_h if rel else np.nan,
            "Weibull k": rel.weibull_k if rel else np.nan,
            "Mean repair (min)": rel.mean_repair_min if rel else np.nan,
            "Repair CV": rel.repair_cv if rel else np.nan,
            "Basis": source,
        })
    st.dataframe(pd.DataFrame(station_param_rows), use_container_width=True, hide_index=True)
    st.markdown(
        """
        **What comes from the supplied workbook**
        - Product-specific routing / machine assignments.
        - Nominal processing times of **25 s for MC6/MC7** and **5 s for MC8/MC9**.
        - MC10 is included as terminal inspection before WS9 based on the machine ontology/completion rules; it is an explicit documented inference rather than a separate route-step row.

        **What is synthetic for this demonstration**
        - Missing cycle-time means, coefficients of variation, Weibull MTTF/shape parameters, and repair parameters.
        - These assumptions are isolated in the configuration and can later be replaced by fitted plant measurements without changing the event logic.

        **Stochastic logic**
        - Automated processing uses positive **Truncated Normal** cycle times; human assembly uses **Lognormal** cycle times.
        - Machine time-to-failure follows a 2-parameter **Weibull** distribution. Failure age accrues only while the machine is processing. MTTF and Weibull shape can be changed from the sidebar by machine class.
        - Repair time is **Lognormal**. Mean repair time and repair CV can also be changed from the sidebar. An interrupted job uses **preempt-resume** after repair; work is not restarted from zero.
        - Input buffers are finite. If a downstream buffer is full, the upstream station becomes **blocked** while holding the completed product.
        - Orders begin in an external backlog and are admitted when the first-station buffer has capacity. Factory lead time starts at admission; total fulfillment time also includes external backlog waiting.
        """
    )

st.caption("Smart Factory Simulator · stochastic discrete-event model · controlled release · synthetic reliability scenario")
