"""fullTDAIL1x.py: Unified FPL Decision Science & Tactical Reasoning Hub

Runs in MSYS2 (UCRT64) Python 3.14 on a single Streamlit process.
Includes:
  1. Pure Parametric Expected Value (EV) MILP Optimizer (HiGHS/PuLP)
  2. Robust CVaR / Quantile Asymmetry MILP Optimizer
  3. Non-Parametric Empirical Telemetry & Distribution Profiler
  4. Resilient Gemini LLM Tactical Advisor (Auto-failover + Closed-socket recovery)
"""

import os
import time
from google import genai
from google.genai import errors
import numpy as np
import pandas as pd
import plotly.express as px
import pulp
from scipy import stats
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & STATE INITIALIZATION
# ==============================================================================
st.set_page_config(
    page_title="FPL Analytical & Tactical Decision Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "chat_history" not in st.session_state:
  st.session_state.chat_history = []
if "ev_solution" not in st.session_state:
  st.session_state.ev_solution = None
if "cvar_solution" not in st.session_state:
  st.session_state.cvar_solution = None


# ==============================================================================
# 2. RESILIENT GOOGLE GENAI WRAPPER (Auto-Retry + Model Failover)
# ==============================================================================
def get_genai_client():
  """Returns an active genai.Client instance without context-manager auto-close."""
  return genai.Client()


def send_chat_resilient(
    prompt_msg: str,
    client=None,
    primary_chat=None,
    primary_model="gemini-3.5-flash",
    fallback_model="gemini-3.5-flash-lite",
    max_retries=3,
    initial_delay=2.0,
):
  """Dispatches chat prompt with exponential backoff on 503s, closed-socket

  auto-recovery, and failover to a secondary model endpoint.
  """
  if client is None:
    client = get_genai_client()
  if primary_chat is None:
    primary_chat = client.chats.create(model=primary_model)

  delay = initial_delay

  # Phase 1: Try Primary Model with Exponential Backoff
  for attempt in range(max_retries):
    try:
      return primary_chat.send_message(prompt_msg)
    except RuntimeError as e:
      if "client has been closed" in str(e):
        client = get_genai_client()
        primary_chat = client.chats.create(model=primary_model)
        continue
      raise e
    except errors.ServerError as e:
      if e.code == 503 and attempt < max_retries - 1:
        time.sleep(delay)
        delay *= 2.0
      else:
        break
    except errors.ClientError as e:
      if e.code in (400, 404):
        break
      raise e
    except Exception as e:
      raise e

  # Phase 2: Failover to Secondary Model
  try:
    if getattr(client, "_api_client", None) and getattr(
        client._api_client, "_httpx_client", None
    ):
      if client._api_client._httpx_client.is_closed:
        client = get_genai_client()

    fallback_chat = client.chats.create(model=fallback_model)
    return fallback_chat.send_message(prompt_msg)
  except Exception as e:
    raise RuntimeError(
        f"Both {primary_model} and {fallback_model} failed: {e}"
    )


# ==============================================================================
# 3. TELEMETRY COMPUTATION ENGINE (Parametric & Non-Parametric)
# ==============================================================================
def compute_robust_metrics(history: np.ndarray) -> dict:
  """Calculates non-parametric shape, dispersion, and parametric Poisson/Gaussian moments."""
  clean = history[~np.isnan(history)]
  if len(clean) < 3:
    return {
        "mean_ev": 0.0,
        "std_dev": 0.0,
        "median": 0.0,
        "trimean": 0.0,
        "iqr": 0.0,
        "bowley_asym": 0.0,
        "p_haul": 0.0,
        "cvar_tail": 0.0,
    }

  # Parametric moments
  mean_ev = float(np.mean(clean))
  std_dev = float(np.std(clean, ddof=1))

  # Non-parametric order statistics
  q25, q50, q75 = np.percentile(clean, [25, 50, 75])
  iqr = q75 - q25

  # Bowley / Galton Quantile Asymmetry: ((Q75 - Q50) - (Q50 - Q25)) / IQR
  bowley = ((q75 - q50) - (q50 - q25)) / iqr if iqr > 0 else 0.0
  trimean = (q25 + 2 * q50 + q75) / 4.0
  p_haul = float(np.mean(clean >= 8.0))

  alpha_val = np.percentile(clean, 20)
  tail = clean[clean <= alpha_val]
  cvar_tail = float(np.mean(tail)) if len(tail) > 0 else float(alpha_val)

  return {
      "mean_ev": mean_ev,
      "std_dev": std_dev,
      "median": float(q50),
      "trimean": float(trimean),
      "iqr": float(iqr),
      "bowley_asym": float(bowley),
      "p_haul": p_haul,
      "cvar_tail": cvar_tail,
  }


@st.cache_data
def get_mock_telemetry_dataset() -> pd.DataFrame:
  """Generates representative dataset with parametric expectations and empirical distributions."""
  rng = np.random.default_rng(42)
  roster = [
      ("Erling Haaland", "FWD", "MCI", 15.2, 12, 1.4),
      ("Mohamed Salah", "MID", "LIV", 12.8, 10, 1.2),
      ("Cole Palmer", "MID", "CHE", 10.8, 8, 1.8),
      ("Bukayo Saka", "MID", "ARS", 10.1, 7, 0.8),
      ("Alexander Isak", "FWD", "NEW", 8.5, 6, 1.1),
      ("Eberechi Eze", "MID", "CRY", 6.8, 5, 1.5),
      ("Trent Alexander-Arnold", "DEF", "LIV", 7.2, 6, 1.3),
      ("Gabriel Magalhães", "DEF", "ARS", 6.0, 5, 0.4),
      ("Tyrick Mitchell", "DEF", "CRY", 5.0, 4, 1.6),
      ("Josko Gvardiol", "DEF", "MCI", 6.0, 5, 0.9),
      ("David Raya", "GKP", "ARS", 5.5, 4, 0.3),
      ("Mark Flekken", "GKP", "BRE", 4.5, 3, 0.2),
      ("Ibrahim Sangaré", "MID", "NFO", 4.5, 2, 0.1),
      ("Wout Faes", "DEF", "LEI", 4.1, 2, -0.2),
      ("Cameron Archer", "FWD", "SOU", 4.9, 3, 0.5),
  ]

  records = []
  for pid, (name, pos, team, cost, shape_k, asym_target) in enumerate(roster, 1):
    sample = rng.gamma(shape=shape_k, scale=1.0, size=38)
    metrics = compute_robust_metrics(sample)

    records.append({
        "id": pid,
        "web_name": name,
        "element_type": {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}[pos],
        "position": pos,
        "team": team,
        "now_cost": cost,
        "mean_ev": metrics["mean_ev"],
        "std_dev": metrics["std_dev"],
        "median": metrics["median"],
        "trimean": metrics["trimean"],
        "iqr": metrics["iqr"],
        "bowley_asym": metrics["bowley_asym"],
        "p_haul": metrics["p_haul"],
        "cvar_tail": metrics["cvar_tail"],
        # Parametric per-gameweek expectations
        "parametric_ev": {
            gw: round(metrics["mean_ev"] * (1.0 + rng.uniform(-0.12, 0.12)), 2)
            for gw in range(4, 8)
        },
        # Non-parametric trimean projections
        "trimean_pts": {
            gw: round(metrics["trimean"] * (1.0 + rng.uniform(-0.12, 0.12)), 2)
            for gw in range(4, 8)
        },
        "raw_history": sample.tolist(),
    })
  return pd.DataFrame(records)


df_players = get_mock_telemetry_dataset()


# ==============================================================================
# 4. SOLVER SUITE (HiGHS / CBC Fallback)
# ==============================================================================
def get_pulp_solver():
  """Selects native HiGHS solver if present in PATH, else falls back to bundled CBC."""
  if pulp.HiGHS_CMD().available():
    return pulp.HiGHS_CMD(msg=False)
  return pulp.PULP_CBC_CMD(msg=False)


def build_base_milp_model(
    players_df: pd.DataFrame,
    gameweeks: list,
    current_squad_ids: list,
    itb: float,
    free_transfers: int,
    force_bb_gw,
    force_wc_gw,
    allow_solver_chips: bool,
):
  """Builds shared structural constraints: 15-man squad, positions, flow continuity, and chips."""
  prob = pulp.LpProblem("FPL_MultiGW_Optimizer", pulp.LpMaximize)
  p_ids = players_df["id"].tolist()
  p_lookup = players_df.set_index("id").to_dict("index")

  x = pulp.LpVariable.dicts("squad", (p_ids, gameweeks), cat=pulp.LpBinary)
  s = pulp.LpVariable.dicts("start", (p_ids, gameweeks), cat=pulp.LpBinary)
  c = pulp.LpVariable.dicts("captain", (p_ids, gameweeks), cat=pulp.LpBinary)
  tin = pulp.LpVariable.dicts("transfer_in", (p_ids, gameweeks), cat=pulp.LpBinary)
  tout = pulp.LpVariable.dicts("transfer_out", (p_ids, gameweeks), cat=pulp.LpBinary)

  bb = pulp.LpVariable.dicts("bench_boost", gameweeks, cat=pulp.LpBinary)
  wc = pulp.LpVariable.dicts("wildcard", gameweeks, cat=pulp.LpBinary)
  cost = pulp.LpVariable.dicts("hit_cost", gameweeks, lowBound=0, cat=pulp.LpInteger)

  # Chip Application Constraints
  for w in gameweeks:
    if force_bb_gw == w:
      prob += bb[w] == 1
    elif not allow_solver_chips:
      prob += bb[w] == 0

    if force_wc_gw == w:
      prob += wc[w] == 1
    elif not allow_solver_chips:
      prob += wc[w] == 0

    prob += bb[w] + wc[w] <= 1

  prob += pulp.lpSum([bb[w] for w in gameweeks]) <= 1
  prob += pulp.lpSum([wc[w] for w in gameweeks]) <= 1

  for i, w in enumerate(gameweeks):
    # Squad composition: 2 GKP, 5 DEF, 5 MID, 3 FWD
    prob += pulp.lpSum([x[p][w] for p in p_ids]) == 15
    prob += pulp.lpSum([x[p][w] for p in p_ids if p_lookup[p]["element_type"] == 1]) == 2
    prob += pulp.lpSum([x[p][w] for p in p_ids if p_lookup[p]["element_type"] == 2]) == 5
    prob += pulp.lpSum([x[p][w] for p in p_ids if p_lookup[p]["element_type"] == 3]) == 5
    prob += pulp.lpSum([x[p][w] for p in p_ids if p_lookup[p]["element_type"] == 4]) == 3

    # Starting XI count (15 if Bench Boost is active)
    prob += pulp.lpSum([s[p][w] for p in p_ids]) == 11 + 4 * bb[w]
    for p in p_ids:
      prob += s[p][w] <= x[p][w]

    # Captaincy: 1 captain chosen from starting XI
    prob += pulp.lpSum([c[p][w] for p in p_ids]) == 1
    for p in p_ids:
      prob += c[p][w] <= s[p][w]

    # Formations when BB is inactive: >=1 GKP, >=3 DEF, >=2 MID, >=1 FWD
    prob += pulp.lpSum([s[p][w] for p in p_ids if p_lookup[p]["element_type"] == 1]) >= 1
    prob += pulp.lpSum([s[p][w] for p in p_ids if p_lookup[p]["element_type"] == 2]) >= 3 * (1 - bb[w])
    prob += pulp.lpSum([s[p][w] for p in p_ids if p_lookup[p]["element_type"] == 3]) >= 2 * (1 - bb[w])
    prob += pulp.lpSum([s[p][w] for p in p_ids if p_lookup[p]["element_type"] == 4]) >= 1 * (1 - bb[w])

    # Budget cap
    prob += pulp.lpSum([x[p][w] * p_lookup[p]["now_cost"] for p in p_ids]) <= 100.0 + itb

    # State Flow Continuity
    prev_squad = (
        {p: (1 if p in current_squad_ids else 0) for p in p_ids}
        if i == 0
        else {p: x[p][gameweeks[i - 1]] for p in p_ids}
    )
    for p in p_ids:
      prob += x[p][w] == prev_squad[p] + tin[p][w] - tout[p][w]

    # Transfer Penalties: Suppressed via Wildcard (M=60)
    outs_count = pulp.lpSum([tout[p][w] for p in p_ids])
    ft = free_transfers if i == 0 else 1
    prob += cost[w] >= 4 * (outs_count - ft) - 60 * wc[w]
    prob += cost[w] >= 0

  return prob, {
      "x": x,
      "s": s,
      "c": c,
      "tin": tin,
      "tout": tout,
      "bb": bb,
      "wc": wc,
      "cost": cost,
      "p_ids": p_ids,
      "p_lookup": p_lookup,
  }


def extract_plan_dataframe(prob, vars_dict, gameweeks):
  """Converts solver variables into an inspection DataFrame."""
  plan = []
  p_lookup = vars_dict["p_lookup"]
  p_ids = vars_dict["p_ids"]

  for w in gameweeks:
    starting = [
        p_lookup[p]["web_name"] for p in p_ids if pulp.value(vars_dict["s"][p][w]) > 0.5
    ]
    bench = [
        p_lookup[p]["web_name"]
        for p in p_ids
        if pulp.value(vars_dict["x"][p][w]) > 0.5 and pulp.value(vars_dict["s"][p][w]) <= 0.5
    ]
    captain = [
        p_lookup[p]["web_name"] for p in p_ids if pulp.value(vars_dict["c"][p][w]) > 0.5
    ]
    t_in = [
        p_lookup[p]["web_name"] for p in p_ids if pulp.value(vars_dict["tin"][p][w]) > 0.5
    ]
    t_out = [
        p_lookup[p]["web_name"] for p in p_ids if pulp.value(vars_dict["tout"][p][w]) > 0.5
    ]
    bb_active = bool(pulp.value(vars_dict["bb"][w]) > 0.5)
    wc_active = bool(pulp.value(vars_dict["wc"][w]) > 0.5)
    hit = int(pulp.value(vars_dict["cost"][w]))

    plan.append({
        "Gameweek": f"GW{w}",
        "Active Chip": "Bench Boost" if bb_active else ("Wildcard" if wc_active else "None"),
        "Captain": captain[0] if captain else "None",
        "Transfers In": ", ".join(t_in) if t_in else "None",
        "Transfers Out": ", ".join(t_out) if t_out else "None",
        "Hits Cost": f"-{hit} pts" if hit > 0 else "0",
        "Starters (Count)": f"{len(starting)} players",
        "Bench (Count)": f"{len(bench)} players",
        "_starters_list": starting,
        "_bench_list": bench,
    })
  return pd.DataFrame(plan)


def solve_parametric_ev_transfers(
    players_df: pd.DataFrame,
    gameweeks=[4, 5, 6, 7],
    current_squad_ids=None,
    itb=0.5,
    free_transfers=1,
    force_bb_gw=None,
    force_wc_gw=None,
    allow_solver_chips=False,
    decay_rate=0.85,
):
  """Solves pure Parametric Expected Value (EV) maximization: max E[Points] - Hits."""
  if current_squad_ids is None:
    current_squad_ids = players_df["id"].tolist()[:15]

  prob, v = build_base_milp_model(
      players_df,
      gameweeks,
      current_squad_ids,
      itb,
      free_transfers,
      force_bb_gw,
      force_wc_gw,
      allow_solver_chips,
  )

  # Pure EV Objective: Expected points for starters + extra multiplier for captain
  obj = []
  for i, w in enumerate(gameweeks):
    discount = decay_rate**i
    pts_gw = pulp.lpSum([
        (v["s"][p][w] + v["c"][p][w]) * v["p_lookup"][p]["parametric_ev"].get(w, 4.0)
        for p in v["p_ids"]
    ])
    obj.append(discount * (pts_gw - v["cost"][w]))

  prob += pulp.lpSum(obj)
  prob.solve(get_pulp_solver())
  return prob, extract_plan_dataframe(prob, v, gameweeks)


def solve_cvar_transfers(
    players_df: pd.DataFrame,
    gameweeks=[4, 5, 6, 7],
    current_squad_ids=None,
    itb=0.5,
    free_transfers=1,
    force_bb_gw=None,
    force_wc_gw=None,
    allow_solver_chips=False,
    decay_rate=0.85,
    cvar_weight=0.20,
):
  """Solves CVaR / Non-Parametric Asymmetry-weighted lineup & transfer planning."""
  if current_squad_ids is None:
    current_squad_ids = players_df["id"].tolist()[:15]

  prob, v = build_base_milp_model(
      players_df,
      gameweeks,
      current_squad_ids,
      itb,
      free_transfers,
      force_bb_gw,
      force_wc_gw,
      allow_solver_chips,
  )

  obj = []
  for i, w in enumerate(gameweeks):
    discount = decay_rate**i
    # Blend Trimean location + Upside Asymmetry tail bonus (Bowley * p_haul)
    pts_gw = pulp.lpSum([
        (v["s"][p][w] + v["c"][p][w])
        * (
            v["p_lookup"][p]["trimean_pts"].get(w, 4.0)
            + cvar_weight
            * v["p_lookup"][p]["bowley_asym"]
            * v["p_lookup"][p]["p_haul"]
            * 10.0
        )
        for p in v["p_ids"]
    ])
    obj.append(discount * (pts_gw - v["cost"][w]))

  prob += pulp.lpSum(obj)
  prob.solve(get_pulp_solver())
  return prob, extract_plan_dataframe(prob, v, gameweeks)


# ==============================================================================
# 5. TABBED DASHBOARD INTERFACE
# ==============================================================================
tab_ev, tab_cvar, tab_nonparam, tab_llm = st.tabs([
    "📊 Parametric EV Optimizer",
    "🎯 Robust CVaR Optimizer",
    "📈 Non-Parametric & Asymmetry Analysis",
    "🤖 LLM Tactical Reasoning",
])

# ------------------------------------------------------------------------------
# TAB 1: PARAMETRIC EXPECTED VALUE OPTIMIZER
# ------------------------------------------------------------------------------
with tab_ev:
  st.subheader("Parametric Expected Value (EV) Transfer & Chip Engine")
  st.caption(
      "Maximizes classic mean expectations E[Points] without distributional skew adjustments."
  )

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    ev_chip_mode = st.selectbox(
        "Chip Mode (EV)",
        ["No Chips", "Force Bench Boost", "Force Wildcard", "Solver Decides"],
        key="ev_chip",
    )
  with col2:
    ev_target_gw = st.selectbox(
        "Target Chip GW",
        options=[4, 5, 6, 7],
        disabled=(ev_chip_mode in ["No Chips", "Solver Decides"]),
        key="ev_gw",
    )
  with col3:
    ev_decay = st.slider("Horizon Decay Rate", 0.70, 1.00, 0.85, 0.05, key="ev_decay")
  with col4:
    ev_itb = st.number_input("In The Bank (£m)", 0.0, 5.0, 0.5, 0.1, key="ev_itb")

  if st.button("Solve Parametric EV Model (HiGHS)", type="primary"):
    with st.spinner("Optimizing Parametric EV Lineup..."):
      bb_gw = ev_target_gw if ev_chip_mode == "Force Bench Boost" else None
      wc_gw = ev_target_gw if ev_chip_mode == "Force Wildcard" else None
      auto_chips = ev_chip_mode == "Solver Decides"

      prob_ev, df_ev_plan = solve_parametric_ev_transfers(
          players_df=df_players,
          gameweeks=[4, 5, 6, 7],
          itb=ev_itb,
          force_bb_gw=bb_gw,
          force_wc_gw=wc_gw,
          allow_solver_chips=auto_chips,
          decay_rate=ev_decay,
      )
      st.session_state.ev_solution = df_ev_plan
      st.success(f"Parametric Model Solved: {pulp.LpStatus[prob_ev.status]}")

  if st.session_state.ev_solution is not None:
    st.dataframe(
        st.session_state.ev_solution[[
            "Gameweek",
            "Active Chip",
            "Captain",
            "Transfers In",
            "Transfers Out",
            "Hits Cost",
            "Starters (Count)",
            "Bench (Count)",
        ]],
        use_container_width=True,
    )
    with st.expander("View Full Parametric Lineup Breakdown"):
      for _, row in st.session_state.ev_solution.iterrows():
        st.markdown(f"**{row['Gameweek']} (Captain: {row['Captain']})**")
        st.write(f"Starters: {', '.join(row['_starters_list'])}")
        st.write(f"Bench: {', '.join(row['_bench_list'])}")
        st.divider()

# ------------------------------------------------------------------------------
# TAB 2: ROBUST CVAR OPTIMIZER
# ------------------------------------------------------------------------------
with tab_cvar:
  st.subheader("Robust CVaR & Quantile Asymmetry Lineup Optimizer")
  st.caption(
      "Blends Tukey Trimean location with Bowley asymmetry and haul probabilities to capture ceiling."
  )

  cvar_c1, cvar_c2, cvar_c3, cvar_c4 = st.columns(4)
  with cvar_c1:
    cvar_chip_mode = st.selectbox(
        "Chip Mode (CVaR)",
        ["No Chips", "Force Bench Boost", "Force Wildcard", "Solver Decides"],
        key="cvar_chip",
    )
  with cvar_c2:
    cvar_target_gw = st.selectbox(
        "Target Chip GW",
        options=[4, 5, 6, 7],
        disabled=(cvar_chip_mode in ["No Chips", "Solver Decides"]),
        key="cvar_gw",
    )
  with cvar_c3:
    cvar_weight = st.slider("Ceiling/Tail Weight (CVaR)", 0.0, 0.5, 0.20, 0.05)
  with cvar_c4:
    cvar_decay = st.slider("Horizon Decay Rate", 0.70, 1.00, 0.85, 0.05, key="cvar_decay")

  if st.button("Solve Robust CVaR Model (HiGHS)", type="primary"):
    with st.spinner("Optimizing Robust CVaR Lineup..."):
      bb_gw = cvar_target_gw if cvar_chip_mode == "Force Bench Boost" else None
      wc_gw = cvar_target_gw if cvar_chip_mode == "Force Wildcard" else None
      auto_chips = cvar_chip_mode == "Solver Decides"

      prob_cvar, df_cvar_plan = solve_cvar_transfers(
          players_df=df_players,
          gameweeks=[4, 5, 6, 7],
          itb=0.5,
          force_bb_gw=bb_gw,
          force_wc_gw=wc_gw,
          allow_solver_chips=auto_chips,
          decay_rate=cvar_decay,
          cvar_weight=cvar_weight,
      )
      st.session_state.cvar_solution = df_cvar_plan
      st.success(f"Robust Model Solved: {pulp.LpStatus[prob_cvar.status]}")

  if st.session_state.cvar_solution is not None:
    st.dataframe(
        st.session_state.cvar_solution[[
            "Gameweek",
            "Active Chip",
            "Captain",
            "Transfers In",
            "Transfers Out",
            "Hits Cost",
            "Starters (Count)",
            "Bench (Count)",
        ]],
        use_container_width=True,
    )

# ------------------------------------------------------------------------------
# TAB 3: NON-PARAMETRIC TELEMETRY & ASYMMETRY
# ------------------------------------------------------------------------------
with tab_nonparam:
  st.subheader("Distributional Telemetry: Robust Scale vs Quantile Asymmetry")

  pos_filter = st.selectbox("Position Group", ["All", "GKP", "DEF", "MID", "FWD"])
  plot_df = (
      df_players.copy()
      if pos_filter == "All"
      else df_players[df_players["position"] == pos_filter]
  )

  fig_scatter = px.scatter(
      plot_df,
      x="iqr",
      y="bowley_asym",
      size="trimean",
      color="position",
      hover_name="web_name",
      hover_data=["mean_ev", "now_cost", "team", "p_haul"],
      labels={
          "iqr": "Dispersion / Volatility (IQR Scale)",
          "bowley_asym": "Upside Ceiling (Bowley Asymmetry: -1 to +1)",
      },
      title="Non-Parametric Player Profiling: Scale vs. Tail Skew",
  )
  fig_scatter.add_hline(y=0.0, line_dash="dash", line_color="gray")
  fig_scatter.add_vline(
      x=float(plot_df["iqr"].median()), line_dash="dash", line_color="gray"
  )
  st.plotly_chart(fig_scatter, use_container_width=True)

  st.divider()
  st.markdown("### Empirical vs Parametric Metric Comparison")
  selected_p = st.selectbox("Select Asset for History Inspection", plot_df["web_name"])
  p_data = plot_df[plot_df["web_name"] == selected_p].iloc[0]

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Parametric Mean (EV)", f"{p_data['mean_ev']:.2f}")
  c2.metric("Trimean Location", f"{p_data['trimean']:.2f}")
  c3.metric("IQR Dispersion", f"{p_data['iqr']:.2f}")
  c4.metric("Galton/Bowley Asymmetry", f"{p_data['bowley_asym']:.2f}")

  fig_hist = px.histogram(
      p_data["raw_history"],
      nbins=15,
      labels={"value": "FPL Points"},
      title=f"Empirical Points Distribution vs Parametric Mean ({selected_p})",
  )
  fig_hist.add_vline(
      x=p_data["mean_ev"],
      line_color="red",
      line_dash="dash",
      annotation_text="Parametric Mean",
  )
  st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: LLM TACTICAL REASONING
# ------------------------------------------------------------------------------
with tab_llm:
  st.subheader("Tactical Assistant (Resilient Gemini 3.5 Client)")

  # Context Auto-Injection from Solvers
  auto_context = ""
  if st.session_state.ev_solution is not None:
    auto_context += (
        "Parametric EV Plan:\n"
        + st.session_state.ev_solution[[
            "Gameweek",
            "Active Chip",
            "Captain",
            "Transfers In",
            "Transfers Out",
            "Hits Cost",
        ]].to_string(index=False)
        + "\n\n"
    )
  if st.session_state.cvar_solution is not None:
    auto_context += (
        "Robust CVaR Plan:\n"
        + st.session_state.cvar_solution[[
            "Gameweek",
            "Active Chip",
            "Captain",
            "Transfers In",
            "Transfers Out",
            "Hits Cost",
        ]].to_string(index=False)
    )

  prompt_input = st.text_area(
      "Tactical Dilemma / Model Comparison Query",
      value="Compare the Parametric EV plan against the Robust CVaR plan. Which strategy handles high-ceiling differential assets more effectively for Gameweek 4?",
      height=100,
  )

  if st.button("Consult Tactical Advisor"):
    full_prompt = (
        f"Model Telemetry Context:\n{auto_context}\n\nUser Question:\n{prompt_input}"
        if auto_context
        else prompt_input
    )

    with st.spinner("Consulting resilient GenAI client..."):
      try:
        resp = send_chat_resilient(
            prompt_msg=full_prompt,
            primary_model="gemini-3.5-flash",
            fallback_model="gemini-3.5-flash-lite",
        )
        st.session_state.chat_history.append((prompt_input, resp.text))
      except Exception as ex:
        st.error(f"GenAI Execution Error: {ex}")

  for user_q, model_a in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
      st.write(user_q)
    with st.chat_message("assistant"):
      st.write(model_a)