import logging
import os
import shutil
import time
from google import genai
from google.genai import errors
from google.genai import types
import pandas as pd
import pulp
import requests
import streamlit as st

# Suppress google-genai library logging
logging.getLogger("google.genai").setLevel(logging.ERROR)

st.set_page_config(page_title="FPL Master Tactical Suite", layout="wide")


# ---------------------------------------------------------
# 0. RESILIENT GENAI CLIENT & DYNAMIC MODEL DISCOVERY
# ---------------------------------------------------------
def get_genai_client(user_api_key: str = None):
    """Instantiates a fresh client using the user's session key first, then env/secrets."""
    api_key = (user_api_key or "").strip()

    if not api_key:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "No Gemini API key provided. Enter your key in the sidebar under 'Gemini API Key'."
        )

    return genai.Client(api_key=api_key)


def resolve_available_models(client) -> list:
    """Queries the API for models enabled on this key and orders preferred Flash models first."""
    preferred = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.0-flash",
        "gemini-2.0-flash",
    ]
    try:
        available = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            short_name = name.replace("models/", "")
            actions = getattr(m, "supported_actions", []) or []
            if "generateContent" in actions or not actions:
                if "flash" in short_name and "image" not in short_name and "audio" not in short_name:
                    available.append(short_name)

        ordered = [m for m in preferred if m in available]
        for m in available:
            if m not in ordered:
                ordered.append(m)

        if ordered:
            return ordered[:3]
    except Exception:
        pass

    return ["gemini-3.5-flash-lite", "gemini-3.5-flash"]


def send_chat_resilient(
    prompt_msg,
    user_api_key=None,
    client=None,
    system_instruction=None,
    max_retries=2,
    initial_delay=2.0,
):
    if client is None:
        client = get_genai_client(user_api_key)

    config = (
        types.GenerateContentConfig(system_instruction=system_instruction)
        if system_instruction
        else None
    )

    candidate_models = resolve_available_models(client)
    last_error = None

    for model_name in candidate_models:
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                chat = client.chats.create(model=model_name, config=config)
                return chat.send_message(message=prompt_msg)
            except RuntimeError as e:
                if "client has been closed" in str(e):
                    client = get_genai_client(user_api_key)
                    continue
                last_error = e
                break
            except errors.ServerError as e:
                last_error = e
                if e.code == 503 and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    break
            except errors.ClientError as e:
                last_error = e
                if e.code in (400, 404):
                    break
                raise e
            except Exception as e:
                last_error = e
                break

    raise RuntimeError(
        f"All candidate models {candidate_models} failed. Last error: {last_error}"
    )


# ---------------------------------------------------------
# 0B. ACCESS PASSWORD GATE
# ---------------------------------------------------------
def check_password() -> bool:
    """Returns True if the user entered the correct passphrase."""

    def password_entered():
        expected = (
            st.secrets.get("app_password", "admin")
            if hasattr(st, "secrets")
            else "admin"
        )
        if st.session_state["password"] == expected:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Enter Access Key",
        type="password",
        on_change=password_entered,
        key="password",
    )
    if "password_correct" in st.session_state:
        st.error("😕 Invalid key.")
    return False


if not check_password():
    st.stop()


# ---------------------------------------------------------
# 1. DATA INGESTION (OFFICIAL FPL API)
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_fpl_core():
    base = "https://fantasy.premierleague.com/api"
    bootstrap = requests.get(f"{base}/bootstrap-static/", timeout=10).json()
    fixtures = requests.get(f"{base}/fixtures/", timeout=10).json()
    return bootstrap, fixtures


@st.cache_data(ttl=120)
def fetch_current_squad(team_id: int, gameweek: int):
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gameweek}/picks/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------
# 2. FDR MATRIX & PLAYER DATASET BUILDER
# ---------------------------------------------------------
def compute_fdr_matrix(fixtures, teams, current_gw, horizon=5):
    """Computes a dict of {team_id: {gw: {"fdr": float, "is_home": bool}}}"""
    fdr_matrix = {t: {} for t in teams.keys()}
    for gw in range(current_gw, current_gw + horizon):
        for t in teams.keys():
            fdr_matrix[t][gw] = {"fdr": 3.0, "is_home": True}

    for f in fixtures:
        gw = f["event"]
        if gw and current_gw <= gw < current_gw + horizon:
            if f["team_h"] in fdr_matrix:
                fdr_matrix[f["team_h"]][gw] = {
                    "fdr": float(f["team_h_difficulty"]),
                    "is_home": True,
                }
            if f["team_a"] in fdr_matrix:
                fdr_matrix[f["team_a"]][gw] = {
                    "fdr": float(f["team_a_difficulty"]),
                    "is_home": False,
                }
    return fdr_matrix


def prepare_player_dataset(
    bootstrap,
    fdr_matrix,
    current_gw,
    horizon=3,
    home_boost=0.10,
    protected_ids=None,
):
    """Builds player dictionary.

    Always retains protected_ids (current squad) even if unavailable
    (chance_of_playing_next_round == 0) so the MILP can sell them and recover
    funds.
    """
    protected_set = set(protected_ids or [])
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    positions = {
        p["id"]: p["singular_name_short"] for p in bootstrap["element_types"]
    }

    players = {}
    for el in bootstrap["elements"]:
        p_id = el["id"]
        chance = el.get("chance_of_playing_next_round")
        is_unavailable = chance == 0 or el.get("status") in ("i", "s", "u", "n")

        # Skip unavailable non-owned players to keep MILP fast, but NEVER drop current squad
        if is_unavailable and p_id not in protected_set:
            continue

        t_id = el["team"]
        form = float(el.get("form", 0.0))
        xgi = float(el.get("expected_goal_involvements", 0.0))
        pts_season = float(el.get("total_points", 0))
        bonus_pts = int(el.get("bonus", 0))

        base_performance = (
            0.5 * form + 0.3 * (xgi * 1.8) + 0.2 * (pts_season / 10.0)
        )

        xp_profile = {}
        fdr_profile = {}
        for idx, gw in enumerate(range(current_gw, current_gw + horizon)):
            fixture_info = fdr_matrix[t_id].get(gw, {"fdr": 3.0, "is_home": True})
            fdr = fixture_info["fdr"]
            is_home = fixture_info["is_home"]

            fdr_factor = (6.0 - fdr) / 3.0
            venue_factor = (1.0 + home_boost) if is_home else (1.0 - home_boost)

            # Zero out immediate GW expected points if flagged 0% chance / suspended
            if is_unavailable and idx == 0:
                raw_xp = 0.0
            elif is_unavailable:
                raw_xp = base_performance * fdr_factor * venue_factor * 0.5
            else:
                raw_xp = max(0.1, base_performance * fdr_factor * venue_factor)

            xp_profile[gw] = round(raw_xp, 2)
            venue_tag = "H" if is_home else "A"
            status_flag = " [FLAG]" if is_unavailable else ""
            fdr_profile[gw] = f"{int(fdr)} ({venue_tag}){status_flag}"

        players[p_id] = {
            "name": el["web_name"],
            "team_id": t_id,
            "team": teams[t_id],
            "position": positions[el["element_type"]],
            "cost": el["now_cost"] / 10.0,
            "ownership": float(el.get("selected_by_percent", 0.0)),
            "form": form,
            "bonus": bonus_pts,
            "xp_profile": xp_profile,
            "fdr_profile": fdr_profile,
        }
    return players, teams, positions


# ---------------------------------------------------------
# 3. ROLLING MULTI-GAMEWEEK MILP SOLVER (PuLP + HiGHS)
# ---------------------------------------------------------
def solve_multi_gw_transfers(
    players: dict,
    initial_squad_ids: list,
    current_gw: int,
    horizon: int = 3,
    budget: float = 100.0,
    initial_ft: int = 1,
    max_penalty_transfers: int = 1,
    locked_ids: list = None,
    excluded_ids: list = None,
    formation_constraint: str = "Any",
    decay_rate: float = 0.85,
):
    highs_path = (
        shutil.which("highs")
        or shutil.which("highs.exe")
        or r"D:\msys64\ucrt64\bin\highs.exe"
    )
    if os.path.exists(highs_path):
        solver = pulp.HiGHS_CMD(path=highs_path, msg=False)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)

    prob = pulp.LpProblem("FPL_Multi_GW_Optimization", pulp.LpMaximize)
    gws = list(range(current_gw, current_gw + horizon))
    p_ids = list(players.keys())

    locked_ids = locked_ids or []
    excluded_ids = excluded_ids or []

    # Decision variables
    x = pulp.LpVariable.dicts(
        "x", ((i, gw) for i in p_ids for gw in gws), cat=pulp.LpBinary
    )
    s = pulp.LpVariable.dicts(
        "s", ((i, gw) for i in p_ids for gw in gws), cat=pulp.LpBinary
    )
    c = pulp.LpVariable.dicts(
        "c", ((i, gw) for i in p_ids for gw in gws), cat=pulp.LpBinary
    )

    tin = pulp.LpVariable.dicts(
        "tin", ((i, gw) for i in p_ids for gw in gws), cat=pulp.LpBinary
    )
    tout = pulp.LpVariable.dicts(
        "tout", ((i, gw) for i in p_ids for gw in gws), cat=pulp.LpBinary
    )

    hits = pulp.LpVariable.dicts(
        "hits", gws, lowBound=0, upBound=max_penalty_transfers, cat=pulp.LpInteger
    )
    ft = pulp.LpVariable.dicts("ft", gws, lowBound=1, upBound=5, cat=pulp.LpInteger)

    # Objective Function with Temporal Exponential Decay
    objective = []
    for step, gw in enumerate(gws):
        discount = decay_rate**step
        gw_xp = pulp.lpSum([
            players[i]["xp_profile"][gw] * s[i, gw]
            + players[i]["xp_profile"][gw] * c[i, gw]
            + 0.1 * players[i]["xp_profile"][gw] * (x[i, gw] - s[i, gw])
            for i in p_ids
        ])
        objective.append(discount * (gw_xp - 4.0 * hits[gw]))

    prob += pulp.lpSum(objective), "Total_Discounted_Net_Points"

    # Multi-Period Structural Constraints
    for idx, gw in enumerate(gws):
        for i in p_ids:
            prob += s[i, gw] <= x[i, gw]
            prob += c[i, gw] <= s[i, gw]

            prev_x = (
                (1 if i in initial_squad_ids else 0)
                if idx == 0
                else x[i, gws[idx - 1]]
            )
            prob += x[i, gw] - prev_x == tin[i, gw] - tout[i, gw]

        transfers_made = pulp.lpSum([tin[i, gw] for i in p_ids])
        if idx == 0:
            prob += ft[gw] == initial_ft
        else:
            prev_gw = gws[idx - 1]
            prev_transfers = pulp.lpSum([tin[i, prev_gw] for i in p_ids])
            prob += ft[gw] <= ft[prev_gw] - prev_transfers + 1
            prob += ft[gw] <= 5

        prob += hits[gw] >= transfers_made - ft[gw]
        prob += transfers_made <= ft[gw] + max_penalty_transfers

        # Squad & Lineup Sizes
        prob += pulp.lpSum([x[i, gw] for i in p_ids]) == 15
        prob += pulp.lpSum([s[i, gw] for i in p_ids]) == 11
        prob += pulp.lpSum([c[i, gw] for i in p_ids]) == 1
        prob += (
            pulp.lpSum([players[i]["cost"] * x[i, gw] for i in p_ids])
            <= budget + 1e-4
        )

        # Club limits (<= 3 per Premier League club)
        for t_id in set(p["team_id"] for p in players.values()):
            club_pids = [i for i in p_ids if players[i]["team_id"] == t_id]
            prob += pulp.lpSum([x[i, gw] for i in club_pids]) <= 3

        # Positional quotas (2 GKP, 5 DEF, 5 MID, 3 FWD)
        for pos, count in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            pos_ids = [i for i in p_ids if players[i]["position"] == pos]
            prob += pulp.lpSum([x[i, gw] for i in pos_ids]) == count

        # Formations
        gk_ids = [i for i in p_ids if players[i]["position"] == "GKP"]
        def_ids = [i for i in p_ids if players[i]["position"] == "DEF"]
        mid_ids = [i for i in p_ids if players[i]["position"] == "MID"]
        fwd_ids = [i for i in p_ids if players[i]["position"] == "FWD"]

        prob += pulp.lpSum([s[i, gw] for i in gk_ids]) == 1

        if formation_constraint != "Any":
            n_def, n_mid, n_fwd = map(int, formation_constraint.split("-"))
            prob += pulp.lpSum([s[i, gw] for i in def_ids]) == n_def
            prob += pulp.lpSum([s[i, gw] for i in mid_ids]) == n_mid
            prob += pulp.lpSum([s[i, gw] for i in fwd_ids]) == n_fwd
        else:
            prob += pulp.lpSum([s[i, gw] for i in def_ids]) >= 3
            prob += pulp.lpSum([s[i, gw] for i in def_ids]) <= 5
            prob += pulp.lpSum([s[i, gw] for i in mid_ids]) >= 2
            prob += pulp.lpSum([s[i, gw] for i in mid_ids]) <= 5
            prob += pulp.lpSum([s[i, gw] for i in fwd_ids]) >= 1
            prob += pulp.lpSum([s[i, gw] for i in fwd_ids]) <= 3

        # Force locks and exclusions
        for l_id in locked_ids:
            if l_id in p_ids:
                prob += x[l_id, gw] == 1
        for e_id in excluded_ids:
            if e_id in p_ids:
                prob += x[e_id, gw] == 0

    prob.writeLP("debug_model.lp")
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None, pulp.LpStatus[prob.status]

    # Parse and structure solver results
    plan = {}
    for gw in gws:
        squad_gw = []
        transfers_in = [
            players[i]["name"] for i in p_ids if pulp.value(tin[i, gw]) > 0.5
        ]
        transfers_out = [
            players[i]["name"] for i in p_ids if pulp.value(tout[i, gw]) > 0.5
        ]
        hit_count = int(round(pulp.value(hits[gw])))

        for i in p_ids:
            if pulp.value(x[i, gw]) > 0.5:
                squad_gw.append({
                    "id": i,
                    "name": players[i]["name"],
                    "pos": players[i]["position"],
                    "club": players[i]["team"],
                    "cost": players[i]["cost"],
                    "fdr": players[i]["fdr_profile"][gw],
                    "bonus": players[i]["bonus"],
                    "xp": players[i]["xp_profile"][gw],
                    "starter": pulp.value(s[i, gw]) > 0.5,
                    "captain": pulp.value(c[i, gw]) > 0.5,
                })
        plan[gw] = {
            "squad": squad_gw,
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "hits": hit_count,
            "hit_cost": hit_count * 4,
        }
    return plan, pulp.value(prob.objective)


# ---------------------------------------------------------
# 4. STREAMLIT APPLICATION
# ---------------------------------------------------------
st.title("🛡️ TDAIL++ FPL Optimization Suite")

bootstrap, fixtures = fetch_fpl_core()
events = bootstrap["events"]
active_gw = next((e["id"] for e in events if e.get("is_current")), 1)

# Sidebar: Preset Dropdown & Manual Input Option
st.sidebar.header("Manager Setup")

PRESET_TEAMS = {
    "GillyGoolies Rampant": 556553,
    "Liquid Courage": 2847869,
    "Killeagh Whalasques": 2586305,
    "Turner Over": 702054,
    "Futbol Fusion": 10549507,
    "Murstarz": 1080006,
    "Dynamo Tippski": 7432471,
    "Rhode Island Reds": 1087397,
    "Carry On": 5878520,
    "✏️ Enter Custom Team ID...": None,
}

selected_team_label = st.sidebar.selectbox(
    "Select FPL Team", options=list(PRESET_TEAMS.keys()), index=0
)

if PRESET_TEAMS[selected_team_label] is None:
    team_id = st.sidebar.number_input(
        "Enter FPL Team ID",
        min_value=1,
        value=1,
        step=1,
        help=(
            "Find your Team ID in the URL on the FPL website:"
            " /entry/{team_id}/event/..."
        ),
    )
else:
    team_id = PRESET_TEAMS[selected_team_label]
    st.sidebar.caption(f"Active Entry ID: `{team_id}` | Current GW: `{active_gw}`")

# Load existing squad picks FIRST so we know exact team value + bank
api_bank = 0.5
try:
    current_picks = fetch_current_squad(team_id, active_gw)
    initial_squad_ids = [p["element"] for p in current_picks["picks"]]
    api_bank = current_picks.get("entry_history", {}).get("bank", 5) / 10.0
except Exception:
    st.sidebar.warning(
        f"Could not load Entry ID {team_id}. Falling back to default squad."
    )
    initial_squad_ids = [el["id"] for el in bootstrap["elements"][:15]]

# Calculate current squad market value so default budget never violates initial roster
cost_lookup = {el["id"]: el["now_cost"] / 10.0 for el in bootstrap["elements"]}
current_squad_val = sum(cost_lookup.get(pid, 0.0) for pid in initial_squad_ids)
detected_total_budget = round(current_squad_val + api_bank, 1)

st.sidebar.info(
    f"Detected Squad Value: **£{current_squad_val:.1f}m** | Bank:"
    f" **£{api_bank:.1f}m**"
)

user_bank = st.sidebar.slider(
    "Total Available Budget (£m)",
    min_value=95.0,
    max_value=115.0,
    value=max(95.0, min(115.0, detected_total_budget)),
    step=0.1,
    help=(
        "Auto-calibrated to your current squad value + bank. Must be >="
        " £{:.1f}m if making 0 transfers.".format(current_squad_val)
    ),
)

free_transfers = st.sidebar.slider("Available Free Transfers (FT)", 1, 5, 1)

max_penalty_transfers = st.sidebar.slider(
    "Allowed Penalty Transfers (Hits)",
    min_value=0,
    max_value=5,
    value=1,
    step=1,
    help=(
        "Extra transfers beyond free transfers. Each penalty costs -4 points."
    ),
)

horizon_weeks = st.sidebar.slider("Planning Horizon (Gameweeks)", 1, 5, 3)

# Sidebar live transfer readouts
total_transfers_allowed = free_transfers + max_penalty_transfers
potential_hit_cost = max_penalty_transfers * 4

st.sidebar.markdown("---")
sb_col1, sb_col2 = st.sidebar.columns(2)
sb_col1.metric("Max Transfers (GW1)", total_transfers_allowed)
sb_col2.metric(
    "Max Hit Cost",
    f"-{potential_hit_cost} pts" if potential_hit_cost > 0 else "0 pts",
)

# Sidebar: Per-User Gemini API Key Input
with st.sidebar.expander("🔑 Gemini API Key (For AI Advisor)", expanded=False):
    user_gemini_key = st.text_input(
        "Your Gemini API Key",
        type="password",
        key="user_gemini_key",
        placeholder="AIzaSy...",
        help="Optional if a host key is configured; otherwise paste your free Google AI Studio key.",
    )
    st.caption("[Get a free API key at Google AI Studio](https://aistudio.google.com/apikey)")

# Sidebar: Advanced Model Weights Expander
with st.sidebar.expander("⚙️ Advanced Model Weights", expanded=False):
    decay_rate = st.slider(
        "Gameweek Decay Factor (γ)",
        min_value=0.70,
        max_value=1.00,
        value=0.85,
        step=0.05,
        help=(
            "Discount multiplier applied to future gameweeks (GW1=1.0, GW2=γ,"
            " GW3=γ²...)"
        ),
    )
    home_adv_pct = st.slider(
        "Home Advantage Split (±%)",
        min_value=0,
        max_value=20,
        value=10,
        step=2,
        help="Percentage boost for home games and penalty for away games.",
    )

# Build Matrices (passing initial_squad_ids as protected_ids so flagged players aren't dropped!)
home_boost_decimal = home_adv_pct / 100.0
fdr_matrix = compute_fdr_matrix(
    fixtures,
    {t["id"]: t["short_name"] for t in bootstrap["teams"]},
    active_gw,
    horizon_weeks,
)
players_dict, team_names, pos_names = prepare_player_dataset(
    bootstrap,
    fdr_matrix,
    active_gw,
    horizon_weeks,
    home_boost=home_boost_decimal,
    protected_ids=initial_squad_ids,
)

# ---------------------------------------------------------
# TABBED INTERFACE
# ---------------------------------------------------------
tab_opt, tab_advisor = st.tabs([
    "⚡ TDAIL++ Multi-GW Squad Optimizer",
    "🧠 TDAIL++ Gemini Tactical Advisor",
])

with tab_opt:
    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])

    with col_c1:
        formation_target = st.selectbox(
            "Preferred Formation",
            ["Any", "3-5-2", "3-4-3", "4-4-2", "4-3-3", "5-3-2"],
        )

    name_id_map = {
        f"{p['name']} ({p['team']} - {p['position']})": p_id
        for p_id, p in players_dict.items()
    }

    with col_c2:
        locked_labels = st.multiselect(
            "🔒 Force-Lock Players", options=list(name_id_map.keys())
        )
        locked_pids = [name_id_map[l] for l in locked_labels]

    with col_c3:
        excluded_labels = st.multiselect(
            "❌ Exclude Players", options=list(name_id_map.keys())
        )
        excluded_pids = [name_id_map[e] for e in excluded_labels]

    if st.button("🚀 Run Multi-Period Solver", type="primary"):
        with st.spinner("Executing HiGHS branch-and-cut optimization..."):
            plan, result_meta = solve_multi_gw_transfers(
                players=players_dict,
                initial_squad_ids=initial_squad_ids,
                current_gw=active_gw,
                horizon=horizon_weeks,
                budget=user_bank,
                initial_ft=free_transfers,
                max_penalty_transfers=max_penalty_transfers,
                locked_ids=locked_pids,
                excluded_ids=excluded_pids,
                formation_constraint=formation_target,
                decay_rate=decay_rate,
            )

            if plan is None:
                st.error(
                    f"Model returned status: **{result_meta}**. Check if your Total"
                    f" Budget (£{user_bank:.1f}m) covers your starting squad"
                    f" (£{current_squad_val:.1f}m) or if Force-Lock/Exclude rules"
                    " conflict."
                )
            else:
                st.session_state["fpl_plan"] = plan
                st.session_state["net_pts"] = result_meta

    if "fpl_plan" in st.session_state:
        plan = st.session_state["fpl_plan"]

        total_horizon_hits = sum(gw_data["hits"] for gw_data in plan.values())
        total_horizon_hit_cost = total_horizon_hits * 4

        st.success(
            "Optimal Schedule Found! Total Discounted Net xP:"
            f" **{st.session_state['net_pts']:.2f}**"
        )

        sum1, sum2, sum3 = st.columns(3)
        sum1.metric("Initial Free Transfers", free_transfers)
        sum2.metric("Total Horizon Hits Taken", f"{total_horizon_hits} transfer(s)")
        sum3.metric(
            "Total Hit Deductions",
            f"-{total_horizon_hit_cost} pts" if total_horizon_hit_cost > 0 else "0 pts",
        )

        st.markdown("---")

        gw_tabs = st.tabs([f"GW {gw}" for gw in plan.keys()])
        for i, gw in enumerate(plan.keys()):
            with gw_tabs[i]:
                gw_data = plan[gw]

                tin_str = (
                    ", ".join(gw_data["transfers_in"])
                    if gw_data["transfers_in"]
                    else "None (Roll FT)"
                )
                tout_str = (
                    ", ".join(gw_data["transfers_out"])
                    if gw_data["transfers_out"]
                    else "None"
                )
                hit_deduction = gw_data["hit_cost"]

                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Transfers In", tin_str)
                t2.metric("Transfers Out", tout_str)
                t3.metric("Penalty Hits Taken", gw_data["hits"])
                t4.metric(
                    "Gameweek Hit Cost",
                    f"-{hit_deduction} pts" if hit_deduction > 0 else "0 pts",
                )

                df_gw = pd.DataFrame(gw_data["squad"])
                starters = df_gw[df_gw["starter"]].copy()
                bench = df_gw[~df_gw["starter"]].copy()

                starters["effective_xp"] = starters.apply(
                    lambda r: r["xp"] * 2.0 if r["captain"] else r["xp"], axis=1
                )
                starters["Role"] = starters.apply(
                    lambda r: "⭐ CAPTAIN" if r["captain"] else "Starter", axis=1
                )
                bench["Role"] = "Sub"

                pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
                starters["rank"] = starters["pos"].map(pos_order)
                bench["rank"] = bench["pos"].map(pos_order)
                starters = starters.sort_values("rank").drop(columns=["rank"])
                bench = bench.sort_values("rank").drop(columns=["rank"])

                starters_raw_xp = starters["xp"].sum()
                captain_bonus_xp = (
                    starters.loc[starters["captain"], "xp"].values[0]
                    if any(starters["captain"])
                    else 0.0
                )
                starting_xi_total_xp = starters["effective_xp"].sum()
                bench_total_xp = bench["xp"].sum()
                net_gameweek_xp = starting_xi_total_xp - hit_deduction

                st.markdown("#### 📈 Gameweek Projected Points Breakdown")
                xp1, xp2, xp3, xp4 = st.columns(4)
                xp1.metric(
                    "Starting XI xP",
                    f"{starting_xi_total_xp:.2f}",
                    help=(
                        f"Base Starter xP ({starters_raw_xp:.2f}) + Captain 2x Bonus"
                        f" (+{captain_bonus_xp:.2f})"
                    ),
                )
                xp2.metric(
                    "Net Gameweek xP",
                    f"{net_gameweek_xp:.2f}",
                    delta=f"-{hit_deduction} hit cost" if hit_deduction > 0 else None,
                    delta_color="inverse",
                    help="Starting XI xP minus penalty hit costs",
                )
                xp3.metric(
                    "Bench xP",
                    f"{bench_total_xp:.2f}",
                    help="Total expected points sitting on your bench",
                )
                squad_spent = df_gw["cost"].sum()
                xp4.metric(
                    "Squad Value / Remaining",
                    f"£{squad_spent:.1f}m / £{user_bank - squad_spent:.1f}m",
                )

                table_cols = {
                    "name": "Player",
                    "pos": "Pos",
                    "club": "Club",
                    "cost": "Cost (£m)",
                    "fdr": "FDR (H/A)",
                    "bonus": "Season Bonus",
                    "xp": "Base xP",
                    "effective_xp": "Lineup xP",
                    "Role": "Role",
                }

                st.subheader("Starting XI")
                st.dataframe(
                    starters.rename(columns=table_cols)[[
                        "Player",
                        "Pos",
                        "Club",
                        "Cost (£m)",
                        "FDR (H/A)",
                        "Season Bonus",
                        "Base xP",
                        "Lineup xP",
                        "Role",
                    ]].style.format({
                        "Cost (£m)": "£{:.1f}m",
                        "Base xP": "{:.2f}",
                        "Lineup xP": "{:.2f}",
                        "Season Bonus": "{:d}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Bench")
                bench_display = bench.rename(columns=table_cols)[[
                    "Player",
                    "Pos",
                    "Club",
                    "Cost (£m)",
                    "FDR (H/A)",
                    "Season Bonus",
                    "Base xP",
                    "Role",
                ]]
                st.dataframe(
                    bench_display.style.format({
                        "Cost (£m)": "£{:.1f}m",
                        "Base xP": "{:.2f}",
                        "Season Bonus": "{:d}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

# ---------------------------------------------------------
# TAB 2: GEMINI TACTICAL ADVISOR
# ---------------------------------------------------------
with tab_advisor:
    st.subheader("🤖 TDAIL++ Tactical Managerial Briefing (Gemini Flash)")

    if "fpl_plan" not in st.session_state:
        st.info(
            "Run the optimizer first to feed structured decision data to the AI"
            " Advisor."
        )
    else:
        if st.button("Generate Tactical Briefing"):
            with st.spinner("Synthesizing solver telemetry and fixture dynamics..."):
                try:
                    gw1_plan = st.session_state["fpl_plan"][active_gw]
                    starters = [p for p in gw1_plan["squad"] if p["starter"]]
                    captain = next((p for p in starters if p["captain"]), starters[0])

                    sorted_teams = sorted(
                        [
                            {team_names[tid]: fdr_matrix[tid][active_gw]["fdr"]}
                            for tid in team_names.keys()
                        ],
                        key=lambda x: list(x.values())[0],
                    )

                    payload = {
                        "active_gameweek": active_gw,
                        "recommended_transfers_in": gw1_plan["transfers_in"],
                        "recommended_transfers_out": gw1_plan["transfers_out"],
                        "penalty_hits_taken": gw1_plan["hits"],
                        "hit_cost_deduction": gw1_plan["hit_cost"],
                        "captain_choice": captain,
                        "top_favorable_fixtures": sorted_teams[:3],
                        "brutal_fixtures_to_bench": sorted_teams[-3:],
                        "starters_summary": [
                            {
                                "name": p["name"],
                                "club": p["club"],
                                "xP": p["xp"],
                                "fdr": p["fdr"],
                                "bonus": p["bonus"],
                            }
                            for p in starters
                        ],
                    }

                    system_instruction = """
                    You are an elite Fantasy Premier League tactical consultant.
                    Analyze the solver's decisions and deliver an executive pre-gameweek briefing.
                    Format your analysis with these clear headers:
                    - **Key Transfer Actions:** Evaluate the logic behind transfers in/out and whether hits taken were justified.
                    - **Captaincy Thesis:** Breakdown why the solver backed this captain over viable alternatives.
                    - **Targeted Fixture Swings:** Explain which club runs to attack and which defensive fixtures to evade.
                    - **Variance & Tactical Risks:** Highlight bench hazards, home/away splits, or potential rotation traps.
                    Keep your language direct, sharp, and data-backed.
                    """

                    prompt_msg = (
                        "Provide a managerial briefing based on this solver"
                        f" data:\n\n{payload}"
                    )
                    response = send_chat_resilient(
                        prompt_msg=prompt_msg,
                        user_api_key=user_gemini_key,
                        system_instruction=system_instruction,
                    )

                    st.markdown("---")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Tactical Briefing Error: {e}")