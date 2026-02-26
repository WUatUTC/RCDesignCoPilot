# ============================================================
# RC DESIGN COPILOT — DEMO (ACI 318-19 teaching demonstration ONLY)
# Natural language prompt -> Agent routes skills -> Beam flexure + shear,
# short tied column (P-M interaction) skill.
#
# Run:
#   pip install streamlit
#   streamlit run app.py
#
# Teaching demo only. Not for real design
# ============================================================

import re
import math
import streamlit as st

# -----------------------------
# Databases (US customary)
# -----------------------------
BAR_DB = {
    "#3": {"area": 0.11, "dia": 0.375},
    "#4": {"area": 0.20, "dia": 0.500},
    "#5": {"area": 0.31, "dia": 0.625},
    "#6": {"area": 0.44, "dia": 0.750},
    "#7": {"area": 0.60, "dia": 0.875},
    "#8": {"area": 0.79, "dia": 1.000},
    "#9": {"area": 1.00, "dia": 1.128},
    "#10": {"area": 1.27, "dia": 1.270},
    "#11": {"area": 1.56, "dia": 1.410},
}

STIRRUP_LEG_AREA = {"#3": 0.11, "#4": 0.20}

Es = 29_000_000.0  # psi
DEFAULT_STIRRUP_SIZE = "#3"
DEFAULT_STIRRUP_LEGS = 2
DEFAULT_TIE_SIZE = "#3"


# ============================================================
# Parser skill (Natural language -> structured inputs)
# ============================================================

def _find_number(text: str, patterns, default=None):
    """
    Returns the first numeric capture group found.
    Works whether the regex has 1 group (number) or multiple groups
    (e.g., 'fc=4 ksi' where group(1) is 'fc' and group(2) is the number).
    """
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            # Try groups from last to first; return the first that parses as float
            for gi in range(m.lastindex, 0, -1):
                try:
                    return float(m.group(gi))
                except (TypeError, ValueError):
                    continue
    return default

def _find_bar(text: str, patterns, default=None):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).replace(" ", "")
    return default

def parse_prompt(prompt: str):
    """
    Supports BEAM or COLUMN or SLAB (punching) style prompts.

    Beam example:
      "Design a 12x24 beam, cover 1.5 in, fc=4 ksi, fy=60 ksi,
       Mu=180 kip-ft, Vu=45 kips, use #8 max bars, stirrups #3."

    Column example:
      "Design a 16x16 tied column, cover 1.5 in, fc=5 ksi, fy=60 ksi,
       Pu=350 kips, Mu=120 kip-ft, use #8 bars, ties #3."

    Punching example (flag/skill):
      "Check punching for slab t=8 in, fc=4 ksi, column=16x16, Vu=220 kips."
    """
    t = prompt.strip()
    warnings = []
    member_type = "beam"

    if re.search(r"\bcolumn\b", t, flags=re.IGNORECASE):
        member_type = "column"
    if re.search(r"\bslab\b|\bpunching\b|\bfooting\b", t, flags=re.IGNORECASE):
        member_type = "slab"

    # Common: fc, fy
    fc_ksi = _find_number(t, [r"\b(f'?c|fc)\s*=?\s*([0-9]*\.?[0-9]+)\s*ksi\b"])
    fc_psi = _find_number(t, [r"\b(f'?c|fc)\s*=?\s*([0-9]*\.?[0-9]+)\s*psi\b"])
    if fc_ksi is not None:
        fc = fc_ksi * 1000.0
    elif fc_psi is not None:
        fc = fc_psi
    else:
        fc = 4000.0
        warnings.append("f'c not found; defaulted to 4.0 ksi.")

    fy_ksi = _find_number(t, [r"\bfy\s*=?\s*([0-9]*\.?[0-9]+)\s*ksi\b"])
    fy_psi = _find_number(t, [r"\bfy\s*=?\s*([0-9]*\.?[0-9]+)\s*psi\b"])
    if fy_ksi is not None:
        fy = fy_ksi * 1000.0
    elif fy_psi is not None:
        fy = fy_psi
    else:
        fy = 60000.0
        warnings.append("fy not found; defaulted to 60 ksi.")

    cover = _find_number(t, [r"\bcover\s*=?\s*([0-9]*\.?[0-9]+)\s*(in|inch|inches)\b"], default=1.5)

    # Geometry for rectangular member: b x h or b=, h=
    b = _find_number(t, [r"\bb\s*=\s*([0-9]*\.?[0-9]+)\s*(in|inch|inches)\b"])
    h = _find_number(t, [r"\bh\s*=\s*([0-9]*\.?[0-9]+)\s*(in|inch|inches)\b"])
    m_dim = re.search(r"\b([0-9]*\.?[0-9]+)\s*[xX]\s*([0-9]*\.?[0-9]+)\b", t)
    if m_dim:
        b = b if b is not None else float(m_dim.group(1))
        h = h if h is not None else float(m_dim.group(2))

    # Beam demands
    Mu = _find_number(t, [r"\bmu\s*=?\s*([0-9]*\.?[0-9]+)\s*(kip[-\s]*ft|kft)\b"])
    Vu = _find_number(t, [r"\bvu\s*=?\s*([0-9]*\.?[0-9]+)\s*kips?\b"])

    # Column demands
    Pu = _find_number(t, [r"\bpu\s*=?\s*([0-9]*\.?[0-9]+)\s*kips?\b"])
    Mu_col = _find_number(t, [r"\bmu\s*=?\s*([0-9]*\.?[0-9]+)\s*(kip[-\s]*ft|kft)\b"])

    # Preferences
    max_bar = _find_bar(t, [r"(#\s*[0-9]+)\s*(max|maximum)"])
    long_bar = _find_bar(t, [r"\b(use|bars?)\s*(#\s*[0-9]+)\b"], default=None)
    stirrup_size = _find_bar(t, [r"stirrups?\s*(#\s*[0-9]+)"], default=None)
    tie_size = _find_bar(t, [r"\bties?\s*(#\s*[0-9]+)"], default=None)

    # Slab punching inputs (very simple)
    slab_t = _find_number(t, [r"\bt\s*=\s*([0-9]*\.?[0-9]+)\s*(in|inch|inches)\b",
                             r"\bthickness\s*=?\s*([0-9]*\.?[0-9]+)\s*(in|inch|inches)\b"])
    col_a = _find_number(t, [r"\bcolumn\s*=\s*([0-9]*\.?[0-9]+)\s*[xX]\s*([0-9]*\.?[0-9]+)\b"], default=None)
    col_b = None
    m_col = re.search(r"\bcolumn\s*=\s*([0-9]*\.?[0-9]+)\s*[xX]\s*([0-9]*\.?[0-9]+)\b", t, flags=re.IGNORECASE)
    if m_col:
        col_a = float(m_col.group(1))
        col_b = float(m_col.group(2))

    # Basic missing checks
    if member_type in ("beam", "column") and (b is None or h is None):
        warnings.append("Section size not fully found. Use '12x24' or 'b=12 in h=24 in'.")
    if member_type == "beam" and Mu is None:
        warnings.append("Beam Mu not found. Example: 'Mu=180 kip-ft'.")
    if member_type == "beam" and Vu is None:
        warnings.append("Beam Vu not found. Example: 'Vu=45 kips'.")
    if member_type == "column" and Pu is None:
        warnings.append("Column Pu not found. Example: 'Pu=350 kips'.")
    if member_type == "column" and Mu_col is None:
        warnings.append("Column Mu not found. Example: 'Mu=120 kip-ft'.")
    if member_type == "slab" and Vu is None:
        warnings.append("Punching Vu not found. Example: 'Vu=220 kips'.")
    if member_type == "slab" and (slab_t is None or col_a is None or col_b is None):
        warnings.append("Punching inputs missing. Example: 'slab t=8 in, column=16x16'.")

    return {
        "member_type": member_type,
        "b_in": b,
        "h_in": h,
        "cover_in": cover,
        "fc_psi": fc,
        "fy_psi": fy,
        "Mu_kipft": Mu,
        "Vu_kips": Vu,
        "Pu_kips": Pu,
        "Mu_col_kipft": Mu_col,
        "max_bar": max_bar,
        "long_bar": long_bar,
        "stirrup_size": stirrup_size,
        "tie_size": tie_size,
        "slab_t_in": slab_t,
        "col_a_in": col_a,
        "col_b_in": col_b,
    }, warnings


# ============================================================
# ACI-style helpers (teaching)
# ============================================================

def beta1_aci(fc_psi: float) -> float:
    if fc_psi <= 4000:
        return 0.85
    beta = 0.85 - 0.05 * ((fc_psi - 4000) / 1000.0)
    return max(0.65, beta)

def eps_y(fy_psi: float) -> float:
    return fy_psi / Es

def phi_flexure_from_eps_t(eps_t: float, fy_psi: float) -> float:
    # Teaching version aligned with ACI 318-19 tension-controlled concept: eps_t >= eps_y + 0.003 -> phi=0.90
    eps_tc = eps_y(fy_psi) + 0.003
    if eps_t >= eps_tc:
        return 0.90
    if eps_t <= 0.002:
        return 0.65
    return 0.65 + (eps_t - 0.002) * (0.90 - 0.65) / (eps_tc - 0.002)

def effective_depth_beam(h_in: float, cover_in: float, bar_dia_in: float, stirrup_dia_in: float = 0.375) -> float:
    return h_in - cover_in - stirrup_dia_in - 0.5 * bar_dia_in

def As_min_beam_us(fc_psi: float, fy_psi: float, b_in: float, d_in: float) -> float:
    # Teaching form commonly used in US units:
    # As,min = max( 3*sqrt(fc')/fy * b*d, 200/fy * b*d )
    term1 = 3.0 * math.sqrt(fc_psi) / fy_psi * b_in * d_in
    term2 = 200.0 / fy_psi * b_in * d_in
    return max(term1, term2)

def shear_Av_over_s_min_us(fc_psi: float, fy_psi: float, bw_in: float) -> float:
    # Minimum area of shear reinforcement (US units) commonly expressed as:
    # Av,min = max(0.75*sqrt(fc')*bw*s/fy, 50*bw*s/fy)  -> Av/s = max(0.75*sqrt(fc')*bw/fy, 50*bw/fy)
    # (Teaching implementation; see Tekla reference for the form.)
    return max(0.75 * math.sqrt(fc_psi) * bw_in / fy_psi, 50.0 * bw_in / fy_psi)

def shear_Vc_simple_us(fc_psi: float, bw_in: float, d_in: float, lam: float = 1.0) -> float:
    # Simple beam shear baseline: Vc = 2*lambda*sqrt(fc')*bw*d (lb) -> kips
    return (2.0 * lam * math.sqrt(fc_psi) * bw_in * d_in) / 1000.0


# ============================================================
# Skill: Beam flexure (singly reinforced, Whitney block)
# ============================================================

def flexure_design_beam(Mu_kipft: float, b_in: float, h_in: float, cover_in: float, fc_psi: float, fy_psi: float,
                        max_bar: str | None):
    steps = []
    if Mu_kipft is None:
        return None, ["Mu missing."], steps

    bar_keys = [k for k in BAR_DB.keys() if k != "#3"]  # tension steel typically #4+
    if max_bar and max_bar in bar_keys:
        bar_keys = bar_keys[: bar_keys.index(max_bar) + 1]

    best = None

    for bar in bar_keys:
        bar_area = BAR_DB[bar]["area"]
        bar_dia = BAR_DB[bar]["dia"]
        d_in = effective_depth_beam(h_in, cover_in, bar_dia)

        if d_in <= 0:
            continue

        # Solve As from Mn = As*fy*(d - a/2), a = As*fy/(0.85*fc*b)
        # Use phi=0.90 guess for As_req, then compute phi from eps_t using provided As.
        phi_guess = 0.90
        Mn_req_kipin = (Mu_kipft * 12.0) / phi_guess

        A = (fy_psi**2) / (2.0 * 0.85 * fc_psi * b_in)
        B = -fy_psi * d_in
        C = Mn_req_kipin * 1000.0

        disc = B*B - 4*A*C
        if disc <= 0:
            continue

        As_req = (-B - math.sqrt(disc)) / (2*A)
        if As_req <= 0:
            continue

        n = math.ceil(As_req / bar_area)
        As_prov = n * bar_area

        beta1 = beta1_aci(fc_psi)
        a = As_prov * fy_psi / (0.85 * fc_psi * b_in)
        c = a / beta1
        eps_t = 0.003 * (d_in - c) / c if c > 0 else 0.0
        phi = phi_flexure_from_eps_t(eps_t, fy_psi)

        Mn_lb_in = As_prov * fy_psi * (d_in - a/2.0)
        Mn_kipft = Mn_lb_in / (1000.0 * 12.0)
        phiMn_kipft = phi * Mn_kipft

        Asmin = As_min_beam_us(fc_psi, fy_psi, b_in, d_in)

        strength_ok = phiMn_kipft >= Mu_kipft
        asmin_ok = As_prov >= Asmin

        if strength_ok:
            score = (n, As_prov - As_req)
            cand = {
                "bar": bar, "n": n, "As_req": As_req, "As_prov": As_prov,
                "d_in": d_in, "beta1": beta1, "a_in": a, "c_in": c,
                "eps_t": eps_t, "phi": phi, "Mn_kipft": Mn_kipft, "phiMn_kipft": phiMn_kipft,
                "Asmin": Asmin, "asmin_ok": asmin_ok
            }
            if best is None:
                best = (cand, score)
            else:
                # Prefer Asmin_ok, then fewer bars/less overage
                if cand["asmin_ok"] and not best[0]["asmin_ok"]:
                    best = (cand, score)
                elif cand["asmin_ok"] == best[0]["asmin_ok"] and score < best[1]:
                    best = (cand, score)

    if best is None:
        return None, ["Could not find bar layout that satisfies flexural strength with current assumptions."], steps

    steps.append("Flexure skill: Whitney block + strain-based ϕ; selected bar size/count to satisfy ϕMn ≥ Mu.")
    return best[0], [], steps


# ============================================================
# Skill: Beam shear (ACI-style minimum Av, spacing cap; teaching)
# ============================================================

def shear_design_beam(Vu_kips: float, bw_in: float, d_in: float, fc_psi: float, fy_psi: float,
                      stirrup_size: str | None, legs: int = DEFAULT_STIRRUP_LEGS, lam: float = 1.0):
    steps = []
    warnings = []
    phi_v = 0.75

    if stirrup_size not in STIRRUP_LEG_AREA:
        stirrup_size = DEFAULT_STIRRUP_SIZE

    Av = legs * STIRRUP_LEG_AREA[stirrup_size]  # in^2

    Vc_kips = shear_Vc_simple_us(fc_psi, bw_in, d_in, lam=lam)
    phiVc = phi_v * Vc_kips

    # One-way shear spacing cap (baseline): s <= min(d/2, 24 in)
    s_cap_in = min(d_in / 2.0, 24.0)

    # Minimum shear reinforcement (Av/s minimum)
    Av_over_s_min = shear_Av_over_s_min_us(fc_psi, fy_psi, bw_in)

    # When is min shear reinforcement required?
    # Teaching trigger (aligned with common ACI 318-19 discussions): provide Av,min in regions where Vu is sufficiently high.
    # (Exact triggers vary by member type/conditions; we keep this as a teachable rule.)
    Vu_trigger = phi_v * lam * math.sqrt(fc_psi) * bw_in * d_in / 1000.0  # kips
    needs_min_shear = (Vu_kips is not None) and (Vu_kips > Vu_trigger)

    if Vu_kips is None:
        warnings.append("Vu not provided → shear design not completed.")
        return {
            "phi_v": phi_v, "Vc_kips": Vc_kips, "phiVc_kips": phiVc,
            "stirrup_size": stirrup_size, "legs": legs, "Av_in2": Av,
            "s_req_in": None, "s_use_in": None, "s_cap_in": s_cap_in,
            "Av_over_s_min": Av_over_s_min, "needs_min_shear": None, "Vu_trigger_kips": Vu_trigger
        }, warnings, ["Shear skill: computed Vc only (Vu missing)."]

    # Required Vs from simple truss model: Vu <= phi*(Vc + Vs)
    Vn_req = Vu_kips / phi_v
    Vs_req = max(0.0, Vn_req - Vc_kips)  # kips
    Vs_req_lb = Vs_req * 1000.0

    if Vs_req <= 1e-9:
        steps.append("Shear skill: Vu ≤ ϕVc (baseline). Stirrups may still be required as minimum reinforcement depending on Vu region.")
        # If min shear is needed, size stirrups to meet Av/s min and spacing cap.
        if needs_min_shear:
            s_from_min = Av / Av_over_s_min
            s_use = min(s_from_min, s_cap_in)
            warnings.append("Minimum shear reinforcement required by trigger; stirrups sized to satisfy Av/s minimum.")
            return {
                "phi_v": phi_v, "Vc_kips": Vc_kips, "phiVc_kips": phiVc,
                "stirrup_size": stirrup_size, "legs": legs, "Av_in2": Av,
                "s_req_in": None, "s_use_in": s_use, "s_cap_in": s_cap_in,
                "Av_over_s_min": Av_over_s_min, "needs_min_shear": True, "Vu_trigger_kips": Vu_trigger,
                "note": "Vu ≤ ϕVc but minimum stirrups provided per trigger (teaching)."
            }, warnings, steps

        return {
            "phi_v": phi_v, "Vc_kips": Vc_kips, "phiVc_kips": phiVc,
            "stirrup_size": stirrup_size, "legs": legs, "Av_in2": Av,
            "s_req_in": None, "s_use_in": None, "s_cap_in": s_cap_in,
            "Av_over_s_min": Av_over_s_min, "needs_min_shear": False, "Vu_trigger_kips": Vu_trigger,
            "note": "Vu ≤ ϕVc and minimum stirrups not triggered (teaching)."
        }, warnings, steps

    # s required for strength from Vs = Av*fy*d/s
    s_strength = (Av * fy_psi * d_in) / Vs_req_lb

    # Also enforce minimum shear reinforcement if needed:
    s_from_min = Av / Av_over_s_min
    s_req = min(s_strength, s_from_min) if needs_min_shear else s_strength

    # Final spacing must satisfy cap
    s_use = min(s_req, s_cap_in)

    # Check if min shear violated after applying cap (rare but possible if Av too small)
    Av_over_s_prov = Av / s_use
    if needs_min_shear and Av_over_s_prov + 1e-12 < Av_over_s_min:
        warnings.append("Provided stirrups cannot meet Av/s minimum with current stirrup size/legs and spacing cap. Increase stirrup size or legs.")

    steps.append("Shear skill: computed Vc, Vs, selected stirrup spacing using Vs=Av*fy*d/s, enforced s ≤ min(d/2,24) and Av/s minimum (teaching).")
    return {
        "phi_v": phi_v, "Vc_kips": Vc_kips, "phiVc_kips": phiVc,
        "stirrup_size": stirrup_size, "legs": legs, "Av_in2": Av,
        "Vs_req_kips": Vs_req,
        "s_strength_in": s_strength,
        "s_from_min_in": s_from_min,
        "s_req_in": s_req,
        "s_use_in": s_use,
        "s_cap_in": s_cap_in,
        "Av_over_s_min": Av_over_s_min,
        "Av_over_s_prov": Av_over_s_prov,
        "needs_min_shear": needs_min_shear,
        "Vu_trigger_kips": Vu_trigger
    }, warnings, steps


# ============================================================
# Skill: Punching shear flag + basic check (teaching)
# ============================================================

def punching_shear_check(Vu_kips: float, slab_t_in: float, cover_in: float, fc_psi: float,
                         col_a_in: float, col_b_in: float, phi_v: float = 0.75):
    """
    Teaching-only punching check:
      - estimate d ≈ t - cover - 0.5*bar_dia (use #5 as default)
      - critical perimeter at ~d/2 from column face (ACI concept)
      - compute v_u = Vu / (b0 * d)
      - compare to a simple v_c baseline (NOT full ACI 318-19 punching provisions)
    """
    steps = []
    warnings = []

    if any(x is None for x in [Vu_kips, slab_t_in, col_a_in, col_b_in]):
        return None, ["Missing punching inputs."], ["Punching skill: missing inputs."]

    bar_dia = BAR_DB["#5"]["dia"]
    d = slab_t_in - cover_in - 0.5 * bar_dia
    if d <= 0:
        return None, ["Computed d <= 0; check thickness/cover assumptions."], ["Punching skill: invalid d."]

    # critical perimeter at d/2 from column face (concept)
    # b0 = perimeter of rectangle (a + d) by (b + d) if offset d/2 on all sides -> (a + d) and (b + d)
    # since distance is d/2, dimensions grow by d (d/2 each side).
    b0 = 2.0 * ((col_a_in + d) + (col_b_in + d))  # in

    vu_psi = (Vu_kips * 1000.0) / (b0 * d)  # lb / in^2 = psi

    # very simple baseline vc ~ 4*sqrt(fc') (psi) (TEACHING PLACEHOLDER)
    vc_psi = 4.0 * math.sqrt(fc_psi)
    phi_vc = phi_v * vc_psi

    ok = vu_psi <= phi_vc

    steps.append("Punching skill: computed b0 at ~d/2 from column face (concept) and checked vu vs a simple baseline vc (teaching placeholder).")
    warnings.append("Punching shear provisions in ACI 318-19 are more detailed than this placeholder. Use this only to demonstrate agent routing/flagging.")

    return {
        "d_in": d,
        "b0_in": b0,
        "vu_psi": vu_psi,
        "vc_psi_baseline": vc_psi,
        "phi_v": phi_v,
        "phi_vc_psi": phi_vc,
        "ok": ok
    }, warnings, steps


# ============================================================
# Skill: Short tied column (P-M interaction, strain compatibility) - teaching
# ============================================================

def _steel_stress_from_strain(eps: float, fy_psi: float) -> float:
    # bilinear, capped at fy
    fs = Es * eps
    if fs > fy_psi:
        return fy_psi
    if fs < -fy_psi:
        return -fy_psi
    return fs

def column_interaction_curve_rect_tied(b_in: float, h_in: float, cover_in: float, fc_psi: float, fy_psi: float,
                                       bar_size: str, n_bars: int, tie_size: str = DEFAULT_TIE_SIZE,
                                       n_points: int = 40):
    """
    Teaching interaction curve (uniaxial bending about strong axis through centroid),
    assuming bars placed symmetrically at top and bottom layers.

    - Concrete strain at extreme compression = 0.003
    - Use Whitney block for concrete compression
    - Steel stress from strain compatibility (capped at fy)
    - Returns list of (Pn_kips, Mn_kipft)
    """
    if bar_size not in BAR_DB:
        raise ValueError("Invalid bar size.")
    if n_bars not in (4, 6, 8):
        raise ValueError("n_bars supported: 4, 6, 8 (teaching).")

    bar_area = BAR_DB[bar_size]["area"]
    bar_dia = BAR_DB[bar_size]["dia"]
    tie_dia = BAR_DB.get(tie_size, BAR_DB["#3"])["dia"]

    # bar layer distances from extreme compression face (top)
    # assume two layers: top layer and bottom layer
    y_top = cover_in + tie_dia + 0.5 * bar_dia
    y_bot = h_in - (cover_in + tie_dia + 0.5 * bar_dia)

    # distribute bars between layers
    # 4 bars: 2 top, 2 bottom
    # 6 bars: 3 top, 3 bottom
    # 8 bars: 4 top, 4 bottom
    n_top = n_bars // 2
    n_bot = n_bars - n_top

    As_top = n_top * bar_area
    As_bot = n_bot * bar_area

    beta1 = beta1_aci(fc_psi)

    # neutral axis depth c from small to large
    # start at small c (tension-controlled) up to very large c (compression-controlled)
    c_values = [0.5 + i * (h_in * 1.5 - 0.5) / (n_points - 1) for i in range(n_points)]

    curve = []
    for c in c_values:
        a = beta1 * c
        a = min(a, h_in)  # cap at section

        # Concrete compressive force
        Cc_lb = 0.85 * fc_psi * b_in * a
        # concrete resultant location from top = a/2
        y_cc = a / 2.0

        # steel strains (linear)
        eps_top = 0.003 * (c - y_top) / c
        eps_bot = 0.003 * (c - y_bot) / c

        fs_top = _steel_stress_from_strain(eps_top, fy_psi)
        fs_bot = _steel_stress_from_strain(eps_bot, fy_psi)

        Fs_top_lb = fs_top * As_top
        Fs_bot_lb = fs_bot * As_bot

        # Axial nominal strength
        Pn_lb = Cc_lb + Fs_top_lb + Fs_bot_lb
        Pn_kips = Pn_lb / 1000.0

        # Moments about centroid (y measured from top)
        y_cg = h_in / 2.0
        Mn_lb_in = (
            Cc_lb * (y_cc - y_cg) +
            Fs_top_lb * (y_top - y_cg) +
            Fs_bot_lb * (y_bot - y_cg)
        )
        Mn_kipft = abs(Mn_lb_in) / (1000.0 * 12.0)

        curve.append((Pn_kips, Mn_kipft))

    # sort by P descending (typical interaction format)
    curve.sort(key=lambda x: -x[0])
    return curve

def column_design_check(Pu_kips: float, Mu_kipft: float, curve, phi: float = 0.65):
    """
    Teaching check:
      - Interpolate Mn at required Pn = Pu/phi
      - Check Mu <= phi*Mn(Pn_req)
    """
    if Pu_kips is None or Mu_kipft is None:
        return None, ["Pu or Mu missing."]

    P_req = Pu_kips / phi

    Ps = [p for p, _m in curve]
    Ms = [m for _p, m in curve]

    # If outside range, fail (teaching)
    if P_req > max(Ps) or P_req < min(Ps):
        return {
            "phi": phi, "P_req": P_req, "Mn_at_Preq": None, "phiMn_at_Preq": None, "ok": False
        }, ["Pu is outside the computed interaction curve range (teaching). Try changing section/rebar."]

    # find bracket
    for i in range(len(Ps) - 1):
        P1, P2 = Ps[i], Ps[i + 1]
        if (P1 >= P_req >= P2) or (P2 >= P_req >= P1):
            M1, M2 = Ms[i], Ms[i + 1]
            # linear interpolation
            if abs(P2 - P1) < 1e-9:
                Mn_req = max(M1, M2)
            else:
                t = (P_req - P1) / (P2 - P1)
                Mn_req = M1 + t * (M2 - M1)
            phiMn = phi * Mn_req
            ok = Mu_kipft <= phiMn
            return {
                "phi": phi, "P_req": P_req, "Mn_at_Preq": Mn_req, "phiMn_at_Preq": phiMn, "ok": ok
            }, []

    return {
        "phi": phi, "P_req": P_req, "Mn_at_Preq": None, "phiMn_at_Preq": None, "ok": False
    }, ["Could not interpolate interaction curve (unexpected)."]

def tie_spacing_limits_teaching(long_bar_dia_in: float, tie_bar_dia_in: float, least_dim_in: float):
    """
    Teaching tie spacing limit often used for tied columns:
      s <= min(16 db_long, 48 db_tie, least dimension, 12 in)
    (This is commonly referenced in design resources; used here as a teaching check.)
    """
    return min(16.0 * long_bar_dia_in, 48.0 * tie_bar_dia_in, least_dim_in, 12.0)


# ============================================================
# Narrative writer
# ============================================================

def write_beam_narrative(inp, flex, shear, shear_type, punch=None):
    lines = []
    lines.append("=== BEAM DESIGN (Teaching) ===")
    lines.append(f"b={inp['b_in']:.1f} in, h={inp['h_in']:.1f} in, cover={inp['cover_in']:.2f} in")
    lines.append(f"f'c={inp['fc_psi']/1000:.2f} ksi, fy={inp['fy_psi']/1000:.0f} ksi")
    lines.append(f"Mu={inp['Mu_kipft']:.2f} kip-ft, Vu={inp['Vu_kips']:.2f} kips")
    lines.append("")
    lines.append("Flexure:")
    lines.append(f"  Selected {flex['n']} {flex['bar']} bars (As,prov={flex['As_prov']:.2f} in^2)")
    lines.append(f"  d≈{flex['d_in']:.2f} in, a={flex['a_in']:.2f} in, c={flex['c_in']:.2f} in")
    lines.append(f"  eps_t≈{flex['eps_t']:.5f} -> phi≈{flex['phi']:.3f}")
    lines.append(f"  phiMn≈{flex['phiMn_kipft']:.1f} kip-ft (check vs Mu)")
    lines.append(f"  As,min≈{flex['Asmin']:.2f} in^2 -> {'OK' if flex['asmin_ok'] else 'NOT OK'}")
    lines.append("")
    lines.append(f"Shear type detected: {shear_type}")
    lines.append("One-way shear:")
    lines.append(f"  Vc≈{shear['Vc_kips']:.1f} kips, phiVc≈{shear['phiVc_kips']:.1f} kips")
    lines.append(f"  Vu trigger for min stirrups (teaching)≈{shear['Vu_trigger_kips']:.1f} kips")
    lines.append(f"  Av/s(min)≈{shear['Av_over_s_min']:.4f} in^2/in")
    if shear.get("s_use_in") is not None:
        lines.append(f"  Provide {shear['legs']}-leg {shear['stirrup_size']} @ s={shear['s_use_in']:.1f} in (cap {shear['s_cap_in']:.1f} in)")
    else:
        lines.append("  Stirrups not required for strength in baseline; minimum stirrups may still govern depending on region (see warnings).")

    if punch is not None:
        lines.append("")
        lines.append("Two-way (punching) shear (teaching placeholder):")
        lines.append(f"  d≈{punch['d_in']:.2f} in, b0≈{punch['b0_in']:.1f} in")
        lines.append(f"  vu≈{punch['vu_psi']:.1f} psi, phi*vc(baseline)≈{punch['phi_vc_psi']:.1f} psi -> {'OK' if punch['ok'] else 'NOT OK'}")

    lines.append("")
    lines.append("DISCLAIMER: Teaching demo only. Not for final design/stamping.")
    return "\n".join(lines)

def write_column_narrative(inp, bar_size, n_bars, check, s_tie_max, tie_size):
    lines = []
    lines.append("=== SHORT TIED COLUMN (Teaching) ===")
    lines.append(f"Section: {inp['b_in']:.1f} x {inp['h_in']:.1f} in, cover={inp['cover_in']:.2f} in")
    lines.append(f"Materials: f'c={inp['fc_psi']/1000:.2f} ksi, fy={inp['fy_psi']/1000:.0f} ksi")
    lines.append(f"Demand: Pu={inp['Pu_kips']:.1f} kips, Mu={inp['Mu_col_kipft']:.1f} kip-ft")
    lines.append("")
    lines.append(f"Rebar trial: {n_bars} {bar_size} bars (symmetric top/bottom layers in this demo)")
    lines.append(f"Interaction check (interpolated at Pn=Pu/phi): phi={check['phi']:.2f}, P_req={check['P_req']:.1f} kips")
    if check["Mn_at_Preq"] is not None:
        lines.append(f"  Mn(P_req)≈{check['Mn_at_Preq']:.1f} kip-ft -> phiMn≈{check['phiMn_at_Preq']:.1f} kip-ft")
        lines.append(f"  Result: {'OK' if check['ok'] else 'NOT OK'}")
    else:
        lines.append("  Could not evaluate Mn(P_req) (outside curve).")
    lines.append("")
    lines.append(f"Tie spacing (teaching check): ties {tie_size}, s_max≈{s_tie_max:.1f} in (limit min(16db_long,48db_tie,least dim,12)).")
    lines.append("")
    lines.append("DISCLAIMER: Teaching demo only. Not for final design/stamping.")
    return "\n".join(lines)


# ============================================================
# Agent
# ============================================================

def agent_run(prompt: str):
    steps = []
    inp, parse_warn = parse_prompt(prompt)
    warnings = list(parse_warn)

    steps.append("Step 1 — Parse prompt; detect member type and extract inputs.")
    mtype = inp["member_type"]

    # Flag shear type
    shear_type = "one-way (beam)"
    if mtype == "slab":
        shear_type = "two-way (punching)"

    if mtype == "beam":
        steps.append("Step 2 — Flexure skill (Whitney block + strain-based phi).")
        flex, wflex, sflex = flexure_design_beam(
            inp["Mu_kipft"], inp["b_in"], inp["h_in"], inp["cover_in"], inp["fc_psi"], inp["fy_psi"], inp["max_bar"]
        )
        warnings += wflex
        steps += sflex
        if flex is None:
            return None, warnings, steps

        steps.append("Step 3 — Shear skill (one-way; Av/s minimum + spacing cap).")
        shear, ws, ss = shear_design_beam(
            inp["Vu_kips"], inp["b_in"], flex["d_in"], inp["fc_psi"], inp["fy_psi"],
            inp["stirrup_size"], legs=DEFAULT_STIRRUP_LEGS
        )
        warnings += ws
        steps += ss

        # Optional: if user explicitly mentions punching/slab keywords, run punching as a demo flag
        punch_res = None
        if re.search(r"\bpunching\b|\bslab\b|\bfooting\b", prompt, flags=re.IGNORECASE):
            steps.append("Step 4 — Two-way shear detected in prompt; running punching placeholder skill.")
            punch_res, wp, sp = punching_shear_check(
                inp["Vu_kips"], inp["slab_t_in"], inp["cover_in"], inp["fc_psi"],
                inp["col_a_in"], inp["col_b_in"]
            )
            warnings += wp
            steps += sp
            shear_type = "one-way + two-way flag"

        narrative = write_beam_narrative(inp, flex, shear, shear_type, punch=punch_res)

        result = {"member_type": "beam", "inputs": inp, "flexure": flex, "shear": shear, "punching": punch_res, "narrative": narrative}
        warnings.append("ACI 318-19 note: shear (Vc, Av,min triggers, spacing) can be more nuanced than this teaching baseline; treat as demonstration only.")
        return result, warnings, steps

    if mtype == "slab":
        steps.append("Step 2 — Two-way shear (punching) skill (teaching placeholder).")
        punch_res, wp, sp = punching_shear_check(
            inp["Vu_kips"], inp["slab_t_in"], inp["cover_in"], inp["fc_psi"],
            inp["col_a_in"], inp["col_b_in"]
        )
        warnings += wp
        steps += sp
        narrative = "Punching-only mode.\n\n" + ("" if punch_res is None else
                                                f"d≈{punch_res['d_in']:.2f} in, b0≈{punch_res['b0_in']:.1f} in, "
                                                f"vu≈{punch_res['vu_psi']:.1f} psi, phi*vc(baseline)≈{punch_res['phi_vc_psi']:.1f} psi")
        result = {"member_type": "slab", "inputs": inp, "punching": punch_res, "narrative": narrative}
        return result, warnings, steps

    if mtype == "column":
        steps.append("Step 2 — Column skill (strain compatibility P-M interaction; teaching).")

        if inp["b_in"] is None or inp["h_in"] is None or inp["Pu_kips"] is None or inp["Mu_col_kipft"] is None:
            warnings.append("Missing required column inputs (need section, Pu, Mu).")
            return None, warnings, steps

        # If user specified a bar size like "use #8 bars", try that first; otherwise iterate.
        preferred_bar = inp["long_bar"] if (inp["long_bar"] in BAR_DB) else None
        bar_trials = [preferred_bar] if preferred_bar else ["#8", "#7", "#6", "#5"]
        n_trials = [8, 6, 4]

        tie_size = inp["tie_size"] if (inp["tie_size"] in BAR_DB) else DEFAULT_TIE_SIZE
        tie_dia = BAR_DB[tie_size]["dia"]

        best = None
        best_check = None
        best_curve = None

        for bar in bar_trials:
            if bar is None or bar not in BAR_DB:
                continue
            for n_bars in n_trials:
                curve = column_interaction_curve_rect_tied(
                    inp["b_in"], inp["h_in"], inp["cover_in"], inp["fc_psi"], inp["fy_psi"],
                    bar_size=bar, n_bars=n_bars, tie_size=tie_size, n_points=45
                )
                check, wchk = column_design_check(inp["Pu_kips"], inp["Mu_col_kipft"], curve, phi=0.65)
                if wchk:
                    # keep warnings but continue searching
                    pass
                if check["ok"]:
                    best = (bar, n_bars)
                    best_check = check
                    best_curve = curve
                    break
            if best is not None:
                break

        if best is None:
            warnings.append("Could not find a simple tied-column rebar trial (4/6/8 bars) that passes Pu-Mu under this teaching model.")
            return None, warnings, steps

        long_bar_dia = BAR_DB[best[0]]["dia"]
        s_tie_max = tie_spacing_limits_teaching(long_bar_dia, tie_dia, least_dim_in=min(inp["b_in"], inp["h_in"]))

        steps.append(f"  • Selected trial that passes: {best[1]} {best[0]} bars (teaching).")
        steps.append("Step 3 — Tie spacing check (teaching).")

        narrative = write_column_narrative(inp, best[0], best[1], best_check, s_tie_max, tie_size)
        result = {
            "member_type": "column",
            "inputs": inp,
            "rebar": {"bar_size": best[0], "n_bars": best[1], "tie_size": tie_size, "s_tie_max_in": s_tie_max},
            "check": best_check,
            "curve": best_curve,  # shown optionally
            "narrative": narrative
        }
        warnings.append("Column note: slenderness, second-order effects, minimum/maximum steel ratio, and detailed tie rules are not fully implemented (teaching demo).")
        return result, warnings, steps

    warnings.append("Unknown member type.")
    return None, warnings, steps


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="ACI 318-19 RC Agent Demo", layout="wide")
st.title("ACI 318-19 RC Design Copilot (Agent + Skills)")
st.info("Teaching demo only — simplified and incomplete checks. Not for real design/stamping.")

left, right = st.columns([1, 1])

with left:
    st.subheader("Natural-language prompt")

    example_prompts = {
        "Beam Example (Flexure + Shear)":
            "Design a 12x24 beam, cover 1.5 in, fc=4 ksi, fy=60 ksi, "
            "Mu=180 kip-ft, Vu=45 kips, use #8 max bars, stirrups #3.",

        "Beam Example (High Shear → min stirrups)":
            "Design a 12x24 beam, cover 1.5 in, fc=4 ksi, fy=60 ksi, "
            "Mu=180 kip-ft, Vu=95 kips, use #8 max bars, stirrups #3.",

        "Punching Example (Flag + Placeholder Check)":
            "Check punching for slab t=8 in, cover 1.5 in, fc=4 ksi, column=16x16, Vu=220 kips.",

        "Column Example (Short Tied Column)":
            "Design a 16x16 tied column, cover 1.5 in, fc=5 ksi, fy=60 ksi, "
            "Pu=350 kips, Mu=120 kip-ft, use #8 bars, ties #3."
    }

    if "prompt_text" not in st.session_state:
        st.session_state.prompt_text = example_prompts["Beam Example (Flexure + Shear)"]

    st.write("### Quick Example Prompts")
    cols = st.columns(2)
    labels = list(example_prompts.keys())
    for i, lab in enumerate(labels):
        if cols[i % 2].button(lab):
            st.session_state.prompt_text = example_prompts[lab]

    prompt = st.text_area("Prompt", value=st.session_state.prompt_text, height=190)
    run = st.button("Run Agent")

with right:
    st.subheader("Agent Output")
    if run:
        result, warnings, steps = agent_run(prompt)

        with st.expander("🧠 Agent Thinking (step-by-step)", expanded=True):
            for s in steps:
                st.write(s)

        if result is None:
            st.error("Agent could not complete the task. See warnings and try adjusting inputs.")
        else:
            st.success(f"Completed: {result['member_type'].upper()} (teaching demo)")

            st.markdown("### Narrative (auto-generated)")
            st.code(result["narrative"])

            # Optional: show a compact summary
            st.markdown("### Key Outputs")
            if result["member_type"] == "beam":
                f = result["flexure"]
                sh = result["shear"]
                st.write({
                    "Flexure bars": f"{f['n']} {f['bar']}",
                    "As_prov (in^2)": round(f["As_prov"], 2),
                    "phi (flexure)": round(f["phi"], 3),
                    "phiMn (kip-ft)": round(f["phiMn_kipft"], 1),
                    "Stirrups": (f"{sh['legs']}-leg {sh['stirrup_size']} @ {sh['s_use_in']:.1f} in"
                                 if sh.get("s_use_in") is not None else "Not required by strength (baseline)"),
                    "s cap (in)": round(sh["s_cap_in"], 1),
                    "needs min shear?": sh["needs_min_shear"],
                })
            elif result["member_type"] == "slab":
                p = result["punching"]
                st.write({
                    "d (in)": round(p["d_in"], 2),
                    "b0 (in)": round(p["b0_in"], 1),
                    "vu (psi)": round(p["vu_psi"], 1),
                    "phi*vc baseline (psi)": round(p["phi_vc_psi"], 1),
                    "OK?": p["ok"]
                })
            elif result["member_type"] == "column":
                rb = result["rebar"]
                ck = result["check"]
                st.write({
                    "Long bars": f"{rb['n_bars']} {rb['bar_size']}",
                    "Tie size": rb["tie_size"],
                    "Max tie spacing (in)": round(rb["s_tie_max_in"], 1),
                    "phi": ck["phi"],
                    "phiMn at Pu/phi (kip-ft)": None if ck["phiMn_at_Preq"] is None else round(ck["phiMn_at_Preq"], 1),
                    "OK?": ck["ok"]
                })

        st.markdown("### Warnings / Assumptions")
        for w in warnings:
            st.warning(w)
