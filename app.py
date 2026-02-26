# ============================================================
# RC DESIGN COPILOT — (ACI 318-19 teaching version)
# Streamlit + Agent + Skills
# ============================================================

import re
import math
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import html

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

RHO_MIN_COL = 0.01
RHO_MAX_COL = 0.08

# ============================================================
# Parser helpers
# ============================================================
def _find_number(text: str, patterns, default=None):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
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

# ============================================================
# Natural language parser
# ============================================================
def parse_prompt(prompt: str):
    t = prompt.strip()
    warnings = []

    member_type = "beam"
    if re.search(r"\bcolumn\b", t, flags=re.IGNORECASE):
        member_type = "column"
    if re.search(r"\bslab\b|\bone-way slab\b|\bone way slab\b", t, flags=re.IGNORECASE):
        member_type = "slab"

    fc_ksi = _find_number(t, [r"\b(?:f'?c|fc)\s*=?\s*([0-9]*\.?[0-9]+)\s*ksi\b"])
    fc_psi = _find_number(t, [r"\b(?:f'?c|fc)\s*=?\s*([0-9]*\.?[0-9]+)\s*psi\b"])
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

    cover = _find_number(
        t,
        [r"\bcover\s*=?\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"],
        default=1.5 if member_type != "slab" else 0.75
    )

    b = _find_number(t, [r"\bb\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"])
    h = _find_number(t, [r"\bh\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"])
    m_dim = re.search(r"\b([0-9]*\.?[0-9]+)\s*[xX]\s*([0-9]*\.?[0-9]+)\b", t)
    if m_dim:
        b = b if b is not None else float(m_dim.group(1))
        h = h if h is not None else float(m_dim.group(2))

    bw = _find_number(t, [r"\bbw\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"], default=None)
    bf = _find_number(t, [r"\bbf\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"], default=None)
    hf = _find_number(t, [r"\bhf\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"], default=None)

    is_t_beam = bool(re.search(r"\bt[-\s]*beam\b|\btbeam\b", t, flags=re.IGNORECASE)) or (bw is not None or bf is not None or hf is not None)

    auto_beff = bool(re.search(r"\bauto\s*beff\b|\bauto\s*b_f\b|\bauto\s*bf\b", t, flags=re.IGNORECASE))

    Ln_ft = _find_number(t, [r"\bLn\s*=\s*([0-9]*\.?[0-9]+)\s*(?:ft|feet)\b"])
    Ln_in = _find_number(t, [r"\bLn\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"])
    if Ln_ft is not None:
        Ln = Ln_ft * 12.0
    elif Ln_in is not None:
        Ln = Ln_in
    else:
        Ln = None

    sw_ft = _find_number(t, [r"\bsw\s*=\s*([0-9]*\.?[0-9]+)\s*(?:ft|feet)\b"])
    sw_in = _find_number(t, [r"\bsw\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"])
    if sw_ft is not None:
        sw = sw_ft * 12.0
    elif sw_in is not None:
        sw = sw_in
    else:
        sw = None

    moment_sign = "positive"
    if re.search(r"\bnegative\b|\bneg\b|\btop tension\b", t, flags=re.IGNORECASE):
        moment_sign = "negative"
    if re.search(r"\bpositive\b|\bpos\b|\bbottom tension\b", t, flags=re.IGNORECASE):
        moment_sign = "positive"

    Mu = _find_number(t, [r"\bmu\s*=?\s*([0-9]*\.?[0-9]+)\s*(?:kip[-\s]*ft|kft)\b"])
    Vu = _find_number(t, [r"\bvu\s*=?\s*([0-9]*\.?[0-9]+)\s*kips?\b"])

    Pu = _find_number(t, [r"\bpu\s*=?\s*([0-9]*\.?[0-9]+)\s*kips?\b"])
    Mu_col = Mu if member_type == "column" else _find_number(t, [r"\bmu_col\s*=?\s*([0-9]*\.?[0-9]+)\s*(?:kip[-\s]*ft|kft)\b"])

    slab_t = _find_number(t, [
        r"\bt\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b",
        r"\bthickness\s*=?\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"
    ])
    L_ft = _find_number(t, [
        r"\bL\s*=\s*([0-9]*\.?[0-9]+)\s*(?:ft|feet)\b",
        r"\bspan\s*=?\s*([0-9]*\.?[0-9]+)\s*(?:ft|feet)\b"
    ])
    wu_psf = _find_number(t, [
        r"\bwu\s*=\s*([0-9]*\.?[0-9]+)\s*psf\b",
        r"\bload\s*=?\s*([0-9]*\.?[0-9]+)\s*psf\b"
    ])

    slab_support = "simply"
    if re.search(r"\bcontinuous\b|\binterior\b", t, flags=re.IGNORECASE):
        slab_support = "continuous"

    max_bar = _find_bar(t, [r"(#\s*[0-9]+)\s*(?:max|maximum)"])
    long_bar = _find_bar(t, [r"\b(?:use|bars?|main bars?)\s*(#\s*[0-9]+)\b"], default=None)
    stirrup_size = _find_bar(t, [r"stirrups?\s*(#\s*[0-9]+)"], default=None)
    tie_size = _find_bar(t, [r"\bties?\s*(#\s*[0-9]+)"], default=None)

    beam_mode = "combined"
    if member_type == "beam":
        if Mu is not None and Vu is None:
            beam_mode = "flexure"
        elif Vu is not None and Mu is None:
            beam_mode = "shear"
        elif Mu is None and Vu is None:
            beam_mode = "flexure"

    if member_type in ("beam", "column"):
        if (b is None or h is None) and not is_t_beam:
            warnings.append("Section size not found. Use '12x24' or 'b=12 in h=24 in'.")
        if is_t_beam:
            if bw is None:
                warnings.append("T-beam web width bw not found.")
            if h is None:
                warnings.append("Overall depth h not found for T-beam.")
            if hf is None:
                warnings.append("Flange thickness hf not found.")
            if (not auto_beff) and (bf is None):
                warnings.append("Effective flange width bf not found.")

        if is_t_beam and auto_beff and bf is None:
            if Ln is None:
                warnings.append("Auto beff requested but Ln not found.")
            if sw is None:
                warnings.append("Auto beff requested but sw not found.")

    if member_type == "beam":
        if Mu is None and beam_mode in ("combined", "flexure"):
            warnings.append("Beam Mu not found. Example: 'Mu=180 kip-ft'.")
        if Vu is None and beam_mode in ("combined", "shear"):
            warnings.append("Beam Vu not found. Example: 'Vu=45 kips'.")
        if beam_mode == "shear" and long_bar is None:
            warnings.append("For shear-only, include a main bar size for d estimate.")

    if member_type == "column":
        if Pu is None:
            warnings.append("Column Pu not found.")
        if Mu_col is None:
            warnings.append("Column Mu not found.")

    if member_type == "slab":
        if slab_t is None:
            warnings.append("Slab thickness not found.")
        if L_ft is None:
            warnings.append("Slab span not found.")
        if wu_psf is None:
            warnings.append("Slab wu not found.")

    return {
        "member_type": member_type,
        "beam_mode": beam_mode,
        "is_t_beam": is_t_beam,
        "auto_beff": auto_beff,
        "Ln_in": Ln,
        "sw_in": sw,
        "moment_sign": moment_sign,
        "b_in": b,
        "h_in": h,
        "bw_in": bw if bw is not None else b,
        "bf_in": bf,
        "hf_in": hf,
        "cover_in": cover,
        "fc_psi": fc,
        "fy_psi": fy,
        "Mu_kipft": Mu,
        "Vu_kips": Vu,
        "Pu_kips": Pu,
        "Mu_col_kipft": Mu_col,
        "slab_t_in": slab_t,
        "slab_L_ft": L_ft,
        "slab_wu_psf": wu_psf,
        "slab_support": slab_support,
        "max_bar": max_bar,
        "long_bar": long_bar,
        "stirrup_size": stirrup_size,
        "tie_size": tie_size,
    }, warnings

# ============================================================
# ACI-style helpers
# ============================================================
def beta1_aci(fc_psi: float) -> float:
    if fc_psi <= 4000:
        return 0.85
    beta = 0.85 - 0.05 * ((fc_psi - 4000) / 1000.0)
    return max(0.65, beta)

def eps_y(fy_psi: float) -> float:
    return fy_psi / Es

def phi_flexure_from_eps_t(eps_t: float, fy_psi: float) -> float:
    eps_tc = eps_y(fy_psi) + 0.003
    if eps_t >= eps_tc:
        return 0.90
    if eps_t <= 0.002:
        return 0.65
    return 0.65 + (eps_t - 0.002) * (0.90 - 0.65) / (eps_tc - 0.002)

def clear_cover_definition_note() -> str:
    return "Cover represents clear cover to the OUTSIDE surface of stirrups/ties."

def effective_depth(h_in: float, cover_in: float, main_bar_dia_in: float, stirrup_dia_in: float = 0.375) -> float:
    return h_in - cover_in - stirrup_dia_in - 0.5 * main_bar_dia_in

def As_min_beam_us(fc_psi: float, fy_psi: float, b_in: float, d_in: float) -> float:
    term1 = 3.0 * math.sqrt(fc_psi) / fy_psi * b_in * d_in
    term2 = 200.0 / fy_psi * b_in * d_in
    return max(term1, term2)

def shear_Av_over_s_min_us(fc_psi: float, fy_psi: float, bw_in: float) -> float:
    return max(0.75 * math.sqrt(fc_psi) * bw_in / fy_psi, 50.0 * bw_in / fy_psi)

def shear_Vc_simple_us(fc_psi: float, bw_in: float, d_in: float, lam: float = 1.0) -> float:
    return (2.0 * lam * math.sqrt(fc_psi) * bw_in * d_in) / 1000.0

def compute_beff_aci_teaching(bw_in: float, hf_in: float, Ln_in: float, sw_in: float) -> float:
    be = min(Ln_in / 8.0, 8.0 * hf_in, sw_in / 2.0)
    return bw_in + 2.0 * be

# ============================================================
# Beam flexure skill
# ============================================================
def tbeam_compression_resultant(fc_psi: float, bw: float, bf: float, hf: float, a: float):
    stress = 0.85 * fc_psi
    if a <= hf:
        C = stress * bf * a
        ybar = a / 2.0
        return C, ybar
    C1 = stress * bf * hf
    y1 = hf / 2.0
    C2 = stress * bw * (a - hf)
    y2 = hf + (a - hf) / 2.0
    C = C1 + C2
    ybar = (C1 * y1 + C2 * y2) / C
    return C, ybar

def solve_a_from_As_Tbeam(As: float, fy_psi: float, fc_psi: float, bw: float, bf: float, hf: float):
    stress = 0.85 * fc_psi
    a_rect = As * fy_psi / (stress * bf)
    if a_rect <= hf:
        return a_rect
    return hf + (As * fy_psi / stress - bf * hf) / bw

def flexure_design_beam_general(inp):
    steps, warnings = [], []
    fc, fy = inp["fc_psi"], inp["fy_psi"]
    Mu, cover, h = inp["Mu_kipft"], inp["cover_in"], inp["h_in"]
    moment_sign, is_t = inp["moment_sign"], inp["is_t_beam"]
    bw, bf, hf = inp["bw_in"], inp["bf_in"], inp["hf_in"]

    if Mu is None or h is None:
        return None, ["Missing Mu or h."], steps

    auto_bf_used = False
    bf_auto = None
    if is_t and inp.get("auto_beff", False):
        Ln_in, sw_in = inp.get("Ln_in"), inp.get("sw_in")
        if (Ln_in is not None) and (sw_in is not None) and (hf is not None) and (bw is not None):
            bf_auto = compute_beff_aci_teaching(bw, hf, Ln_in, sw_in)
            if bf is None:
                bf = bf_auto
                auto_bf_used = True
        else:
            warnings.append("Auto beff is ON but could not compute bf.")

    bar_keys = [k for k in BAR_DB.keys() if k != "#3"]
    if inp["max_bar"] and inp["max_bar"] in bar_keys:
        bar_keys = bar_keys[: bar_keys.index(inp["max_bar"]) + 1]

    beta1 = beta1_aci(fc)
    best = None

    for bar in bar_keys:
        Ab, db = BAR_DB[bar]["area"], BAR_DB[bar]["dia"]
        d = effective_depth(h, cover, db, stirrup_dia_in=BAR_DB["#3"]["dia"])

        for n in range(2, 16):
            As = n * Ab
            if not is_t:
                b_rect = inp["b_in"] if inp["b_in"] is not None else bw
                a = As * fy / (0.85 * fc * b_rect)
                ybar = a / 2.0
            else:
                if moment_sign == "negative":
                    b_rect = bw
                    a = As * fy / (0.85 * fc * b_rect)
                    ybar = a / 2.0
                else:
                    if (bw is None) or (bf is None) or (hf is None): continue
                    a = solve_a_from_As_Tbeam(As, fy, fc, bw=bw, bf=bf, hf=hf)
                    _C, ybar = tbeam_compression_resultant(fc, bw=bw, bf=bf, hf=hf, a=a)

            T = As * fy
            Mn_lbin = T * (d - ybar)
            Mn_kipft = Mn_lbin / (1000.0 * 12.0)
            c = a / beta1 if beta1 > 0 else 0.0
            eps_t = 0.003 * (d - c) / c if c > 1e-9 else 0.0
            phi = phi_flexure_from_eps_t(eps_t, fy)
            phiMn = phi * Mn_kipft

            b_for_Asmin = inp["b_in"] if inp["b_in"] is not None else bw
            Asmin = As_min_beam_us(fc, fy, b_for_Asmin, d)

            if phiMn >= Mu:
                score = (0 if As >= Asmin else 1, n, As)
                cand = {
                    "shape": "T-beam" if is_t else "Rectangular",
                    "moment_sign": moment_sign,
                    "bar": bar,
                    "n": n,
                    "As_prov": As,
                    "Asmin": Asmin,
                    "asmin_ok": (As >= Asmin),
                    "d_in": d,
                    "a_in": a,
                    "c_in": c,
                    "eps_t": eps_t,
                    "phi": phi,
                    "Mn_kipft": Mn_kipft,
                    "phiMn_kipft": phiMn,
                    "ybar_in": ybar,
                    "bw_in": bw if bw is not None else inp["b_in"],
                    "bf_in": bf,
                    "hf_in": hf,
                    "h_in": h,
                    "b_in": inp["b_in"] if inp["b_in"] is not None else bw,
                    "cover_in": cover,
                    "stirrup_dia_in": BAR_DB["#3"]["dia"],
                    "auto_beff": inp.get("auto_beff", False),
                    "bf_auto_in": bf_auto,
                    "bf_auto_used": auto_bf_used,
                }
                if best is None or score < best[1]:
                    best = (cand, score)
                break

    if best is None:
        warnings.append("Flexure design could not find a bar arrangement satisfying ϕMn ≥ Mu.")
        return None, warnings, steps

    steps.append("Beam flexure: searched bar sizes/counts to satisfy ϕMn ≥ Mu.")
    return best[0], warnings, steps

# ============================================================
# Beam shear skill
# ============================================================
def shear_design_beam(inp, d_in: float | None):
    steps, warnings = [], []
    Vu, fc, fy = inp["Vu_kips"], inp["fc_psi"], inp["fy_psi"]
    bw = inp["bw_in"] if inp["bw_in"] is not None else inp["b_in"]

    if Vu is None or bw is None:
        return None, ["Missing Vu or bw/b."], steps

    phi_v = 0.75
    if d_in is None:
        main_bar = inp["long_bar"] if (inp["long_bar"] in BAR_DB) else "#8"
        db = BAR_DB[main_bar]["dia"]
        if inp["h_in"] is None: return None, ["Missing h for shear."], steps
        d_in = effective_depth(inp["h_in"], inp["cover_in"], db, stirrup_dia_in=BAR_DB["#3"]["dia"])

    stirrup_size = inp["stirrup_size"] if (inp["stirrup_size"] in STIRRUP_LEG_AREA) else DEFAULT_STIRRUP_SIZE
    Av = DEFAULT_STIRRUP_LEGS * STIRRUP_LEG_AREA[stirrup_size]

    Vc = shear_Vc_simple_us(fc, bw, d_in)
    phiVc = phi_v * Vc
    s_cap = min(d_in / 2.0, 24.0)

    Av_over_s_min = shear_Av_over_s_min_us(fc, fy, bw)
    Vu_trigger = phi_v * math.sqrt(fc) * bw * d_in / 1000.0
    needs_min = Vu > Vu_trigger

    Vn_req = Vu / phi_v
    Vs_req = max(0.0, Vn_req - Vc)

    if Vs_req <= 1e-9:
        s_use = min(Av / Av_over_s_min, s_cap) if needs_min else None
        steps.append("Beam shear: Vu ≤ ϕVc.")
        return {
            "phi_v": phi_v, "d_in": d_in, "bw_in": bw,
            "Vc_kips": Vc, "phiVc_kips": phiVc,
            "stirrup_size": stirrup_size, "legs": DEFAULT_STIRRUP_LEGS, "Av_in2": Av,
            "needs_min": needs_min, "Av_over_s_min": Av_over_s_min,
            "s_use_in": s_use, "s_cap_in": s_cap, "Vs_req_kips": 0.0
        }, warnings, steps

    s_strength = (Av * fy * d_in) / (Vs_req * 1000.0)
    s_from_min = Av / Av_over_s_min
    s_req = min(s_strength, s_from_min) if needs_min else s_strength
    s_use = min(s_req, s_cap)

    if needs_min and (Av / s_use) + 1e-12 < Av_over_s_min:
        warnings.append("Cannot satisfy Av/s(min). Increase stirrup size or legs.")

    steps.append("Beam shear: computed Vc and Vs; sized stirrups.")
    return {
        "phi_v": phi_v, "d_in": d_in, "bw_in": bw,
        "Vc_kips": Vc, "phiVc_kips": phiVc,
        "stirrup_size": stirrup_size, "legs": DEFAULT_STIRRUP_LEGS, "Av_in2": Av,
        "needs_min": needs_min, "Av_over_s_min": Av_over_s_min,
        "s_strength_in": s_strength, "s_from_min_in": s_from_min, "s_req_in": s_req,
        "s_use_in": s_use, "s_cap_in": s_cap, "Vs_req_kips": Vs_req
    }, warnings, steps

# ============================================================
# Column skill
# ============================================================
def column_bar_layout_perimeter(b_in, h_in, cover_in, tie_size, bar_size, n_bars):
    tie_dia = BAR_DB[tie_size]["dia"]
    bar_dia = BAR_DB[bar_size]["dia"]
    off = cover_in + tie_dia + 0.5 * bar_dia
    xL, xR = off, b_in - off
    yT, yB = off, h_in - off

    pts = []
    def add_unique(p):
        if p not in pts: pts.append(p)

    add_unique((xL, yT)); add_unique((xR, yT)); add_unique((xR, yB)); add_unique((xL, yB))
    if n_bars == 4: return pts

    def interior_points(a, b, k):
        return [a + (i + 1) * (b - a) / (k + 1) for i in range(k)] if k > 0 else []

    k = 1 if n_bars == 8 else (2 if n_bars == 12 else 3)
    for x in interior_points(xL, xR, k): add_unique((x, yT)); add_unique((x, yB))
    for y in interior_points(yT, yB, k): add_unique((xL, y)); add_unique((xR, y))
    return pts[:n_bars]

def column_interaction_curve(b_in, h_in, cover_in, fc_psi, fy_psi, bar_size, n_bars, tie_size):
    Ab = BAR_DB[bar_size]["area"]
    beta1 = beta1_aci(fc_psi)
    bars = column_bar_layout_perimeter(b_in, h_in, cover_in, tie_size, bar_size, n_bars)

    c_vals = [0.5 + i * (1.5 * h_in - 0.5) / 59 for i in range(60)]
    y_cg, curve = h_in / 2.0, []

    for c in c_vals:
        a = min(beta1 * c, h_in)
        Cc = 0.85 * fc_psi * b_in * a
        y_cc = a / 2.0
        Psteel, Msteel = 0.0, 0.0

        for (_x, y) in bars:
            eps = 0.003 * (c - y) / c
            fs = max(-fy_psi, min(fy_psi, Es * eps))
            Fs = fs * Ab
            Psteel += Fs
            Msteel += Fs * (y - y_cg)

        Pn = (Cc + Psteel) / 1000.0
        Mn = abs(Cc * (y_cc - y_cg) + Msteel) / (1000.0 * 12.0)
        curve.append((Pn, Mn))

    curve.sort(key=lambda x: -x[0])
    return curve

def _interp_M_at_P(P_req, Ps, Ms):
    if P_req > max(Ps) or P_req < min(Ps): return None
    for i in range(len(Ps) - 1):
        if (Ps[i] >= P_req >= Ps[i + 1]) or (Ps[i + 1] >= P_req >= Ps[i]):
            if abs(Ps[i + 1] - Ps[i]) < 1e-12: return max(Ms[i], Ms[i + 1])
            return Ms[i] + (P_req - Ps[i]) / (Ps[i + 1] - Ps[i]) * (Ms[i + 1] - Ms[i])
    return None

def column_design_tied(inp):
    steps, warnings = [], []
    b, h, Pu, Mu = inp["b_in"], inp["h_in"], inp["Pu_kips"], inp["Mu_col_kipft"]
    fc, fy, cover = inp["fc_psi"], inp["fy_psi"], inp["cover_in"]

    if None in (b, h, Pu, Mu): return None, ["Missing inputs."], steps
    tie_size = inp["tie_size"] if inp["tie_size"] in BAR_DB else DEFAULT_TIE_SIZE
    
    bar_trials = [inp["long_bar"]] if (inp["long_bar"] in BAR_DB) else []
    for x in ["#11", "#10", "#9", "#8", "#7", "#6", "#5"]:
        if x not in bar_trials: bar_trials.append(x)

    best, phi = None, 0.65
    P_req = Pu / phi

    for bar in bar_trials:
        for n_bars in [16, 12, 8, 4]:
            rho = (n_bars * BAR_DB[bar]["area"]) / (b * h)
            if not (RHO_MIN_COL <= rho <= RHO_MAX_COL): continue

            curve = column_interaction_curve(b, h, cover, fc, fy, bar, n_bars, tie_size)
            Mn_req = _interp_M_at_P(P_req, [p for p, _m in curve], [m for _p, m in curve])
            if Mn_req is None: continue

            if Mu <= phi * Mn_req:
                smax = min(16.0 * BAR_DB[bar]["dia"], 48.0 * BAR_DB[tie_size]["dia"], min(b, h), 12.0)
                best = {"bar_size": bar, "n_bars": n_bars, "Ast_in2": n_bars * BAR_DB[bar]["area"],
                        "rho": rho, "phi": phi, "P_req": P_req, "Mn_at_Preq": Mn_req,
                        "phiMn_at_Preq": phi * Mn_req, "ok": True, "tie_size": tie_size, "s_tie_max_in": smax}
                break
        if best: break

    if not best: warnings.append("No column trial passed Pu–Mu.")
    return best, warnings, steps

# ============================================================
# One-way slab skill
# ============================================================
def slab_design_one_way(inp):
    steps, warnings = [], []
    t_in, L_ft, wu_psf = inp["slab_t_in"], inp["slab_L_ft"], inp["slab_wu_psf"]
    fc, fy, cover = inp["fc_psi"], inp["fy_psi"], inp["cover_in"]

    if None in (t_in, L_ft, wu_psf): return None, ["Missing slab inputs."], steps
    b, h = 12.0, t_in
    As_min_req = 0.0018 * b * h

    if inp["slab_support"] == "continuous":
        Mu_kipft = (wu_psf * (L_ft ** 2) / 12.0) / 1000.0
        support_note = "Continuous interior (wL²/12)"
    else:
        Mu_kipft = (wu_psf * (L_ft ** 2) / 8.0) / 1000.0
        support_note = "Simply supported (wL²/8)"

    candidates, phi = [], 0.90
    for bar in ["#4", "#5"]:
        d = h - cover - 0.5 * BAR_DB[bar]["dia"]
        if d <= 0: continue

        for s in range(4, int(math.floor(min(3.0 * h, 18.0))) + 1):
            As = BAR_DB[bar]["area"] * (12.0 / s)
            if As < As_min_req: continue

            a = As * fy / (0.85 * fc * b)
            phiMn = phi * ((As * fy) * (d - a / 2.0) / 12000.0)

            if phiMn >= Mu_kipft:
                candidates.append({"bar": bar, "s_in": s, "As_in2_per_ft": As, "d_in": d, "a_in": a,
                                   "phi": phi, "phiMn_kipft": phiMn, "Mu_kipft": Mu_kipft, "support_note": support_note})
                break

    if not candidates: return None, ["Could not find spacing to satisfy moment AND min steel."], steps
    return sorted(candidates, key=lambda x: (-x["s_in"], x["bar"]))[0], warnings, steps

# ============================================================
# HTML NARRATIVE BUILDERS (Upgraded from Plain Text)
# ============================================================
def narrative_beam_flexure_html(inp, f):
    lines = []
    lines.append("<div style='margin-bottom: 8px;'><strong style='color: #1f77b4;'>=== BEAM FLEXURE ===</strong></div>")
    lines.append(f"<div style='font-size: 0.85em; color: #666; margin-bottom: 12px;'><em>{clear_cover_definition_note()}</em></div>")
    lines.append("<ul style='margin-top:0; padding-left: 20px;'>")
    lines.append(f"<li><strong>Shape:</strong> {f['shape']} | <strong>Moment:</strong> {f['moment_sign']}</li>")
    
    if f["shape"] == "T-beam":
        lines.append(f"<li><strong>Dimensions:</strong> bw = {f['bw_in']}\", bf = {f['bf_in']}\", hf = {f['hf_in']}\"</li>")
        if f.get("auto_beff", False):
            if f.get("bf_auto_used", False):
                lines.append(f"<li><strong>Auto beff ON:</strong> computed &approx; {f.get('bf_auto_in'):.1f}\"</li>")
            elif f.get("bf_auto_in") is not None:
                lines.append(f"<li><strong>Auto beff ON:</strong> &approx; {f.get('bf_auto_in'):.1f}\" (user bf applied)</li>")
            else:
                lines.append("<li><strong>Auto beff ON:</strong> could not compute (needs Ln and sw)</li>")

    lines.append(f"<li><strong>Materials:</strong> f'c = {inp['fc_psi']/1000:.2f} ksi, fy = {inp['fy_psi']/1000:.0f} ksi</li>")
    lines.append(f"<li><strong>Demand:</strong> Mu = {inp['Mu_kipft']:.2f} kip-ft</li>")
    lines.append(f"<li><strong>Selected Steel:</strong> <span style='background-color: #e6f2ff; padding: 2px 6px; border-radius: 4px; color: #0056b3;'><strong>{f['n']} {f['bar']} bars</strong></span> &rarr; As = {f['As_prov']:.2f} in&sup2; (As,min = {f['Asmin']:.2f} in&sup2;)</li>")
    lines.append(f"<li><strong>Section Props:</strong> d &approx; {f['d_in']:.2f}\", a = {f['a_in']:.2f}\", c &approx; {f['c_in']:.2f}\"</li>")
    lines.append(f"<li><strong>Strain Check:</strong> &epsilon;<sub>t</sub> &approx; {f['eps_t']:.5f} &rarr; &phi; = {f['phi']:.3f}</li>")
    lines.append(f"<li><strong>Capacity:</strong> Mn &approx; {f['Mn_kipft']:.1f} kip-ft &rarr; <strong>&phi;Mn &approx; {f['phiMn_kipft']:.1f} kip-ft</strong></li>")
    lines.append("</ul>")
    return "".join(lines)

def narrative_beam_shear_html(inp, sh):
    lines = []
    lines.append("<div style='margin-bottom: 8px; margin-top: 24px;'><strong style='color: #1f77b4;'>=== BEAM SHEAR ===</strong></div>")
    lines.append("<ul style='margin-top:0; padding-left: 20px;'>")
    lines.append(f"<li><strong>Section:</strong> Vu = {inp['Vu_kips']:.2f} kips, bw = {sh['bw_in']:.1f}\", d &approx; {sh['d_in']:.2f}\"</li>")
    lines.append(f"<li><strong>Concrete Capacity:</strong> Vc &approx; {sh['Vc_kips']:.1f} kips &rarr; <strong>&phi;Vc &approx; {sh['phiVc_kips']:.1f} kips</strong> (&phi;={sh['phi_v']:.2f})</li>")
    if sh["s_use_in"] is None:
        lines.append("<li><strong>Result:</strong> Stirrups not required by strength in baseline.</li>")
    else:
        lines.append(f"<li><strong>Result:</strong> Provide <span style='background-color: #e6f2ff; padding: 2px 6px; border-radius: 4px; color: #0056b3;'><strong>{sh['legs']}-leg {sh['stirrup_size']} @ s = {sh['s_use_in']:.1f}\"</strong></span> (cap = {sh['s_cap_in']:.1f}\")</li>")
    lines.append("</ul>")
    return "".join(lines)

def narrative_column_html(inp, col):
    lines = []
    lines.append("<div style='margin-bottom: 8px;'><strong style='color: #1f77b4;'>=== SHORT TIED COLUMN ===</strong></div>")
    lines.append(f"<div style='font-size: 0.85em; color: #666; margin-bottom: 12px;'><em>{clear_cover_definition_note()}</em></div>")
    lines.append("<ul style='margin-top:0; padding-left: 20px;'>")
    lines.append(f"<li><strong>Section:</strong> {inp['b_in']:.1f}\" x {inp['h_in']:.1f}\"</li>")
    lines.append(f"<li><strong>Materials:</strong> f'c = {inp['fc_psi']/1000:.2f} ksi, fy = {inp['fy_psi']/1000:.0f} ksi</li>")
    lines.append(f"<li><strong>Demands:</strong> Pu = {inp['Pu_kips']:.1f} kips, Mu = {inp['Mu_col_kipft']:.1f} kip-ft</li>")
    lines.append(f"<li><strong>Selected Long. Steel:</strong> <span style='background-color: #e6f2ff; padding: 2px 6px; border-radius: 4px; color: #0056b3;'><strong>{col['n_bars']} {col['bar_size']} bars</strong></span> &rarr; Ast = {col['Ast_in2']:.2f} in&sup2; (&rho; = {col['rho']:.3f})</li>")
    lines.append(f"<li><strong>Capacity:</strong> P_req = {col['P_req']:.1f} kips &rarr; &phi;Mn(P_req) &approx; {col['phiMn_at_Preq']:.1f} kip-ft (OK: {col['ok']})</li>")
    lines.append(f"<li><strong>Ties:</strong> Use {col['tie_size']} ties &rarr; s_max &approx; {col['s_tie_max_in']:.1f}\"</li>")
    lines.append("</ul>")
    return "".join(lines)

def narrative_slab_html(inp, sres):
    lines = []
    lines.append("<div style='margin-bottom: 8px;'><strong style='color: #1f77b4;'>=== ONE-WAY SLAB ===</strong></div>")
    lines.append("<ul style='margin-top:0; padding-left: 20px;'>")
    lines.append(f"<li><strong>Model:</strong> {sres['support_note']}</li>")
    lines.append(f"<li><strong>Parameters:</strong> L = {inp['slab_L_ft']:.2f} ft, wu = {inp['slab_wu_psf']:.1f} psf</li>")
    lines.append(f"<li><strong>Demand:</strong> Mu &approx; {sres['Mu_kipft']:.2f} kip-ft (per 1-ft strip)</li>")
    lines.append(f"<li><strong>Result:</strong> Provide <span style='background-color: #e6f2ff; padding: 2px 6px; border-radius: 4px; color: #0056b3;'><strong>{sres['bar']} @ {sres['s_in']}\"</strong></span> &rarr; As &approx; {sres['As_in2_per_ft']:.3f} in&sup2;/ft</li>")
    lines.append(f"<li><strong>Capacity:</strong> d &approx; {sres['d_in']:.2f}\", &phi;Mn &approx; {sres['phiMn_kipft']:.2f} kip-ft</li>")
    lines.append("</ul>")
    return "".join(lines)


# ============================================================
# Design Sketch 
# ============================================================
def _dim_arrow(ax, x1, y1, x2, y2, label, text_offset=(0, 0), fontsize=10):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="<->", linewidth=1.5))
    xm = 0.5 * (x1 + x2) + text_offset[0]
    ym = 0.5 * (y1 + y2) + text_offset[1]
    ax.text(xm, ym, label, fontsize=fontsize, va="center", ha="center")

def draw_design_sketch_section(flex: dict):
    shape = flex.get("shape", "Rectangular")
    h, cover, stirrup_dia = float(flex.get("h_in", 24.0)), float(flex.get("cover_in", 1.5)), float(flex.get("stirrup_dia_in", 0.375))
    moment_sign = flex.get("moment_sign", "positive")
    tension_face = "bottom" if moment_sign == "positive" else "top"
    compression_face = "top" if tension_face == "bottom" else "bottom"

    bar = flex.get("bar", "#8")
    db = BAR_DB.get(bar, BAR_DB["#8"])["dia"]
    n_bars = int(flex.get("n", 4))
    a, c, d = float(flex.get("a_in", 0.0)), float(flex.get("c_in", 0.0)), float(flex.get("d_in", h - cover - stirrup_dia - 0.5 * db))

    if shape == "T-beam":
        bw, bf, hf = float(flex.get("bw_in", 12.0)), float(flex.get("bf_in", 36.0)), float(flex.get("hf_in", 4.0))
        b_web, b_total = bw, bf
    else:
        b_web, b_total, hf, bw, bf = float(flex.get("b_in", 12.0)), float(flex.get("b_in", 12.0)), 0.0, float(flex.get("b_in", 12.0)), float(flex.get("b_in", 12.0))

    # REDUCED SKETCH SIZE (75% of original 5.75x3.25)
    fig, ax = plt.subplots(figsize=(4.3, 2.4))

    if shape == "T-beam":
        web_left = (b_total - b_web) / 2.0
        ax.add_patch(patches.Rectangle((0, h - hf), b_total, hf, fill=False, linewidth=2))
        ax.add_patch(patches.Rectangle((web_left, 0), b_web, h - hf, fill=False, linewidth=2))
        sec_left, sec_right, web_right = 0.0, b_total, web_left + b_web
    else:
        ax.add_patch(patches.Rectangle((0, 0), b_total, h, fill=False, linewidth=2))
        sec_left, sec_right, web_left, web_right = 0.0, b_total, 0.0, b_total

    if 0 < a < h:
        y0, height = (h - a, a) if compression_face == "top" else (0.0, a)
        comp_left, comp_w = (0.0, b_total) if shape == "T-beam" and moment_sign == "positive" else (web_left, web_right - web_left)
        ax.add_patch(patches.Rectangle((comp_left, y0), comp_w, height, fill=True, alpha=0.12, linewidth=0))
        ax.text(sec_right + 0.8, y0 + 0.5 * height, f"a={a:.2f} in", va="center")

    if 0 < c < h:
        y_na = (h - c) if compression_face == "top" else c
        ax.plot([sec_left, sec_right], [y_na, y_na], linestyle="--", linewidth=1.5)
        ax.text(sec_right + 0.8, y_na, f"c={c:.2f} in", va="center")

    if tension_face == "bottom":
        y_steel = cover + stirrup_dia + 0.5 * db
        y_d_start, y_d_end = h, y_steel
    else:
        y_steel = h - (cover + stirrup_dia + 0.5 * db)
        y_d_start, y_d_end = 0.0, y_steel

    xL, xR = web_left + 0.9, web_right - 0.9
    xs = [(xL + xR) / 2.0] if n_bars <= 1 else [xL + i * (xR - xL) / (n_bars - 1) for i in range(n_bars)]
    for x in xs:
        ax.add_patch(patches.Circle((x, y_steel), radius=max(0.18, 0.18 * db), fill=True))

    ax.text(sec_left, y_steel - (1.4 if tension_face == "bottom" else -1.0), f"{n_bars} {bar}", ha="left", va="center")
    
    ax.annotate("", xy=(sec_right + 0.2, y_d_start), xytext=(sec_right + 0.2, y_d_end), arrowprops=dict(arrowstyle="<->", linewidth=1.5))
    ax.text(sec_right + 0.6, 0.5 * (y_d_start + y_d_end), f"d≈{d:.2f} in", va="center")

    _dim_arrow(ax, sec_left, -0.8, sec_right, -0.8, f"b={b_total:.1f} in", text_offset=(0, -0.2))
    _dim_arrow(ax, -0.8, 0.0, -0.8, h, f"h={h:.1f} in", text_offset=(-0.2, 0))

    if shape == "T-beam":
        _dim_arrow(ax, 0.0, h + 0.8, b_total, h + 0.8, f"bf={b_total:.1f} in", text_offset=(0, 0.2))
        _dim_arrow(ax, web_left, (h - hf) / 2.0, web_right, (h - hf) / 2.0, f"bw={b_web:.1f} in")
        _dim_arrow(ax, sec_right + 1.6, h - hf, sec_right + 1.6, h, f"hf={hf:.1f} in")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(sec_left - 2.5, sec_right + 8.5)
    ax.set_ylim(-2.2, h + 2.2)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ============================================================
# Agent router
# ============================================================
def agent_run(prompt: str):
    steps = []
    inp, parse_warn = parse_prompt(prompt)
    warnings = list(parse_warn)
    steps.append("Parse prompt; extract inputs.")

    if inp["member_type"] == "beam":
        mode = inp["beam_mode"]
        results = {"member_type": "beam", "mode": mode, "inputs": inp}
        flex, sh = None, None

        if mode in ("flexure", "combined"):
            steps.append("Run beam flexure skill.")
            flex, w, s = flexure_design_beam_general(inp)
            warnings += w; steps += s
            if flex is not None:
                results["flexure"] = flex
                results["flexure_narrative"] = narrative_beam_flexure_html(inp, flex)

        if mode in ("shear", "combined"):
            steps.append("Run beam shear skill.")
            d_for_shear = flex["d_in"] if flex is not None else None
            sh, w, s = shear_design_beam(inp, d_for_shear)
            warnings += w; steps += s
            if sh is not None:
                results["shear"] = sh
                results["shear_narrative"] = narrative_beam_shear_html(inp, sh)

        if (mode in ("flexure", "combined")) and flex is None: return None, warnings, steps
        if (mode in ("shear", "combined")) and sh is None: return None, warnings, steps
        return results, warnings, steps

    if inp["member_type"] == "column":
        col, w, s = column_design_tied(inp)
        warnings += w; steps += s
        if col is None: return None, warnings, steps
        return {"member_type": "column", "inputs": inp, "column": col, "narrative": narrative_column_html(inp, col)}, warnings, steps

    sres, w, s = slab_design_one_way(inp)
    warnings += w; steps += s
    if sres is None: return None, warnings, steps
    return {"member_type": "slab", "inputs": inp, "slab": sres, "narrative": narrative_slab_html(inp, sres)}, warnings, steps


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="ACI 318-19 RC Design Agent", layout="wide")
st.title("ACI 318-19 RC Design Copilot — Demo")

# Upgraded CSS: Larger fonts, better spacing, and cleaner table styles
st.markdown(
    """
    <style>
      .panel {
        border: 1px solid rgba(49,51,63,0.15);
        border-radius: 12px;
        padding: 24px;
        min-height: 520px;
        background: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        overflow: auto;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 1.15rem;
        line-height: 1.6;
        color: #31333F;
      }
      .panel h4 { 
        margin: 0 0 16px 0; 
        padding-bottom: 8px;
        border-bottom: 2px solid #f0f2f6;
        font-size: 1.4rem; 
      }
      .output-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 1.1rem;
      }
      .output-table td {
        padding: 8px 4px;
        border-bottom: 1px solid #eaeaea;
      }
      .val-col {
        text-align: right;
        font-family: ui-monospace, monospace;
        color: #0056b3;
      }
    </style>
    """,
    unsafe_allow_html=True
)

EXAMPLES = {
    "Beam Flexure (Rect, positive)": "Beam flexure: Design a 12x24 rectangular beam, cover 1.5 in, fc=4 ksi, fy=60 ksi, Mu=180 kip-ft, positive moment, use #8 max bars.",
    "Beam Shear (standalone)": "Beam shear: Design shear for a 12x24 beam, cover 1.5 in, fc=4 ksi, fy=60 ksi, Vu=70 kips, main bars #8, stirrups #3.",
    "Beam Flexure + Shear": "Design a 12x24 beam, cover 1.5 in, fc=4 ksi, fy=60 ksi, Mu=180 kip-ft, Vu=70 kips, positive moment, use #8 max bars, stirrups #3.",
    "Column (tied)": "Design a 16x16 tied column, cover 1.5 in, fc=5 ksi, fy=60 ksi, Pu=350 kips, Mu=120 kip-ft, use #8 bars, ties #3.",
    "One-way slab": "One-way slab design: t=8 in, cover 0.75 in, fc=4 ksi, fy=60 ksi, L=15 ft, wu=120 psf, simply supported."
}

if "prompt_text" not in st.session_state: st.session_state.prompt_text = EXAMPLES["Beam Flexure (Rect, positive)"]
if "show_thinking" not in st.session_state: st.session_state.show_thinking = True
if "show_sketch" not in st.session_state: st.session_state.show_sketch = True

st.sidebar.title("RC Design Copilot")
choice = st.sidebar.selectbox("Example prompt", list(EXAMPLES.keys()))
rowA = st.sidebar.columns(2)
if rowA[0].button("Load Example", use_container_width=True): st.session_state.prompt_text = EXAMPLES[choice]
if rowA[1].button("Reset", use_container_width=True): st.session_state.prompt_text = EXAMPLES["Beam Flexure (Rect, positive)"]

prompt = st.sidebar.text_area("Prompt", value=st.session_state.prompt_text, height=230)
st.session_state.show_thinking = st.sidebar.checkbox("Show agent thinking panel", value=st.session_state.show_thinking)
st.session_state.show_sketch = st.sidebar.checkbox("Show Design Sketch", value=st.session_state.show_sketch)

rowB = st.sidebar.columns(2)
run = rowB[0].button("▶ Run Agent", use_container_width=True)
clear = rowB[1].button("🧹 Clear Output", use_container_width=True)

if clear:
    st.session_state.last_result = None
    st.session_state.last_warnings = []
    st.session_state.last_steps = []

if run:
    with st.spinner("Agent is thinking..."):
        result, warnings, steps = agent_run(prompt)
    st.session_state.last_result = result
    st.session_state.last_warnings = warnings
    st.session_state.last_steps = steps

result = st.session_state.last_result
warnings = st.session_state.last_warnings
steps = st.session_state.last_steps

if result is None:
    st.info("Run the agent using the sidebar. Results will appear here.")
else:
    st.success(f"Completed: {result['member_type'].upper()}")

    if st.session_state.show_thinking:
        with st.expander("🧠 Agent Thinking", expanded=False):
            for s in steps: st.write(f"- {s}")

    # Build the HTML Table for Outputs
    def build_html_table(out_dict):
        table_html = "<table class='output-table'><tbody>"
        for k, v in out_dict.items():
            table_html += f"<tr><td><strong>{k}</strong></td><td class='val-col'>{v}</td></tr>"
        table_html += "</tbody></table>"
        return table_html

    # Beam Logic
    if result["member_type"] == "beam":
        colL, colR = st.columns([1, 1], gap="large")

        narrative_parts = []
        if "flexure_narrative" in result: narrative_parts.append(result["flexure_narrative"])
        if "shear_narrative" in result: narrative_parts.append(result["shear_narrative"])
        narrative_text = "".join(narrative_parts)

        out = {}
        if "flexure" in result:
            f = result["flexure"]
            out.update({
                "Flexure shape": f["shape"],
                "Bars": f"{f['n']} {f['bar']}",
                "As (in²)": round(f["As_prov"], 2),
                "d (in)": round(f["d_in"], 2),
                "a (in)": round(f["a_in"], 2),
                "c (in)": round(f["c_in"], 2),
                "phi": round(f["phi"], 3),
                "phiMn (kip-ft)": round(f["phiMn_kipft"], 1),
            })
        if "shear" in result:
            sh = result["shear"]
            out.update({
                "Vu (kips)": result["inputs"]["Vu_kips"],
                "Vc (kips)": round(sh["Vc_kips"], 1),
                "phiVc (kips)": round(sh["phiVc_kips"], 1),
                "Stirrups": ("None req." if sh["s_use_in"] is None else f"{sh['legs']}-leg {sh['stirrup_size']} @ {sh['s_use_in']:.1f}\""),
            })

        with colL:
            st.markdown(f"<div class='panel'><h4>Beam Narrative</h4>{narrative_text}</div>", unsafe_allow_html=True)
        with colR:
            st.markdown(f"<div class='panel'><h4>Key Outputs</h4>{build_html_table(out)}</div>", unsafe_allow_html=True)

        if st.session_state.show_sketch and "flexure" in result:
            st.markdown("### Design Sketch")
            fig = draw_design_sketch_section(result["flexure"])
            st.pyplot(fig, use_container_width=False)

    # Column Logic
    elif result["member_type"] == "column":
        colL, colR = st.columns([1, 1], gap="large")
        c = result["column"]
        out = {
            "Bars": f"{c['n_bars']} {c['bar_size']}",
            "rho": round(c["rho"], 3),
            "phi": c["phi"],
            "phiMn at Pu/phi (kip-ft)": round(c["phiMn_at_Preq"], 1),
            "Tie size": c["tie_size"],
            "Max tie spacing (in)": round(c["s_tie_max_in"], 1),
            "OK?": c["ok"]
        }
        with colL:
            st.markdown(f"<div class='panel'><h4>Column Narrative</h4>{result['narrative']}</div>", unsafe_allow_html=True)
        with colR:
            st.markdown(f"<div class='panel'><h4>Key Outputs</h4>{build_html_table(out)}</div>", unsafe_allow_html=True)

    # Slab Logic
    elif result["member_type"] == "slab":
        colL, colR = st.columns([1, 1], gap="large")
        sres = result["slab"]
        out = {
            "Bars": f"{sres['bar']} @ {sres['s_in']}\"",
            "As (in²/ft)": round(sres["As_in2_per_ft"], 3),
            "Mu (kip-ft/ft)": round(sres["Mu_kipft"], 2),
            "phiMn (kip-ft)": round(sres["phiMn_kipft"], 2),
        }
        with colL:
            st.markdown(f"<div class='panel'><h4>Slab Narrative</h4>{result['narrative']}</div>", unsafe_allow_html=True)
        with colR:
            st.markdown(f"<div class='panel'><h4>Key Outputs</h4>{build_html_table(out)}</div>", unsafe_allow_html=True)

    if warnings:
        st.markdown("### Warnings / Assumptions")
        for w in warnings: st.warning(w)
