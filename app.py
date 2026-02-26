# ============================================================
# RC DESIGN COPILOT — (ACI 318-19 teaching version)
# Streamlit + Agent + Skills:
#   - Beam flexure (Rect or T-beam; positive or negative moment)
#   - Beam shear (ACI beam shear; can run standalone)
#   - Short tied column (P-M interaction; perimeter rebar layout)
#   - One-way slab design (12-in strip)
#
# UI (this version):
#   - Sidebar inputs + equal-size buttons
#   - Beam output shown in two equal-height panels:
#       (A) Beam flexural/shear narrative
#       (B) Key outputs
#   - Design Sketch (section only, larger):
#       - Rectangular or T-beam outline
#       - Dimensions (b, h, and for T-beam bf, bw, hf)
#       - Tension steel on correct face (+/- moment)
#       - Rebar label (n and bar size)
#       - Compression block depth a, neutral axis c, effective depth d (teaching visuals)
#
# Optional "Auto beff" for T-beams:
#   be (each side) = min(Ln/8, 8hf, sw/2)
#   bf = bw + 2be
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Teaching demo only. Not for real design/stamping.
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

# Column longitudinal steel ratio limits (teaching common ranges)
RHO_MIN_COL = 0.01
RHO_MAX_COL = 0.08


# ============================================================
# Parser helpers
# ============================================================

def _find_number(text: str, patterns, default=None):
    """
    Robust: returns the first capture group in the regex match that can be parsed as float.
    This prevents errors when group(1) is a word like 'fc' and group(2) is the number.
    """
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            # Try groups from last to first, pick the first numeric one
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

    # Detect member type
    member_type = "beam"
    if re.search(r"\bcolumn\b", t, flags=re.IGNORECASE):
        member_type = "column"
    if re.search(r"\bslab\b|\bone-way slab\b|\bone way slab\b", t, flags=re.IGNORECASE):
        member_type = "slab"

    # Materials
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

    # Cover definition used: clear cover to outside of stirrup/tie
    cover = _find_number(
        t,
        [r"\bcover\s*=?\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"],
        default=1.5 if member_type != "slab" else 0.75
    )

    # Common geometry for beam/column
    b = _find_number(t, [r"\bb\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"])
    h = _find_number(t, [r"\bh\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"])
    m_dim = re.search(r"\b([0-9]*\.?[0-9]+)\s*[xX]\s*([0-9]*\.?[0-9]+)\b", t)
    if m_dim:
        b = b if b is not None else float(m_dim.group(1))
        h = h if h is not None else float(m_dim.group(2))

    # T-beam inputs (bw, bf, hf)
    bw = _find_number(t, [r"\bbw\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"], default=None)
    bf = _find_number(t, [r"\bbf\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"], default=None)
    hf = _find_number(t, [r"\bhf\s*=\s*([0-9]*\.?[0-9]+)\s*(?:in|inch|inches)\b"], default=None)

    is_t_beam = bool(re.search(r"\bt[-\s]*beam\b|\btbeam\b", t, flags=re.IGNORECASE)) or (bw is not None or bf is not None or hf is not None)

    # Auto beff flag + inputs (Ln and sw)
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

    # Moment sign / tension face for beams
    moment_sign = "positive"  # positive: bottom tension; negative: top tension
    if re.search(r"\bnegative\b|\bneg\b|\btop tension\b", t, flags=re.IGNORECASE):
        moment_sign = "negative"
    if re.search(r"\bpositive\b|\bpos\b|\bbottom tension\b", t, flags=re.IGNORECASE):
        moment_sign = "positive"

    # Beam demands
    Mu = _find_number(t, [r"\bmu\s*=?\s*([0-9]*\.?[0-9]+)\s*(?:kip[-\s]*ft|kft)\b"])
    Vu = _find_number(t, [r"\bvu\s*=?\s*([0-9]*\.?[0-9]+)\s*kips?\b"])

    # Column demands
    Pu = _find_number(t, [r"\bpu\s*=?\s*([0-9]*\.?[0-9]+)\s*kips?\b"])
    Mu_col = Mu if member_type == "column" else _find_number(t, [r"\bmu_col\s*=?\s*([0-9]*\.?[0-9]+)\s*(?:kip[-\s]*ft|kft)\b"])

    # Slab inputs
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

    # Preferences
    max_bar = _find_bar(t, [r"(#\s*[0-9]+)\s*(?:max|maximum)"])
    long_bar = _find_bar(t, [r"\b(?:use|bars?|main bars?)\s*(#\s*[0-9]+)\b"], default=None)
    stirrup_size = _find_bar(t, [r"stirrups?\s*(#\s*[0-9]+)"], default=None)
    tie_size = _find_bar(t, [r"\bties?\s*(#\s*[0-9]+)"], default=None)

    # Identify beam mode
    beam_mode = "combined"
    if member_type == "beam":
        if Mu is not None and Vu is None:
            beam_mode = "flexure"
        elif Vu is not None and Mu is None:
            beam_mode = "shear"
        elif Mu is None and Vu is None:
            beam_mode = "flexure"

    # Missing checks
    if member_type in ("beam", "column"):
        if (b is None or h is None) and not is_t_beam:
            warnings.append("Section size not found. Use '12x24' or 'b=12 in h=24 in'.")
        if is_t_beam:
            if bw is None:
                warnings.append("T-beam web width bw not found (e.g., 'bw=12 in').")
            if h is None:
                warnings.append("Overall depth h not found for T-beam (e.g., 'h=28 in').")
            if hf is None:
                warnings.append("Flange thickness hf not found (e.g., 'hf=4 in').")
            if (not auto_beff) and (bf is None):
                warnings.append("Effective flange width bf not found (e.g., 'bf=48 in') or enable 'auto beff' with Ln and sw.")

        if is_t_beam and auto_beff and bf is None:
            if Ln is None:
                warnings.append("Auto beff requested but Ln not found (e.g., 'Ln=24 ft').")
            if sw is None:
                warnings.append("Auto beff requested but sw not found (e.g., 'sw=72 in').")

    if member_type == "beam":
        if Mu is None and beam_mode in ("combined", "flexure"):
            warnings.append("Beam Mu not found. Example: 'Mu=180 kip-ft'.")
        if Vu is None and beam_mode in ("combined", "shear"):
            warnings.append("Beam Vu not found. Example: 'Vu=45 kips'.")
        if beam_mode == "shear" and long_bar is None:
            warnings.append("For shear-only, include a main bar size for d estimate (e.g., 'main bars #8').")

    if member_type == "column":
        if Pu is None:
            warnings.append("Column Pu not found. Example: 'Pu=350 kips'.")
        if Mu_col is None:
            warnings.append("Column Mu not found. Example: 'Mu=120 kip-ft'.")

    if member_type == "slab":
        if slab_t is None:
            warnings.append("Slab thickness not found. Example: 't=8 in'.")
        if L_ft is None:
            warnings.append("Slab span not found. Example: 'L=15 ft'.")
        if wu_psf is None:
            warnings.append("Slab wu not found. Example: 'wu=120 psf'.")

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
    eps_tc = eps_y(fy_psi) + 0.003
    if eps_t >= eps_tc:
        return 0.90
    if eps_t <= 0.002:
        return 0.65
    return 0.65 + (eps_t - 0.002) * (0.90 - 0.65) / (eps_tc - 0.002)

def clear_cover_definition_note() -> str:
    return (
        "Cover in this app = clear cover to the OUTSIDE surface of stirrups/ties "
        "(measured from concrete surface to outside of transverse reinforcement)."
    )

def effective_depth(h_in: float, cover_in: float, main_bar_dia_in: float, stirrup_dia_in: float = 0.375) -> float:
    return h_in - cover_in - stirrup_dia_in - 0.5 * main_bar_dia_in

def As_min_beam_us(fc_psi: float, fy_psi: float, b_in: float, d_in: float) -> float:
    term1 = 3.0 * math.sqrt(fc_psi) / fy_psi * b_in * d_in
    term2 = 200.0 / fy_psi * b_in * d_in
    return max(term1, term2)

def shear_Av_over_s_min_us(fc_psi: float, fy_psi: float, bw_in: float) -> float:
    return max(0.75 * math.sqrt(fc_psi) * bw_in / fy_psi, 50.0 * bw_in / fy_psi)

def shear_Vc_simple_us(fc_psi: float, bw_in: float, d_in: float, lam: float = 1.0) -> float:
    return (2.0 * lam * math.sqrt(fc_psi) * bw_in * d_in) / 1000.0  # kips

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

    fc = inp["fc_psi"]
    fy = inp["fy_psi"]
    Mu = inp["Mu_kipft"]
    cover = inp["cover_in"]
    h = inp["h_in"]
    moment_sign = inp["moment_sign"]

    is_t = inp["is_t_beam"]
    bw = inp["bw_in"]
    bf = inp["bf_in"]
    hf = inp["hf_in"]

    if Mu is None or h is None:
        return None, ["Missing Mu or h."], steps

    # Auto beff
    auto_bf_used = False
    bf_auto = None
    if is_t and inp.get("auto_beff", False):
        Ln_in = inp.get("Ln_in")
        sw_in = inp.get("sw_in")
        if (Ln_in is not None) and (sw_in is not None) and (hf is not None) and (bw is not None):
            bf_auto = compute_beff_aci_teaching(bw, hf, Ln_in, sw_in)
            if bf is None:
                bf = bf_auto
                auto_bf_used = True
        else:
            warnings.append("Auto beff is ON but could not compute bf (need Ln, sw, bw, hf).")

    # Candidate bars
    bar_keys = [k for k in BAR_DB.keys() if k != "#3"]
    if inp["max_bar"] and inp["max_bar"] in bar_keys:
        bar_keys = bar_keys[: bar_keys.index(inp["max_bar"]) + 1]

    beta1 = beta1_aci(fc)
    best = None

    for bar in bar_keys:
        Ab = BAR_DB[bar]["area"]
        db = BAR_DB[bar]["dia"]
        d = effective_depth(h, cover, db, stirrup_dia_in=BAR_DB["#3"]["dia"])

        for n in range(2, 16):
            As = n * Ab

            if not is_t:
                b_rect = inp["b_in"] if inp["b_in"] is not None else bw
                a = As * fy / (0.85 * fc * b_rect)
                ybar = a / 2.0
            else:
                if moment_sign == "negative":
                    # teaching simplification: compression in web for negative moment
                    b_rect = bw
                    a = As * fy / (0.85 * fc * b_rect)
                    ybar = a / 2.0
                else:
                    if (bw is None) or (bf is None) or (hf is None):
                        continue
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
        warnings.append("Flexure design could not find a bar arrangement satisfying ϕMn ≥ Mu under current assumptions.")
        return None, warnings, steps

    steps.append("Beam flexure: searched bar sizes/counts to satisfy ϕMn ≥ Mu (Rect/T; +/- moment; Auto beff optional).")
    return best[0], warnings, steps


# ============================================================
# Beam shear skill (ACI beam shear; teaching)
# ============================================================

def shear_design_beam(inp, d_in: float | None):
    steps, warnings = [], []

    Vu = inp["Vu_kips"]
    fc = inp["fc_psi"]
    fy = inp["fy_psi"]
    bw = inp["bw_in"] if inp["bw_in"] is not None else inp["b_in"]

    if Vu is None or bw is None:
        return None, ["Missing Vu or bw/b."], steps

    phi_v = 0.75

    if d_in is None:
        main_bar = inp["long_bar"] if (inp["long_bar"] in BAR_DB) else "#8"
        db = BAR_DB[main_bar]["dia"]
        if inp["h_in"] is None:
            return None, ["Missing h for shear-only d estimate."], steps
        d_in = effective_depth(inp["h_in"], inp["cover_in"], db, stirrup_dia_in=BAR_DB["#3"]["dia"])
        warnings.append(f"Shear-only: estimated d using main bars {main_bar}.")

    stirrup_size = inp["stirrup_size"] if (inp["stirrup_size"] in STIRRUP_LEG_AREA) else DEFAULT_STIRRUP_SIZE
    Av = DEFAULT_STIRRUP_LEGS * STIRRUP_LEG_AREA[stirrup_size]

    Vc = shear_Vc_simple_us(fc, bw, d_in)
    phiVc = phi_v * Vc

    s_cap = min(d_in / 2.0, 24.0)

    Av_over_s_min = shear_Av_over_s_min_us(fc, fy, bw)
    Vu_trigger = phi_v * math.sqrt(fc) * bw * d_in / 1000.0  # teaching trigger
    needs_min = Vu > Vu_trigger

    Vn_req = Vu / phi_v
    Vs_req = max(0.0, Vn_req - Vc)  # kips

    if Vs_req <= 1e-9:
        if needs_min:
            s_from_min = Av / Av_over_s_min
            s_use = min(s_from_min, s_cap)
            steps.append("Beam shear: Vu ≤ ϕVc, but minimum stirrups triggered (teaching).")
            return {
                "phi_v": phi_v, "d_in": d_in, "bw_in": bw,
                "Vc_kips": Vc, "phiVc_kips": phiVc,
                "stirrup_size": stirrup_size, "legs": DEFAULT_STIRRUP_LEGS, "Av_in2": Av,
                "needs_min": True, "Av_over_s_min": Av_over_s_min,
                "s_use_in": s_use, "s_cap_in": s_cap,
                "Vs_req_kips": 0.0
            }, warnings, steps

        steps.append("Beam shear: Vu ≤ ϕVc (baseline).")
        return {
            "phi_v": phi_v, "d_in": d_in, "bw_in": bw,
            "Vc_kips": Vc, "phiVc_kips": phiVc,
            "stirrup_size": stirrup_size, "legs": DEFAULT_STIRRUP_LEGS, "Av_in2": Av,
            "needs_min": False, "Av_over_s_min": Av_over_s_min,
            "s_use_in": None, "s_cap_in": s_cap,
            "Vs_req_kips": 0.0
        }, warnings, steps

    Vs_req_lb = Vs_req * 1000.0
    s_strength = (Av * fy * d_in) / Vs_req_lb
    s_from_min = Av / Av_over_s_min
    s_req = min(s_strength, s_from_min) if needs_min else s_strength
    s_use = min(s_req, s_cap)

    if needs_min and (Av / s_use) + 1e-12 < Av_over_s_min:
        warnings.append("Cannot satisfy Av/s(min) with current stirrup size/legs and spacing cap. Increase stirrup size or legs.")

    steps.append("Beam shear: computed Vc and Vs; sized stirrups; enforced s cap and Av/s(min) (teaching).")
    return {
        "phi_v": phi_v, "d_in": d_in, "bw_in": bw,
        "Vc_kips": Vc, "phiVc_kips": phiVc,
        "stirrup_size": stirrup_size, "legs": DEFAULT_STIRRUP_LEGS, "Av_in2": Av,
        "needs_min": needs_min, "Av_over_s_min": Av_over_s_min,
        "s_strength_in": s_strength, "s_from_min_in": s_from_min, "s_req_in": s_req,
        "s_use_in": s_use, "s_cap_in": s_cap,
        "Vs_req_kips": Vs_req
    }, warnings, steps


# ============================================================
# Column skill (teaching)
# ============================================================

def _steel_stress_from_strain(eps: float, fy_psi: float) -> float:
    fs = Es * eps
    return max(-fy_psi, min(fy_psi, fs))

def column_bar_layout_perimeter(b_in: float, h_in: float, cover_in: float, tie_size: str, bar_size: str, n_bars: int):
    if tie_size not in BAR_DB:
        tie_size = DEFAULT_TIE_SIZE
    if bar_size not in BAR_DB:
        raise ValueError("Invalid bar size.")
    tie_dia = BAR_DB[tie_size]["dia"]
    bar_dia = BAR_DB[bar_size]["dia"]

    off = cover_in + tie_dia + 0.5 * bar_dia
    xL, xR = off, b_in - off
    yT, yB = off, h_in - off

    pts = []

    def add_unique(p):
        if p not in pts:
            pts.append(p)

    add_unique((xL, yT))
    add_unique((xR, yT))
    add_unique((xR, yB))
    add_unique((xL, yB))

    if n_bars == 4:
        return pts

    def interior_points(a, b, k):
        if k <= 0:
            return []
        return [a + (i + 1) * (b - a) / (k + 1) for i in range(k)]

    if n_bars == 8:
        k = 1
    elif n_bars == 12:
        k = 2
    elif n_bars == 16:
        k = 3
    else:
        raise ValueError("n_bars supported: 4, 8, 12, 16 (teaching).")

    xs = interior_points(xL, xR, k)
    for x in xs:
        add_unique((x, yT))
        add_unique((x, yB))

    ys = interior_points(yT, yB, k)
    for y in ys:
        add_unique((xL, y))
        add_unique((xR, y))

    return pts[:n_bars]

def column_interaction_curve_rect_tied_perimeter(b_in: float, h_in: float, cover_in: float, fc_psi: float, fy_psi: float,
                                                 bar_size: str, n_bars: int, tie_size: str, n_points: int = 60):
    Ab = BAR_DB[bar_size]["area"]
    beta1 = beta1_aci(fc_psi)
    bars = column_bar_layout_perimeter(b_in, h_in, cover_in, tie_size, bar_size, n_bars)

    c_min, c_max = 0.5, 1.5 * h_in
    c_values = [c_min + i * (c_max - c_min) / (n_points - 1) for i in range(n_points)]

    y_cg = h_in / 2.0
    curve = []
    for c in c_values:
        a = min(beta1 * c, h_in)
        Cc = 0.85 * fc_psi * b_in * a
        y_cc = a / 2.0

        Psteel = 0.0
        Msteel = 0.0

        for (_x, y) in bars:
            eps = 0.003 * (c - y) / c
            fs = _steel_stress_from_strain(eps, fy_psi)
            Fs = fs * Ab
            Psteel += Fs
            Msteel += Fs * (y - y_cg)

        Pn = (Cc + Psteel) / 1000.0
        Mn = abs(Cc * (y_cc - y_cg) + Msteel) / (1000.0 * 12.0)
        curve.append((Pn, Mn))

    curve.sort(key=lambda x: -x[0])
    return curve

def tie_spacing_limit_teaching(db_long: float, db_tie: float, least_dim: float) -> float:
    return min(16.0 * db_long, 48.0 * db_tie, least_dim, 12.0)

def _interp_M_at_P(P_req, Ps, Ms):
    if P_req > max(Ps) or P_req < min(Ps):
        return None
    for i in range(len(Ps) - 1):
        P1, P2 = Ps[i], Ps[i + 1]
        if (P1 >= P_req >= P2) or (P2 >= P_req >= P1):
            M1, M2 = Ms[i], Ms[i + 1]
            if abs(P2 - P1) < 1e-12:
                return max(M1, M2)
            t = (P_req - P1) / (P2 - P1)
            return M1 + t * (M2 - M1)
    return None

def column_design_tied(inp):
    steps, warnings = [], []

    b, h = inp["b_in"], inp["h_in"]
    Pu, Mu = inp["Pu_kips"], inp["Mu_col_kipft"]
    if None in (b, h, Pu, Mu):
        return None, ["Missing required column inputs (b,h,Pu,Mu)."], steps

    fc, fy = inp["fc_psi"], inp["fy_psi"]
    cover = inp["cover_in"]
    Ag = b * h

    tie_size = inp["tie_size"] if inp["tie_size"] in BAR_DB else DEFAULT_TIE_SIZE
    db_tie = BAR_DB[tie_size]["dia"]

    preferred = inp["long_bar"] if (inp["long_bar"] in BAR_DB) else None
    bar_trials = [preferred] if preferred else []
    for x in ["#11", "#10", "#9", "#8", "#7", "#6", "#5"]:
        if x not in bar_trials:
            bar_trials.append(x)
    n_trials = [16, 12, 8, 4]

    best = None
    phi = 0.65  # teaching conservative
    P_req = Pu / phi

    for bar in bar_trials:
        Ab = BAR_DB[bar]["area"]
        db_long = BAR_DB[bar]["dia"]
        for n_bars in n_trials:
            Ast = n_bars * Ab
            rho = Ast / Ag
            if rho < RHO_MIN_COL or rho > RHO_MAX_COL:
                continue

            curve = column_interaction_curve_rect_tied_perimeter(
                b, h, cover, fc, fy, bar_size=bar, n_bars=n_bars, tie_size=tie_size
            )
            Ps = [p for p, _m in curve]
            Ms = [m for _p, m in curve]

            Mn_req = _interp_M_at_P(P_req, Ps, Ms)
            if Mn_req is None:
                continue

            ok = Mu <= phi * Mn_req
            if ok:
                smax = tie_spacing_limit_teaching(db_long, db_tie, min(b, h))
                best = {
                    "bar_size": bar,
                    "n_bars": n_bars,
                    "Ast_in2": Ast,
                    "rho": rho,
                    "phi": phi,
                    "P_req": P_req,
                    "Mn_at_Preq": Mn_req,
                    "phiMn_at_Preq": phi * Mn_req,
                    "ok": ok,
                    "tie_size": tie_size,
                    "s_tie_max_in": smax,
                }
                break
        if best is not None:
            break

    if best is None:
        warnings.append("No column trial passed Pu–Mu while meeting 1%–8% steel ratio (teaching).")
        return None, warnings, steps

    steps.append("Column: selected perimeter rebar + checked Pu–Mu using an interaction-style curve (teaching).")
    steps.append("Column: reported tie spacing limit (teaching).")
    return best, warnings, steps


# ============================================================
# One-way slab skill (12-in strip)
# ============================================================

def slab_design_one_way(inp):
    steps, warnings = [], []

    t_in = inp["slab_t_in"]
    L_ft = inp["slab_L_ft"]
    wu_psf = inp["slab_wu_psf"]
    fc, fy = inp["fc_psi"], inp["fy_psi"]
    cover = inp["cover_in"]

    if None in (t_in, L_ft, wu_psf):
        return None, ["Missing slab inputs (t, L, wu)."], steps

    b = 12.0
    h = t_in

    # ACI Shrinkage and Temperature minimum steel (Grade 60 assumed for teaching)
    As_min_req = 0.0018 * b * h

    w_plf = wu_psf * 1.0
    if inp["slab_support"] == "continuous":
        Mu_kipft = (w_plf * (L_ft ** 2) / 12.0) / 1000.0
        support_note = "continuous interior (teaching): wL^2/12"
    else:
        Mu_kipft = (w_plf * (L_ft ** 2) / 8.0) / 1000.0
        support_note = "simply supported: wL^2/8"

    phi = 0.90
    candidates = []

    # ACI Max spacing limit: lesser of 3h or 18 in
    s_max_limit = min(3.0 * h, 18.0)

    for bar in ["#4", "#5"]:
        Ab = BAR_DB[bar]["area"]
        db = BAR_DB[bar]["dia"]
        d = h - cover - 0.5 * db  # slab: no stirrups
        if d <= 0:
            continue

        for s in range(4, int(math.floor(s_max_limit)) + 1):
            As_per_ft = Ab * (12.0 / s)
            As = As_per_ft

            # Skip if it doesn't meet minimum T&S steel
            if As < As_min_req:
                continue

            a = As * fy / (0.85 * fc * b)
            ybar = a / 2.0
            Mn = (As * fy) * (d - ybar)  # lb-in
            phiMn = phi * (Mn / (1000.0 * 12.0))

            if phiMn >= Mu_kipft:
                candidates.append({
                    "bar": bar,
                    "s_in": s,
                    "As_in2_per_ft": As_per_ft,
                    "d_in": d,
                    "a_in": a,
                    "phi": phi,
                    "phiMn_kipft": phiMn,
                    "Mu_kipft": Mu_kipft,
                    "support_note": support_note
                })
                break

    if not candidates:
        warnings.append("Slab: could not find #4/#5 spacing to satisfy moment AND minimum steel under assumptions.")
        return None, warnings, steps

    best = sorted(candidates, key=lambda x: (-x["s_in"], x["bar"]))[0]
    steps.append("Slab: 1-ft strip + simple bar-spacing search to satisfy ϕMn ≥ Mu (teaching).")
    return best, warnings, steps


# ============================================================
# Narratives
# ============================================================

def narrative_beam_flexure(inp, f):
    lines = []
    lines.append("=== BEAM FLEXURE (Teaching) ===")
    lines.append(clear_cover_definition_note())
    lines.append(f"Shape: {f['shape']}, moment: {f['moment_sign']}")
    if f["shape"] == "T-beam":
        lines.append(f"T-beam: bw={f['bw_in']} in, bf={f['bf_in']} in, hf={f['hf_in']} in")
        if f.get("auto_beff", False):
            if f.get("bf_auto_used", False):
                lines.append(f"Auto beff ON: bf computed ≈ {f.get('bf_auto_in'):.1f} in from Ln and sw.")
            elif f.get("bf_auto_in") is not None:
                lines.append(f"Auto beff ON: bf_auto ≈ {f.get('bf_auto_in'):.1f} in (user bf used).")
            else:
                lines.append("Auto beff ON: could not compute bf (need Ln and sw).")
    lines.append(f"f'c={inp['fc_psi']/1000:.2f} ksi, fy={inp['fy_psi']/1000:.0f} ksi")
    lines.append(f"Mu={inp['Mu_kipft']:.2f} kip-ft")
    lines.append(f"Selected: {f['n']} {f['bar']} bars -> As={f['As_prov']:.2f} in^2 (As,min={f['Asmin']:.2f} in^2)")
    lines.append(f"d≈{f['d_in']:.2f} in, a={f['a_in']:.2f} in, c≈{f['c_in']:.2f} in")
    lines.append(f"εt≈{f['eps_t']:.5f} -> ϕ≈{f['phi']:.3f}")
    lines.append(f"Mn≈{f['Mn_kipft']:.1f} kip-ft; ϕMn≈{f['phiMn_kipft']:.1f} kip-ft")
    lines.append("DISCLAIMER: Teaching demo only.")
    return "\n".join(lines)

def narrative_beam_shear(inp, sh):
    lines = []
    lines.append("=== BEAM SHEAR (ACI beam shear; teaching) ===")
    lines.append(clear_cover_definition_note())
    lines.append(f"f'c={inp['fc_psi']/1000:.2f} ksi, fy={inp['fy_psi']/1000:.0f} ksi")
    lines.append(f"Vu={inp['Vu_kips']:.2f} kips, bw={sh['bw_in']:.1f} in, d≈{sh['d_in']:.2f} in")
    lines.append(f"Vc≈{sh['Vc_kips']:.1f} kips; ϕVc≈{sh['phiVc_kips']:.1f} kips (ϕ={sh['phi_v']:.2f})")
    if sh["s_use_in"] is None:
        lines.append("Stirrups not required by strength in baseline; minimum reinforcement may still govern depending on region.")
    else:
        lines.append(f"Provide {sh['legs']}-leg {sh['stirrup_size']} @ s={sh['s_use_in']:.1f} in (cap={sh['s_cap_in']:.1f} in)")
    lines.append("DISCLAIMER: Teaching demo only.")
    return "\n".join(lines)

def narrative_column(inp, col):
    lines = []
    lines.append("=== SHORT TIED COLUMN (Teaching) ===")
    lines.append(clear_cover_definition_note())
    lines.append(f"Section: {inp['b_in']:.1f} x {inp['h_in']:.1f} in")
    lines.append(f"f'c={inp['fc_psi']/1000:.2f} ksi, fy={inp['fy_psi']/1000:.0f} ksi")
    lines.append(f"Pu={inp['Pu_kips']:.1f} kips, Mu={inp['Mu_col_kipft']:.1f} kip-ft")
    lines.append(f"Selected: {col['n_bars']} {col['bar_size']} bars; Ast={col['Ast_in2']:.2f} in^2; ρ={col['rho']:.3f}")
    lines.append(f"ϕ={col['phi']:.2f}; P_req=Pu/ϕ={col['P_req']:.1f} kips")
    lines.append(f"Mn(P_req)≈{col['Mn_at_Preq']:.1f} kip-ft; ϕMn≈{col['phiMn_at_Preq']:.1f} kip-ft -> OK={col['ok']}")
    lines.append(f"Ties: {col['tie_size']}; s_max≈{col['s_tie_max_in']:.1f} in (teaching limit)")
    lines.append("DISCLAIMER: Teaching demo only.")
    return "\n".join(lines)

def narrative_slab(inp, sres):
    lines = []
    lines.append("=== ONE-WAY SLAB (Teaching) ===")
    lines.append(f"Support model: {sres['support_note']}")
    lines.append(f"t={inp['slab_t_in']:.1f} in, cover={inp['cover_in']:.2f} in (typical slab assumption)")
    lines.append(f"f'c={inp['fc_psi']/1000:.2f} ksi, fy={inp['fy_psi']/1000:.0f} ksi")
    lines.append(f"L={inp['slab_L_ft']:.2f} ft, wu={inp['slab_wu_psf']:.1f} psf -> Mu≈{sres['Mu_kipft']:.2f} kip-ft (per 1-ft strip)")
    lines.append(f"Provide {sres['bar']} @ {sres['s_in']} in -> As≈{sres['As_in2_per_ft']:.3f} in^2/ft")
    lines.append(f"d≈{sres['d_in']:.2f} in, ϕ={sres['phi']:.2f}, ϕMn≈{sres['phiMn_kipft']:.2f} kip-ft")
    lines.append("DISCLAIMER: Teaching demo only.")
    return "\n".join(lines)


# ============================================================
# Design Sketch (section only, larger; dimensions + rebar label)
# ============================================================

def _dim_arrow(ax, x1, y1, x2, y2, label, text_offset=(0, 0), fontsize=10):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="<->", linewidth=1.5),
    )
    xm = 0.5 * (x1 + x2) + text_offset[0]
    ym = 0.5 * (y1 + y2) + text_offset[1]
    ax.text(xm, ym, label, fontsize=fontsize, va="center", ha="center")

def draw_design_sketch_section(flex: dict):
    """
    Larger, teaching-style cross-section sketch with:
      - Rectangular or T-beam outline
      - Dimensions: b, h; (T-beam: bf, bw, hf)
      - Rebar shown on tension face (+/- moment) with label "n #bar"
      - a, c, d (teaching labels)
    """
    shape = flex.get("shape", "Rectangular")
    h = float(flex.get("h_in", 24.0))
    cover = float(flex.get("cover_in", 1.5))
    stirrup_dia = float(flex.get("stirrup_dia_in", 0.375))

    moment_sign = flex.get("moment_sign", "positive")
    tension_face = "bottom" if moment_sign == "positive" else "top"
    compression_face = "top" if tension_face == "bottom" else "bottom"

    bar = flex.get("bar", "#8")
    db = BAR_DB.get(bar, BAR_DB["#8"])["dia"]
    n_bars = int(flex.get("n", 4))

    a = float(flex.get("a_in", 0.0))
    c = float(flex.get("c_in", 0.0))
    d = float(flex.get("d_in", h - cover - stirrup_dia - 0.5 * db))

    if shape == "T-beam":
        bw = float(flex.get("bw_in", 12.0))
        bf = float(flex.get("bf_in", 36.0))
        hf = float(flex.get("hf_in", 4.0))
        b_web = bw
        b_total = bf
    else:
        b_web = float(flex.get("b_in", 12.0))
        b_total = b_web
        hf = 0.0
        bw = b_web
        bf = b_total

    # Half-sized figure (single sketch)
    fig, ax = plt.subplots(figsize=(5.75, 3.25))

    # Coordinate convention: y from bottom (0) to top (h)
    # Draw section
    if shape == "T-beam":
        # Draw flange and web with web centered under flange
        web_left = (b_total - b_web) / 2.0
        flange = patches.Rectangle((0, h - hf), b_total, hf, fill=False, linewidth=2)
        web = patches.Rectangle((web_left, 0), b_web, h - hf, fill=False, linewidth=2)
        ax.add_patch(flange)
        ax.add_patch(web)
        sec_left, sec_right = 0.0, b_total
        web_right = web_left + b_web
    else:
        rect = patches.Rectangle((0, 0), b_total, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        sec_left, sec_right = 0.0, b_total
        web_left, web_right = sec_left, sec_right

    # Compression block shading (teaching visual)
    if 0 < a < h:
        if compression_face == "top":
            y0 = h - a
            height = a
        else:
            y0 = 0.0
            height = a

        # For a visual: use bf in positive T-beam; otherwise use web width
        if (shape == "T-beam") and (moment_sign == "positive"):
            comp_left, comp_w = 0.0, b_total
        else:
            comp_left, comp_w = web_left, (web_right - web_left)

        comp = patches.Rectangle((comp_left, y0), comp_w, height, fill=True, alpha=0.12, linewidth=0)
        ax.add_patch(comp)

        # a label
        ax.text(sec_right + 0.8, y0 + 0.5 * height, f"a={a:.2f} in", va="center")

    # Neutral axis c (dashed)
    if 0 < c < h:
        y_na = (h - c) if compression_face == "top" else c
        ax.plot([sec_left, sec_right], [y_na, y_na], linestyle="--", linewidth=1.5)
        ax.text(sec_right + 0.8, y_na, f"c={c:.2f} in", va="center")

    # Tension steel layer position
    if tension_face == "bottom":
        y_steel = cover + stirrup_dia + 0.5 * db
        y_d_start, y_d_end = h, y_steel  # d measured from top to steel
    else:
        y_steel = h - (cover + stirrup_dia + 0.5 * db)
        y_d_start, y_d_end = 0.0, y_steel  # d measured from bottom to steel (teaching flip)

    # Place bars across web width
    margin = 0.9
    xL = web_left + margin
    xR = web_right - margin
    if n_bars <= 1:
        xs = [(xL + xR) / 2.0]
    else:
        xs = [xL + i * (xR - xL) / (n_bars - 1) for i in range(n_bars)]

    for x in xs:
        circ = patches.Circle((x, y_steel), radius=max(0.18, 0.18 * db), fill=True)
        ax.add_patch(circ)

    # Rebar label
    ax.text(sec_left, y_steel - (1.4 if tension_face == "bottom" else -1.0),
            f"{n_bars} {bar} (tension)", ha="left", va="center")

    # d dimension arrow
    ax.annotate("", xy=(sec_right + 0.2, y_d_start), xytext=(sec_right + 0.2, y_d_end),
                arrowprops=dict(arrowstyle="<->", linewidth=1.5))
    ax.text(sec_right + 0.6, 0.5 * (y_d_start + y_d_end), f"d≈{d:.2f} in", va="center")

    # Section dimensions: b and h
    # b along bottom
    _dim_arrow(ax, sec_left, -0.8, sec_right, -0.8, f"b={b_total:.1f} in", text_offset=(0, -0.2))
    # h along left
    _dim_arrow(ax, -0.8, 0.0, -0.8, h, f"h={h:.1f} in", text_offset=(-0.2, 0))

    # T-beam extra dimensions
    if shape == "T-beam":
        # bf along flange top
        _dim_arrow(ax, 0.0, h + 0.8, b_total, h + 0.8, f"bf={b_total:.1f} in", text_offset=(0, 0.2))
        # bw along web mid
        _dim_arrow(ax, web_left, (h - hf) / 2.0, web_right, (h - hf) / 2.0, f"bw={b_web:.1f} in")
        # hf along right side near flange
        _dim_arrow(ax, sec_right + 1.6, h - hf, sec_right + 1.6, h, f"hf={hf:.1f} in")

    # Formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(sec_left - 2.5, sec_right + 8.5)
    ax.set_ylim(-2.2, h + 2.2)
    ax.axis("off")
    ax.set_title("Design Sketch (Teaching)", fontsize=14)
    fig.tight_layout()
    return fig


# ============================================================
# Agent router
# ============================================================

def agent_run(prompt: str):
    steps = []
    inp, parse_warn = parse_prompt(prompt)
    warnings = list(parse_warn)
    steps.append("Parse prompt; detect member type and extract inputs.")

    if inp["member_type"] == "beam":
        mode = inp["beam_mode"]
        results = {"member_type": "beam", "mode": mode, "inputs": inp}

        flex = None
        sh = None

        if mode in ("flexure", "combined"):
            steps.append("Run beam flexure skill (Rect/T; +/- moment; Auto beff optional).")
            flex, w, s = flexure_design_beam_general(inp)
            warnings += w
            steps += s
            if flex is not None:
                results["flexure"] = flex
                results["flexure_narrative"] = narrative_beam_flexure(inp, flex)

        if mode in ("shear", "combined"):
            steps.append("Run beam shear skill (ACI beam shear; teaching).")
            d_for_shear = flex["d_in"] if flex is not None else None
            sh, w, s = shear_design_beam(inp, d_for_shear)
            warnings += w
            steps += s
            if sh is not None:
                results["shear"] = sh
                results["shear_narrative"] = narrative_beam_shear(inp, sh)

        if (mode in ("flexure", "combined")) and flex is None:
            return None, warnings, steps
        if (mode in ("shear", "combined")) and sh is None:
            return None, warnings, steps

        steps.append("Done — beam workflow completed.")
        return results, warnings, steps

    if inp["member_type"] == "column":
        steps.append("Run column skill (teaching interaction + detailing limits).")
        col, w, s = column_design_tied(inp)
        warnings += w
        steps += s
        if col is None:
            return None, warnings, steps
        steps.append("Done — column workflow completed.")
        return {"member_type": "column", "inputs": inp, "column": col, "narrative": narrative_column(inp, col)}, warnings, steps

    # slab
    steps.append("Run one-way slab skill (1-ft strip; teaching).")
    sres, w, s = slab_design_one_way(inp)
    warnings += w
    steps += s
    if sres is None:
        return None, warnings, steps
    steps.append("Done — slab workflow completed.")
    return {"member_type": "slab", "inputs": inp, "slab": sres, "narrative": narrative_slab(inp, sres)}, warnings, steps


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="ACI 318-19 RC Design Agent", layout="wide")
st.title("ACI 318-19 RC Design Copilot — Demo (Agent + Skills)")
st.caption("Teaching demo only — simplified/incomplete checks. Not for real design/stamping.")

# CSS for equal-height panels
st.markdown(
    """
    <style>
      .panel {
        border: 1px solid rgba(49,51,63,0.2);
        border-radius: 12px;
        padding: 14px 14px;
        min-height: 520px;
        background: rgba(250,250,252,0.25);
        overflow: auto;
      }
      .panel h4 { margin: 0 0 8px 0; }
      .panel pre {
        white-space: pre-wrap;
        word-wrap: break-word;
        margin: 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.9rem;
        line-height: 1.25rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

EXAMPLES = {
    "Beam Flexure (Rect, positive)":
        "Beam flexure: Design a 12x24 rectangular beam, cover 1.5 in, fc=4 ksi, fy=60 ksi, "
        "Mu=180 kip-ft, positive moment, use #8 max bars.",
    "Beam Flexure (T-beam, positive, manual bf)":
        "Beam flexure: Design a T-beam bw=12 in h=28 in bf=48 in hf=4 in, cover 1.5 in, fc=4 ksi, fy=60 ksi, "
        "Mu=220 kip-ft, positive moment, use #8 max bars.",
    "Beam Flexure (T-beam, Auto beff)":
        "Beam flexure: Design a T-beam bw=12 in h=28 in hf=4 in, auto beff, Ln=24 ft, sw=72 in, "
        "cover 1.5 in, fc=4 ksi, fy=60 ksi, Mu=220 kip-ft, positive moment, use #8 max bars.",
    "Beam Flexure (T-beam, negative)":
        "Beam flexure: Design a T-beam bw=12 in h=28 in bf=48 in hf=4 in, cover 1.5 in, fc=4 ksi, fy=60 ksi, "
        "Mu=180 kip-ft, negative moment, use #8 max bars.",
    "Beam Shear (standalone)":
        "Beam shear: Design shear for a 12x24 beam, cover 1.5 in, fc=4 ksi, fy=60 ksi, "
        "Vu=70 kips, main bars #8, stirrups #3.",
    "Beam Flexure + Shear":
        "Design a 12x24 beam, cover 1.5 in, fc=4 ksi, fy=60 ksi, "
        "Mu=180 kip-ft, Vu=70 kips, positive moment, use #8 max bars, stirrups #3.",
    "Column (tied)":
        "Design a 16x16 tied column, cover 1.5 in, fc=5 ksi, fy=60 ksi, "
        "Pu=350 kips, Mu=120 kip-ft, use #8 bars, ties #3.",
    "One-way slab":
        "One-way slab design: t=8 in, cover 0.75 in, fc=4 ksi, fy=60 ksi, "
        "L=15 ft, wu=120 psf, simply supported.",
}

# Session state init
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = EXAMPLES["Beam Flexure (Rect, positive)"]
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True
if "show_sketch" not in st.session_state:
    st.session_state.show_sketch = True

# Sidebar
st.sidebar.title("RC Design Copilot")
st.sidebar.caption("User Inputs")

choice = st.sidebar.selectbox("Example prompt", list(EXAMPLES.keys()))
rowA = st.sidebar.columns(2)
load_example = rowA[0].button("Load Example", use_container_width=True)
reset_prompt = rowA[1].button("Reset", use_container_width=True)

if load_example:
    st.session_state.prompt_text = EXAMPLES[choice]
if reset_prompt:
    st.session_state.prompt_text = EXAMPLES["Beam Flexure (Rect, positive)"]

st.sidebar.markdown("---")
prompt = st.sidebar.text_area("Prompt", value=st.session_state.prompt_text, height=230)

st.session_state.show_thinking = st.sidebar.checkbox("Show agent thinking panel", value=st.session_state.show_thinking)
st.session_state.show_sketch = st.sidebar.checkbox("Show Design Sketch", value=st.session_state.show_sketch)

st.sidebar.markdown("---")
rowB = st.sidebar.columns(2)
run = rowB[0].button("▶ Run Agent", use_container_width=True)
clear = rowB[1].button("🧹 Clear Output", use_container_width=True)

# Output state
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_warnings" not in st.session_state:
    st.session_state.last_warnings = []
if "last_steps" not in st.session_state:
    st.session_state.last_steps = []

if clear:
    st.session_state.last_result = None
    st.session_state.last_warnings = []
    st.session_state.last_steps = []

# Main
st.markdown("### Output")

if run:
    with st.spinner("Agent is thinking..."):
        with st.status("Agent progress", expanded=True) as status:
            status.write("Parsing prompt...")
            result, warnings, steps = agent_run(prompt)
            status.write("Finalizing results...")
            status.update(label="Agent finished", state="complete", expanded=False)

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
        with st.expander("🧠 Agent Thinking (step-by-step)", expanded=True):
            for s in steps:
                st.write(f"- {s}")

    # Beam
    if result["member_type"] == "beam":
        colL, colR = st.columns([1, 1], gap="large")

        # Build narrative text (single string)
        narrative_parts = []
        if "flexure_narrative" in result:
            narrative_parts.append("=== FLEXURE ===\n" + result["flexure_narrative"])
        if "shear_narrative" in result:
            narrative_parts.append("\n=== SHEAR ===\n" + result["shear_narrative"])
        narrative_text = "\n".join(narrative_parts).strip()

        # Build key outputs dict
        out = {}
        if "flexure" in result:
            f = result["flexure"]
            out.update({
                "Flexure shape": f["shape"],
                "Moment sign": f["moment_sign"],
                "Bars": f"{f['n']} {f['bar']}",
                "As (in^2)": round(f["As_prov"], 2),
                "d (in)": round(f["d_in"], 2),
                "a (in)": round(f["a_in"], 2),
                "c (in)": round(f["c_in"], 2),
                "phi": round(f["phi"], 3),
                "phiMn (kip-ft)": round(f["phiMn_kipft"], 1),
            })
            if f["shape"] == "T-beam":
                out.update({
                    "bw (in)": f["bw_in"],
                    "bf (in)": f["bf_in"],
                    "hf (in)": f["hf_in"],
                    "Auto beff?": f.get("auto_beff", False),
                    "bf_auto (in)": (None if f.get("bf_auto_in") is None else round(f["bf_auto_in"], 1)),
                    "bf_auto used?": f.get("bf_auto_used", False),
                })
        if "shear" in result:
            sh = result["shear"]
            out.update({
                "Vu (kips)": result["inputs"]["Vu_kips"],
                "Vc (kips)": round(sh["Vc_kips"], 1),
                "phiVc (kips)": round(sh["phiVc_kips"], 1),
                "Stirrups": ("N/A" if sh["s_use_in"] is None else f"{sh['legs']}-leg {sh['stirrup_size']} @ {sh['s_use_in']:.1f} in"),
            })

        # Render equal-height panels using HTML (works reliably)
        with colL:
            st.markdown(
                f"""
                <div class="panel">
                  <h4>Beam Narrative</h4>
                  <pre>{html.escape(narrative_text if narrative_text else "No narrative available.")}</pre>
                </div>
                """,
                unsafe_allow_html=True
            )

        with colR:
            st.markdown(
                f"""
                <div class="panel">
                  <h4>Key Outputs</h4>
                  <pre>{html.escape(json.dumps(out, indent=2))}</pre>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Design Sketch (smaller)
        if st.session_state.show_sketch and "flexure" in result:
            st.markdown("### Design Sketch")
            fig = draw_design_sketch_section(result["flexure"])
            st.pyplot(fig, use_container_width=False)

    # Column
    elif result["member_type"] == "column":
        st.markdown("#### Narrative")
        st.code(result["narrative"])
        c = result["column"]
        st.markdown("#### Key outputs")
        st.write({
            "Bars": f"{c['n_bars']} {c['bar_size']}",
            "rho": round(c["rho"], 3),
            "phi": c["phi"],
            "phiMn at Pu/phi (kip-ft)": round(c["phiMn_at_Preq"], 1),
            "Tie size": c["tie_size"],
            "Max tie spacing (in)": round(c["s_tie_max_in"], 1),
            "OK?": c["ok"]
        })

    # Slab
    elif result["member_type"] == "slab":
        st.markdown("#### Narrative")
        st.code(result["narrative"])
        sres = result["slab"]
        st.markdown("#### Key outputs")
        st.write({
            "Bars": f"{sres['bar']} @ {sres['s_in']} in",
            "As (in^2/ft)": round(sres["As_in2_per_ft"], 3),
            "Mu (kip-ft per ft strip)": round(sres["Mu_kipft"], 2),
            "phiMn (kip-ft)": round(sres["phiMn_kipft"], 2),
        })

    # Warnings
    st.markdown("### Warnings / Assumptions")
    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.write("No warnings.")
