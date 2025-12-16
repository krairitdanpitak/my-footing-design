import streamlit as st
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import io
import base64
import streamlit.components.v1 as components

# ==========================================
# 1. SETUP & CSS
# ==========================================
st.set_page_config(page_title="RC Pile Cap Design (ACI 318-19)", layout="wide")

st.markdown("""
<style>
    /* ปุ่มพิมพ์ */
    .print-btn-internal {
        background-color: #008CBA;
        border: none;
        color: white !important;
        padding: 12px 28px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 0px;
        cursor: pointer;
        border-radius: 5px;
        font-family: 'Sarabun', sans-serif;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .print-btn-internal:hover { background-color: #005f7f; }

    /* ตาราง */
    .report-table {width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px;}
    .report-table th, .report-table td {border: 1px solid #ddd; padding: 8px;}
    .report-table th {background-color: #f2f2f2; text-align: center; font-weight: bold;}

    .pass-ok {color: green; font-weight: bold;}
    .pass-no {color: red; font-weight: bold;}
    .sec-row {background-color: #e0e0e0; font-weight: bold; font-size: 15px;}
    .load-value {color: #D32F2F !important; font-weight: bold;}

    /* Layout รูปภาพ */
    .drawing-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
    }
    .drawing-box {
        border: 1px solid #ddd;
        padding: 10px;
        background-color: #fff;
        text-align: center;
        min-width: 300px;
    }

    /* Footer Style */
    .footer-section {
        margin-top: 50px;
        page-break-inside: avoid;
        width: 100%;
    }
    .signature-block {
        float: left; /* ชิดซ้าย */
        width: 300px;
        text-align: left;
    }
    .sign-line {
        border-bottom: 1px solid #000;
        margin: 40px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATABASE & HELPER
# ==========================================
BAR_INFO = {
    'RB6': {'A_cm2': 0.283, 'd_mm': 6},
    'RB9': {'A_cm2': 0.636, 'd_mm': 9},
    'DB10': {'A_cm2': 0.785, 'd_mm': 10},
    'DB12': {'A_cm2': 1.131, 'd_mm': 12},
    'DB16': {'A_cm2': 2.011, 'd_mm': 16},
    'DB20': {'A_cm2': 3.142, 'd_mm': 20},
    'DB25': {'A_cm2': 4.909, 'd_mm': 25},
    'DB28': {'A_cm2': 6.158, 'd_mm': 28},
    'DB32': {'A_cm2': 8.042, 'd_mm': 32}
}


def fmt(n, digits=3):
    try:
        val = float(n)
        if math.isnan(val): return "-"
        return f"{val:,.{digits}f}"
    except:
        return "-"


# ==========================================
# 3. CALCULATION LOGIC (ACI 318-19)
# ==========================================
def get_pile_coordinates(n_pile, s):
    if n_pile == 1:
        return [(0, 0)]
    elif n_pile == 2:
        return [(-s / 2, 0), (s / 2, 0)]
    elif n_pile == 3:
        h_tri = s * math.sqrt(3) / 2
        return [(-s / 2, -h_tri / 3), (s / 2, -h_tri / 3), (0, 2 * h_tri / 3)]
    elif n_pile == 4:
        return [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2)]
    elif n_pile == 5:
        return [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2), (0, 0)]
    return []


def check_shear_capacity_silent(h_trial, inputs, coords, width_x, width_y):
    """ฟังก์ชันคำนวณ Shear ภายในสำหรับ Auto-Design"""
    fc = inputs['fc'] * 0.0980665
    Pu_tf = inputs['Pu']
    n_pile = int(inputs['n_pile'])
    col_x = inputs['cx'] * 1000
    col_y = inputs['cy'] * 1000
    cover = 75.0
    bar_key = inputs['mainBar']
    db = BAR_INFO[bar_key]['d_mm']

    d = h_trial - cover - db
    if d <= 0: return False

    P_avg_N = (Pu_tf * 9806.65) / n_pile if n_pile > 0 else 0
    phi_v = 0.75

    # 1. Punching Shear
    c1 = col_x + d;
    c2 = col_y + d
    Vu_punch_N = sum([P_avg_N for px, py in coords if (abs(px) > c1 / 2) or (abs(py) > c2 / 2)])
    bo = 2 * (c1 + c2)
    # Simplified Vc check for iteration
    Vc_punch_N = 0.33 * math.sqrt(fc) * bo * d
    if Vu_punch_N > phi_v * Vc_punch_N: return False

    # 2. Beam Shear (Approximate check)
    dist_crit = col_x / 2 + d
    Vu_beam_N = sum([P_avg_N for px, py in coords if abs(px) > dist_crit])
    Vc_beam_N = 0.17 * math.sqrt(fc) * width_y * d
    if Vu_beam_N > phi_v * Vc_beam_N: return False

    return True


def process_footing_calculation(inputs):
    rows = []

    def sec(title):
        rows.append(["SECTION", title, "", "", "", "", ""])

    def row(item, formula, subs, result, unit, status=""):
        rows.append([item, formula, subs, result, unit, status])

    # Inputs
    fc = inputs['fc'] * 0.0980665  # MPa
    fy = inputs['fy'] * 0.0980665  # MPa
    Pu_tf = inputs['Pu']
    Pu_N = Pu_tf * 9806.65
    PileCap_tf = inputs['PileCap']
    n_pile = int(inputs['n_pile'])
    s = inputs['spacing'] * 1000
    edge = inputs['edge'] * 1000
    col_x = inputs['cx'] * 1000
    col_y = inputs['cy'] * 1000
    dp = inputs['dp'] * 1000
    cover = 75.0
    bar_key = inputs['mainBar']
    db = BAR_INFO[bar_key]['d_mm']

    # Geometry Setup
    coords = get_pile_coordinates(n_pile, s)
    if n_pile == 1:
        width_x = max(dp + 2 * edge, col_x + 2 * edge)
        width_y = max(dp + 2 * edge, col_y + 2 * edge)
    else:
        xs = [c[0] for c in coords];
        ys = [c[1] for c in coords]
        width_x = (max(xs) - min(xs)) + dp + 2 * edge
        width_y = (max(ys) - min(ys)) + dp + 2 * edge
        if n_pile == 3:
            width_x = s + dp + 2 * edge
            width_y = (s * math.sqrt(3) / 2) + dp + 2 * edge

    # Auto-Design Thickness
    h_final = inputs['h'] * 1000
    is_auto = inputs.get('auto_h', False)
    if is_auto and n_pile > 1:
        h_try = 300.0
        for _ in range(50):
            if check_shear_capacity_silent(h_try, inputs, coords, width_x, width_y):
                h_final = h_try;
                break
            h_try += 50.0

    d = h_final - cover - db  # Effective depth

    # --- 1. GEOMETRY & MATERIALS ---
    sec("1. GEOMETRY & PROPERTIES")
    row("Materials", "fc', fy", f"{fmt(fc, 2)}, {fmt(fy, 0)}", "-", "MPa")
    row("Pile Cap Size", "B x L", f"{fmt(width_x, 0)} x {fmt(width_y, 0)}", f"h={h_final:.0f}", "mm")
    row("Effective Depth", "d = h - cover - db", f"{h_final:.0f} - {cover} - {db:.0f}", f"{d:.1f}", "mm")

    lambda_s = math.sqrt(2.0 / (1 + 0.004 * d))
    if lambda_s > 1.0: lambda_s = 1.0
    row("Size Effect (λs)", "√(2 / (1 + 0.004d))", f"√(2 / (1 + 0.004*{d:.0f}))", f"{fmt(lambda_s, 3)}", "≤ 1.0")

    # --- 2. PILE REACTION ---
    sec("2. PILE REACTION CHECK")
    P_avg_tf = Pu_tf / n_pile if n_pile > 0 else 0
    P_avg_N = Pu_N / n_pile if n_pile > 0 else 0
    status_pile = "PASS" if P_avg_tf <= PileCap_tf else "FAIL"
    row("Avg Reaction (Ru)", "Pu / N", f"{fmt(Pu_tf, 2)} / {n_pile}", f"{fmt(P_avg_tf, 2)}", "tf", status_pile)
    row("Capacity Check", "Ru ≤ P_pile_max", f"{fmt(P_avg_tf, 2)} ≤ {fmt(PileCap_tf, 2)}", status_pile, "-", "")

    # --- 3. FLEXURAL DESIGN ---
    sec("3. FLEXURAL DESIGN")

    # 3.1 Long Direction (X-Moment, Y-Bars)
    Mx_Nmm = 0
    if n_pile > 1:
        face_dist_x = col_x / 2
        for (px, py) in coords:
            lever = abs(px) - face_dist_x
            if lever > 0: Mx_Nmm += P_avg_N * lever
    Mx_tfm = Mx_Nmm / 9806650.0

    phi_f = 0.9
    req_As_x = Mx_Nmm / (phi_f * fy * 0.9 * d) if Mx_Nmm > 0 else 0
    As_min_x = 0.0018 * width_y * h_final
    As_design_x = max(req_As_x, As_min_x)

    bar_area = BAR_INFO[bar_key]['A_cm2'] * 100
    nx_bars = math.ceil(As_design_x / bar_area)
    if n_pile == 1 and nx_bars < 4: nx_bars = 4
    As_prov_x = nx_bars * bar_area

    # Detailed Sub for Moment X
    sub_mx = f"Sum(P*{fmt(abs(coords[0][0]) - col_x / 2, 2)})" if n_pile > 1 else "-"
    row("Mu-X (Long)", "Σ P·(x - cx/2)", sub_mx, f"{fmt(Mx_tfm, 2)}", "tf-m")

    sub_as_req_x = f"{fmt(Mx_Nmm, 0)} / (0.9·{fy:.0f}·0.9·{d:.0f})"
    row("As-X Req", "Mu / (0.9·fy·0.9d)", sub_as_req_x, f"{fmt(req_As_x, 0)}", "mm²")

    sub_as_min_x = f"0.0018 · {width_y:.0f} · {h_final:.0f}"
    row("As-X Min", "0.0018 · B · h", sub_as_min_x, f"{fmt(As_min_x, 0)}", "mm²")

    row("Provide X-Dir", f"Use {bar_key}", f"Req {fmt(As_design_x, 0)} -> {nx_bars} bars", f"{nx_bars}-{bar_key}", "-",
        "OK")

    # 3.2 Short Direction (Y-Moment, X-Bars)
    My_Nmm = 0
    if n_pile > 1:
        face_dist_y = col_y / 2
        for (px, py) in coords:
            lever = abs(py) - face_dist_y
            if lever > 0: My_Nmm += P_avg_N * lever
    My_tfm = My_Nmm / 9806650.0

    req_As_y = My_Nmm / (phi_f * fy * 0.9 * d) if My_Nmm > 0 else 0
    As_min_y = 0.0018 * width_x * h_final
    As_design_y = max(req_As_y, As_min_y)

    ny_bars = math.ceil(As_design_y / bar_area)
    if n_pile == 1 and ny_bars < 4: ny_bars = 4
    As_prov_y = ny_bars * bar_area

    sub_my = f"Sum(P*{fmt(abs(coords[0][1]) - col_y / 2, 2)})" if n_pile > 1 else "-"
    row("Mu-Y (Short)", "Σ P·(y - cy/2)", sub_my, f"{fmt(My_tfm, 2)}", "tf-m")
    row("Provide Y-Dir", f"Use {bar_key}", f"Req {fmt(As_design_y, 0)} -> {ny_bars} bars", f"{ny_bars}-{bar_key}", "-",
        "OK")

    # Calculate Rho for Shear
    rho_w = As_prov_x / (width_y * d)
    rho_term = math.pow(rho_w, 1 / 3)

    # --- 4. SHEAR CHECKS ---
    if n_pile > 1:
        sec("4. SHEAR CHECKS (ACI 318-19)")
        phi_v = 0.75

        # 4.1 Punching Shear
        c1 = col_x + d;
        c2 = col_y + d
        bo = 2 * (c1 + c2)
        beta = max(col_x, col_y) / min(col_x, col_y)
        alpha_s = 40

        Vu_punch_N = sum([P_avg_N for px, py in coords if (abs(px) > c1 / 2 or abs(py) > c2 / 2)])

        # Vc Formulas
        vc1 = 0.33 * lambda_s * math.sqrt(fc)
        vc2 = 0.17 * (1 + 2 / beta) * lambda_s * math.sqrt(fc)
        vc3 = 0.083 * (2 + alpha_s * d / bo) * lambda_s * math.sqrt(fc)
        vc_punch = min(vc1, vc2, vc3)

        Vc_punch_N = vc_punch * bo * d
        phiVc_punch_N = phi_v * Vc_punch_N

        row("Punching Perimeter", "bo = 2(c1+c2)", f"2({c1:.0f}+{c2:.0f})", f"{bo:.0f}", "mm")
        row("Vu (Punching)", "Sum Piles Outside", f"Sum({fmt(P_avg_tf, 2)} tf)", f"{fmt(Vu_punch_N / 9806.65, 2)}",
            "tf")

        # Detailed Sub for Vc Stress
        sub_vc_punch = f"min({fmt(vc1, 2)}, {fmt(vc2, 2)}, {fmt(vc3, 2)})"
        row("vc (Stress)", "min(eq a,b,c)", sub_vc_punch, f"{fmt(vc_punch, 2)}", "MPa")

        # Detailed Sub for Phi Vc
        sub_phi_vc_p = f"0.75 · {fmt(vc_punch, 2)} · {fmt(bo, 0)} · {fmt(d, 0)}"

        st_p = "PASS" if Vu_punch_N <= phiVc_punch_N else "FAIL"
        row("Check Punching", "φVc ≥ Vu", sub_phi_vc_p,
            f"{fmt(phiVc_punch_N / 9806.65, 2)} ≥ {fmt(Vu_punch_N / 9806.65, 2)}", "tf", st_p)

        # 4.2 Beam Shear (One-Way)
        dist_x = col_x / 2 + d
        Vu_beam_N = sum([P_avg_N for px, py in coords if abs(px) > dist_x])

        # ACI 318-19 Eq 22.5.5.1
        vc_beam = 0.66 * lambda_s * rho_term * math.sqrt(fc)
        Vc_beam_N = vc_beam * width_y * d
        phiVc_beam_N = phi_v * Vc_beam_N

        row("Crit. Section", "d from col face", f"{dist_x:.0f} mm from center", "-", "-")
        row("Vu (Beam)", "Sum Piles Outside", f"Sum({fmt(P_avg_tf, 2)} tf)", f"{fmt(Vu_beam_N / 9806.65, 2)}", "tf")

        sub_rho_term = f"({fmt(rho_w * 100, 2)}%)^(1/3)"
        row("ρw Factor", "(ρw)^1/3", sub_rho_term, f"{fmt(rho_term, 2)}", "-")

        # Detailed Sub for Vc Beam Stress
        sub_vc_beam = f"0.66·{fmt(lambda_s, 2)}·{fmt(rho_term, 2)}·√{fmt(fc, 0)}"
        row("vc (Stress)", "0.66λs(ρ)^1/3√fc'", sub_vc_beam, f"{fmt(vc_beam, 2)}", "MPa")

        # Detailed Sub for Phi Vc Beam
        sub_phi_vc_b = f"0.75 · {fmt(vc_beam, 2)} · {fmt(width_y, 0)} · {fmt(d, 0)}"

        st_b = "PASS" if Vu_beam_N <= phiVc_beam_N else "FAIL"
        row("Beam Shear Check", "φVc ≥ Vu", sub_phi_vc_b,
            f"{fmt(phiVc_beam_N / 9806.65, 2)} ≥ {fmt(Vu_beam_N / 9806.65, 2)}", "tf", st_b)
    else:
        st_p = "PASS";
        st_b = "PASS"

    sec("5. FINAL STATUS")
    overall = "OK" if (status_pile == "PASS" and st_p == "PASS" and st_b == "PASS") else "NOT OK"
    row("Overall", "-", "-", "DESIGN COMPLETE", "-", overall)

    return rows, coords, width_x, width_y, nx_bars, ny_bars, overall, h_final


# ==========================================
# 4. PLOTTING WITH DIMENSIONS (TRUE SCALE)
# ==========================================
def fig_to_base64(fig):
    buf = io.BytesIO();
    fig.savefig(buf, format='png', bbox_inches='tight');
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def draw_dim(ax, p1, p2, text, offset=50, color='black'):
    x1, y1 = p1;
    x2, y2 = p2
    dx = x2 - x1;
    dy = y2 - y1;
    angle = math.atan2(dy, dx);
    perp = angle + math.pi / 2
    ox = offset * math.cos(perp);
    oy = offset * math.sin(perp)
    p1o = (x1 + ox, y1 + oy);
    p2o = (x2 + ox, y2 + oy)
    ax.plot([x1, p1o[0]], [y1, p1o[1]], color=color, lw=0.5)
    ax.plot([x2, p2o[0]], [y2, p2o[1]], color=color, lw=0.5)
    ax.annotate('', xy=p1o, xytext=p2o, arrowprops=dict(arrowstyle='<->', color=color, lw=0.8))
    mx = (p1o[0] + p2o[0]) / 2;
    my = (p1o[1] + p2o[1]) / 2
    deg = math.degrees(angle)
    if 90 < deg <= 270:
        deg -= 180
    elif -270 <= deg < -90:
        deg += 180
    tx = mx + 15 * math.cos(perp);
    ty = my + 15 * math.sin(perp)
    ax.text(tx, ty, text, ha='center', va='center', rotation=deg, fontsize=9, color=color,
            bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.8))


def plot_plan(coords, bx, by, cx, cy, dp, nx, ny, bar):
    # Set figsize aspect roughly to match bx/by
    ratio = bx / by
    fig_w = 6;
    fig_h = 6 / ratio if ratio > 1 else 6
    if fig_h < 4: fig_h = 4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.add_patch(patches.Rectangle((-bx / 2, -by / 2), bx, by, lw=2, ec='k', fc='#f9f9f9'))

    # Bars
    ys = np.linspace(-by / 2 + 75, by / 2 - 75, min(ny, 8))
    for y in ys: ax.plot([-bx / 2 + 50, bx / 2 - 50], [y, y], 'b-', lw=1, alpha=0.5)
    xs = np.linspace(-bx / 2 + 75, bx / 2 - 75, min(nx, 8))
    for x in xs: ax.plot([x, x], [-by / 2 + 50, by / 2 - 50], 'r-', lw=1, alpha=0.5)

    ax.add_patch(patches.Rectangle((-cx / 2, -cy / 2), cx, cy, lw=1.5, ec='#333', fc='#ddd', hatch='//', zorder=5))
    for px, py in coords:
        ax.add_patch(patches.Circle((px, py), dp / 2, ec='k', fc='white', ls='--'))

    # Dimensions
    off = max(bx, by) * 0.15
    draw_dim(ax, (-bx / 2, -by / 2 - off), (bx / 2, -by / 2 - off), f"L = {bx / 1000:.2f} m", 0)
    draw_dim(ax, (-bx / 2 - off, -by / 2), (-bx / 2 - off, by / 2), f"B = {by / 1000:.2f} m", 0)

    # Labels (Bubble style)
    ax.annotate(f"{nx}-{bar} (Y-Dir)", xy=(0, by / 2 - 50), xytext=(0, by / 2 + off),
                arrowprops=dict(arrowstyle='->', color='red'), color='red', ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red"))
    ax.annotate(f"{ny}-{bar} (X-Dir)", xy=(bx / 2 - 50, 0), xytext=(bx / 2 + off, 0),
                arrowprops=dict(arrowstyle='->', color='blue'), color='blue', ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue"))

    # Force Equal Aspect Ratio for True Scale
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("PLAN VIEW", fontweight='bold')
    return fig


def plot_sect(bx, h, cx, dp, cov, bar, npile):
    # Set figsize based on aspect ratio
    ratio = bx / h
    fig_w = 6;
    fig_h = 6 / ratio if ratio > 1 else 6
    if fig_h < 4: fig_h = 4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot([-bx, bx], [0, 0], 'k-', lw=0.5)
    ax.add_patch(patches.Rectangle((-bx / 2, -h), bx, h, lw=2, ec='k', fc='#f0f0f0'))
    ax.add_patch(patches.Rectangle((-cx / 2, 0), cx, h * 0.5, lw=1.5, ec='k', fc='#fff', hatch='///'))

    ph = h * 0.5
    if npile == 1:
        ax.add_patch(patches.Rectangle((-dp / 2, -h - ph), dp, ph, ec='k', fc='w'))
    else:
        off = bx / 2 - dp / 2 - 150
        ax.add_patch(patches.Rectangle((-off - dp / 2, -h - ph), dp, ph, ec='k', fc='w'))
        ax.add_patch(patches.Rectangle((off - dp / 2, -h - ph), dp, ph, ec='k', fc='w'))

    by = -h + cov
    ax.plot([-bx / 2 + cov, bx / 2 - cov], [by, by], 'r-', lw=3)
    ax.plot([-bx / 2 + cov, -bx / 2 + cov], [by, by + h * 0.6], 'r-', lw=3)
    ax.plot([bx / 2 - cov, bx / 2 - cov], [by, by + h * 0.6], 'r-', lw=3)

    # Dimensions
    off = max(bx, h) * 0.15
    draw_dim(ax, (bx / 2 + off, 0), (bx / 2 + off, -h), f"h={h / 1000:.2f}m", 50)
    draw_dim(ax, (-bx / 2, -h - ph - off), (bx / 2, -h - ph - off), f"Width={bx / 1000:.2f}m", 0)

    ax.text(0, by - off, f"Main: {bar}", ha='center', color='red', fontweight='bold')

    # Force Equal Aspect Ratio
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("SECTION DETAIL", fontweight='bold')
    return fig


# ==========================================
# 5. UI & REPORT
# ==========================================
st.title("RC Pile Cap Design SDM")

with st.sidebar.form("inputs"):
    st.header("Project Info")
    project = st.text_input("Project Name", "อาคารสำนักงาน 2 ชั้น")
    f_id = st.text_input("Footing ID", "F-01")
    engineer = st.text_input("Engineer Name", "นายไกรฤทธิ์ ด่านพิทักษ์")

    c1, c2 = st.columns(2)
    fc = c1.number_input("fc' (ksc)", 240);
    fy = c2.number_input("fy (ksc)", 4000)
    cx = c1.number_input("Col X (m)", 0.25);
    cy = c2.number_input("Col Y (m)", 0.25)

    n_pile = st.selectbox("Number of Piles", [1, 2, 3, 4, 5], index=3)
    dp = c1.number_input("Pile Dia (m)", 0.22);
    spacing = c2.number_input("Spacing (m)", 0.80)

    auto_h = st.checkbox("ออกแบบความหนาอัตโนมัติ (Auto-Design)", True)
    h = st.number_input("Thickness (m) [Init]", 0.50, help="ความหนาของฐานราก (เมตร)")
    edge = st.number_input("Edge Dist (m)", 0.25, help="ระยะจากศูนย์กลางเสาเข็มต้นริมสุด ถึงขอบฐานราก")
    mainBar = st.selectbox("Main Rebar", list(BAR_INFO.keys()), index=4)

    Pu = st.number_input("Axial Load Pu (tf)", 0.0, value=60.0)
    PileCap = st.number_input("Max Load/Pile (tf)", 0.0, value=30.0)

    run_btn = st.form_submit_button("Run Design")

if run_btn:
    data = {
        'project': project, 'f_id': f_id, 'engineer': engineer,
        'fc': fc, 'fy': fy, 'cx': cx, 'cy': cy, 'n_pile': n_pile, 'dp': dp,
        'spacing': spacing, 'h': h, 'edge': edge, 'mainBar': mainBar,
        'Pu': Pu, 'PileCap': PileCap, 'auto_h': auto_h
    }

    rows, coords, bx, by, nx, ny, stt, fh = process_footing_calculation(data)

    fig1 = plot_plan(coords, bx, by, cx * 1000, cy * 1000, dp * 1000, nx, ny, mainBar)
    fig2 = plot_sect(bx, fh, cx * 1000, dp * 1000, 75, f"{max(nx, ny)}-{mainBar}", n_pile)

    # HTML Report Generation
    t_rows = "".join([
                         f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td class='load-value'>{r[3]}</td><td>{r[4]}</td><td class='{('pass-ok' if 'PASS' in r[5] or 'OK' in r[5] else 'pass-no')}'>{r[5]}</td></tr>" if
                         r[0] != "SECTION" else f"<tr class='sec-row'><td colspan='6'>{r[1]}</td></tr>" for r in rows])

    html = f"""
    <div style="font-family: Sarabun, sans-serif; padding: 20px;">
        <div style="text-align:center; border-bottom: 2px solid #333; margin-bottom: 20px;">
            <div style="float:right; border:2px solid #333; padding:5px 10px; font-weight:bold;">{f_id}</div>
            <h2>ENGINEERING DESIGN REPORT</h2>
            <h4>RC Pile Cap Design (ACI 318-19)</h4>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:20px;">
            <div style="border:1px solid #ddd; padding:10px; width:48%;">
                <strong>Project:</strong> {project}<br><strong>Engineer:</strong> {engineer}
            </div>
            <div style="border:1px solid #ddd; padding:10px; width:48%;">
                <strong>Materials:</strong> fc'={fc} ksc, fy={fy} ksc<br><strong>Pile:</strong> {n_pile} x Dia {dp} m
            </div>
        </div>
        <div style="display:flex; justify-content:center; gap:20px; margin-bottom:20px;">
            <img src="{fig_to_base64(fig1)}" style="border:1px solid #ddd; padding:5px; width:45%;">
            <img src="{fig_to_base64(fig2)}" style="border:1px solid #ddd; padding:5px; width:45%;">
        </div>
        <table class="report-table">
            <thead><tr><th width="20%">Item</th><th width="25%">Formula</th><th width="30%">Substitution</th><th>Result</th><th>Unit</th><th>Status</th></tr></thead>
            <tbody>{t_rows}</tbody>
        </table>
        <div class="footer-section">
            <div class="signature-block">
                <div style="text-align: left; font-weight: bold;">Designed by:</div>
                <div class="sign-line"></div>
                <div>({engineer})</div>
                <div>วิศวกรโครงสร้าง</div>
            </div>
        </div>
    </div>
    """
    st.components.v1.html(html, height=1200, scrolling=True)
else:
    st.info("👈 กรุณากรอกข้อมูลและกด Run Design")