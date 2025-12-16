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
        background-color: #008CBA; border: none; color: white !important;
        padding: 12px 28px; text-align: center; text-decoration: none;
        display: inline-block; font-size: 16px; margin: 10px 0px;
        cursor: pointer; border-radius: 5px; font-family: 'Sarabun', sans-serif;
        font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .print-btn-internal:hover { background-color: #005f7f; }

    /* ตารางรายการคำนวณ */
    .report-table {width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 13px;}
    .report-table th, .report-table td {border: 1px solid #ddd; padding: 8px;}
    .report-table th {background-color: #f2f2f2; text-align: center; font-weight: bold;}
    .sec-row {background-color: #e0e0e0; font-weight: bold; font-size: 14px; text-align: left;}

    /* สถานะ */
    .pass-ok {color: green; font-weight: bold; text-align: center;}
    .pass-no {color: red; font-weight: bold; text-align: center;}
    .load-value {color: #D32F2F !important; font-weight: bold;}

    /* รูปภาพ */
    .drawing-container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 20px; }
    .drawing-box { border: 1px solid #ddd; padding: 10px; background-color: #fff; text-align: center; min-width: 300px; }
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
        return [(-s / 2, -s * math.sqrt(3) / 6), (s / 2, -s * math.sqrt(3) / 6), (0, s * math.sqrt(3) / 3)]
    elif n_pile == 4:
        return [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2)]
    elif n_pile == 5:
        return [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2), (0, 0)]
    return []


def check_shear_capacity_silent(h_trial, inputs, coords, width_x, width_y):
    """Internal shear check for Auto-Design loop"""
    fc = inputs['fc'] * 0.0980665;
    Pu_tf = inputs['Pu'];
    n_pile = int(inputs['n_pile'])
    col_x = inputs['cx'] * 1000;
    col_y = inputs['cy'] * 1000
    cover = 75.0;
    db = BAR_INFO[inputs['mainBar']]['d_mm']
    d = h_trial - cover - db
    if d <= 0: return False

    P_avg_N = (Pu_tf * 9806.65) / n_pile if n_pile > 0 else 0
    phi_v = 0.75

    # Punching
    c1 = col_x + d;
    c2 = col_y + d
    Vu_punch = sum([P_avg_N for px, py in coords if (abs(px) > c1 / 2) or (abs(py) > c2 / 2)])
    bo = 2 * (c1 + c2)
    Vc_punch = 0.33 * math.sqrt(fc) * bo * d
    if Vu_punch > phi_v * Vc_punch: return False

    # Beam Shear (Approx check without detailed rho)
    dist_crit = col_x / 2 + d
    Vu_beam = sum([P_avg_N for px, py in coords if abs(px) > dist_crit])
    Vc_beam = 0.17 * math.sqrt(fc) * width_y * d
    if Vu_beam > phi_v * Vc_beam: return False

    return True


def process_footing_calculation(inputs):
    rows = []

    def sec(title):
        rows.append(["SECTION", title, "", "", "", "", ""])

    def row(item, formula, subs, result, unit, status=""):
        rows.append([item, formula, subs, result, unit, status])

    # 1. Inputs & Conversions
    fc = inputs['fc'] * 0.0980665  # MPa
    fy = inputs['fy'] * 0.0980665  # MPa
    Pu_tf = inputs['Pu'];
    Pu_N = Pu_tf * 9806.65
    PileCap_tf = inputs['PileCap']
    n_pile = int(inputs['n_pile'])
    s = inputs['spacing'] * 1000
    edge = inputs['edge'] * 1000
    col_x = inputs['cx'] * 1000;
    col_y = inputs['cy'] * 1000
    dp = inputs['dp'] * 1000
    cover = 75.0;
    bar_key = inputs['mainBar']
    db = BAR_INFO[bar_key]['d_mm']

    # 2. Geometry
    coords = get_pile_coordinates(n_pile, s)
    if n_pile == 1:
        width_x = max(dp + 2 * edge, col_x + 2 * edge)
        width_y = max(dp + 2 * edge, col_y + 2 * edge)
    else:
        xs = [c[0] for c in coords];
        ys = [c[1] for c in coords]
        width_x = (max(xs) - min(xs)) + dp + 2 * edge
        width_y = (max(ys) - min(ys)) + dp + 2 * edge
        if n_pile == 3: width_x = s + dp + 2 * edge; width_y = (s * math.sqrt(3) / 2) + dp + 2 * edge

    # 3. Auto-Design Thickness
    h_final = inputs['h'] * 1000
    if inputs.get('auto_h', False) and n_pile > 1:
        h_try = 300.0
        for _ in range(50):
            if check_shear_capacity_silent(h_try, inputs, coords, width_x, width_y):
                h_final = h_try;
                break
            h_try += 50.0

    d = h_final - cover - db  # Effective depth

    # --- REPORT GENERATION START ---
    sec("1. PROPERTIES & GEOMETRY")
    row("Materials", "fc', fy", f"{fmt(fc, 2)}, {fmt(fy, 0)}", "-", "MPa")
    row("Pile Cap Size", "B x L", f"{fmt(width_x, 0)} x {fmt(width_y, 0)}", f"h={h_final:.0f}", "mm")
    row("Effective Depth", "d = h - cov - db", f"{h_final:.0f} - {cover} - {db:.0f}", f"{d:.1f}", "mm")

    # ACI 318-19 Size Effect
    lambda_s = math.sqrt(2.0 / (1 + 0.004 * d))
    if lambda_s > 1.0: lambda_s = 1.0
    row("Size Factor (λs)", "√(2/(1+0.004d))", f"√(2/(1+0.004*{d:.0f}))", f"{fmt(lambda_s, 3)}", "≤ 1.0")

    sec("2. PILE REACTION")
    P_avg_tf = Pu_tf / n_pile if n_pile > 0 else 0
    P_avg_N = Pu_N / n_pile if n_pile > 0 else 0
    status_pile = "PASS" if P_avg_tf <= PileCap_tf else "FAIL"
    row("Avg Reaction (Ru)", "Pu / N", f"{fmt(Pu_tf, 2)} / {n_pile}", f"{fmt(P_avg_tf, 2)}", "tf", status_pile)

    sec("3. FLEXURAL DESIGN")
    # X-Moment (Bars parallel to X, Moment about Y axis) - Wait, usually Long bars take Long moment?
    # Convention: Mu-X implies moment causing bending in X-direction length. Bars run along X.
    # Lever arm is y-distance from x-axis.
    # Let's align with typical Beam convention: Bars along L resist moment from P * (L_arm).

    # X-Direction Design (Bars running along X, Moment arm is distance Y from Center)
    # Note: previous code mixed this. Let's be precise.
    # Bars along X (width_x length) resist Moment My (about X axis).
    # Let's label explicitly "Reinforcement Parallel to X"

    # 3.1 Bars Parallel to X (Resisting Moment about Y?? No, resisting Moment about Y requires bars along X) -> NO.
    # Moment about Y-axis (My) is caused by loads at x-distance. Resisted by bars along X.
    # Correct: Mu_y_axis = Sum(P * x). Resisted by As_x.

    Mx_Nmm = 0  # Moment about Y-axis (Lever x)
    if n_pile > 1:
        for (px, py) in coords:
            lever = abs(px) - col_x / 2
            if lever > 0: Mx_Nmm += P_avg_N * lever
    Mx_tfm = Mx_Nmm / 9806650.0

    phi_f = 0.9
    req_As_x = Mx_Nmm / (phi_f * fy * 0.9 * d) if Mx_Nmm > 0 else 0
    As_min_x = 0.0018 * width_y * h_final
    As_design_x = max(req_As_x, As_min_x)

    nx_bars = math.ceil(As_design_x / BAR_INFO[bar_key]['A_cm2'] / 100)
    if n_pile == 1: nx_bars = max(nx_bars, 4)
    As_prov_x = nx_bars * BAR_INFO[bar_key]['A_cm2'] * 100

    row("Mu (Long Dir)", "Σ P·(x - cx/2)", f"Sum(P * Lever_x)", f"{fmt(Mx_tfm, 2)}", "tf-m")
    row("As Req", "Mu / (0.9·fy·0.9d)", f"{fmt(Mx_Nmm, 2)} / ...", f"{fmt(req_As_x, 0)}", "mm²")
    row("As Min", "0.0018 · B · h", f"0.0018 · {width_y:.0f} · {h_final:.0f}", f"{fmt(As_min_x, 0)}", "mm²")
    row("Provide X-Bars", f"{nx_bars}-{bar_key}", f"As={As_prov_x:.0f} > {As_design_x:.0f}", "OK", "-", "")

    # 3.2 Bars Parallel to Y
    My_Nmm = 0  # Moment about X-axis (Lever y)
    if n_pile > 1:
        for (px, py) in coords:
            lever = abs(py) - col_y / 2
            if lever > 0: My_Nmm += P_avg_N * lever
    My_tfm = My_Nmm / 9806650.0

    req_As_y = My_Nmm / (phi_f * fy * 0.9 * d) if My_Nmm > 0 else 0
    As_min_y = 0.0018 * width_x * h_final
    As_design_y = max(req_As_y, As_min_y)

    ny_bars = math.ceil(As_design_y / BAR_INFO[bar_key]['A_cm2'] / 100)
    if n_pile == 1: ny_bars = max(ny_bars, 4)
    As_prov_y = ny_bars * BAR_INFO[bar_key]['A_cm2'] * 100

    row("Mu (Short Dir)", "Σ P·(y - cy/2)", f"Sum(P * Lever_y)", f"{fmt(My_tfm, 2)}", "tf-m")
    row("Provide Y-Bars", f"{ny_bars}-{bar_key}", f"As={As_prov_y:.0f} > {As_design_y:.0f}", "OK", "-", "")

    # Shear Parameters
    rho_w = As_prov_x / (width_y * d)  # Use X-dir as representative
    rho_term = math.pow(rho_w, 1 / 3)

    if n_pile > 1:
        sec("4. PUNCHING SHEAR (Two-Way)")
        c1 = col_x + d;
        c2 = col_y + d;
        bo = 2 * (c1 + c2)
        Vu_punch = sum([P_avg_N for px, py in coords if (abs(px) > c1 / 2 or abs(py) > c2 / 2)])

        # ACI 318-19 Eq 22.6.5.2
        beta = max(col_x, col_y) / min(col_x, col_y);
        alpha_s = 40
        vc1 = 0.33 * lambda_s * math.sqrt(fc)
        vc2 = 0.17 * (1 + 2 / beta) * lambda_s * math.sqrt(fc)
        vc3 = 0.083 * (2 + alpha_s * d / bo) * lambda_s * math.sqrt(fc)
        vc_punch = min(vc1, vc2, vc3)
        phiVc_punch = 0.75 * vc_punch * bo * d

        row("Perimeter bo", "2(c1+c2)", f"2({c1:.0f}+{c2:.0f})", f"{bo:.0f}", "mm")
        row("Vu (Punching)", "Sum Piles Outside", "-", f"{fmt(Vu_punch / 9806.65, 2)}", "tf")
        row("vc (Stress)", "min(eq a,b,c)", f"min({fmt(vc1, 2)}, {fmt(vc2, 2)}, {fmt(vc3, 2)})", f"{fmt(vc_punch, 2)}",
            "MPa")
        st_p = "PASS" if Vu_punch <= phiVc_punch else "FAIL"
        row("Check", "φVc ≥ Vu", f"{fmt(phiVc_punch / 9806.65, 1)} ≥ {fmt(Vu_punch / 9806.65, 1)}", st_p, "tf", st_p)

        sec("5. BEAM SHEAR (One-Way)")
        # Check Critical Section X
        dist_x = col_x / 2 + d
        Vu_beam = sum([P_avg_N for px, py in coords if abs(px) > dist_x])

        # ACI 318-19 Eq 22.5.5.1
        vc_beam_s = 0.66 * lambda_s * rho_term * math.sqrt(fc)
        phiVc_beam = 0.75 * vc_beam_s * width_y * d

        row("Crit. Section", "d from face", f"{dist_x:.0f} mm from center", "-", "-")
        row("Vu (Beam)", "Sum Piles Outside", "-", f"{fmt(Vu_beam / 9806.65, 2)}", "tf")
        row("ρw Factor", "(ρw)^1/3", f"({fmt(rho_w * 100, 2)}%)^1/3", f"{fmt(rho_term, 2)}", "-")
        row("vc (Stress)", "0.66λs(ρ)^1/3√fc'", f"0.66·{lambda_s:.2f}·{rho_term:.2f}·√{fc:.0f}", f"{fmt(vc_beam_s, 2)}",
            "MPa")
        st_b = "PASS" if Vu_beam <= phiVc_beam else "FAIL"
        row("Check", "φVc ≥ Vu", f"{fmt(phiVc_beam / 9806.65, 1)} ≥ {fmt(Vu_beam / 9806.65, 1)}", st_b, "tf", st_b)
    else:
        st_p = "PASS";
        st_b = "PASS"

    sec("6. CONCLUSION")
    overall = "OK" if (status_pile == "PASS" and st_p == "PASS" and st_b == "PASS") else "NOT OK"
    row("Design Status", "-", "-", overall, "-", overall)

    return rows, coords, width_x, width_y, nx_bars, ny_bars, overall, h_final


# ==========================================
# 4. PLOTTING
# ==========================================
def fig_to_base64(fig):
    buf = io.BytesIO();
    fig.savefig(buf, format='png', bbox_inches='tight');
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def draw_dim(ax, p1, p2, text, offset=50):
    x1, y1 = p1;
    x2, y2 = p2
    angle = math.atan2(y2 - y1, x2 - x1);
    perp = angle + math.pi / 2
    ox = offset * math.cos(perp);
    oy = offset * math.sin(perp)
    p1o = (x1 + ox, y1 + oy);
    p2o = (x2 + ox, y2 + oy)
    ax.plot([x1, p1o[0]], [y1, p1o[1]], 'k-', lw=0.5)
    ax.plot([x2, p2o[0]], [y2, p2o[1]], 'k-', lw=0.5)
    ax.annotate('', xy=p1o, xytext=p2o, arrowprops=dict(arrowstyle='<->', lw=0.8))
    mx = (p1o[0] + p2o[0]) / 2;
    my = (p1o[1] + p2o[1]) / 2
    deg = math.degrees(angle)
    if 90 < deg <= 270:
        deg -= 180
    elif -270 <= deg < -90:
        deg += 180
    tx = mx + 15 * math.cos(perp);
    ty = my + 15 * math.sin(perp)
    ax.text(tx, ty, text, ha='center', va='center', rotation=deg, fontsize=9,
            bbox=dict(fc='white', ec='none', alpha=0.7))


def plot_plan(coords, bx, by, cx, cy, dp, nx, ny, bar):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(patches.Rectangle((-bx / 2, -by / 2), bx, by, lw=2, ec='k', fc='#f9f9f9'))

    # Draw Grid Bars
    ys = np.linspace(-by / 2 + 75, by / 2 - 75, min(ny, 10))
    for y in ys: ax.plot([-bx / 2 + 50, bx / 2 - 50], [y, y], 'b-', lw=1, alpha=0.5)
    xs = np.linspace(-bx / 2 + 75, bx / 2 - 75, min(nx, 10))
    for x in xs: ax.plot([x, x], [-by / 2 + 50, by / 2 - 50], 'r-', lw=1, alpha=0.5)

    ax.add_patch(patches.Rectangle((-cx / 2, -cy / 2), cx, cy, lw=1.5, ec='#333', fc='#ddd', hatch='//', zorder=5))
    for px, py in coords:
        ax.add_patch(patches.Circle((px, py), dp / 2, ec='k', fc='white', ls='--'))

    draw_dim(ax, (-bx / 2, -by / 2 - 250), (bx / 2, -by / 2 - 250), f"L = {bx / 1000:.2f} m", 0)
    draw_dim(ax, (-bx / 2 - 250, -by / 2), (-bx / 2 - 250, by / 2), f"B = {by / 1000:.2f} m", 0)

    ax.text(0, by / 2 + 100, f"{nx}-{bar} (Along Y)", color='red', ha='center', fontweight='bold',
            bbox=dict(boxstyle="round", fc="white", ec="red"))
    ax.text(bx / 2 + 100, 0, f"{ny}-{bar} (Along X)", color='blue', va='center', rotation=90, fontweight='bold',
            bbox=dict(boxstyle="round", fc="white", ec="blue"))

    ax.set_xlim(-bx / 1.1, bx / 1.1);
    ax.set_ylim(-by / 1.1, by / 1.1);
    ax.axis('off');
    ax.set_title("PLAN VIEW", fontweight='bold')
    return fig


def plot_sect(bx, h, cx, dp, cov, bar, npile):
    fig, ax = plt.subplots(figsize=(6, 4))
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

    draw_dim(ax, (bx / 2 + 200, 0), (bx / 2 + 200, -h), f"h={h / 1000:.2f}m", 50)

    ax.text(0, by - 150, f"Main: {bar}", ha='center', color='red', fontweight='bold')
    ax.set_xlim(-bx / 1.2, bx / 1.2);
    ax.set_ylim(-h * 2, h);
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

    auto_h = st.checkbox("Auto-Design Thickness", True)
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
        <div class="drawing-container">
            <div class="drawing-box"><img src="{fig_to_base64(fig1)}" style="max-width:100%;"></div>
            <div class="drawing-box"><img src="{fig_to_base64(fig2)}" style="max-width:100%;"></div>
        </div>
        <br>
        <table class="report-table">
            <thead><tr><th width="20%">Item</th><th width="25%">Formula</th><th width="30%">Substitution</th><th>Result</th><th>Unit</th><th>Status</th></tr></thead>
            <tbody>{t_rows}</tbody>
        </table>
        <div style="margin-top:40px; text-align:center;">
            <div style="display:inline-block; width:250px; text-align:left;">
                <strong>Designed by:</strong><br><br><div style="border-bottom:1px solid #000;"></div>
                <div style="text-align:center; margin-top:5px;">({engineer})<br>วิศวกรโครงสร้าง</div>
            </div>
        </div>
    </div>
    """
    st.components.v1.html(html, height=1200, scrolling=True)
else:
    st.info("👈 กรุณากรอกข้อมูลและกด Run Design")
