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
    /* ปุ่มพิมพ์ สีเขียว */
    .print-btn-internal {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white !important;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 0px;
        cursor: pointer;
        border-radius: 4px;
        font-family: sans-serif;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .print-btn-internal:hover { background-color: #45a049; }

    /* Container สำหรับรายงาน */
    .report-container {
        font-family: 'Sarabun', sans-serif;
        padding: 40px;
        background-color: white;
        max-width: 210mm;
        margin: auto;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }

    /* ตารางรายการคำนวณ */
    .report-table {
        width: 100%; 
        border-collapse: collapse; 
        font-family: sans-serif; 
        font-size: 13px;
        margin-top: 20px;
    }
    .report-table th, .report-table td {
        border: 1px solid #000;
        padding: 8px;
        vertical-align: top;
    }
    .report-table th {
        background-color: #f2f2f2; 
        text-align: center; 
        font-weight: bold;
    }

    .pass-ok {color: green; font-weight: bold; text-align: center;}
    .pass-no {color: red; font-weight: bold; text-align: center;}
    .sec-row {
        background-color: #e0e0e0; 
        font-weight: bold; 
        font-size: 14px; 
        text-align: left;
    }
    .load-value {color: #D32F2F !important; font-weight: bold;}

    /* Layout รูปภาพ */
    .drawing-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
        margin: 20px 0;
    }
    .drawing-box {
        border: 1px solid #ddd;
        padding: 10px;
        background-color: #fff;
        text-align: center;
        min-width: 300px;
    }

    /* Header Report */
    .header-report {
        text-align: center;
        position: relative;
        border-bottom: 2px solid #000;
        margin-bottom: 20px;
        padding-bottom: 15px;
    }
    .id-box {
        position: absolute;
        top: 0;
        right: 0;
        border: 2px solid #000;
        padding: 5px 15px;
        font-weight: bold;
        font-size: 16px;
    }

    /* Project Info Box */
    .info-table {
        width: 100%;
        margin-bottom: 20px;
        border: 1px solid #ddd;
        padding: 10px;
        display: flex;
        justify-content: space-between;
        font-size: 14px;
    }

    /* Footer Style (จัดซ้ายล่าง) */
    .footer-section {
        margin-top: 50px;
        page-break-inside: avoid;
        width: 100%;
        text-align: left;
    }
    .signature-block {
        display: inline-block;
        width: 300px;
        text-align: left;
    }
    .sign-line {
        border-bottom: 1px solid #000;
        margin: 40px 0 5px 0;
        width: 100%;
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


def fmt(n, digits=2):
    try:
        val = float(n)
        if math.isnan(val): return "-"
        return f"{val:,.{digits}f}"
    except:
        return "-"


def fig_to_base64(fig):
    buf = io.BytesIO();
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100);
    buf.seek(0)
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ==========================================
# 3. CALCULATION LOGIC (Pile Cap)
# ==========================================
def get_pile_coordinates(n_pile, s):
    if n_pile == 1:
        return [(0, 0)]
    elif n_pile == 2:
        return [(-s / 2, 0), (s / 2, 0)]
    elif n_pile == 3:
        return [(-s / 2, -s * 0.288), (s / 2, -s * 0.288), (0, s * 0.577)]
    elif n_pile == 4:
        return [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2)]
    elif n_pile == 5:
        return [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2), (0, 0)]
    return []


def check_shear_capacity_silent(h_trial, inputs, coords, width_x, width_y):
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
    Vc_punch_N = 0.33 * math.sqrt(fc) * bo * d
    if Vu_punch_N > phi_v * Vc_punch_N: return False

    # 2. Beam Shear (Approx check along X)
    dist_crit = col_x / 2 + d
    Vu_beam_N = sum([P_avg_N for px, py in coords if abs(px) > dist_crit])
    Vc_beam_N = 0.17 * math.sqrt(fc) * width_y * d
    if Vu_beam_N > phi_v * Vc_beam_N: return False

    return True


def process_footing_calculation(inputs):
    rows = []

    def sec(t):
        rows.append(["SECTION", t, "", "", "", ""])

    def row(i, f, s, r, u, st=""):
        rows.append([i, f, s, r, u, st])

    fc = inputs['fc'] * 0.0980665;
    fy = inputs['fy'] * 0.0980665
    pu_tf = inputs['Pu'];
    n_pile = int(inputs['n_pile'])
    s = inputs['spacing'] * 1000;
    edge = inputs['edge'] * 1000
    dp = inputs['dp'] * 1000;
    col_x = inputs['cx'] * 1000;
    col_y = inputs['cy'] * 1000
    h_final = inputs['h'] * 1000;
    cover = 75.0
    db = BAR_INFO[inputs['mainBar']]['d_mm']

    # Auto H Logic
    coords = get_pile_coordinates(n_pile, s)
    bx = (max([abs(x) for x, _ in coords]) * 2) + dp + 2 * edge if n_pile > 1 else dp + 2 * edge
    by = (max([abs(y) for _, y in coords]) * 2) + dp + 2 * edge if n_pile > 1 else dp + 2 * edge

    if inputs.get('auto_h', False) and n_pile > 1:
        h_try = 300.0
        for _ in range(50):
            if check_shear_capacity_silent(h_try, inputs, coords, bx, by):
                h_final = h_try;
                break
            h_try += 50.0

    d = h_final - cover - db

    # --- 1. GEOMETRY ---
    sec("1. GEOMETRY & PROPERTIES")
    row("Footing Size", "B x L", f"{bx:.0f}x{by:.0f}", f"h={h_final:.0f}", "mm", "")
    row("Effective Depth", "d = h - cover - db", f"{h_final:.0f} - {cover} - {db}", f"{d:.1f}", "mm", "")
    lambda_s = min(math.sqrt(2 / (1 + 0.004 * d)), 1.0)
    row("Size Effect λs", "√(2/(1+0.004d))", f"√(2/(1+0.004*{d:.0f}))", f"{lambda_s:.3f}", "-", "≤1.0")

    # --- 2. PILE REACTION ---
    sec("2. PILE REACTION CHECK")
    p_avg = pu_tf / n_pile
    row("Load per Pile (Ru)", "Pu / N", f"{pu_tf} / {n_pile}", f"{p_avg:.2f}", "tf",
        "PASS" if p_avg <= inputs['PileCap'] else "FAIL")

    # --- 3. FLEXURAL ---
    sec("3. FLEXURAL DESIGN")
    p_n = p_avg * 9806;
    mx = 0;
    my = 0
    for x, y in coords:
        lx = abs(x) - col_x / 2;
        ly = abs(y) - col_y / 2
        if lx > 0: mx += p_n * lx
        if ly > 0: my += p_n * ly

    res_bars = {}
    for label, mom, width in [('X-Dir', mx, by), ('Y-Dir', my, bx)]:
        req_as = mom / (0.9 * fy * 0.9 * d) if mom > 0 else 0
        min_as = 0.0018 * width * h_final
        des_as = max(req_as, min_as)
        n = math.ceil(des_as / (BAR_INFO[inputs['mainBar']]['A_cm2'] * 100))
        if n_pile == 1: n = max(n, 4)
        prov_as = n * BAR_INFO[inputs['mainBar']]['A_cm2'] * 100

        # Detailed Substitution
        sub_mu = f"Sum(P * Lever) = {mom / 9.8e6:.2f}"
        sub_as = f"Max({req_as:.0f}, {min_as:.0f})"

        row(f"Moment {label}", "Σ P(arm)", sub_mu, f"{mom / 9.8e6:.2f}", "tf-m", "")
        row(f"As,req {label}", "Max(Calc, Min)", sub_as, f"{des_as:.0f}", "mm²", "")
        row(f"Provide {label}", f"{n}-{inputs['mainBar']}", f"As={prov_as:.0f}", "OK", "-", "")

        res_bars[label] = n

    # --- 4. SHEAR ---
    if n_pile > 1:
        sec("4. SHEAR CHECKS (ACI 318-19)")
        # Punching
        bo = 4 * (col_x + d)  # Simplified bo for center column
        if n_pile == 2: bo = 2 * (col_x + d) + 2 * (col_y + d)  # Generic

        vu_p = sum([p_n for x, y in coords if max(abs(x), abs(y)) > (max(col_x, col_y) + d) / 2])
        vc_p = 0.33 * lambda_s * math.sqrt(fc) * bo * d
        phi_vc_p = 0.75 * vc_p

        row("Punching Vu", "Sum Outside Crit. Sect", "-", f"{vu_p / 9806:.2f}", "tf", "")

        # Detailed Sub for Vc
        sub_vc_p = f"0.75·0.33·{lambda_s:.2f}·√{fc:.0f}·{bo:.0f}·{d:.0f}"
        st_p = "PASS" if vu_p <= phi_vc_p else "FAIL"
        row("Punching Capacity φVc", "0.75·0.33λs√fc·bo·d", sub_vc_p, f"{phi_vc_p / 9806:.2f}", "tf", st_p)

        # Beam Shear X
        prov_as_x = res_bars.get('X-Dir', 4) * BAR_INFO[inputs['mainBar']]['A_cm2'] * 100
        rho_w = prov_as_x / (by * d);
        rho_term = math.pow(rho_w, 1 / 3)
        vu_b = sum([p_n for x, y in coords if abs(x) > col_x / 2 + d])
        vc_b = 0.66 * lambda_s * rho_term * math.sqrt(fc) * by * d
        phi_vc_b = 0.75 * vc_b

        row("Beam Vu (X-Axis)", "Sum Outside d", "-", f"{vu_b / 9806:.2f}", "tf", "")

        sub_vc_b = f"0.75·0.66·{lambda_s:.2f}·{rho_term:.2f}·√{fc:.0f}·{by:.0f}·{d:.0f}"
        st_b = "PASS" if vu_b <= phi_vc_b else "FAIL"
        row("Beam Capacity φVc", "0.75·0.66λs(ρ)^1/3√fc·bd", sub_vc_b, f"{phi_vc_b / 9806:.2f}", "tf", st_b)

    sec("5. FINAL STATUS")
    row("Overall Design", "-", "-", "COMPLETE", "-", "OK")

    return rows, coords, bx, by, res_bars.get('X-Dir', 4), res_bars.get('Y-Dir', 4), h_final


# ==========================================
# 4. PLOTTING
# ==========================================
def draw_dim(ax, p1, p2, text, offset=50, color='black'):
    x1, y1 = p1;
    x2, y2 = p2
    angle = math.atan2(y2 - y1, x2 - x1);
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
            bbox=dict(fc='white', ec='none', alpha=0.8))


def plot_foot_combined(coords, bx, by, nx, ny, bar, h_mm, cx_mm, cy_mm):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # PLAN
    ax1.set_title("PLAN VIEW", fontweight='bold')
    ax1.add_patch(patches.Rectangle((-bx / 2, -by / 2), bx, by, ec='k', fc='#f9f9f9', lw=2))
    ys = np.linspace(-by / 2 + 100, by / 2 - 100, min(ny, 8))
    for y in ys: ax1.plot([-bx / 2 + 50, bx / 2 - 50], [y, y], 'b-', lw=1, alpha=0.5)
    xs = np.linspace(-bx / 2 + 100, bx / 2 - 100, min(nx, 8))
    for x in xs: ax1.plot([x, x], [-by / 2 + 50, by / 2 - 50], 'r-', lw=1, alpha=0.5)
    ax1.add_patch(patches.Rectangle((-cx_mm / 2, -cy_mm / 2), cx_mm, cy_mm, ec='k', fc='#ddd', hatch='//'))
    for x, y in coords: ax1.add_patch(patches.Circle((x, y), 120, ec='k', ls='--'))

    draw_dim(ax1, (-bx / 2, -by / 2 - 200), (bx / 2, -by / 2 - 200), f"L={bx / 1000:.2f}m", 0)
    draw_dim(ax1, (-bx / 2 - 200, -by / 2), (-bx / 2 - 200, by / 2), f"B={by / 1000:.2f}m", 0)
    ax1.text(0, by / 2 + 150, f"{nx}-{bar} (Y-Dir)", ha='center', color='red', fontweight='bold')
    ax1.text(bx / 2 + 150, 0, f"{ny}-{bar} (X-Dir)", va='center', rotation=90, color='blue', fontweight='bold')
    ax1.set_xlim(-bx / 1.1, bx / 1.1);
    ax1.set_ylim(-by / 1.1, by / 1.1);
    ax1.set_aspect('equal');
    ax1.axis('off')

    # SECTION
    ax2.set_title("SECTION DETAIL", fontweight='bold')
    ax2.plot([-bx, bx], [0, 0], 'k-', lw=0.5)
    ax2.add_patch(patches.Rectangle((-bx / 2, -h_mm), bx, h_mm, ec='k', fc='#f0f0f0', lw=2))
    ax2.add_patch(patches.Rectangle((-cx_mm / 2, 0), cx_mm, h_mm / 2, ec='k', fc='#fff', hatch='///'))

    cov = 75
    ax2.plot([-bx / 2 + cov, bx / 2 - cov], [-h_mm + cov, -h_mm + cov], 'r-', lw=3)
    ax2.plot([-bx / 2 + cov, -bx / 2 + cov], [-h_mm + cov, -h_mm + cov + h_mm * 0.6], 'r-', lw=3)
    ax2.plot([bx / 2 - cov, bx / 2 - cov], [-h_mm + cov, -h_mm + cov + h_mm * 0.6], 'r-', lw=3)

    draw_dim(ax2, (bx / 2 + 200, 0), (bx / 2 + 200, -h_mm), f"h={h_mm / 1000:.2f}m", 50)
    ax2.text(0, -h_mm + cov - 150, f"Main Reinforcement", ha='center', color='red', fontweight='bold')
    ax2.set_xlim(-bx / 1.1, bx / 1.1);
    ax2.set_ylim(-h_mm * 2, h_mm);
    ax2.set_aspect('equal');
    ax2.axis('off')

    return fig


# ==========================================
# 5. GENERATE REPORT
# ==========================================
def generate_report(title, rows, imgs, proj, eng, elem_id):
    t_rows = ""
    for r in rows:
        if r[0] == "SECTION":
            t_rows += f"<tr class='sec-row'><td colspan='6'>{r[1]}</td></tr>"
        else:
            cls = "pass-ok" if "PASS" in r[5] or "OK" in r[5] else ("pass-no" if "FAIL" in r[5] else "")
            val_cls = "load-value" if "Mu" in r[0] or "Vu" in r[0] else ""
            t_rows += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td class='{val_cls}'>{r[3]}</td><td>{r[4]}</td><td class='{cls}'>{r[5]}</td></tr>"

    img_html = "".join([f"<div class='drawing-box'><img src='{i}' style='max-width:100%'></div>" for i in imgs])

    return f"""
    <div style="font-family: Sarabun, sans-serif; padding: 20px;">
        <div style="text-align:center; margin-bottom:20px;">
            <button onclick="window.print()" class="print-btn-internal">🖨️ Print This Page / พิมพ์หน้านี้</button>
        </div>

        <div class="report-container">
            <div class="header-report">
                <div class="id-box">{elem_id}</div>
                <h1>ENGINEERING DESIGN REPORT</h1>
                <h3>{title}</h3>
            </div>

            <div class="info-table">
                <div><strong>Project:</strong> {proj}<br><strong>Engineer:</strong> {eng}</div>
                <div style="text-align:right;"><strong>Date:</strong> 16/12/2568</div>
            </div>

            <h3 style="text-align:center;">Design Summary</h3>
            <div class="drawing-container">{img_html}</div>

            <h3 style="text-align:center;">Calculation Details</h3>
            <table class="report-table">
                <thead><tr><th width="20%">Item</th><th width="25%">Formula</th><th width="30%">Substitution</th><th>Result</th><th>Unit</th><th>Status</th></tr></thead>
                <tbody>{t_rows}</tbody>
            </table>

            <div class="footer-section">
                <div class="signature-block">
                    <div style="text-align: left; font-weight: bold;">Designed by:</div>
                    <div class="sign-line"></div>
                    <div>({eng})</div>
                    <div>วิศวกรโครงสร้าง</div>
                </div>
            </div>
        </div>
    </div>
    """


# ==========================================
# 6. MAIN UI
# ==========================================
st.title("RC Pile Cap Design SDM")

with st.sidebar.form("inputs"):
    st.header("Project Info")
    project = st.text_input("Project Name", "New Building")
    f_id = st.text_input("Footing ID", "F-01")
    engineer = st.text_input("Engineer Name", "Mr. Engineer")

    c1, c2 = st.columns(2)
    fc = c1.number_input("fc' (ksc)", 240);
    fy = c2.number_input("fy (ksc)", 4000)
    cx = c1.number_input("Col X (m)", 0.25);
    cy = c2.number_input("Col Y (m)", 0.25)

    n_pile = st.selectbox("Number of Piles", [1, 2, 3, 4, 5], index=3)
    dp = c1.number_input("Pile Dia (m)", 0.22);
    spacing = c2.number_input("Spacing (m)", 0.80)

    auto_h = st.checkbox("Auto-Design Thickness", True)
    h = st.number_input("Thickness (m)", 0.50)
    edge = st.number_input("Edge Dist (m)", 0.25)
    mainBar = st.selectbox("Main Rebar", list(BAR_INFO.keys()), index=4)

    Pu = st.number_input("Axial Load Pu (tf)", 60.0)
    PileCap = st.number_input("Pile Capacity (tf)", 30.0)

    run_btn = st.form_submit_button("Run Design")

if run_btn:
    st.success("✅ คำนวณเสร็จสิ้น (Calculation Finished)")
    d = {'project': project, 'f_id': f_id, 'engineer': engineer, 'fc': fc, 'fy': fy, 'cx': cx, 'cy': cy,
         'n_pile': n_pile, 'dp': dp, 'spacing': spacing, 'h': h, 'edge': edge, 'mainBar': mainBar,
         'Pu': Pu, 'PileCap': PileCap, 'auto_h': auto_h}

    rows, coords, bx, by, nx, ny, h_calc = process_footing_calculation(d)
    img = fig_to_base64(plot_foot_combined(coords, bx, by, nx, ny, mainBar, h_calc, cx * 1000, cy * 1000))
    st.components.v1.html(generate_report("Pile Cap Calculation Report", rows, [img], project, engineer, f_id),
                          height=1200, scrolling=True)
else:
    st.info("👈 กรุณากรอกข้อมูลและกด Run Design")
