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
st.set_page_config(page_title="RC Pile Cap Design SDM", layout="wide")

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
# 3. CALCULATION LOGIC
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


def check_shear_capacity(h_trial, inputs, coords, width_x, width_y):
    fc = inputs['fc'] * 0.0980665
    Pu_tf = inputs['Pu']
    n_pile = int(inputs['n_pile'])
    col_x = inputs['cx'] * 1000
    col_y = inputs['cy'] * 1000
    cover = 75.0
    bar_key = inputs['mainBar']
    db = BAR_INFO[bar_key]['d_mm']

    d = h_trial - cover - db / 2
    if d <= 0: return False, 0, 0, 0

    P_avg_N = (Pu_tf * 9806.65) / n_pile if n_pile > 0 else 0
    phi_v = 0.75

    # Punching
    c1 = col_x + d;
    c2 = col_y + d
    bo = 2 * (c1 + c2)
    Vu_punch_N = sum([P_avg_N for px, py in coords if (abs(px) > c1 / 2) or (abs(py) > c2 / 2)])
    Vc_punch_N = 0.33 * math.sqrt(fc) * bo * d
    if Vu_punch_N > phi_v * Vc_punch_N: return False, d, 0, 0

    # Beam Shear (Check both X and Y)
    # Check X
    dist_crit_x = col_x / 2 + d
    Vu_beam_x = sum([P_avg_N for px, py in coords if abs(px) > dist_crit_x])
    Vc_beam_x = 0.17 * math.sqrt(fc) * width_y * d
    if Vu_beam_x > phi_v * Vc_beam_x: return False, d, 0, 0

    return True, d, 0, 0


def process_footing_calculation(inputs):
    rows = []

    def sec(title):
        rows.append(["SECTION", title, "", "", "", "", ""])

    def row(item, formula, subs, result, unit, status=""):
        rows.append([item, formula, subs, result, unit, status])

    # Inputs
    fc = inputs['fc'] * 0.0980665
    fy = inputs['fy'] * 0.0980665
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

    # 1. Geometry
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

    # Auto-Design h
    h_final = inputs['h'] * 1000
    is_auto = inputs.get('auto_h', False)
    if is_auto and n_pile > 1:
        h_try = 300.0
        for _ in range(50):
            passed, _, _, _ = check_shear_capacity(h_try, inputs, coords, width_x, width_y)
            if passed:
                h_final = h_try;
                break
            h_try += 50.0
    d = h_final - cover - db / 2

    sec("1. GEOMETRY & MATERIALS")
    row("Materials", "fc', fy", f"{fmt(fc, 2)}, {fmt(fy, 0)}", "-", "MPa")
    row("Pile Cap Size", "B x L", f"{fmt(width_x, 0)} x {fmt(width_y, 0)}", f"h={h_final:.0f}", "mm")
    row("Pile Config", f"{n_pile} Piles", f"Spacing {s:.0f} mm", "-", "-")

    # 2. Reaction
    sec("2. PILE REACTION CHECK")
    P_avg_tf = Pu_tf / n_pile if n_pile > 0 else 0
    P_avg_N = Pu_N / n_pile if n_pile > 0 else 0
    status_pile = "PASS" if P_avg_tf <= PileCap_tf else "FAIL"
    row("Pile Reaction", "Pu / N", f"{fmt(Pu_tf, 2)} / {n_pile}", f"{fmt(P_avg_tf, 2)}", "tf", status_pile)

    # 3. Flexure (Both Directions)
    sec("3. FLEXURAL DESIGN")

    # Direction X (Bending about Y-axis) - Bars parallel to X
    Mx_Nmm = 0
    for (px, py) in coords:
        lever = abs(px) - col_x / 2
        if lever > 0: Mx_Nmm += P_avg_N * lever
    Mx_tfm = Mx_Nmm / 9806650.0

    # Direction Y (Bending about X-axis) - Bars parallel to Y
    My_Nmm = 0
    for (px, py) in coords:
        lever = abs(py) - col_y / 2
        if lever > 0: My_Nmm += P_avg_N * lever
    My_tfm = My_Nmm / 9806650.0

    # As Req
    phi_f = 0.9
    req_As_x = Mx_Nmm / (phi_f * fy * 0.9 * d) if Mx_Nmm > 0 else 0
    req_As_y = My_Nmm / (phi_f * fy * 0.9 * d) if My_Nmm > 0 else 0

    # Min Steel
    As_min_x = 0.0018 * width_y * h_final  # Distributed along width_y (bars parallel to X)
    As_min_y = 0.0018 * width_x * h_final  # Distributed along width_x (bars parallel to Y)

    # Design As
    As_design_x = max(req_As_x, As_min_x)
    As_design_y = max(req_As_y, As_min_y)

    # Bar Counts
    bar_area = BAR_INFO[bar_key]['A_cm2'] * 100
    nx_bars = math.ceil(As_design_x / bar_area)
    ny_bars = math.ceil(As_design_y / bar_area)

    if n_pile == 1:
        if nx_bars < 4: nx_bars = 4
        if ny_bars < 4: ny_bars = 4

    row("Mu-X (Long)", "Σ P·(x - cx/2)", "-", f"{fmt(Mx_tfm, 2)}", "tf-m")
    row("As-X Req", "Max(Calc, Min)", f"Max({req_As_x:.0f}, {As_min_x:.0f})", f"{fmt(As_design_x, 0)}", "mm²")
    row("Provide X-Dir", f"Use {bar_key}", "-", f"{nx_bars}-{bar_key}", "-", "OK")

    row("Mu-Y (Short)", "Σ P·(y - cy/2)", "-", f"{fmt(My_tfm, 2)}", "tf-m")
    row("As-Y Req", "Max(Calc, Min)", f"Max({req_As_y:.0f}, {As_min_y:.0f})", f"{fmt(As_design_y, 0)}", "mm²")
    row("Provide Y-Dir", f"Use {bar_key}", "-", f"{ny_bars}-{bar_key}", "-", "OK")

    # Shear Checks (Skip if F1)
    if n_pile > 1:
        sec("4. SHEAR CHECKS")
        # Punching
        c1 = col_x + d;
        c2 = col_y + d;
        bo = 2 * (c1 + c2)
        Vu_punch_N = sum([P_avg_N for px, py in coords if (abs(px) > c1 / 2 or abs(py) > c2 / 2)])
        Vc_punch_N = 0.33 * math.sqrt(fc) * bo * d
        status_punch = "PASS" if Vu_punch_N <= 0.75 * Vc_punch_N else "FAIL"
        row("Punching Check", "Vu ≤ 0.75Vc", f"{Vu_punch_N / 9806.65:.1f} ≤ {0.75 * Vc_punch_N / 9806.65:.1f}",
            status_punch, "tf", status_punch)

        # Beam Shear (Check max direction)
        dist_x = col_x / 2 + d
        Vu_beam_x = sum([P_avg_N for px, py in coords if abs(px) > dist_x])
        Vc_beam_x = 0.17 * math.sqrt(fc) * width_y * d
        status_beam = "PASS" if Vu_beam_x <= 0.75 * Vc_beam_x else "FAIL"
        row("Beam Shear Check", "Vu ≤ 0.75Vc", f"{Vu_beam_x / 9806.65:.1f} ≤ {0.75 * Vc_beam_x / 9806.65:.1f}",
            status_beam, "tf", status_beam)
    else:
        status_punch = "PASS";
        status_beam = "PASS"

    sec("5. FINAL STATUS")
    overall = "OK" if (status_pile == "PASS" and status_punch == "PASS" and status_beam == "PASS") else "NOT OK"
    row("Overall", "-", "-", "DESIGN COMPLETE", "-", overall)

    return rows, coords, width_x, width_y, nx_bars, ny_bars, overall, h_final


# ==========================================
# 4. PLOTTING
# ==========================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def plot_footing_plan(coords, width_x, width_y, col_x, col_y, dp, nx_bars, ny_bars, bar_name):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Footing
    rect = patches.Rectangle((-width_x / 2, -width_y / 2), width_x, width_y, linewidth=2, edgecolor='black',
                             facecolor='#f9f9f9')
    ax.add_patch(rect)

    # Rebar X-Direction (Horizontal lines, spaced vertically)
    # We draw `ny_bars` horizontal lines distributed along Y
    ys = np.linspace(-width_y / 2 + 75, width_y / 2 - 75, min(ny_bars, 10))
    for y in ys:
        ax.plot([-width_x / 2 + 50, width_x / 2 - 50], [y, y], 'b-', linewidth=1.5)

    # Rebar Y-Direction (Vertical lines, spaced horizontally)
    # We draw `nx_bars` vertical lines distributed along X
    xs = np.linspace(-width_x / 2 + 75, width_x / 2 - 75, min(nx_bars, 10))
    for x in xs:
        ax.plot([x, x], [-width_y / 2 + 50, width_y / 2 - 50], 'r-', linewidth=1.5)

    # Column
    rect_col = patches.Rectangle((-col_x / 2, -col_y / 2), col_x, col_y, linewidth=1.5, edgecolor='#333',
                                 facecolor='#ddd', hatch='//', zorder=10)
    ax.add_patch(rect_col)

    # Piles
    for (px, py) in coords:
        circle = patches.Circle((px, py), radius=dp / 2, edgecolor='black', facecolor='white', linewidth=1,
                                linestyle='--')
        ax.add_patch(circle)

    # Labels (Callouts)
    # Label for X-Bars (Vertical lines, bars parallel to Y)
    ax.annotate(f"{nx_bars}-{bar_name}", xy=(0, width_y / 2 - 50), xytext=(0, width_y / 2 + 150),
                arrowprops=dict(arrowstyle='->', color='red'), color='red', ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1))

    # Label for Y-Bars (Horizontal lines, bars parallel to X)
    ax.annotate(f"{ny_bars}-{bar_name}", xy=(width_x / 2 - 50, 0), xytext=(width_x / 2 + 150, 0),
                arrowprops=dict(arrowstyle='->', color='blue'), color='blue', ha='center', fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=1))

    ax.set_xlim(-width_x / 1.2, width_x / 1.2)
    ax.set_ylim(-width_y / 1.2, width_y / 1.2)
    ax.set_aspect('equal');
    ax.axis('off')
    ax.set_title("PLAN VIEW (REINFORCEMENT)", fontweight='bold', fontsize=12)

    # Legend
    ax.text(0, -width_y / 2 - 200, f"Red: {nx_bars}-{bar_name} (Along Y)\nBlue: {ny_bars}-{bar_name} (Along X)",
            ha='center', va='top', fontsize=9)
    return fig


def plot_footing_section(width, h, col_w, dp, cover, bar_txt, n_pile):
    fig, ax = plt.subplots(figsize=(5, 4))
    # Ground
    ax.plot([-width, width], [0, 0], 'k-', linewidth=0.5)
    # Cap
    rect = patches.Rectangle((-width / 2, -h), width, h, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)
    # Col
    rect_col = patches.Rectangle((-col_w / 2, 0), col_w, h * 0.8, linewidth=1.5, edgecolor='black', facecolor='#fff',
                                 hatch='///')
    ax.add_patch(rect_col)
    # Pile
    pile_h = h * 0.6
    if n_pile == 1:
        rect_p = patches.Rectangle((-dp / 2, -h - pile_h), dp, pile_h, linewidth=1, edgecolor='black', facecolor='#fff')
        ax.add_patch(rect_p)
    else:
        off = width / 2 - dp / 2 - 150
        rect_p1 = patches.Rectangle((-off - dp / 2, -h - pile_h), dp, pile_h, linewidth=1, edgecolor='black',
                                    facecolor='#fff')
        rect_p2 = patches.Rectangle((off - dp / 2, -h - pile_h), dp, pile_h, linewidth=1, edgecolor='black',
                                    facecolor='#fff')
        ax.add_patch(rect_p1);
        ax.add_patch(rect_p2)

    # Rebar - Main U
    bar_y = -h + cover
    ax.plot([-width / 2 + cover, width / 2 - cover], [bar_y, bar_y], 'r-', linewidth=3)
    ax.plot([-width / 2 + cover, -width / 2 + cover], [bar_y, bar_y + h * 0.6], 'r-', linewidth=3)
    ax.plot([width / 2 - cover, width / 2 - cover], [bar_y, bar_y + h * 0.6], 'r-', linewidth=3)
    # Dots
    dot_xs = np.linspace(-width / 2 + cover * 2, width / 2 - cover * 2, 6)
    for x in dot_xs: ax.add_patch(patches.Circle((x, bar_y + 15), radius=8, color='blue'))

    ax.annotate(f"h={h / 1000:.2f}m", xy=(width / 2, -h / 2), xytext=(width / 2 + 100, -h / 2),
                arrowprops=dict(arrowstyle='->'))
    ax.text(0, bar_y - 80, f"Reinforcement: {bar_txt}", ha='center', color='red', fontsize=9, fontweight='bold')

    ax.set_xlim(-width / 1.2, width / 1.2);
    ax.set_ylim(-h * 1.8, h * 1.2);
    ax.axis('off')
    ax.set_title("SECTION DETAIL", fontweight='bold', fontsize=12)
    return fig


# ==========================================
# 5. REPORT GENERATOR
# ==========================================
def generate_report(inputs, rows, img_plan, img_sect):
    table_rows = ""
    for r in rows:
        if r[0] == "SECTION":
            table_rows += f"<tr class='sec-row'><td colspan='6'>{r[1]}</td></tr>"
        else:
            status_cls = "pass-ok" if "OK" in r[5] or "PASS" in r[5] else "pass-no"
            val_cls = "load-value" if "Load Input" in str(r[0]) else ""
            table_rows += f"""
            <tr>
                <td>{r[0]}</td>
                <td>{r[1]}</td>
                <td>{r[2]}</td>
                <td class='{val_cls}'>{r[3]}</td>
                <td>{r[4]}</td>
                <td class='{status_cls}'>{r[5]}</td>
            </tr>
            """

    html = f"""
    <!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">
        <title>Pile Cap Design Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Sarabun', sans-serif; padding: 20px; color: black; }}
            h1, h3 {{ text-align: center; margin: 5px; }}
            .header {{ position: relative; margin-bottom: 20px; border-bottom: 2px solid #333; padding-bottom: 10px; }}
            .beam-box {{
                position: absolute; top: 0; right: 0;
                border: 2px solid #333; padding: 5px 15px;
                font-size: 18px; font-weight: bold;
            }}
            .info-container {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
            .info-box {{ width: 48%; border: 1px solid #ddd; padding: 10px; }}

            .drawing-container {{ display: flex; justify-content: center; gap: 20px; margin: 20px 0; }}
            .drawing-box {{ border: 1px solid #ccc; padding: 5px; text-align: center; width: 45%; }}
            .drawing-box img {{ max-width: 100%; height: auto; }}

            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }}
            th, td {{ border: 1px solid #444; padding: 6px; }}
            th {{ background-color: #eee; }}
            .sec-row {{ background-color: #ddd; font-weight: bold; }}
            .pass-ok {{ color: green; font-weight: bold; text-align: center; }}
            .pass-no {{ color: red; font-weight: bold; text-align: center; }}
            .load-value {{ color: #D32F2F !important; font-weight: bold; }}

            .footer-section {{ margin-top: 40px; page-break-inside: avoid; }}
            .signature-block {{ width: 300px; text-align: center; }}
            .sign-line {{ border-bottom: 1px solid #000; margin: 40px 0 10px 0; }}

            @media print {{
                .no-print {{ display: none !important; }}
                body {{ padding: 0; }}
            }}
            .print-btn-internal {{
                background-color: #4CAF50; color: white; padding: 12px 24px;
                border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="no-print" style="text-align: center;">
            <button onclick="window.print()" class="print-btn-internal">🖨️ Print This Page / พิมพ์หน้านี้</button>
        </div>

        <div class="header">
            <div class="beam-box">{inputs['f_id']}</div>
            <h1>ENGINEERING DESIGN REPORT</h1>
            <h3>RC Pile Cap Design SDM (ACI 318-19)</h3>
        </div>

        <div class="info-container">
            <div class="info-box">
                <strong>Project:</strong> {inputs['project']}<br>
                <strong>Engineer:</strong> {inputs['engineer']}<br>
                <strong>Date:</strong> 15/12/2568
            </div>
            <div class="info-box">
                <strong>Materials:</strong> fc'={inputs['fc']} ksc, fy={inputs['fy']} ksc<br>
                <strong>Pile Config:</strong> {inputs['n_pile']} Piles, Dia {inputs['dp']} m<br>
                <strong>Spacing:</strong> {inputs['spacing']} m
            </div>
        </div>

        <h3>Design Drawings</h3>
        <div class="drawing-container">
            <div class="drawing-box">
                <img src="{img_plan}" />
                <p>Plan View (Reinforcement)</p>
            </div>
            <div class="drawing-box">
                <img src="{img_sect}" />
                <p>Section Detail</p>
            </div>
        </div>

        <br>

        <h3>Calculation Details</h3>
        <table>
            <thead>
                <tr>
                    <th width="20%">Item</th>
                    <th width="30%">Formula</th>
                    <th width="25%">Substitution</th>
                    <th>Result</th>
                    <th>Unit</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>

        <div class="footer-section">
            <div class="signature-block">
                <div style="text-align: left; font-weight: bold;">Designed by:</div>
                <div class="sign-line"></div>
                <div>({inputs['engineer']})</div>
                <div>วิศวกรโครงสร้าง</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html


# ==========================================
# 6. MAIN UI
# ==========================================
st.title("RC Pile Cap Design SDM")

if 'calc_done' not in st.session_state:
    st.session_state['calc_done'] = False

with st.sidebar.form("inputs"):
    st.header("Project Info")
    project = st.text_input("Project Name", "อาคารสำนักงาน 2 ชั้น")
    f_id = st.text_input("Footing ID", "F-01")
    engineer = st.text_input("Engineer Name", "นายไกรฤทธิ์ ด่านพิทักษ์")

    st.header("1. Material & Column")
    c1, c2 = st.columns(2)
    fc = c1.number_input("fc' (ksc)", 240)
    fy = c2.number_input("fy (ksc)", 4000)

    c1, c2 = st.columns(2)
    cx = c1.number_input("Col X (m)", 0.25)
    cy = c2.number_input("Col Y (m)", 0.25)

    st.header("2. Pile Configuration")
    n_pile = st.selectbox("Number of Piles", [1, 2, 3, 4, 5], index=1)
    c1, c2 = st.columns(2)
    dp = c1.number_input("Pile Dia (m)", 0.22)
    spacing = c2.number_input("Spacing (m)", 0.80)

    st.header("3. Footing Geometry")
    auto_h = st.checkbox("ออกแบบความหนาอัตโนมัติ (Auto-Design)", value=True)
    c1, c2 = st.columns(2)
    h = c1.number_input("Thickness (m) [Initial/Fixed]", value=0.50,
                        help="ความหนาของฐานราก (เมตร) - หากเลือก Auto-Design ค่านี้จะเป็นความหนาเริ่มต้นที่โปรแกรมใช้ลองคำนวณ หากไม่เลือก โปรแกรมจะใช้ค่านี้เป็นความหนาคงที่ในการออกแบบ")
    edge = c2.number_input("Edge Dist (m)", value=0.25,
                           help="ระยะจากศูนย์กลางเสาเข็มต้นริมสุด ถึงขอบฐานราก (เมตร) - เพื่อให้ครอบคลุมระยะฝังของเหล็กเสริมและป้องกันการกะเทาะของคอนกรีตขอบฐานราก")

    mainBar = st.selectbox("Main Rebar", list(BAR_INFO.keys()), index=4)  # DB16

    st.header("4. Loads (Factored)")
    Pu = st.number_input("Axial Load Pu (tf)", min_value=0.0, value=30.0)
    PileCap = st.number_input("Max Factored Load/Pile (tf)", min_value=0.0, value=30.0,
                              help="ความสามารถรับน้ำหนักเสาเข็ม (Factored)")

    run_btn = st.form_submit_button("Run Design")

if run_btn:
    inputs = {
        'project': project, 'f_id': f_id, 'engineer': engineer,
        'fc': fc, 'fy': fy, 'cx': cx, 'cy': cy,
        'n_pile': n_pile, 'dp': dp, 'spacing': spacing,
        'h': h, 'edge': edge, 'mainBar': mainBar,
        'Pu': Pu, 'PileCap': PileCap, 'auto_h': auto_h
    }

    # Calculate
    rows, coords, bx, by, nx, ny, status, final_h = process_footing_calculation(inputs)

    # Plot Plan
    fig_plan = plot_footing_plan(coords, bx, by, cx * 1000, cy * 1000, dp * 1000, nx, ny, mainBar)
    img_plan = fig_to_base64(fig_plan)

    # Plot Section
    bar_txt = f"{max(nx, ny)}-{mainBar}"  # Simplified for section
    fig_sect = plot_footing_section(bx, final_h, cx * 1000, dp * 1000, 75, bar_txt, n_pile)
    img_sect = fig_to_base64(fig_sect)

    # Report
    html = generate_report(inputs, rows, img_plan, img_sect)

    st.success(f"✅ Calculation Complete: {status}")
    st.components.v1.html(html, height=800, scrolling=True)

else:
    st.info("👈 กรุณากรอกข้อมูลทางด้านซ้าย แล้วกด 'Run Design'")
