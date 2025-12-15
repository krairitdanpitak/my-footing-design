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
    /* CSS ปุ่มพิมพ์ */
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

    /* CSS ตาราง */
    .report-table {width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px;}
    .report-table th, .report-table td {border: 1px solid #ddd; padding: 8px;}
    .report-table th {background-color: #f2f2f2; text-align: center; font-weight: bold;}

    .pass-ok {color: green; font-weight: bold;}
    .pass-no {color: red; font-weight: bold;}
    .sec-row {background-color: #e0e0e0; font-weight: bold; font-size: 15px;}
    .load-value {color: #D32F2F !important; font-weight: bold;}
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
# 3. CALCULATION LOGIC (PILE CAP)
# ==========================================
def get_pile_coordinates(n_pile, s):
    """Return list of (x, y) tuples for pile positions relative to center"""
    coords = []
    if n_pile == 2:
        coords = [(-s / 2, 0), (s / 2, 0)]
    elif n_pile == 3:
        h_tri = s * math.sqrt(3) / 2
        # Centroid is at h/3 from base
        coords = [(-s / 2, -h_tri / 3), (s / 2, -h_tri / 3), (0, 2 * h_tri / 3)]
    elif n_pile == 4:
        coords = [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2)]
    elif n_pile == 5:
        coords = [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2), (0, 0)]
    return coords


def process_footing_calculation(inputs):
    rows = []

    def sec(title):
        rows.append(["SECTION", title, "", "", "", "", ""])

    def row(item, formula, subs, result, unit, status=""):
        rows.append([item, formula, subs, result, unit, status])

    # --- 1. Inputs ---
    fc = inputs['fc'] * 0.0980665  # MPa
    fy = inputs['fy'] * 0.0980665
    Pu_tf = inputs['Pu']
    Pu_N = Pu_tf * 9806.65
    PileCap_tf = inputs[
        'PileCap']  # Capacity per pile (Service? usually check factored in SDM, but let's assume input is Safe Load x FS or Factored Capacity. Here assume Safe Load for checking number of piles, usually strictly we check Pu/N <= Q_all_factored)
    # Standard practice: Check Pu <= Phi * Pn_pile. Here we simplify: Pu/N <= P_pile_design

    n_pile = int(inputs['n_pile'])
    s = inputs['spacing'] * 1000  # m to mm
    edge = inputs['edge'] * 1000
    col_x = inputs['cx'] * 1000
    col_y = inputs['cy'] * 1000
    h_cap = inputs['h'] * 1000
    cover = 75.0  # mm (Standard for footing against earth)

    bar_key = inputs['mainBar']
    db = BAR_INFO[bar_key]['d_mm']
    d = h_cap - cover - db  # Effective depth

    # Pile Size (for Punching)
    dp = inputs['dp'] * 1000

    # Generate Geometry
    coords = get_pile_coordinates(n_pile, s)

    # Calculate Cap Dimensions (Box bounding)
    xs = [c[0] for c in coords];
    ys = [c[1] for c in coords]
    width_x = (max(xs) - min(xs)) + dp + 2 * edge  # Simplified logic
    width_y = (max(ys) - min(ys)) + dp + 2 * edge

    # Override for standard shapes
    if n_pile == 3:  # Triangle shape approx
        width_x = s + dp + 2 * edge
        width_y = (s * math.sqrt(3) / 2) + dp + 2 * edge  # Approx height

    sec("1. GEOMETRY & MATERIALS")
    row("Materials", "fc', fy", f"{fmt(fc, 2)}, {fmt(fy, 0)}", "-", "MPa")
    row("Footing Size", "B x L x h", f"{fmt(width_x, 0)}x{fmt(width_y, 0)}x{fmt(h_cap, 0)}", "-", "mm")
    row("Effective Depth", "d = h - cover - db", f"{h_cap:.0f} - {cover} - {db}", f"{d:.0f}", "mm")
    row("Pile Config", f"{n_pile} Piles", f"Spacing {s:.0f} mm", "-", "-")

    # --- 2. PILE REACTION CHECK ---
    sec("2. PILE REACTION CHECK")
    # For simplicity, assume concentric load Pu equally distributed
    P_avg_N = Pu_N / n_pile
    P_avg_tf = P_avg_N / 9806.65

    row("Load Input (Pu)", "-", "-", f"{fmt(Pu_tf, 3)}", "tf", "", )

    # Capacity Check
    # Assuming input 'Pile Capacity' is Safe Load (Allowable).
    # For SDM, we compare Factored Load to Factored Resistance.
    # Often in practice: P_pile_service <= Q_allowable.
    # Here input is Pu (Factored). We need P_pile_factored_cap.
    # Let's assume User inputs "Safe Load" and we treat check as Pu_pile <= Q_safe * FS?
    # OR User inputs "Factored Capacity". Let's assume User inputs Safe Load and we check Service Load...
    # BUT this app is SDM (Factored inputs).
    # ADJUSTMENT: Check P_avg <= P_pile_max_input. User should input Factored Capacity of Pile or we warn.
    # Let's label input as "Max Factored Load per Pile".

    status_pile = "PASS" if P_avg_tf <= PileCap_tf else "FAIL"
    row("Pile Reaction (Ru)", "Pu / N", f"{fmt(Pu_tf, 2)} / {n_pile}", f"{fmt(P_avg_tf, 2)}", "tf")
    row("Check Capacity", "Ru ≤ P_pile(max)", f"{fmt(P_avg_tf, 2)} ≤ {fmt(PileCap_tf, 2)}", status_pile, "-",
        status_pile)

    # --- 3. PUNCHING SHEAR (Two-Way) ---
    sec("3. PUNCHING SHEAR (at d/2)")
    # Critical perimeter b0 around column
    # Rectangular critical section: (cx+d) * (cy+d)
    c1 = col_x + d;
    c2 = col_y + d
    bo = 2 * (c1 + c2)

    # Calculate Vu_punch: Sum of piles OUTSIDE the critical perimeter
    # Distance check from center
    # Critical boundary limits: [-c1/2, c1/2] and [-c2/2, c2/2]
    Vu_punch_N = 0
    piles_out = 0
    for (px, py) in coords:
        # Check if pile center is outside
        # Simple check: if pile center is strictly outside critical rect
        is_outside = (abs(px) > c1 / 2) or (abs(py) > c2 / 2)
        if is_outside:
            Vu_punch_N += P_avg_N
            piles_out += 1

    # If standard 2-pile or closely spaced, shear might be different, but using general ACI logic:
    Vu_punch_tf = Vu_punch_N / 9806.65

    # Capacity phi*Vc
    # phi = 0.75
    # Vc = 0.33 * sqrt(fc) * bo * d (simplified ACI for column)
    phi_v = 0.75
    Vc_punch_N = 0.33 * math.sqrt(fc) * bo * d
    phiVc_punch_N = phi_v * Vc_punch_N
    phiVc_punch_tf = phiVc_punch_N / 9806.65

    row("Critical Perimeter", "bo = 2(c1+d + c2+d)", f"2({c1:.0f}+{c2:.0f})", f"{bo:.0f}", "mm")
    row("Vu (Punching)", f"Sum Piles Outside ({piles_out})", "-", f"{fmt(Vu_punch_tf, 2)}", "tf")

    status_punch = "PASS" if Vu_punch_N <= phiVc_punch_N else "FAIL"
    row("Capacity φVc", "0.75·0.33√fc'·bo·d", f"0.75·0.33√{fc:.1f}·{bo:.0f}·{d:.0f}", f"{fmt(phiVc_punch_tf, 2)}", "tf")
    row("Check Punching", "Vu ≤ φVc", f"{fmt(Vu_punch_tf, 2)} ≤ {fmt(phiVc_punch_tf, 2)}", status_punch, "-",
        status_punch)

    # --- 4. ONE-WAY SHEAR (Beam Action) ---
    sec("4. BEAM SHEAR (One-Way at d)")
    # Check Critical Section at d from column face
    # Distance from center = col_x/2 + d
    dist_crit = col_x / 2 + d

    # Sum piles beyond this distance
    Vu_beam_N = 0
    piles_beam = 0
    for (px, py) in coords:
        if abs(px) > dist_crit:  # Check X direction mainly
            Vu_beam_N += P_avg_N
            piles_beam += 1

    Vu_beam_tf = Vu_beam_N / 9806.65

    # Capacity
    # Vc = 0.17 * sqrt(fc) * bw * d
    # bw is width of footing
    Vc_beam_N = 0.17 * math.sqrt(fc) * width_y * d
    phiVc_beam_N = phi_v * Vc_beam_N
    phiVc_beam_tf = phiVc_beam_N / 9806.65

    row("Critical Section", "from center = c/2 + d", f"{col_x / 2:.0f} + {d:.0f}", f"{dist_crit:.0f}", "mm")
    row("Vu (One-Way)", f"Sum Piles Outside ({piles_beam})", "-", f"{fmt(Vu_beam_tf, 2)}", "tf")
    row("Capacity φVc", "0.75·0.17√fc'·B·d", f"0.75·0.17...·{width_y:.0f}·{d:.0f}", f"{fmt(phiVc_beam_tf, 2)}", "tf")

    status_beam = "PASS" if Vu_beam_N <= phiVc_beam_N else "FAIL"
    row("Check Beam Shear", "Vu ≤ φVc", "-", status_beam, "-", status_beam)

    # --- 5. FLEXURE DESIGN ---
    sec("5. FLEXURAL DESIGN")
    # Critical section at column face
    face_dist = col_x / 2

    # Calculate Moment from piles
    Mu_calc_Nmm = 0
    for (px, py) in coords:
        lever_arm = abs(px) - face_dist
        if lever_arm > 0:
            Mu_calc_Nmm += P_avg_N * lever_arm

    Mu_calc_tfm = Mu_calc_Nmm / 9806650.0
    row("Design Moment Mu", "Σ P_pile · (x - c/2)", f"At Col Face", f"{fmt(Mu_calc_tfm, 2)}", "tf-m")

    # Calculate Steel
    # As approx = Mu / (0.9 * fy * 0.9d)
    phi_f = 0.9
    # Solve exact a
    # Mn = As fy (d - a/2) -> a = As fy / 0.85 fc b
    # Iterative or quadratic. Let's use standard approx for footing

    req_As = 0
    if Mu_calc_Nmm > 0:
        # Simple formula Mu = phi * As * fy * (d - a/2)
        # Assume j = 0.9
        req_As = Mu_calc_Nmm / (phi_f * fy * 0.9 * d)

    # Check Min Steel (Temp & Shrinkage usually 0.0018 b h)
    As_min = 0.0018 * width_y * h_cap

    As_design = max(req_As, As_min)

    row("As (Strength)", "Mu / (φ·fy·0.9d)", "-", f"{fmt(req_As, 0)}", "mm²")
    row("As (Min)", "0.0018 · B · h", f"0.0018·{width_y:.0f}·{h_cap:.0f}", f"{fmt(As_min, 0)}", "mm²")

    # Provide Bars
    bar_area = BAR_INFO[bar_key]['A_cm2'] * 100
    num_bars = math.ceil(As_design / bar_area)
    # Check spacing
    spacing_prov = (width_y - 2 * cover) / num_bars

    row("Provide Steel", f"Use {bar_key}", f"Req: {fmt(As_design, 0)}", f"{num_bars}-{bar_key}", "mm²", "OK")
    row("Spacing", "-", "-", f"@{spacing_prov / 10:.0f} cm", "-", "")

    sec("6. FINAL STATUS")
    overall = "OK" if (status_pile == "PASS" and status_punch == "PASS" and status_beam == "PASS") else "NOT OK"
    row("Overall", "-", "-", "DESIGN COMPLETE", "-", overall)

    return rows, coords, width_x, width_y, num_bars


# ==========================================
# 4. PLOTTING
# ==========================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def plot_footing(coords, width_x, width_y, col_x, col_y, dp, bar_txt):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Footing Outline
    rect = patches.Rectangle((-width_x / 2, -width_y / 2), width_x, width_y, linewidth=2, edgecolor='black',
                             facecolor='#f0f0f0')
    ax.add_patch(rect)

    # Column
    rect_col = patches.Rectangle((-col_x / 2, -col_y / 2), col_x, col_y, linewidth=1, edgecolor='black',
                                 facecolor='#bbb', hatch='//')
    ax.add_patch(rect_col)

    # Piles
    for (px, py) in coords:
        circle = patches.Circle((px, py), radius=dp / 2, edgecolor='black', facecolor='white', linewidth=1.5)
        ax.add_patch(circle)
        ax.plot(px, py, 'k+', markersize=5)

    ax.set_xlim(-width_x / 1.5, width_x / 1.5)
    ax.set_ylim(-width_y / 1.5, width_y / 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Pile Cap Plan View", fontweight='bold')

    # Info Text
    info = f"Size: {width_x / 1000:.2f} x {width_y / 1000:.2f} m\nRebar: {bar_txt}"
    ax.text(0, -width_y / 2 - 200, info, ha='center', va='top', fontsize=10)

    return fig


# ==========================================
# 5. REPORT
# ==========================================
def generate_report(inputs, rows, img_base64):
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

            .images {{ text-align: center; margin: 20px 0; }}
            .images img {{ width: 50%; border: 1px solid #ddd; padding: 10px; }}

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
                <strong>Config:</strong> {inputs['n_pile']} Piles, Dia {inputs['dp']} m<br>
                <strong>Spacing:</strong> {inputs['spacing']} m
            </div>
        </div>

        <h3>Design Summary</h3>
        <div class="images">
            <img src="{img_base64}" />
        </div>

        <br><br><br>

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
    n_pile = st.selectbox("Number of Piles", [2, 3, 4, 5], index=2)
    c1, c2 = st.columns(2)
    dp = c1.number_input("Pile Dia (m)", 0.22)
    spacing = c2.number_input("Spacing (m)", 0.80)

    st.header("3. Footing Geometry")
    c1, c2 = st.columns(2)
    h = c1.number_input("Thickness (m)", 0.50)
    edge = c2.number_input("Edge Dist (m)", 0.25)
    mainBar = st.selectbox("Main Rebar", list(BAR_INFO.keys()), index=4)  # DB16

    st.header("4. Loads (Factored)")
    Pu = st.number_input("Axial Load Pu (tf)", 60.0)
    PileCap = st.number_input("Max Factored Load/Pile (tf)", 30.0, help="ความสามารถรับน้ำหนักเสาเข็ม (Factored)")

    run_btn = st.form_submit_button("Run Design")

if run_btn:
    inputs = {
        'project': project, 'f_id': f_id, 'engineer': engineer,
        'fc': fc, 'fy': fy, 'cx': cx, 'cy': cy,
        'n_pile': n_pile, 'dp': dp, 'spacing': spacing,
        'h': h, 'edge': edge, 'mainBar': mainBar,
        'Pu': Pu, 'PileCap': PileCap
    }

    # Calculate
    rows, coords, bx, by, n_bars = process_footing_calculation(inputs)

    # Plot
    fig = plot_footing(coords, bx, by, cx * 1000, cy * 1000, dp * 1000, f"{n_bars}-{mainBar}")
    img = fig_to_base64(fig)

    # Report
    html = generate_report(inputs, rows, img)

    st.success("✅ Calculation Complete")
    st.components.v1.html(html, height=800, scrolling=True)

else:
    st.info("👈 กรุณากรอกข้อมูลทางด้านซ้าย แล้วกด 'Run Design'")
