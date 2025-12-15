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
    """Return list of (x, y) tuples"""
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


def process_footing_calculation(inputs):
    rows = []

    def sec(title):
        rows.append(["SECTION", title, "", "", "", "", ""])

    def row(item, formula, subs, result, unit, status=""):
        rows.append([item, formula, subs, result, unit, status])

    # Inputs
    fc = inputs['fc'] * 0.0980665  # MPa
    fy = inputs['fy'] * 0.0980665
    Pu_tf = inputs['Pu']
    Pu_N = Pu_tf * 9806.65
    PileCap_tf = inputs['PileCap']

    n_pile = int(inputs['n_pile'])
    s = inputs['spacing'] * 1000  # mm
    edge = inputs['edge'] * 1000
    col_x = inputs['cx'] * 1000
    col_y = inputs['cy'] * 1000
    h_cap = inputs['h'] * 1000
    dp = inputs['dp'] * 1000
    cover = 75.0

    bar_key = inputs['mainBar']
    db = BAR_INFO[bar_key]['d_mm']
    d = h_cap - cover - db

    # Geometry Calculation
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

    # --- 1. GEOMETRY & MATERIALS ---
    sec("1. GEOMETRY & MATERIALS")
    row("Materials", "fc', fy", f"{fmt(fc, 2)}, {fmt(fy, 0)}", "-", "MPa")
    row("Pile Cap Size", "B x L x h", f"{fmt(width_x, 0)} x {fmt(width_y, 0)} x {fmt(h_cap, 0)}", "-", "mm")
    row("Effective Depth", "d = h - cover - db", f"{h_cap:.0f} - {cover} - {db}", f"{d:.0f}", "mm")
    row("Pile Config", f"{n_pile} Piles, Dia {dp / 1000:.2f}m", f"Spacing {s:.0f} mm", "-", "-")

    # --- 2. PILE REACTION ---
    sec("2. PILE REACTION CHECK")
    P_avg_tf = Pu_tf / n_pile
    P_avg_N = (Pu_tf * 9806.65) / n_pile

    row("Load Input (Pu)", "-", "-", f"{fmt(Pu_tf, 3)}", "tf", "")
    row("Pile Reaction (Ru)", "Pu / N", f"{fmt(Pu_tf, 2)} / {n_pile}", f"{fmt(P_avg_tf, 2)}", "tf")

    status_pile = "PASS" if P_avg_tf <= PileCap_tf else "FAIL"
    row("Capacity Check", "Ru ≤ P_pile(max)", f"{fmt(P_avg_tf, 2)} ≤ {fmt(PileCap_tf, 2)}", status_pile, "-",
        status_pile)

    # --- 3. ONE PILE (Special Case) ---
    if n_pile == 1:
        sec("3. REINFORCEMENT (F1)")
        row("Note", "-", "Single Pile Cap", "-", "-", "INFO")

        # Min Steel (Temp & Shrinkage)
        As_min = 0.0018 * width_x * h_cap
        row("As,min (Temp)", "0.0018 · B · h", f"0.0018 · {width_x:.0f} · {h_cap:.0f}", f"{fmt(As_min, 0)}", "mm²")

        bar_area = BAR_INFO[bar_key]['A_cm2'] * 100
        n_bars = math.ceil(As_min / bar_area)
        if n_bars < 4: n_bars = 4  # Cage min
        As_prov = n_bars * bar_area

        row("Provide Steel", f"Use {bar_key}", f"Req: {fmt(As_min, 0)}", f"{n_bars}-{bar_key}", "mm²", "OK")
        row("As Provided", f"{n_bars} x {bar_area:.0f}", "-", f"{fmt(As_prov, 0)}", "mm²")

        return rows, coords, width_x, width_y, n_bars, "PASS"

    # --- 4. PUNCHING SHEAR (N > 1) ---
    sec("3. PUNCHING SHEAR (at d/2 from Col)")
    c1 = col_x + d;
    c2 = col_y + d
    bo = 2 * (c1 + c2)

    # Calculate Vu
    Vu_punch_N = 0
    piles_out = 0
    for (px, py) in coords:
        if (abs(px) > c1 / 2) or (abs(py) > c2 / 2):
            Vu_punch_N += P_avg_N
            piles_out += 1

    Vu_punch_tf = Vu_punch_N / 9806.65

    # Capacity
    phi_v = 0.75
    # Vc = 0.33 * sqrt(fc) * bo * d
    Vc_punch_N = 0.33 * math.sqrt(fc) * bo * d
    phiVc_punch_N = phi_v * Vc_punch_N
    phiVc_punch_tf = phiVc_punch_N / 9806.65

    row("Critical Perimeter", "bo = 2(cx+d + cy+d)", f"2({c1:.0f} + {c2:.0f})", f"{bo:.0f}", "mm")
    row("Vu (Punching)", f"Sum({piles_out} Piles Outside)", f"{piles_out} x {fmt(P_avg_tf, 2)}",
        f"{fmt(Vu_punch_tf, 2)}", "tf")
    row("Capacity φVc", "0.75·0.33√fc'·bo·d", f"0.75·0.33·{math.sqrt(fc):.2f}·{bo:.0f}·{d:.0f}",
        f"{fmt(phiVc_punch_tf, 2)}", "tf")

    status_punch = "PASS" if Vu_punch_N <= phiVc_punch_N else "FAIL"
    row("Check Punching", "Vu ≤ φVc", f"{fmt(Vu_punch_tf, 2)} ≤ {fmt(phiVc_punch_tf, 2)}", status_punch, "-",
        status_punch)

    # --- 5. BEAM SHEAR (One-Way) ---
    sec("4. BEAM SHEAR (One-Way at d)")
    dist_crit = col_x / 2 + d

    Vu_beam_N = 0
    piles_beam = 0
    for (px, py) in coords:
        if abs(px) > dist_crit:
            Vu_beam_N += P_avg_N
            piles_beam += 1

    Vu_beam_tf = Vu_beam_N / 9806.65

    # Capacity Vc = 0.17 * sqrt(fc) * B * d
    # width B here depends on direction check (take width_y for checking X direction shear)
    Vc_beam_N = 0.17 * math.sqrt(fc) * width_y * d
    phiVc_beam_N = phi_v * Vc_beam_N
    phiVc_beam_tf = phiVc_beam_N / 9806.65

    row("Critical Section", "from center = c/2 + d", f"{col_x / 2:.0f} + {d:.0f}", f"{dist_crit:.0f}", "mm")
    row("Vu (One-Way)", f"Sum({piles_beam} Piles Outside)", f"{piles_beam} x {fmt(P_avg_tf, 2)}",
        f"{fmt(Vu_beam_tf, 2)}", "tf")
    row("Capacity φVc", "0.75·0.17√fc'·B·d", f"0.75·0.17·{math.sqrt(fc):.2f}·{width_y:.0f}·{d:.0f}",
        f"{fmt(phiVc_beam_tf, 2)}", "tf")

    status_beam = "PASS" if Vu_beam_N <= phiVc_beam_N else "FAIL"
    row("Check Beam Shear", "Vu ≤ φVc", f"{fmt(Vu_beam_tf, 2)} ≤ {fmt(phiVc_beam_tf, 2)}", status_beam, "-",
        status_beam)

    # --- 6. FLEXURE ---
    sec("5. FLEXURAL DESIGN")
    face_dist = col_x / 2
    Mu_calc_Nmm = 0

    # Calculate Moment
    calc_str = ""
    for (px, py) in coords:
        lever = abs(px) - face_dist
        if lever > 0:
            Mu_calc_Nmm += P_avg_N * lever
            calc_str += f"+ P·{lever:.0f} "

    Mu_calc_tfm = Mu_calc_Nmm / 9806650.0

    if calc_str == "": calc_str = "No Tension"

    row("Design Moment Mu", "Σ P_pile · (x - c/2)", "Sum moments at face", f"{fmt(Mu_calc_tfm, 2)}", "tf-m")

    # Steel Calculation
    phi_f = 0.9
    req_As = 0
    if Mu_calc_Nmm > 0:
        req_As = Mu_calc_Nmm / (phi_f * fy * 0.9 * d)
        row("As (Strength)", "Mu / (φ·fy·0.9d)", f"{fmt(Mu_calc_tfm, 2)}e7 / (0.9·{fy:.0f}·0.9·{d:.0f})",
            f"{fmt(req_As, 0)}", "mm²")
    else:
        row("As (Strength)", "-", "Mu approx 0", "0", "mm²")

    As_min = 0.0018 * width_y * h_cap
    row("As (Min)", "0.0018 · B · h", f"0.0018 · {width_y:.0f} · {h_cap:.0f}", f"{fmt(As_min, 0)}", "mm²")

    As_design = max(req_As, As_min)

    # Provide
    bar_area = BAR_INFO[bar_key]['A_cm2'] * 100
    n_bars = math.ceil(As_design / bar_area)
    As_prov = n_bars * bar_area
    s_prov = (width_y - 2 * cover) / n_bars

    row("Provide Steel", f"Use {bar_key}", f"Req: {fmt(As_design, 0)}", f"{n_bars}-{bar_key}", "mm²", "OK")
    row("As Provided", f"{n_bars} x {bar_area:.0f}", "-", f"{fmt(As_prov, 0)}", "mm²")
    row("Spacing", "-", "-", f"@{s_prov / 10:.0f} cm", "-", "")

    sec("6. FINAL STATUS")
    overall = "OK" if (status_pile == "PASS" and status_punch == "PASS" and status_beam == "PASS") else "NOT OK"
    row("Overall", "-", "-", "DESIGN COMPLETE", "-", overall)

    return rows, coords, width_x, width_y, n_bars, overall


# ==========================================
# 4. PLOTTING (Plan & Section)
# ==========================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def plot_footing_plan(coords, width_x, width_y, col_x, col_y, dp, bar_txt):
    fig, ax = plt.subplots(figsize=(5, 5))
    rect = patches.Rectangle((-width_x / 2, -width_y / 2), width_x, width_y, linewidth=2, edgecolor='black',
                             facecolor='#f9f9f9')
    ax.add_patch(rect)
    for i in np.linspace(-width_x / 2 + 75, width_x / 2 - 75, 5):
        ax.plot([i, i], [-width_y / 2 + 75, width_y / 2 - 75], 'r-', alpha=0.3, linewidth=1)
    for i in np.linspace(-width_y / 2 + 75, width_y / 2 - 75, 5):
        ax.plot([-width_x / 2 + 75, width_x / 2 - 75], [i, i], 'r-', alpha=0.3, linewidth=1)
    rect_col = patches.Rectangle((-col_x / 2, -col_y / 2), col_x, col_y, linewidth=1.5, edgecolor='#333',
                                 facecolor='#ddd', hatch='//')
    ax.add_patch(rect_col)
    for (px, py) in coords:
        circle = patches.Circle((px, py), radius=dp / 2, edgecolor='black', facecolor='white', linewidth=1.5,
                                linestyle='--')
        ax.add_patch(circle)
        ax.text(px, py, "P", ha='center', va='center', fontsize=8)
    ax.set_xlim(-width_x / 1.4, width_x / 1.4);
    ax.set_ylim(-width_y / 1.4, width_y / 1.4)
    ax.set_aspect('equal');
    ax.axis('off')
    ax.set_title("PLAN VIEW", fontweight='bold', fontsize=12)
    ax.text(0, -width_y / 2 - 100, f"Size: {width_x / 1000:.2f}x{width_y / 1000:.2f}m", ha='center', va='top')
    return fig


def plot_footing_section(width, h, col_w, dp, cover, bar_txt, n_pile):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([-width, width], [0, 0], 'k-', linewidth=0.5)
    rect = patches.Rectangle((-width / 2, -h), width, h, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)
    rect_col = patches.Rectangle((-col_w / 2, 0), col_w, h * 0.8, linewidth=1.5, edgecolor='black', facecolor='#fff',
                                 hatch='///')
    ax.add_patch(rect_col)
    ax.plot([-col_w / 2, -col_w / 2], [h * 0.8, h * 1.0], 'k--', linewidth=0.5)
    ax.plot([col_w / 2, col_w / 2], [h * 0.8, h * 1.0], 'k--', linewidth=0.5)
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
    bar_y = -h + cover
    ax.plot([-width / 2 + cover, width / 2 - cover], [bar_y, bar_y], 'r-', linewidth=3)
    ax.plot([-width / 2 + cover, -width / 2 + cover], [bar_y, bar_y + h * 0.6], 'r-', linewidth=3)
    ax.plot([width / 2 - cover, width / 2 - cover], [bar_y, bar_y + h * 0.6], 'r-', linewidth=3)
    dot_xs = np.linspace(-width / 2 + cover * 2, width / 2 - cover * 2, 6)
    for x in dot_xs: ax.add_patch(patches.Circle((x, bar_y + 15), radius=8, color='blue'))
    top_y = -cover
    ax.plot([-width / 2 + cover, width / 2 - cover], [top_y, top_y], 'b-', linewidth=1.5)
    ax.annotate(f"h={h / 1000:.2f}m", xy=(width / 2, -h / 2), xytext=(width / 2 + 100, -h / 2),
                arrowprops=dict(arrowstyle='->'))
    ax.text(0, bar_y - 80, f"Main: {bar_txt}", ha='center', color='red', fontsize=9, fontweight='bold')
    ax.text(0, -h / 2, "Concrete Cover 7.5cm", ha='center', fontsize=8, alpha=0.6)
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
                <p>Plan View</p>
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
    # Added '1' for Single Pile
    n_pile = st.selectbox("Number of Piles", [1, 2, 3, 4, 5], index=1)
    c1, c2 = st.columns(2)
    dp = c1.number_input("Pile Dia (m)", 0.22)
    spacing = c2.number_input("Spacing (m)", 0.80)

    st.header("3. Footing Geometry")
    c1, c2 = st.columns(2)
    h = c1.number_input("Thickness (m)", 0.50)
    edge = c2.number_input("Edge Dist (m)", 0.25)
    mainBar = st.selectbox("Main Rebar", list(BAR_INFO.keys()), index=4)  # DB16

    st.header("4. Loads (Factored)")
    Pu = st.number_input("Axial Load Pu (tf)", 30.0)
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
    rows, coords, bx, by, n_bars, status = process_footing_calculation(inputs)

    # Plot Plan
    fig_plan = plot_footing_plan(coords, bx, by, cx * 1000, cy * 1000, dp * 1000, f"{n_bars}-{mainBar}")
    img_plan = fig_to_base64(fig_plan)

    # Plot Section
    fig_sect = plot_footing_section(bx, h * 1000, cx * 1000, dp * 1000, 75, f"{n_bars}-{mainBar}", n_pile)
    img_sect = fig_to_base64(fig_sect)

    # Report
    html = generate_report(inputs, rows, img_plan, img_sect)

    st.success(f"✅ Calculation Complete: {status}")
    st.components.v1.html(html, height=800, scrolling=True)

else:
    st.info("👈 กรุณากรอกข้อมูลทางด้านซ้าย แล้วกด 'Run Design'")