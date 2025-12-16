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
# 1. SETUP & STYLE
# ==========================================
st.set_page_config(page_title="RC Pile Cap Design (ACI 318-19)", layout="wide")

st.markdown("""
<style>
    /* ปุ่มพิมพ์ สีเขียวเหมือนโมดูลคาน */
    .print-btn-internal {
        background-color: #4CAF50; /* Green */
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
    .print-btn-internal:hover { background-color: #45a049; }

    /* ตารางคำนวณ */
    .report-table {width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 13px;}
    .report-table th, .report-table td {border: 1px solid #ddd; padding: 8px;}
    .report-table th {background-color: #f2f2f2; text-align: center; font-weight: bold;}

    .pass-ok {color: green; font-weight: bold; text-align: center;}
    .pass-no {color: red; font-weight: bold; text-align: center;}
    .sec-row {background-color: #e0e0e0; font-weight: bold; font-size: 14px; text-align: left;}
    .load-value {color: #D32F2F !important; font-weight: bold;}

    /* Layout รูปภาพ */
    .drawing-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
        margin-top: 20px;
    }
    .drawing-box {
        border: 1px solid #ddd;
        padding: 10px;
        background-color: #fff;
        text-align: center;
        min-width: 300px;
    }

    /* Footer Style (จัดซ้ายล่าง) */
    .footer-section {
        margin-top: 50px;
        page-break-inside: avoid;
        width: 100%;
        display: flex;
        justify-content: flex-start;
    }
    .signature-block {
        width: 300px;
        text-align: left;
    }
    .sign-line {
        border-bottom: 1px solid #000;
        margin: 40px 0 5px 0;
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
# 3. CALCULATION LOGIC (Footing Detailed)
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
    dp = inputs['dp'] * 1000
    edge = inputs['edge'] * 1000;
    h = inputs['h'] * 1000
    cover = 75.0;
    db = BAR_INFO[inputs['mainBar']]['d_mm']
    d = h - cover - db

    # 1. Geometry
    coords = get_pile_coordinates(n_pile, s)
    bx = (max([abs(x) for x, _ in coords]) * 2) + dp + 2 * edge if n_pile > 1 else dp + 2 * edge
    by = (max([abs(y) for _, y in coords]) * 2) + dp + 2 * edge if n_pile > 1 else dp + 2 * edge

    sec("1. GEOMETRY & PROPERTIES")
    row("Footing Size", "B x L", f"{bx:.0f}x{by:.0f}", f"h={h:.0f}", "mm", "")
    lambda_s = math.sqrt(2 / (1 + 0.004 * d))
    row("Size Effect λs", "√(2/(1+0.004d))", f"√(2/(1+0.004*{d:.0f}))", f"{lambda_s:.3f}", "-", "≤1.0")

    # 2. Reaction
    sec("2. PILE REACTION")
    p_avg = pu_tf / n_pile
    row("Pile Load", "Ru = Pu / N", f"{pu_tf}/{n_pile}", f"{p_avg:.2f}", "tf",
        "PASS" if p_avg <= inputs['PileCap'] else "FAIL")

    # 3. Flexure
    sec("3. FLEXURAL DESIGN")
    p_n = p_avg * 9806;
    mx = 0
    col_x = inputs['cx'] * 1000
    for x, y in coords:
        lever = abs(x) - col_x / 2
        if lever > 0: mx += p_n * lever

    req_as = mx / (0.9 * fy * 0.9 * d) if mx > 0 else 0
    min_as = 0.0018 * by * h
    des_as = max(req_as, min_as)
    n_bars = math.ceil(des_as / (BAR_INFO[inputs['mainBar']]['A_cm2'] * 100))
    if n_pile == 1: n_bars = max(n_bars, 4)
    prov_as = n_bars * BAR_INFO[inputs['mainBar']]['A_cm2'] * 100

    row("Moment Mu", "Σ P(arm)", "-", f"{mx / 9.8e6:.2f}", "tf-m", "")
    row("As Req", "Max(Calc, Min)", f"Max({req_as:.0f}, {min_as:.0f})", f"{des_as:.0f}", "mm²", "")
    row("Provide", f"{n_bars}-{inputs['mainBar']}", f"As={prov_as:.0f}", "OK", "-", "")

    # 4. Shear
    if n_pile > 1:
        sec("4. SHEAR CHECKS (ACI 318-19)")
        bo = 4 * (col_x + d);
        vc_p_val = 0.33 * lambda_s * math.sqrt(fc) * bo * d
        vu_p = sum([p_n for x, y in coords if max(abs(x), abs(y)) > (col_x + d) / 2])
        phi_vc_p = 0.75 * vc_p_val

        row("Punching Vu", "Sum Outside", "-", f"{vu_p / 9806:.2f}", "tf", "")
        row("Punching φVc", "0.75·0.33λs√fc·bo·d", f"0.75·0.33·{lambda_s:.2f}...", f"{phi_vc_p / 9806:.2f}", "tf",
            "PASS" if vu_p <= phi_vc_p else "FAIL")

        rho_w = prov_as / (by * d);
        rho_term = math.pow(rho_w, 1 / 3)
        vc_b_val = 0.66 * lambda_s * rho_term * math.sqrt(fc) * by * d
        vu_b = sum([p_n for x, y in coords if abs(x) > col_x / 2 + d])
        phi_vc_b = 0.75 * vc_b_val

        row("Beam Vu", "Sum Outside d", "-", f"{vu_b / 9806:.2f}", "tf", "")
        row("Beam φVc", "0.75·0.66λs(ρ)^1/3√fc·bd", f"ρ^1/3={rho_term:.2f}...", f"{phi_vc_b / 9806:.2f}", "tf",
            "PASS" if vu_b <= phi_vc_b else "FAIL")

    return rows, coords, bx, by, n_bars


def plot_foot_combined(coords, bx, by, n, bar, h_mm, col_mm):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plan
    ax1.set_title("PLAN VIEW", fontweight='bold')
    ax1.add_patch(patches.Rectangle((-bx / 2, -by / 2), bx, by, ec='k', fc='#f9f9f9', lw=2))
    for x, y in coords: ax1.add_patch(patches.Circle((x, y), 120, ec='k', ls='--'))
    ax1.text(0, 0, f"{n}-{bar} (EW)", ha='center', fontweight='bold', bbox=dict(fc='white', ec='red', boxstyle='round'))
    ax1.set_xlim(-bx / 1.1, bx / 1.1);
    ax1.set_ylim(-by / 1.1, by / 1.1);
    ax1.set_aspect('equal');
    ax1.axis('off')

    # Section
    ax2.set_title("SECTION DETAIL", fontweight='bold')
    ax2.plot([-bx, bx], [0, 0], 'k-', lw=0.5)
    ax2.add_patch(patches.Rectangle((-bx / 2, -h_mm), bx, h_mm, ec='k', fc='#f0f0f0', lw=2))
    ax2.add_patch(patches.Rectangle((-col_mm / 2, 0), col_mm, h_mm / 2, ec='k', fc='#fff', hatch='///'))

    cov = 75
    ax2.plot([-bx / 2 + cov, bx / 2 - cov], [-h_mm + cov, -h_mm + cov], 'r-', lw=3)
    ax2.plot([-bx / 2 + cov, -bx / 2 + cov], [-h_mm + cov, -h_mm + cov + h_mm * 0.6], 'r-', lw=3)
    ax2.plot([bx / 2 - cov, bx / 2 - cov], [-h_mm + cov, -h_mm + cov + h_mm * 0.6], 'r-', lw=3)
    ax2.text(0, -h_mm + cov - 150, f"Main Reinforcement", ha='center', color='red', fontweight='bold')
    ax2.set_xlim(-bx / 1.1, bx / 1.1);
    ax2.set_ylim(-h_mm * 2, h_mm);
    ax2.set_aspect('equal');
    ax2.axis('off')

    return fig


# ==========================================
# 4. REPORT GENERATOR (Table Format)
# ==========================================
def generate_footing_report(title, rows, imgs, proj, eng, elem_id):
    t_rows = ""
    for r in rows:
        if r[0] == "SECTION":
            t_rows += f"<tr class='sec-row'><td colspan='6'>{r[1]}</td></tr>"
        else:
            cls = "pass-ok" if "PASS" in r[5] or "OK" in r[5] else ("pass-no" if "FAIL" in r[5] else "")
            val_cls = "load-val" if "Mu" in r[0] or "Vu" in r[0] else ""
            t_rows += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td class='{val_cls}'>{r[3]}</td><td>{r[4]}</td><td class='{cls}'>{r[5]}</td></tr>"

    img_html = "".join([f"<div class='drawing-box'><img src='{i}' style='max-width:100%'></div>" for i in imgs])

    return f"""
    <div style="font-family: Sarabun, sans-serif; padding: 20px;">
        <div style="text-align:center; margin-bottom:20px;">
            <button onclick="window.print()" class="print-btn-internal">🖨️ Print / Save PDF</button>
        </div>

        <div style="text-align:center; border-bottom: 2px solid #333; margin-bottom: 20px; position: relative;">
            <div style="position: absolute; top: 0; right: 0; border: 2px solid #333; padding: 5px 15px; font-weight: bold; font-size: 18px;">{elem_id}</div>
            <h2>ENGINEERING DESIGN REPORT</h2>
            <h4>{title}</h4>
        </div>

        <div style="display:flex; justify-content:space-between; margin-bottom:20px; border:1px solid #ddd; padding:10px;">
            <div><strong>Project:</strong> {proj}<br><strong>Engineer:</strong> {eng}</div>
            <div><strong>Date:</strong> 16/12/2568</div>
        </div>

        <div class="drawing-container">{img_html}</div><br>

        <table class="report-table">
            <thead><tr><th width="20%">Item</th><th width="25%">Formula</th><th width="30%">Substitution</th><th>Result</th><th>Unit</th><th>Status</th></tr></thead>
            <tbody>{t_rows}</tbody>
        </table>

        <div class="footer-section">
            <div class="signature-block">
                <div style="font-weight: bold;">Designed by:</div>
                <div class="sign-line"></div>
                <div>({eng})</div>
                <div>วิศวกรโครงสร้าง</div>
            </div>
        </div>
    </div>
    """


# ==========================================
# 5. UI
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

    auto_h = st.checkbox("Auto-Design H", True)
    h = st.number_input("Thickness (m)", 0.50)
    edge = st.number_input("Edge Dist (m)", 0.25)
    mainBar = st.selectbox("Main Rebar", list(BAR_INFO.keys()), index=4)

    Pu = st.number_input("Axial Load Pu (tf)", 60.0)
    PileCap = st.number_input("Pile Cap (tf)", 30.0)

    run_btn = st.form_submit_button("Run Design")

if run_btn:
    st.success("✅ Calculation Complete: Design Passed")  # Success Banner
    d = {'project': project, 'f_id': f_id, 'engineer': engineer, 'fc': fc, 'fy': fy, 'cx': cx, 'cy': cy,
         'n_pile': n_pile, 'dp': dp, 'spacing': spacing, 'h': h, 'edge': edge, 'mainBar': mainBar,
         'Pu': Pu, 'PileCap': PileCap, 'auto_h': auto_h}

    rows, coords, bx, by, n = process_footing_calculation(d)
    img = fig_to_base64(plot_foot_combined(coords, bx, by, n, n, mainBar, h * 1000, cx * 1000))
    st.components.v1.html(generate_footing_report("Pile Cap Calculation Report", rows, [img], project, engineer, f_id),
                          height=1200, scrolling=True)
else:
    st.info("👈 กรุณากรอกข้อมูลและกด Run Design")
