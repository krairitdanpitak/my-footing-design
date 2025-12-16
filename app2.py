import streamlit as st
import streamlit.components.v1 as components
import matplotlib

matplotlib.use('Agg')  # ป้องกัน Error เรื่อง Thread ของ Matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import io
import base64

# ==========================================
# 1. SETUP (ต้องอยู่บรรทัดแรกๆ เสมอ)
# ==========================================
st.set_page_config(page_title="RC Pile Cap Design", layout="wide")

# CSS: ปรับให้แสดงผลสวยงาม และซ่อนปุ่มเมื่อสั่ง Print
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');

    /* สไตล์สำหรับหน้าจอปกติ */
    .report-container {
        background-color: white;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-top: 20px;
        font-family: 'Sarabun', sans-serif;
    }

    /* ปุ่มพิมพ์สีเขียว */
    .print-btn {
        background-color: #28a745;
        color: white; 
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border-radius: 5px;
        cursor: pointer;
        border: none;
        margin-bottom: 20px;
    }
    .print-btn:hover { background-color: #218838; }

    /* การตั้งค่าเมื่อสั่ง Print (ซ่อน Sidebar และปุ่ม) */
    @media print {
        .stApp > header {display: none !important;}
        .sidebar {display: none !important;}
        [data-testid="stSidebar"] {display: none !important;}
        .print-btn-area {display: none !important;}
        .no-print {display: none !important;}
        body { margin: 0; padding: 0; }
        .report-container { border: none; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA & FUNCTIONS
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


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def get_pile_coords(n, s):
    # s = spacing (mm)
    if n == 1:
        return [(0, 0)]
    elif n == 2:
        return [(-s / 2, 0), (s / 2, 0)]
    elif n == 3:
        return [(-s / 2, -s * 0.288), (s / 2, -s * 0.288), (0, s * 0.577)]
    elif n == 4:
        return [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2)]
    elif n == 5:
        return [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2), (0, 0)]
    return []


def calculate_design(inputs):
    # Unpack Inputs
    fc = inputs['fc'] * 0.0980665  # MPa
    fy = inputs['fy'] * 0.0980665  # MPa
    Pu_load = inputs['Pu'] * 9806.65  # N
    n_pile = int(inputs['n_pile'])
    s = inputs['spacing'] * 1000  # mm
    edge = inputs['edge'] * 1000  # mm
    dp = inputs['dp'] * 1000  # mm
    cx = inputs['cx'] * 1000  # mm
    cy = inputs['cy'] * 1000  # mm
    h = inputs['h'] * 1000  # mm

    # 1. Geometry
    coords = get_pile_coords(n_pile, s)
    if n_pile == 1:
        bx, by = dp + 2 * edge, dp + 2 * edge
    else:
        mx = max([abs(x) for x, y in coords])
        my = max([abs(y) for x, y in coords])
        bx, by = (mx * 2) + dp + 2 * edge, (my * 2) + dp + 2 * edge

    cover = 75
    db = BAR_INFO[inputs['mainBar']]['d_mm']
    d = h - cover - db

    rows = []

    def add_row(item, formula, sub, res, unit, status):
        rows.append([item, formula, sub, res, unit, status])

    # 2. Check Loads
    P_avg_tf = inputs['Pu'] / n_pile
    status_pile = "PASS" if P_avg_tf <= inputs['PileCap'] else "FAIL"
    add_row("Load / Pile", "Pu / N", f"{inputs['Pu']} / {n_pile}", f"{P_avg_tf:.2f}", "tf", status_pile)

    # 3. Moment (Simplified Envelope)
    P_node_N = Pu_load / n_pile
    Mx, My = 0, 0
    for x, y in coords:
        lx = abs(x) - cx / 2
        ly = abs(y) - cy / 2
        if lx > 0: Mx += P_node_N * lx
        if ly > 0: My += P_node_N * ly

    # Check X-Dir Steel (Resist Moment around Y-Axis -> Uses lever arm x)
    # Note: Structural notation varies, assuming Main Steel handles max moment
    M_max_Nmm = max(Mx, My)
    M_max_tfm = M_max_Nmm / 9806650

    req_As = M_max_Nmm / (0.9 * fy * 0.9 * d) if M_max_Nmm > 0 else 0
    min_As = 0.0018 * bx * h  # Temp & Shrinkage (Approx)
    design_As = max(req_As, min_As)

    bar_area = BAR_INFO[inputs['mainBar']]['A_cm2'] * 100
    n_bars = math.ceil(design_As / bar_area)
    if n_pile == 1: n_bars = max(n_bars, 4)
    prov_As = n_bars * bar_area

    add_row("Max Moment", "Σ P*arm", "Envelope", f"{M_max_tfm:.2f}", "tf-m", "")
    add_row("As Required", "Mu/φfjd", "-", f"{req_As:.0f}", "mm²", "")
    add_row("As Minimum", "0.0018 bh", "-", f"{min_As:.0f}", "mm²", "")
    add_row("Provide Rebar", f"{n_bars}-{inputs['mainBar']}", f"As={prov_As:.0f}", "OK", "mm²", "PASS")

    return rows, coords, bx, by, n_bars, h


def plot_pilecap(coords, bx, by, n_bars, bar_name, h, cx, cy):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plan
    ax1.set_title("PLAN VIEW")
    ax1.add_patch(patches.Rectangle((-bx / 2, -by / 2), bx, by, ec='k', fc='white', lw=2))
    ax1.add_patch(patches.Rectangle((-cx / 2, -cy / 2), cx, cy, ec='k', fc='#ddd', hatch='//'))
    for x, y in coords:
        ax1.add_patch(patches.Circle((x, y), 150, ec='k', ls='--'))
    ax1.text(0, by / 2 + 200, f"Reinforce: {n_bars}-{bar_name} (BW)", ha='center', color='blue', fontweight='bold')
    ax1.axis('equal');
    ax1.axis('off')

    # Section
    ax2.set_title("SECTION VIEW")
    ax2.plot([-bx, bx], [0, 0], 'k-', lw=1)
    ax2.add_patch(patches.Rectangle((-bx / 2, -h), bx, h, ec='k', fc='#f9f9f9', lw=2))
    ax2.add_patch(patches.Rectangle((-cx / 2, 0), cx, h / 2, ec='k', fc='white', hatch='///'))

    # Rebar
    cov = 75
    ax2.plot([-bx / 2 + cov, bx / 2 - cov], [-h + cov, -h + cov], 'r-', lw=3)
    ax2.plot([-bx / 2 + cov, -bx / 2 + cov], [-h + cov, -h + cov + 200], 'r-', lw=3)
    ax2.plot([bx / 2 - cov, bx / 2 - cov], [-h + cov, -h + cov + 200], 'r-', lw=3)

    ax2.text(bx / 2 + 100, -h / 2, f"h={h / 1000:.2f}m", color='red')
    ax2.axis('equal');
    ax2.axis('off')

    return fig


# ==========================================
# 3. HTML GENERATOR (Report Layout)
# ==========================================
def generate_html_report(info, rows, img_b64):
    # สร้างแถวตาราง HTML
    tr_html = ""
    for r in rows:
        color = "green" if "PASS" in r[5] or "OK" in r[5] else ("red" if "FAIL" in r[5] else "black")
        tr_html += f"""
        <tr>
            <td style="border:1px solid #000; padding:5px;">{r[0]}</td>
            <td style="border:1px solid #000; padding:5px;">{r[1]}</td>
            <td style="border:1px solid #000; padding:5px;">{r[2]}</td>
            <td style="border:1px solid #000; padding:5px; text-align:right;">{r[3]}</td>
            <td style="border:1px solid #000; padding:5px; text-align:center;">{r[4]}</td>
            <td style="border:1px solid #000; padding:5px; text-align:center; color:{color}; font-weight:bold;">{r[5]}</td>
        </tr>
        """

    return f"""
    <div class="report-container">
        <div class="print-btn-area" style="text-align:center;">
            <button onclick="window.print()" class="print-btn">🖨️ Print Report / พิมพ์รายงาน</button>
        </div>

        <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
            <tr>
                <td style="border:1px solid #000; padding:10px;" width="70%">
                    <h2 style="margin:0;">CALCULATION SHEET</h2>
                    <p style="margin:5px 0;"><strong>Project:</strong> {info['project']}</p>
                    <p style="margin:5px 0;"><strong>Location:</strong> {info['location']}</p>
                </td>
                <td style="border:1px solid #000; padding:10px;">
                    <p style="margin:5px 0;"><strong>ID:</strong> {info['id']}</p>
                    <p style="margin:5px 0;"><strong>Eng:</strong> {info['eng']}</p>
                    <p style="margin:5px 0;"><strong>Date:</strong> {info['date']}</p>
                </td>
            </tr>
        </table>

        <div style="text-align:center; border:1px solid #000; padding:10px; margin-bottom:20px;">
            <img src="{img_b64}" style="max-width:100%; height:auto;">
        </div>

        <table style="width:100%; border-collapse:collapse; font-size:14px;">
            <thead style="background-color:#f0f0f0;">
                <tr>
                    <th style="border:1px solid #000; padding:8px;">Item</th>
                    <th style="border:1px solid #000; padding:8px;">Formula</th>
                    <th style="border:1px solid #000; padding:8px;">Sub</th>
                    <th style="border:1px solid #000; padding:8px;">Result</th>
                    <th style="border:1px solid #000; padding:8px;">Unit</th>
                    <th style="border:1px solid #000; padding:8px;">Status</th>
                </tr>
            </thead>
            <tbody>
                {tr_html}
            </tbody>
        </table>

        <div style="margin-top:50px; display:flex; justify-content:space-between;">
            <div style="width:40%; text-align:center;">
                <div style="border-bottom:1px solid #000; margin-bottom:5px; height:30px;"></div>
                <div>Designed By ({info['eng']})</div>
            </div>
            <div style="width:40%; text-align:center;">
                <div style="border-bottom:1px solid #000; margin-bottom:5px; height:30px;"></div>
                <div>Approved By (..........................)</div>
            </div>
        </div>
    </div>
    """


# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.title("🏗️ RC Pile Cap Design (Simple)")

with st.sidebar.form("input_form"):
    st.header("1. ข้อมูลโครงการ (Project)")
    project_name = st.text_input("Project Name", "New Warehouse")
    location = st.text_input("Location", "BKK")
    footing_id = st.text_input("Footing ID", "F-01")
    engineer = st.text_input("Engineer", "Admin")

    st.header("2. ข้อมูลเสาเข็มและน้ำหนัก")
    c1, c2 = st.columns(2)
    Pu = c1.number_input("Axial Load Pu (tf)", value=50.0)
    PileCap = c2.number_input("Pile Safe Load (tf)", value=25.0)

    n_pile = st.selectbox("จำนวนเสาเข็ม (No. of Pile)", [1, 2, 3, 4, 5], index=3)
    dp = c1.number_input("ขนาดเสาเข็ม (m)", value=0.22)
    spacing = c2.number_input("ระยะห่างเสาเข็ม (m)", value=0.80)

    st.header("3. ฐานรากและวัสดุ")
    h_input = c1.number_input("ความหนาฐานราก (m)", value=0.50)
    edge_input = c2.number_input("ระยะขอบ (m)", value=0.25)
    cx = c1.number_input("ตอม่อ X (m)", value=0.25)
    cy = c2.number_input("ตอม่อ Y (m)", value=0.25)

    fc = c1.number_input("fc' (ksc)", value=240)
    fy = c2.number_input("fy (ksc)", value=4000)
    mainBar = st.selectbox("เหล็กเสริมหลัก", list(BAR_INFO.keys()), index=4)

    submit_btn = st.form_submit_button("✅ Run Calculation")

# Logic การแสดงผล
if submit_btn:
    try:
        # 1. รวบรวมข้อมูล
        inputs = {
            'fc': fc, 'fy': fy, 'Pu': Pu, 'PileCap': PileCap,
            'n_pile': n_pile, 'dp': dp, 'spacing': spacing,
            'h': h_input, 'edge': edge_input, 'cx': cx, 'cy': cy,
            'mainBar': mainBar
        }

        # 2. คำนวณ
        rows, coords, bx, by, n_bars, h_final = calculate_design(inputs)

        # 3. สร้างกราฟ
        fig = plot_pilecap(coords, bx, by, n_bars, mainBar, h_final, cx * 1000, cy * 1000)
        img_b64 = fig_to_base64(fig)

        # 4. สร้าง Report
        info = {'project': project_name, 'location': location, 'id': footing_id, 'eng': engineer, 'date': '16/12/2025'}
        html_code = generate_html_report(info, rows, img_b64)

        # 5. แสดงผล
        st.success("คำนวณเสร็จสิ้น! (Calculation Complete)")
        components.html(html_code, height=1000, scrolling=True)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
        st.write("กรุณาตรวจสอบการป้อนข้อมูล หรือ Library ที่ติดตั้ง")
else:
    st.info("👈 กรุณากรอกข้อมูลด้านซ้ายแล้วกดปุ่ม 'Run Calculation'")
