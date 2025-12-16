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
# 1. SETUP & CSS (ปรับปรุงรูปแบบเอกสาร)
# ==========================================
st.set_page_config(page_title="RC Pile Cap Design", layout="wide")

# CSS สำหรับจำลองหน้ากระดาษ A4 และปุ่มพิมพ์สีเขียว
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');

    /* ปุ่มพิมพ์ สีเขียว */
    .print-btn-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .print-btn {
        background-color: #28a745; /* สีเขียวสด */
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 5px;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: background 0.3s;
        text-decoration: none;
        display: inline-block;
    }
    .print-btn:hover {
        background-color: #218838;
    }

    /* ซ่อน Elements ของ Streamlit เมื่อสั่งพิมพ์ */
    @media print {
        .stApp > header {display: none !important;}
        .stApp {margin: 0; padding: 0;}
        .sidebar .sidebar-content {display: none;}
        iframe {height: 100% !important;}
        .print-btn-container {display: none !important;}
        /* บังคับให้พิมพ์พื้นหลัง */
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS
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
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


# ==========================================
# 3. CALCULATION LOGIC
# ==========================================
def get_pile_coordinates(n_pile, s):
    # s = spacing
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
    # ฟังก์ชันตรวจสอบ Shear แบบเงียบๆ เพื่อหาความหนา Auto
    fc = inputs['fc'] * 0.0980665  # MPa
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
    phi_v = 0.75  # ACI Shear

    # 1. Punching Shear
    c1 = col_x + d;
    c2 = col_y + d
    Vu_punch_N = sum([P_avg_N for px, py in coords if (abs(px) > c1 / 2) or (abs(py) > c2 / 2)])
    bo = 2 * (c1 + c2)
    Vc_punch_N = 0.33 * math.sqrt(fc) * bo * d
    if Vu_punch_N > phi_v * Vc_punch_N: return False

    # 2. Beam Shear (Approx Check along X axis only for auto sizing)
    dist_crit = col_x / 2 + d
    Vu_beam_N = sum([P_avg_N for px, py in coords if abs(px) > dist_crit])
    Vc_beam_N = 0.17 * math.sqrt(fc) * width_y * d
    if Vu_beam_N > phi_v * Vc_beam_N: return False

    return True


def process_footing_calculation(inputs):
    rows = []

    # Helper to add row: [Item, Symbol/Formula, Substitution, Result, Unit, Status]
    def sec(t):
        rows.append(["HEADER", t, "", "", "", ""])

    def row(item, formula, sub, res, unit, status=""):
        rows.append([item, formula, sub, res, unit, status])

    fc = inputs['fc'] * 0.0980665  # ksc -> MPa
    fy = inputs['fy'] * 0.0980665  # ksc -> MPa
    pu_tf = inputs['Pu']
    n_pile = int(inputs['n_pile'])
    s = inputs['spacing'] * 1000
    edge = inputs['edge'] * 1000
    dp = inputs['dp'] * 1000
    col_x = inputs['cx'] * 1000
    col_y = inputs['cy'] * 1000

    # Initial H
    h_final = inputs['h'] * 1000
    cover = 75.0
    db = BAR_INFO[inputs['mainBar']]['d_mm']

    # Coordinates & Footing Size
    coords = get_pile_coordinates(n_pile, s)

    if n_pile == 1:
        bx = dp + 2 * edge
        by = dp + 2 * edge
    else:
        max_x = max([abs(x) for x, y in coords])
        max_y = max([abs(y) for x, y in coords])
        bx = (max_x * 2) + dp + 2 * edge
        by = (max_y * 2) + dp + 2 * edge

    # Auto H Logic
    if inputs.get('auto_h', False) and n_pile > 1:
        h_try = 300.0
        found = False
        for _ in range(50):  # Max 50 iterations
            if check_shear_capacity_silent(h_try, inputs, coords, bx, by):
                h_final = h_try
                found = True
                break
            h_try += 50.0

    d = h_final - cover - db

    # --- 1. DESIGN DATA ---
    sec("1. DESIGN PARAMETERS (ข้อกำหนดการออกแบบ)")
    row("Axial Load", "Pu", "-", f"{pu_tf:.2f}", "tf", "")
    row("Concrete Strength", "fc'", f"{inputs['fc']:.0f} ksc", f"{fc:.1f}", "MPa", "")
    row("Steel Strength", "fy", f"{inputs['fy']:.0f} ksc", f"{fy:.1f}", "MPa", "")
    row("Pile Diameter", "Ø pile", "-", f"{dp:.0f}", "mm", "")
    row("Pile Capacity", "R_allow", "-", f"{inputs['PileCap']:.2f}", "tf/pile", "")

    # --- 2. GEOMETRY ---
    sec("2. GEOMETRY & PROPERTIES (หน้าตัดฐานราก)")
    row("Footing Size", "B x L", f"{bx:.0f}x{by:.0f}", f"h={h_final:.0f}", "mm", "")
    row("Effective Depth", "d = h - cover - db", f"{h_final:.0f} - {cover} - {db}", f"{d:.1f}", "mm", "")

    # Size effect factor (ACI 318-19)
    lambda_s = math.sqrt(2 / (1 + 0.004 * d))
    if lambda_s > 1.0: lambda_s = 1.0
    row("Size Effect Factor", "λs = √(2/(1+0.004d))", f"√(2/(1+0.004*{d:.0f}))", f"{lambda_s:.3f}", "-", "≤ 1.0")

    # --- 3. PILE REACTION ---
    sec("3. PILE REACTION CHECK (ตรวจสอบน้ำหนักลงเสาเข็ม)")
    p_avg = pu_tf / n_pile
    status_pile = "PASS" if p_avg <= inputs['PileCap'] else "FAIL"
    row("Load per Pile", "Ru = Pu / N", f"{pu_tf:.2f} / {n_pile}", f"{p_avg:.2f}", "tf", status_pile)

    # --- 4. FLEXURAL DESIGN ---
    sec("4. FLEXURAL DESIGN (ออกแบบเหล็กเสริมรับโมเมนต์)")

    # Convert to Newton for calc
    p_n = p_avg * 9806.65

    # Calculate Moments at face of column
    mx = 0  # Moment causing bending about Y axis (along X direction arms) -> Reinforcement X-Dir? No, Mx usually means M around X axis.
    # Let's align standard: 
    # M_x_axis uses lever arm in Y direction. Reinforcement runs Parallel to Y.
    # M_y_axis uses lever arm in X direction. Reinforcement runs Parallel to X.

    # However, common notation: