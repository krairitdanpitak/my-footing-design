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
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');

    .stApp { font-family: 'Sarabun', sans-serif; }
    h1, h2, h3 { font-family: 'Sarabun', sans-serif; }

    /* ซ่อน Header และ Sidebar ของ Streamlit เมื่อสั่งพิมพ์ */
    @media print {
        [data-testid="stHeader"] { display: none !important; }
        [data-testid="stSidebar"] { display: none !important; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        footer { display: none !important; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATABASE & HELPER FUNCTIONS
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
        return f"{val:,.{digits}f}"
    except:
        return str(n)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def check_development_length(inputs, bar_dia_mm, avail_mm):
    """
    ตรวจสอบ Development Length ตาม ACI 318-19
    คืนค่า: (Length Required, Type (Straight/Hook), Status, Detailed Info)
    """
    fc_mpa = inputs['fc'] * 0.0980665
    fy_mpa = inputs['fy'] * 0.0980665

    # 1. Straight Ld (ACI 318-19 Sec 25.4.2)
    psi_t = 1.0;
    psi_e = 1.0;
    lam = 1.0

    if bar_dia_mm <= 19:
        ld_val = (fy_mpa * psi_t * psi_e) / (2.1 * lam * math.sqrt(fc_mpa)) * bar_dia_mm
    else:
        ld_val = (fy_mpa * psi_t * psi_e) / (1.7 * lam * math.sqrt(fc_mpa)) * bar_dia_mm

    ld_req = max(ld_val, 300.0)

    # 2. Hook Ldh (ACI 318-19 Sec 25.4.3)
    psi_c = 0.7
    psi_r = 1.0
    ldh_val = (0.24 * fy_mpa * psi_e * psi_c * psi_r) / (lam * math.sqrt(fc_mpa)) * bar_dia_mm
    ldh_req = max(ldh_val, 8 * bar_dia_mm, 150.0)

    # 3. Decision
    if avail_mm >= ld_req:
        return ld_req, "Straight", "PASS", f"Avail({avail_mm:.0f}) > Ld({ld_req:.0f})"
    elif avail_mm >= ldh_req:
        return ldh_req, "Hook 90°", "PASS", f"Avail({avail_mm:.0f}) > Ldh({ldh_req:.0f})"
    else:
        return ldh_req, "Hook 90°", "FAIL", f"Avail({avail_mm:.0f}) < Ldh({ldh_req:.0f})"


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
    """
    ฟังก์ชันตรวจสอบแรงเฉือนแบบเงียบ (ใช้สำหรับ Loop หาความหนาอัตโนมัติ)
    """
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

    # Punching Shear Check
    c1 = col_x + d;
    c2 = col_y + d
    Vu_punch_N = sum([P_avg_N for px, py in coords if (abs(px) > c1 / 2) or (abs(py) > c2 / 2)])
    bo = 2 * (c1 + c2)
    # Using simplified Vc formula for loop
    Vc_punch_N = 0.33 * math.sqrt(fc) * bo * d
    if Vu_punch_N > phi_v * Vc_punch_N: return False

    # Beam Shear Check (Simplified: Check only one direction for Auto H)
    dist_crit = col_x / 2 + d
    Vu_beam_N = sum([P_avg_N for px, py in coords if abs(px) > dist_crit])
    Vc_beam_N = 0.17 * math.sqrt(fc) * width_y * d
    if Vu_beam_N > phi_v * Vc_beam_N: return False

    return True


# ==========================================
# 3. MAIN CALCULATION LOGIC (UPDATED)
# ==========================================
def process_footing_calculation(inputs):
    rows = []

    # Helper functions for report formatting
    def sec(t):
        rows.append(["SECTION", t, "", "", "", ""])

    def row(i, f, s, r, u, st=""):
        rows.append([i, f, s, r, u, st])

    # 1. Prepare Variables
    fc = inputs['fc'] * 0.0980665  # ksc -> MPa
    fy = inputs['fy'] * 0.0980665  # ksc -> MPa
    pu_tf = inputs['Pu']
    n_pile = int(inputs['n_pile'])
    s_m = inputs['spacing']
    s = s_m * 1000
    edge = inputs['edge'] * 1000
    dp = inputs['dp'] * 1000
    col_x = inputs['cx'] * 1000
    col_y = inputs['cy'] * 1000
    h_final = inputs['h'] * 1000
    cover = 75.0
    bar_key = inputs['mainBar']
    db = BAR_INFO[bar_key]['d_mm']

    # Auto H Logic (Simple loop)
    coords = get_pile_coordinates(n_pile, s)

    # Determine Footing Size (bx, by)
    if n_pile == 1:
        bx = dp + 2 * edge
        by = dp + 2 * edge
    else:
        max_x = max([abs(x) for x, _ in coords])
        max_y = max([abs(y) for _, y in coords])
        bx = (max_x * 2) + dp + 2 * edge
        by = (max_y * 2) + dp + 2 * edge

    # Auto H Adjustment
    if inputs.get('auto_h', False) and n_pile > 1:
        h_try = 300.0
        for _ in range(50):
            if check_shear_capacity_silent(h_try, inputs, coords, bx, by):
                h_final = h_try
                break
            h_try += 50.0

    # Effective Depth
    d = h_final - cover - db

    # --- 1. GEOMETRY ---
    sec("1. GEOMETRY & PROPERTIES")
    row("Footing Size", "B x L", f"{bx:.0f}x{by:.0f}", f"h={h_final:.0f}", "mm", "")
    row("Effective Depth", "d = h - cover - db", f"{h_final:.0f} - {cover} - {db}", f"{d:.1f}", "mm", "")

    # ACI 318-19 Size Effect Factor (Lambda_s)
    lambda_s = min(math.sqrt(2 / (1 + 0.004 * d)), 1.0)
    row("Size Effect λs", "√(2/(1+0.004d))", f"√(2/(1+0.004*{d:.0f}))", f"{lambda_s:.3f}", "-", "≤1.0")

    # --- 2. PILE REACTION ---
    sec("2. PILE REACTION CHECK")
    p_avg = pu_tf / n_pile
    row("Load per Pile (Ru)", "Pu / N", f"{pu_tf} / {n_pile}", f"{p_avg:.2f}", "tf",
        "PASS" if p_avg <= inputs['PileCap'] else "FAIL")

    # --- 3. FLEXURAL ---
    sec("3. FLEXURAL DESIGN")
    p_n_newton = p_avg * 9806.65  # Convert ton to Newton for calculation

    # Calculate Moments
    mx_newton_mm = 0  # Moment around X-axis (Check Y direction bars)
    my_newton_mm = 0  # Moment around Y-axis (Check X direction bars)

    for x, y in coords:
        lx = abs(x) - col_x / 2
        ly = abs(y) - col_y / 2
        if lx > 0: my_newton_mm += p_n_newton * lx  # Force * Arm X
        if ly > 0: mx_newton_mm += p_n_newton * ly  # Force * Arm Y

    res_bars = {}

    # Loop for X-Direction Bars (Moment My) and Y-Direction Bars (Moment Mx)
    # Note: X-Dir bars resist Moment around Y-axis (My)
    # Note: Y-Dir bars resist Moment around X-axis (Mx)

    design_params = [
        {'label': 'X-Dir (M around Y)', 'mom': my_newton_mm, 'width': by, 'key': 'X-Dir'},
        {'label': 'Y-Dir (M around X)', 'mom': mx_newton_mm, 'width': bx, 'key': 'Y-Dir'}
    ]

    for param in design_params:
        mom = param['mom']
        width = param['width']
        label = param['label']

        # Design As
        req_as = mom / (0.9 * fy * 0.9 * d) if mom > 0 else 0
        min_as = 0.0018 * width * h_final  # Temp & Shrinkage
        des_as = max(req_as, min_as)

        bar_area_mm2 = BAR_INFO[inputs['mainBar']]['A_cm2'] * 100
        n_bars = math.ceil(des_as / bar_area_mm2)
        if n_pile == 1: n_bars = max(n_bars, 4)  # Minimum for single pile cap

        prov_as = n_bars * bar_area_mm2

        # Add to Report
        row(f"Moment {label}", "Σ P(arm)", "-", f"{mom / 9.80665e7:.2f}", "tf-m", "")  # N-mm to tf-m
        row(f"As Req {label}", "Max(Calc, Min)", f"Max({req_as:.0f}, {min_as:.0f})", f"{des_as:.0f}", "mm²", "")
        row(f"Provide {label}", f"{n_bars}-{inputs['mainBar']}", f"As={prov_as:.0f}", "OK", "-", "OK")

        res_bars[param['key']] = n_bars

    # --- 4. SHEAR CHECKS (ACI 318-19) ---
    if n_pile > 1:
        phi_shear = 0.75

        # 4.1 One-Way Shear (X-Axis Analysis -> Critical Section cuts Y-Axis)
        # Critical section at d from face of column
        sec("4.1 ONE-WAY SHEAR (CRITICAL SECTION Y)")
        dist_crit_y = (col_y / 2) + d

        # Sum piles outside critical section in Y direction
        n_piles_shear_y = sum([1 for _, y in coords if abs(y) > dist_crit_y])
        vu_y_newton = n_piles_shear_y * p_n_newton

        # Vc Capacity (Simplified ACI 318-19: 0.17 * lambda * sqrt(fc) * bw * d)
        # bw here is bx (width perpendicular to shear force)
        vc_y_newton = 0.17 * lambda_s * math.sqrt(fc) * bx * d
        phi_vc_y = phi_shear * vc_y_newton

        row("Critical Dist (d)", "Cy/2 + d", f"{col_y / 2:.0f} + {d:.0f}", f"{dist_crit_y:.0f}", "mm", "")
        row("Vu (One-Way Y)", "Σ P_out", f"{n_piles_shear_y} Piles", f"{vu_y_newton / 9806.65:.2f}", "tf", "")
        row("Capacity φVc", "0.75·0.17λs√fc·B·d", f"0.75·0.17·{lambda_s:.2f}·√{fc:.0f}·{bx:.0f}·{d:.0f}",
            f"{phi_vc_y / 9806.65:.2f}", "tf",
            "PASS" if vu_y_newton <= phi_vc_y else "FAIL")

        # 4.2 One-Way Shear (Y-Axis Analysis -> Critical Section cuts X-Axis)
        sec("4.2 ONE-WAY SHEAR (CRITICAL SECTION X)")
        dist_crit_x = (col_x / 2) + d

        # Sum piles outside critical section in X direction
        n_piles_shear_x = sum([1 for x, _ in coords if abs(x) > dist_crit_x])
        vu_x_newton = n_piles_shear_x * p_n_newton

        # bw here is by
        vc_x_newton = 0.17 * lambda_s * math.sqrt(fc) * by * d
        phi_vc_x = phi_shear * vc_x_newton

        row("Critical Dist (d)", "Cx/2 + d", f"{col_x / 2:.0f} + {d:.0f}", f"{dist_crit_x:.0f}", "mm", "")
        row("Vu (One-Way X)", "Σ P_out", f"{n_piles_shear_x} Piles", f"{vu_x_newton / 9806.65:.2f}", "tf", "")
        row("Capacity φVc", "0.75·0.17λs√fc·B·d", f"0.75·0.17·{lambda_s:.2f}·√{fc:.0f}·{by:.0f}·{d:.0f}",
            f"{phi_vc_x / 9806.65:.2f}", "tf",
            "PASS" if vu_x_newton <= phi_vc_x else "FAIL")

        # 4.3 Two-Way Shear (Punching)
        sec("4.3 TWO-WAY SHEAR (PUNCHING)")

        # Critical Perimeter (bo) at d/2
        c1 = col_x + d
        c2 = col_y + d

        # Handle special case where pile cap is small relative to d/2
        # But for general rect column:
        bo = 2 * (c1 + c2)

        # Sum loads outside critical perimeter
        # Check if pile center is outside (Load is outside if center is outside per simplified assumption)
        vu_punch_newton = sum([p_n_newton for px, py in coords if (abs(px) > c1 / 2) or (abs(py) > c2 / 2)])

        # Vc Punching (ACI 318-19 Table 22.6.5.2)
        # Using minimum of 3 equations, usually 0.33 limit controls for stocky members
        # Vc = 0.33 * lambda_s * sqrt(fc) * bo * d
        vc_punch_newton = 0.33 * lambda_s * math.sqrt(fc) * bo * d
        phi_vc_punch = phi_shear * vc_punch_newton

        row("Critical Perim (bo)", "2(c1+d + c2+d)", f"2({c1:.0f}+{c2:.0f})", f"{bo:.0f}", "mm", "")
        row("Vu (Punching)", "Σ P_out", "-", f"{vu_punch_newton / 9806.65:.2f}", "tf", "")
        row("Capacity φVc", "0.75·0.33λs√fc·bo·d", f"0.75·0.33·{lambda_s:.2f}·√{fc:.0f}·{bo:.0f}·{d:.0f}",
            f"{phi_vc_punch / 9806.65:.2f}", "tf",
            "PASS" if vu_punch_newton <= phi_vc_punch else "FAIL")

    # --- 5. DETAILING CHECK ---
    sec("5. DETAILING CHECK (ACI 318-19)")
    avail_x = (bx / 2) - (col_x / 2) - cover
    avail_y = (by / 2) - (col_y / 2) - cover

    req_len_x, type_x, stat_x, info_x = check_development_length(inputs, db, avail_x)
    row("Dev. Length (X-Dir)", f"Avail: {avail_x:.0f} mm", info_x, f"{req_len_x:.0f} ({type_x})", "mm", stat_x)

    req_len_y, type_y, stat_y, info_y = check_development_length(inputs, db, avail_y)
    row("Dev. Length (Y-Dir)", f"Avail: {avail_y:.0f} mm", info_y, f"{req_len_y:.0f} ({type_y})", "mm", stat_y)

    sec("6. FINAL STATUS")
    final_st = "OK" if "FAIL" not in [r[5] for r in rows] else "CHECK"
    final_msg = "DESIGN COMPLETE" if final_st == "OK" else "CALCULATION DONE"
    row("Overall Design", "-", "-", final_msg, "-", final_st)

    return rows, coords, bx, by, res_bars.get('X-Dir', 4), res_bars.get('Y-Dir', 4), h_final


# ==========================================
# 4. PLOTTING & REPORTING
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


def generate_report_html(title, rows, imgs, proj, eng, elem_id, inputs):
    style = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
        body { font-family: 'Sarabun', sans-serif; color: #000; }
        .print-btn { 
            background-color: #4CAF50; border: none; color: white; padding: 10px 24px; 
            text-align: center; text-decoration: none; display: inline-block; 
            font-size: 16px; margin: 10px 2px; cursor: pointer; border-radius: 4px; font-weight: bold;
        }
        .print-btn:hover { background-color: #45a049; }
        .report-container { 
            width: 210mm; min-height: 297mm; padding: 20mm; margin: 10mm auto; 
            border: 1px solid #d3d3d3; border-radius: 5px; background: white; 
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.1); 
        }
        .header { text-align: center; margin-bottom: 20px; position: relative; }
        .header h1 { margin: 0; font-size: 24px; font-weight: bold; text-transform: uppercase; }
        .header h2 { margin: 5px 0 0 0; font-size: 18px; font-weight: bold; color: #333; }
        .id-box { position: absolute; top: 0; right: 0; border: 2px solid #000; padding: 5px 10px; font-weight: bold; font-size: 16px; }
        .info-box { 
            border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; 
            display: flex; justify-content: space-between; background-color: #f9f9f9; font-size: 14px;
        }
        .info-col { width: 48%; }
        .info-item { margin-bottom: 5px; }
        .summary-title { text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 10px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
        .img-container { text-align: center; margin-bottom: 20px; }
        .img-container img { max-width: 100%; height: auto; border: 1px solid #eee; padding: 5px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 20px; }
        th, td { border: 1px solid #000; padding: 6px 8px; vertical-align: top; }
        th { background-color: #f2f2f2; font-weight: bold; text-align: center; }
        .sec-row td { background-color: #e6e6e6; font-weight: bold; text-align: left; padding-left: 10px; }
        .status-pass { color: green; font-weight: bold; text-align: center; }
        .status-fail { color: red; font-weight: bold; text-align: center; }
        .col-result { font-weight: bold; text-align: right; }
        .col-unit { text-align: center; }
        .footer { margin-top: 50px; display: flex; justify-content: space-between; page-break-inside: avoid; }
        .sign-box { width: 250px; text-align: center; }
        .sign-line { border-bottom: 1px solid #000; margin-bottom: 5px; height: 40px; }
        @media print {
            body { background: none; -webkit-print-color-adjust: exact; }
            .report-container { width: 100%; margin: 0; padding: 0; border: none; box-shadow: none; }
            .print-btn-container { display: none; }
            @page { margin: 2cm; }
        }
    </style>
    """

    table_html = ""
    for r in rows:
        if r[0] == "SECTION":
            table_html += f"<tr class='sec-row'><td colspan='6'>{r[1]}</td></tr>"
        else:
            status_class = "status-pass" if "PASS" in str(r[5]) or "OK" in str(r[5]) else "status-fail"
            table_html += f"""
            <tr>
                <td>{r[0]}</td>
                <td style="font-family: monospace;">{r[1]}</td>
                <td>{r[2]}</td>
                <td class="col-result">{r[3]}</td>
                <td class="col-unit">{r[4]}</td>
                <td class="{status_class}">{r[5]}</td>
            </tr>
            """

    html_content = f"""
    <html>
    <head>{style}</head>
    <body>
        <div class="print-btn-container" style="text-align: center;">
            <button class="print-btn" onclick="window.print()">🖨️ Print This Page / พิมพ์หน้านี้</button>
        </div>

        <div class="report-container">
            <div class="header">
                <div class="id-box">{elem_id}</div>
                <h1>ENGINEERING DESIGN REPORT</h1>
                <h2>{title}</h2>
            </div>

            <div class="info-box">
                <div class="info-col">
                    <div class="info-item"><strong>Project:</strong> {proj}</div>
                    <div class="info-item"><strong>Engineer:</strong> {eng}</div>
                    <div class="info-item"><strong>Date:</strong> 16/12/2568</div>
                </div>
                <div class="info-col">
                    <div class="info-item"><strong>Materials:</strong> fc'={inputs['fc']} ksc, fy={inputs['fy']} ksc</div>
                    <div class="info-item"><strong>Pile:</strong> {inputs['n_pile']}xØ{inputs['dp']}m @ {inputs['spacing']}m</div>
                    <div class="info-item"><strong>Cap Size:</strong> {inputs['h']}m Thick</div>
                </div>
            </div>

            <div class="summary-title">Design Summary</div>
            <div class="img-container">
                <img src="{imgs[0]}" />
            </div>

            <div class="summary-title" style="margin-top: 20px;">Calculation Details</div>
            <table>
                <thead>
                    <tr>
                        <th width="25%">Item</th>
                        <th width="20%">Formula</th>
                        <th width="25%">Substitution</th>
                        <th width="15%">Result</th>
                        <th width="5%">Unit</th>
                        <th width="10%">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>

            <div class="footer">
                <div class="sign-box">
                    <div style="text-align: left;"><strong>Designed by:</strong></div>
                    <div class="sign-line"></div>
                    <div>({eng})</div>
                    <div>Structural Engineer</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


# ==========================================
# 5. MAIN UI
# ==========================================
st.title("RC Pile Cap Design SDM")

with st.sidebar.form("inputs"):
    st.header("Project Info")
    project = st.text_input("Project Name", "อาคารสำนักงาน 2 ชั้น")
    f_id = st.text_input("Footing ID", "F-01")
    engineer = st.text_input("Engineer Name", "นายไกรฤทธิ์ ด่านพิทักษ์")

    st.markdown("---")
    st.header("Parameters")
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

    Pu = st.number_input("Axial Load Pu (tf)", min_value=0.0, value=60.0, step=1.0)
    PileCap = st.number_input("Pile Capacity (tf)", min_value=0.0, value=30.0, step=1.0)

    run_btn = st.form_submit_button("Run Design")

if run_btn:
    st.success("✅ คำนวณเสร็จสิ้น (Calculation Finished)")
    d_inputs = {
        'project': project, 'f_id': f_id, 'engineer': engineer,
        'fc': fc, 'fy': fy, 'cx': cx, 'cy': cy,
        'n_pile': n_pile, 'dp': dp, 'spacing': spacing,
        'h': h, 'edge': edge, 'mainBar': mainBar,
        'Pu': Pu, 'PileCap': PileCap, 'auto_h': auto_h
    }

    # 1. Calculate (Uses Updated Function)
    rows, coords, bx, by, nx, ny, h_calc = process_footing_calculation(d_inputs)
    d_inputs['h'] = fmt(h_calc / 1000)  # Update H for report

    # 2. Plot
    img_b64 = fig_to_base64(plot_foot_combined(coords, bx, by, nx, ny, mainBar, h_calc, cx * 1000, cy * 1000))

    # 3. Generate HTML Report
    html_report = generate_report_html("RC Pile Cap Design SDM", rows, [img_b64], project, engineer, f_id, d_inputs)

    # 4. Render
    st.components.v1.html(html_report, height=1200, scrolling=True)

else:
    st.info("👈 กรุณากรอกข้อมูลและกด Run Design")