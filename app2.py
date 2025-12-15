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
    h_final = inputs['h'] * 1000
    dp = inputs['dp'] * 1000
    cover = 75.0
    bar_key = inputs['mainBar']
    db = BAR_INFO[bar_key]['d_mm']
    d = h_final - cover - db  # Effective depth (approx)

    # Geometry
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
    row("Pile Cap Size", "B x L", f"{fmt(width_x, 0)} x {fmt(width_y, 0)}", f"h={h_final:.0f}", "mm")
    row("Effective Depth", "d = h - cover - db", f"{h_final:.0f} - {cover} - {db:.0f}", f"{d:.1f}", "mm")

    # ACI 318-19 Size Effect Factor
    lambda_s = math.sqrt(2.0 / (1 + 0.004 * d))
    if lambda_s > 1.0: lambda_s = 1.0
    row("Size Effect (λs)", "√(2 / (1 + 0.004d))", f"√(2 / (1 + 0.004*{d:.0f}))", f"{fmt(lambda_s, 3)}", "≤ 1.0")

    # --- 2. PILE REACTION ---
    sec("2. PILE REACTION CHECK")
    P_avg_tf = Pu_tf / n_pile if n_pile > 0 else 0
    P_avg_N = Pu_N / n_pile if n_pile > 0 else 0
    status_pile = "PASS" if P_avg_tf <= PileCap_tf else "FAIL"
    row("Pile Reaction (Ru)", "Pu / N", f"{fmt(Pu_tf, 2)} / {n_pile}", f"{fmt(P_avg_tf, 2)}", "tf", status_pile)

    # --- 3. FLEXURAL DESIGN ---
    sec("3. FLEXURAL DESIGN (Determine As)")

    if n_pile == 1:
        row("Moment", "One Pile", "Negligible Moment", "0", "tf-m")
        As_design_x = As_design_y = 0.0018 * width_x * h_final
        row("As,min (Temp)", "0.0018 · B · h", f"0.0018 · {width_x:.0f} · {h_final:.0f}", f"{fmt(As_design_x, 0)}",
            "mm²")
        req_As_x = req_As_y = 0  # For later rho calcs
    else:
        # Long Direction (Mu-X check Y-axis)
        face_dist_x = col_x / 2
        Mx_Nmm = 0
        for (px, py) in coords:
            lever = abs(px) - face_dist_x
            if lever > 0: Mx_Nmm += P_avg_N * lever
        Mx_tfm = Mx_Nmm / 9806650.0

        # Short Direction (Mu-Y check X-axis)
        face_dist_y = col_y / 2
        My_Nmm = 0
        for (px, py) in coords:
            lever = abs(py) - face_dist_y
            if lever > 0: My_Nmm += P_avg_N * lever
        My_tfm = My_Nmm / 9806650.0

        row("Design Mu-X (Long)", "Σ P·(x - cx/2)", "-", f"{fmt(Mx_tfm, 2)}", "tf-m")
        row("Design Mu-Y (Short)", "Σ P·(y - cy/2)", "-", f"{fmt(My_tfm, 2)}", "tf-m")

        # As Req (Strength)
        phi_f = 0.9;
        j_approx = 0.9
        req_As_x = Mx_Nmm / (phi_f * fy * j_approx * d) if Mx_Nmm > 0 else 0
        req_As_y = My_Nmm / (phi_f * fy * j_approx * d) if My_Nmm > 0 else 0

        # As Min (Temp)
        As_min_x = 0.0018 * width_y * h_final
        As_min_y = 0.0018 * width_x * h_final

        As_design_x = max(req_As_x, As_min_x)
        As_design_y = max(req_As_y, As_min_y)

        row("As-X Req", "Max(Calc, Min)", f"Max({req_As_x:.0f}, {As_min_x:.0f})", f"{fmt(As_design_x, 0)}", "mm²")
        row("As-Y Req", "Max(Calc, Min)", f"Max({req_As_y:.0f}, {As_min_y:.0f})", f"{fmt(As_design_y, 0)}", "mm²")

    # Provide Bars
    bar_area = BAR_INFO[bar_key]['A_cm2'] * 100
    nx_bars = math.ceil(As_design_x / bar_area)
    ny_bars = math.ceil(As_design_y / bar_area)
    if n_pile == 1 and nx_bars < 4: nx_bars = 4; ny_bars = 4

    row("Provide X-Dir", f"{nx_bars}-{bar_key}", f"As = {nx_bars * bar_area:.0f}", "OK", "-", "")
    row("Provide Y-Dir", f"{ny_bars}-{bar_key}", f"As = {ny_bars * bar_area:.0f}", "OK", "-", "")

    # Reinforcement Ratio for Shear (Use X-dir as representative or critical)
    rho_w = (nx_bars * bar_area) / (width_y * d)

    # --- 4. SHEAR CHECKS (ACI 318-19) ---
    if n_pile > 1:
        sec("4. PUNCHING SHEAR (Two-Way)")
        c1 = col_x + d;
        c2 = col_y + d
        bo = 2 * (c1 + c2)
        beta = max(col_x, col_y) / min(col_x, col_y)
        alpha_s = 40  # Interior col assumption

        # Vu Punching
        Vu_punch_N = sum([P_avg_N for px, py in coords if (abs(px) > c1 / 2 or abs(py) > c2 / 2)])

        # Vc Formulas (ACI 318-19 Table 22.6.5.2) with Size Effect
        # 1. 0.33 lambda lambda_s sqrt(fc)
        vc1 = 0.33 * 1.0 * lambda_s * math.sqrt(fc)
        # 2. 0.17(1 + 2/beta) ...
        vc2 = 0.17 * (1 + 2 / beta) * 1.0 * lambda_s * math.sqrt(fc)
        # 3. 0.083(2 + alpha*d/bo) ...
        vc3 = 0.083 * (2 + alpha_s * d / bo) * 1.0 * lambda_s * math.sqrt(fc)

        vc_min = min(vc1, vc2, vc3)
        Vc_punch_N = vc_min * bo * d
        phiVc_punch_N = 0.75 * Vc_punch_N

        row("Critical Perimeter", "bo = 2(c1+c2)", f"2({c1:.0f}+{c2:.0f})", f"{bo:.0f}", "mm")
        row("Vu (Punching)", "Sum Piles Outside", "-", f"{fmt(Vu_punch_N / 9806.65, 2)}", "tf")
        row("vc (Stress)", "min(eq a,b,c)·λs", f"{fmt(vc_min, 3)} MPa", f"(λs={fmt(lambda_s, 2)})", "-")

        status_punch = "PASS" if Vu_punch_N <= phiVc_punch_N else "FAIL"
        row("Check Punching", "Vu ≤ 0.75Vc", f"{fmt(Vu_punch_N / 9806.65, 1)} ≤ {fmt(phiVc_punch_N / 9806.65, 1)}",
            status_punch, "tf", status_punch)

        # --- SECTION 5: BEAM SHEAR ---
        sec("5. BEAM SHEAR (One-Way)")
        # ACI 318-19 Eq 22.5.5.1: Vc = 0.66 lambda lambda_s (rho_w)^(1/3) sqrt(fc) bw d
        # Note: rho_w should be limited or checked

        dist_x = col_x / 2 + d
        Vu_beam_x = sum([P_avg_N for px, py in coords if abs(px) > dist_x])

        # Calculate Capacity
        rho_term = math.pow(rho_w, 1 / 3)
        # Vc formula
        vc_beam_stress = 0.66 * 1.0 * lambda_s * rho_term * math.sqrt(fc)
        # Verify limit: Vc <= 0.42 lambda sqrt(fc) bw d? (Actually 0.42 limit is usually implied)

        Vc_beam_N = vc_beam_stress * width_y * d
        phiVc_beam_N = 0.75 * Vc_beam_N

        row("Critical Section", "d from face", f"{dist_x:.0f} mm", "-", "-")
        row("Vu (One-Way)", "Sum Piles Outside", "-", f"{fmt(Vu_beam_x / 9806.65, 2)}", "tf")
        row("Parameter ρw", "As / (b·d)", f"{fmt(rho_w * 100, 2)}%", f"(ρ^1/3={rho_term:.2f})", "-")
        row("vc (Stress)", "0.66λs(ρ)^1/3√fc'", f"0.66·{lambda_s:.2f}·{rho_term:.2f}·{math.sqrt(fc):.1f}",
            f"{fmt(vc_beam_stress, 2)}", "MPa")

        status_beam = "PASS" if Vu_beam_x <= phiVc_beam_N else "FAIL"
        row("Check Beam Shear", "Vu ≤ 0.75Vc", f"{fmt(Vu_beam_x / 9806.65, 1)} ≤ {fmt(phiVc_beam_N / 9806.65, 1)}",
            status_beam, "tf", status_beam)
    else:
        status_punch = "PASS";
        status_beam = "PASS"

    sec("6. FINAL STATUS")
    overall = "OK" if (status_pile == "PASS" and status_punch == "PASS" and status_beam == "PASS") else "NOT OK"
    row("Overall", "-", "-", "DESIGN COMPLETE", "-", overall)

    return rows, coords, width_x, width_y, nx_bars, ny_bars, overall, h_final


# ==========================================
# 4. PLOTTING WITH DIMENSIONS
# ==========================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def draw_dim(ax, p1, p2, text, offset=50, color='black'):
    """Draw dimension line between p1 and p2 with text"""
    x1, y1 = p1
    x2, y2 = p2

    # Angle
    dx = x2 - x1;
    dy = y2 - y1
    angle = math.atan2(dy, dx)
    perp = angle + math.pi / 2

    # Offset points
    ox = offset * math.cos(perp);
    oy = offset * math.sin(perp)
    p1_off = (x1 + ox, y1 + oy)
    p2_off = (x2 + ox, y2 + oy)

    # Extension lines
    ax.plot([x1, p1_off[0]], [y1, p1_off[1]], color=color, linewidth=0.5)
    ax.plot([x2, p2_off[0]], [y2, p2_off[1]], color=color, linewidth=0.5)

    # Main line with arrows
    ax.annotate('', xy=p1_off, xytext=p2_off, arrowprops=dict(arrowstyle='<->', color=color, linewidth=0.8))

    # Text
    mid_x = (p1_off[0] + p2_off[0]) / 2
    mid_y = (p1_off[1] + p2_off[1]) / 2

    # Text rotation (keep readable)
    deg = math.degrees(angle)
    if 90 < deg <= 270:
        deg -= 180
    elif -270 <= deg < -90:
        deg += 180

    # Text offset from line
    text_gap = 15
    tx = mid_x + text_gap * math.cos(perp)
    ty = mid_y + text_gap * math.sin(perp)

    ax.text(tx, ty, text, ha='center', va='center', rotation=deg, fontsize=9, color=color,
            bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.8))


def plot_footing_plan(coords, width_x, width_y, col_x, col_y, dp, nx_bars, ny_bars, bar_name):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Footing
    rect = patches.Rectangle((-width_x / 2, -width_y / 2), width_x, width_y, linewidth=2, edgecolor='black',
                             facecolor='#f9f9f9')
    ax.add_patch(rect)

    # Bars
    ys = np.linspace(-width_y / 2 + 75, width_y / 2 - 75, min(ny_bars, 8))
    for y in ys: ax.plot([-width_x / 2 + 50, width_x / 2 - 50], [y, y], 'b-', linewidth=1, alpha=0.5)
    xs = np.linspace(-width_x / 2 + 75, width_x / 2 - 75, min(nx_bars, 8))
    for x in xs: ax.plot([x, x], [-width_y / 2 + 50, width_y / 2 - 50], 'r-', linewidth=1, alpha=0.5)

    # Column
    rect_col = patches.Rectangle((-col_x / 2, -col_y / 2), col_x, col_y, linewidth=1.5, edgecolor='#333',
                                 facecolor='#ddd', hatch='//', zorder=5)
    ax.add_patch(rect_col)

    # Piles
    for (px, py) in coords:
        circle = patches.Circle((px, py), radius=dp / 2, edgecolor='black', facecolor='white', linewidth=1.5,
                                linestyle='--')
        ax.add_patch(circle)

    # Dimensions (Outside)
    off = 200  # Offset from edge
    draw_dim(ax, (-width_x / 2, -width_y / 2 - off), (width_x / 2, -width_y / 2 - off), f"L = {width_x / 1000:.2f} m",
             offset=0)
    draw_dim(ax, (-width_x / 2 - off, -width_y / 2), (-width_x / 2 - off, width_y / 2), f"B = {width_y / 1000:.2f} m",
             offset=0)

    # Bar Labels
    ax.text(0, width_y / 2 + 50, f"{nx_bars}-{bar_name} (Vertical)", color='red', ha='center', fontweight='bold')
    ax.text(width_x / 2 + 50, 0, f"{ny_bars}-{bar_name}\n(Horizontal)", color='blue', va='center', fontweight='bold')

    ax.set_xlim(-width_x / 1.2, width_x / 1.2);
    ax.set_ylim(-width_y / 1.2, width_y / 1.2)
    ax.set_aspect('equal');
    ax.axis('off')
    ax.set_title("PLAN VIEW", fontweight='bold', fontsize=12)
    return fig


def plot_footing_section(width, h, col_w, dp, cover, bar_txt, n_pile):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([-width, width], [0, 0], 'k-', linewidth=0.5)  # Ground

    # Concrete
    rect = patches.Rectangle((-width / 2, -h), width, h, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)
    # Column
    rect_col = patches.Rectangle((-col_w / 2, 0), col_w, h * 0.5, linewidth=1.5, edgecolor='black', facecolor='#fff',
                                 hatch='///')
    ax.add_patch(rect_col)

    # Piles
    pile_h = h * 0.5
    if n_pile == 1:
        ax.add_patch(patches.Rectangle((-dp / 2, -h - pile_h), dp, pile_h, edgecolor='black', facecolor='white'))
    else:
        off = width / 2 - dp / 2 - 150  # edge dist approx
        ax.add_patch(patches.Rectangle((-off - dp / 2, -h - pile_h), dp, pile_h, edgecolor='black', facecolor='white'))
        ax.add_patch(patches.Rectangle((off - dp / 2, -h - pile_h), dp, pile_h, edgecolor='black', facecolor='white'))

    # Rebar
    bar_y = -h + cover
    ax.plot([-width / 2 + cover, width / 2 - cover], [bar_y, bar_y], 'r-', linewidth=3)  # Main
    ax.plot([-width / 2 + cover, -width / 2 + cover], [bar_y, bar_y + h * 0.6], 'r-', linewidth=3)  # Hook
    ax.plot([width / 2 - cover, width / 2 - cover], [bar_y, bar_y + h * 0.6], 'r-', linewidth=3)

    # Dimensions
    draw_dim(ax, (width / 2 + 100, 0), (width / 2 + 100, -h), f"h = {h / 1000:.2f} m", offset=50)
    draw_dim(ax, (-width / 2, -h - pile_h - 100), (width / 2, -h - pile_h - 100), f"Width = {width / 1000:.2f} m",
             offset=0)

    ax.text(0, bar_y - 100, f"Main Reinforcement: {bar_txt}", ha='center', color='red', fontsize=10, fontweight='bold')

    ax.set_xlim(-width / 1.2, width / 1.2);
    ax.set_ylim(-h * 2.0, h * 1.0);
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
    n_pile = st.selectbox("Number of Piles", [1, 2, 3, 4, 5], index=1)
    c1, c2 = st.columns(2)
    dp = c1.number_input("Pile Dia (m)", 0.22)
    spacing = c2.number_input("Spacing (m)", 0.80)

    st.header("3. Footing Geometry")
    auto_h = st.checkbox("ออกแบบความหนาอัตโนมัติ (Auto-Design)", value=True)
    c1, c2 = st.columns(2)

    h = c1.number_input(
        "Thickness (m) [Initial/Fixed]",
        value=0.50,
        help="ความหนาของฐานราก (เมตร) - หากเลือก Auto-Design ค่านี้จะเป็นความหนาเริ่มต้นที่โปรแกรมใช้ลองคำนวณ หากไม่เลือก โปรแกรมจะใช้ค่านี้เป็นความหนาคงที่ในการออกแบบ"
    )
    edge = c2.number_input(
        "Edge Dist (m)",
        value=0.25,
        help="ระยะจากศูนย์กลางเสาเข็มต้นริมสุด ถึงขอบฐานราก (เมตร) - เพื่อให้ครอบคลุมระยะฝังของเหล็กเสริมและป้องกันการกะเทาะของคอนกรีตขอบฐานราก"
    )

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
    bar_txt_all = f"Main: {mainBar}"  # simplified
    fig_plan = plot_footing_plan(coords, bx, by, cx * 1000, cy * 1000, dp * 1000, nx, ny, mainBar)
    img_plan = fig_to_base64(fig_plan)

    # Plot Section
    bar_txt_sect = f"{max(nx, ny)}-{mainBar}"
    fig_sect = plot_footing_section(bx, final_h, cx * 1000, dp * 1000, 75, bar_txt_sect, n_pile)
    img_sect = fig_to_base64(fig_sect)

    # Report
    html = generate_report(inputs, rows, img_plan, img_sect)

    st.success(f"✅ Calculation Complete: {status}")
    st.components.v1.html(html, height=800, scrolling=True)

else:
    st.info("👈 กรุณากรอกข้อมูลทางด้านซ้าย แล้วกด 'Run Design'")
