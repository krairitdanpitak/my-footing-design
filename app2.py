# ==========================================
# 6. MAIN UI (เฉพาะส่วน sidebar ที่มีการปรับปรุง)
# ==========================================
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

    # เพิ่ม Tooltip (help) ภาษาไทยตรงนี้ครับ
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